import os
import logging
import re
import asyncio 
import requests 
import json 
import io 
import base64 
import html
import aiohttp
from aiohttp import web
from telegram import Update, ChatPermissions, Poll
from telegram.ext import (
    Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackContext, PollHandler, PollAnswerHandler, ChatMemberHandler
)
from datetime import datetime, timedelta, timezone
# Import firestore package directly for SERVER_TIMESTAMP
from firebase_admin import initialize_app, firestore, credentials 
import functools
import random
# Try to import pypdf for PDF text extraction
try:
    import pypdf
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False

# --- Configuration and Initialization ---

# 1. Environment Variables (Required for security and hosting)
BOT_TOKEN = os.environ.get("BOT_TOKEN")
OWNER_ID = int(os.environ.get("OWNER_ID", 0)) # Your personal Telegram User ID
WEBHOOK_URL = os.environ.get("WEBHOOK_URL")
PORT = int(os.environ.get("PORT", 8080))
frozen_chats = {}
freeze_message = (
    "❄The bot is frozen🥶 in this chat by the bot owner because of its misuse💢🗯.\n"
    "The GC owner or the user(for private chat) will have to contact owner ke bhaiya @Tota_ton to resume the bot.🥱"
)
# Global dictionaries for quiz ranking system
# Maps chat_id -> { 'scores': { user_id: { 'name': str, 'score': int, 'time': float } }, 'poll_ids': list }
quiz_sessions = {}

# Maps poll_id -> { 'chat_id': int, 'correct_option': int, 'start_time': datetime }
active_polls = {}

# Maps group_id -> asyncio.Task (The background supervisor task)
autonomous_tasks = {}

MOOD_REACTIONS = {
    "question": ["🤔", "❓", "🧠"],
    "joke": ["😂", "🤣", "😆"],
    "hype": ["🔥", "💯", "🚀", "👏"],
    "love": ["❤️", "😍", "🥰"],
    "sad": ["😢", "💔", "🫂"],
    "angry": ["😡", "🤬"],
    "agree": ["👍", "🤝"],
    "neutral": ["👍", "🙂", "😎"]
}


# Gemini Configuration (Updated for Gemini 2.5 Flash)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") # Use this environment variable now
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent"
GEMINI_MODEL = "gemini-2.5-flash-preview-09-2025"

# OPEN weather API
OPENWEATHER_API_KEY = "71f9cff3b805051c09d0c73f0ff81424"

# 2. Logging Setup
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=os.getenv("LOG_LEVEL", "WARNING")
)

logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# 3. Firestore Database Setup
db = None
try:
    # Load Firestore Service Account Key from environment variable
    firebase_config_json = os.environ.get("FIREBASE_CREDENTIALS")
    if firebase_config_json:
        import json
        service_account_info = json.loads(firebase_config_json)
        cred = credentials.Certificate(service_account_info)
        initialize_app(cred)
        db = firestore.client()
        logger.info("Firestore connected successfully.")
    else:
        logger.error("FIREBASE_CREDENTIALS environment variable not set. Persistence will fail.")
except Exception as e:
    logger.error(f"Error initializing Firestore: {e}")

# --- Global Bot State Management (For Owner Control Panel) ---

def get_welcome_image_file_id():
    try:
        doc = db.collection("bot_config").document("welcome_image").get()
        if doc.exists:
            return doc.to_dict().get("file_id")
    except:
        pass
    return None


# Fetches the global bot state (used for /freeze and /resume)
async def get_bot_state() -> dict:
    """Fetches the global bot settings document from Firestore."""
    if not db:
        return {"is_frozen": False}
    try:
        doc_ref = db.collection("global_settings").document("bot_state")
        doc = await asyncio.to_thread(doc_ref.get)
        return doc.to_dict() or {"is_frozen": False}
    except Exception as e:
        logger.error(f"Error fetching bot state: {e}")
        return {"is_frozen": False}

# Decorator to check if the bot is frozen before executing a command/handler
def check_frozen(func):
    @functools.wraps(func)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs):
        state = await get_bot_state()
        if state.get("is_frozen", False):
            # Only allow owner to run commands when frozen
            if update.effective_user and update.effective_user.id == OWNER_ID:
                return await func(update, context, *args, **kwargs)
            
            # Send a silent notification or a non-command message for frozen state
            if update.message and (update.message.text.startswith('/') or update.message.caption_startswith('/')):
                 await update.message.reply_text("❄️ The bot is currently frozen by the owner and cannot respond to commands.")
            return
        
        return await func(update, context, *args, **kwargs)
    return wrapper

# Decorator to skip frozen check for owner commands that are supposed to work while frozen
def owner_override(func):
    @functools.wraps(func)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs):
        if is_owner(update.effective_user.id):
            return await func(update, context, *args, **kwargs)
        
        state = await get_bot_state()
        if state.get("is_frozen", False):
            await update.message.reply_text("❄️ The bot is currently frozen by the owner.")
            return
            
        return await func(update, context, *args, **kwargs)
    return wrapper
# ============================
# OWNER: Freeze a particular chat
# ============================
async def deny(update: Update, context: CallbackContext):
    if update.effective_user.id != OWNER_ID:
        return
    
    if len(context.args) < 2:
        return await update.message.reply_text(
            "Usage: /deny <chat_id> <reason>"
        )
    
    chat_id = context.args[0]
    reason = " ".join(context.args[1:])
    
    try:
        chat_id = int(chat_id)
    except:
        return await update.message.reply_text("Invalid chat ID.")
    
    # Freeze
    frozen_chats[chat_id] = reason
    
    await update.message.reply_text(
        f"❄ Bot frozen in chat **{chat_id}**.\nReason: {reason}",
        parse_mode="Markdown"
    )


# ============================
# OWNER: Allow a frozen chat
# ============================
async def allow(update: Update, context: CallbackContext):
    if update.effective_user.id != OWNER_ID:
        return
    
    if len(context.args) < 1:
        return await update.message.reply_text(
            "Usage: /allow <chat_id>"
        )
    
    chat_id = context.args[0]
    
    try:
        chat_id = int(chat_id)
    except:
        return await update.message.reply_text("Invalid chat ID.")
    
    if chat_id in frozen_chats:
        frozen_chats.pop(chat_id)
        await update.message.reply_text(
            f"✅ Bot resumed in chat **{chat_id}**.",
            parse_mode="Markdown"
        )
    else:
        await update.message.reply_text(
            f"Chat {chat_id} is not frozen."
        )


def detect_mood(text: str) -> str:
    t = text.lower()

    # ❓ Question
    if "?" in t or t.startswith(("why", "how", "what", "when", "where", "can", "is")):
        return "question"

    # 😂 Joke / fun
    if any(w in t for w in ["lol", "lmao", "haha", "😂", "🤣", "funny"]):
        return "joke"

    # 🔥 Hype / achievement
    if any(w in t for w in ["wow", "awesome", "great", "op", "legend", "fire", "lit"]):
        return "hype"

    # ❤️ Love / appreciation
    if any(w in t for w in ["love", "thanks", "thank you", "❤️", "😍"]):
        return "love"

    # 😢 Sad
    if any(w in t for w in ["sad", "cry", "depressed", "tired", "hurt"]):
        return "sad"

    # 😡 Angry
    if any(w in t for w in ["angry", "hate", "stupid", "idiot", "mad"]):
        return "angry"

    # 👍 Agreement
    if any(w in t for w in ["yes", "true", "right", "agreed", "correct"]):
        return "agree"

    return "neutral"

async def save_allowed_reactions(group_id: int, reactions: list[str]):
    if not db:
        return
    try:
        ref = db.collection("group_settings").document(str(group_id))
        await asyncio.to_thread(
            ref.set,
            {"allowed_reactions": reactions},
            merge=True
        )
    except Exception as e:
        logger.error(f"Failed to save reactions: {e}")


async def is_auto_reaction_enabled(group_id: int) -> bool:
    if not db:
        return False
    try:
        ref = db.collection("group_settings").document(str(group_id))
        doc = await asyncio.to_thread(ref.get)
        return doc.to_dict().get("auto_reaction", False) if doc.exists else False
    except Exception as e:
        logger.error(f"Auto reaction check failed: {e}")
        return False


# --- AI Personality Mode Management ---

AI_PERSONALITIES = {
    "friendly": "You are extremely warm, approachable, and encouraging. Your explanations are gentle and you always cheer the user on. Use lots of positive affirmations and happy emojis.",
    "strict": "You are a serious, no-nonsense mentor. Your explanations are concise and highly focused. Demand respect and discourage any off-topic chatter. Use formal language and stern emojis.",
    "sarcastic": "You are highly intelligent but incredibly snarky and dry-witted. Answer all questions correctly but with a thick layer of sarcasm and light mockery. Use rolling-eye or confused emojis.",
    "big brother style": "You are a helpful but playfully dominant big brother. You offer simplified, easy-to-digest advice and constantly check on the user's focus and well-being. Use casual language and brotherly advice.",
    "teacher/neet mentor style": "You are a professional NEET preparation mentor. Your language is formal, highly technical, and uses NCERT/NEET-specific terminology. Your goal is to prepare the student for the exam environment. Maintain an encouraging but professional demeanor."
}

async def get_ai_personality_instruction(group_id: int) -> str:
    """Fetches the current AI personality mode and returns the corresponding instruction text."""
    if not db:
        return "" # Default to base instruction
    try:
        ref = db.collection("group_settings").document(str(group_id))
        doc = await asyncio.to_thread(ref.get)
        data = doc.to_dict() or {}
        mode = data.get("ai_personality_mode", "default").lower()
        return AI_PERSONALITIES.get(mode, "")
    except Exception as e:
        logger.error(f"Error fetching AI personality: {e}")
        return ""

@check_frozen
async def set_ai_personality(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat.id in frozen_chats:
        return await update.message.reply_text(freeze_message)

    """Sets the AI personality mode for the group."""
    if not await check_admin(update, context): return
    if not db:
        await update.message.reply_text("Database not available.")
        return

    if not context.args:
        modes = ", ".join(AI_PERSONALITIES.keys())
        await update.message.reply_text(
            f"Usage: `/set_personality <mode>`\nAvailable modes: `default`, {modes}",
            parse_mode="Markdown"
        )
        return

    mode = context.args[0].lower().strip()
    group_id = update.effective_chat.id
    
    # Check if the mode is valid or 'default'
    if mode != "default" and mode not in AI_PERSONALITIES:
        modes = ", ".join(AI_PERSONALITIES.keys())
        await update.message.reply_text(f"Invalid mode **'{mode}'**. Choose from: `default`, {modes}", parse_mode="Markdown")
        return

    ref = db.collection("group_settings").document(str(group_id))
    try:
        if mode == "default":
            await asyncio.to_thread(ref.update, {"ai_personality_mode": firestore.DELETE_FIELD})
            await update.message.reply_text("AI personality successfully **RESET** to default (NEET study helper).", parse_mode="Markdown")
        else:
            await asyncio.to_thread(ref.set, {"ai_personality_mode": mode}, merge=True)
            await update.message.reply_text(f"AI personality successfully set to **{mode.upper()}** mode!", parse_mode="Markdown")
    except Exception as e:
        logger.error(f"Error setting AI personality: {e}")
        await update.message.reply_text(f"Failed to set AI personality due to a database error: {e}")


# --- AI Chatbot Functionality (Updated for Gemini Multimodal & Personality) ---

async def get_ai_response(prompt: str, group_id: int, image_base64: str = None, mime_type: str = None) -> str:
    """
    Synchronously calls the Gemini API, optionally with an image, and extracts the response content.
    The model used is gemini-2.5-flash-preview-09-2025 which supports multimodal input.
    """
    
    personality_instruction = await get_ai_personality_instruction(group_id)

    # We define the API call logic inside a synchronous function
    def sync_api_call(prompt: str, personality_instruction: str, image_base64: str = None, mime_type: str = None):
        if not GEMINI_API_KEY:
            return "❌ Error: The Gemini API key is not configured. Please set the GEMINI_API_KEY environment variable."
        
        headers = {
            "Content-Type": "application/json"
        }
        
        # Base System instruction to guide the Gemini model's behavior (UNCHANGED)
        system_instruction_text_base = (
            r"You are a helpful, Telegram bot to help in study related doubts running inside a group chat or in a private chat. You are created by Saniya🐣, she is your owner. Your telegram username is @Chhoti_don_bot. You are here for chit-chat with users. Try to write text normally because telegram can't understand markdown formatting. Use some emojis in case if you need to provide special emphasis on a part of text. Be very careful about not giving the user about any medical, financial or family related adivce. Don't give the user anything illegal. Whenever someone asks you anything in hindi, then don't reply using hindi letters. Instead use hinglish. Write hindi words in english. Don't give any idea about your language model to the user and don't give them any codes for making a bot like you. Don't give them your API key. Don't give reply to any sexually abusive things or slang words rather than their biology related doubts. At last don't forget your identity; your name is Sîdæ. You don't have to mention your name or the name of your owner everytime. Just mention it whenever they ask you about your identity; like- who are you? or who is your owner or creator. Just mention the name of your owner in introduction. Also you don't have to write your knowledge base in every message. Just write them when they ask you. You can help users by translating something off-topic  that they don't understand.  Make the message understandable and readable by the users. Always write math in plain text, not LaTeX. Do not use $ or \frac or \mu or any TeX formatting."
            "If an image is provided, focus your analysis on that image and the user's question about it. "
        )
        
        # ADD Personality Instruction
        full_system_instruction_text = system_instruction_text_base
        if personality_instruction:
            full_system_instruction_text += f"\n\n--- CURRENT PERSONALITY INSTRUCTION ---\n{personality_instruction}"
            
        system_instruction = {"parts": [{"text": full_system_instruction_text}]}
        
        # Construct contents array: [Image Part (Optional), Text Part]
        contents_parts = []
        
        # 1. Add Image if available
        if image_base64 and mime_type:
            contents_parts.append({
                "inlineData": {
                    "mimeType": mime_type,
                    "data": image_base64
                }
            })

        # 2. Add Text Prompt
        contents_parts.append({"text": prompt})
        
        # Define the request payload for Gemini
        payload = {
            "contents": [{"parts": contents_parts}],
            "systemInstruction": system_instruction,
        }
        
        api_url_with_key = f"{GEMINI_API_URL}?key={GEMINI_API_KEY}"
        
        try:
            # Make the synchronous HTTP request
            response = requests.post(api_url_with_key, headers=headers, json=payload, timeout=30)
            response.raise_for_status() # Raise HTTP errors (4xx or 5xx)
            data = response.json()
            
            # Extract the content from the Gemini API response structure
            text = data.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text')
            
            if text:
                return text.strip()
            else:
                safety_ratings = data.get('candidates', [[]])[0].get('safetyRatings', [])
                if safety_ratings:
                    return "The AI response was blocked due to content policy. Please try rephrasing your request."
                return "The AI returned an empty or unrecognized response."

        except requests.exceptions.HTTPError as e:
            logger.error(f"Gemini HTTP Error: {e.response.status_code} - {e.response.text}")
            return f"""👋 Welcome to the group!
We’re glad to have you here. Feel free to introduce yourself and join the discussions! 😄

📌 Some quick tips:

Be respectful and kind to everyone.

No spam or unrelated promotions.

Check the pinned messages for group rules and important info.

Let’s make this space fun and helpful for everyone! 🚀"""
        except requests.exceptions.RequestException as e:
            logger.error(f"Gemini Request Error: {e}")
            return "⚠️ Please try again later."
        except Exception as e:
            logger.error(f"General AI Error: {e}")
            return "An unexpected error occurred while processing the AI request."

    # Use asyncio.to_thread to run the synchronous API call in a thread pool
    return await asyncio.to_thread(sync_api_call, prompt, personality_instruction, image_base64, mime_type)


@check_frozen
async def ask_ai(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat.id in frozen_chats:
        return await update.message.reply_text(freeze_message)

    """
    Handles the /ask command, mentions, replies to the bot, and QUIZ SOLVING.
    """
    
    if not GEMINI_API_KEY:
        await update.message.reply_text("The Gemini AI service is not configured. Please set the GEMINI_API_KEY environment variable.")
        return

    # --- 1. Determine Prompt & Intent ---
    message = update.message
    reply = message.reply_to_message
    
    # Get raw text (handle captions if image is sent)
    text_body = message.text or message.caption or ""
    
    # Initialize prompt
    prompt = ""
    
    # Check if this is a Command, a Mention, or a Reply
    is_command = text_body.startswith("/ask")
    
    # Logic to determine if we should process this message
    should_process = False
    
    if is_command:
        # It's a command: /ask <query>
        prompt = " ".join(context.args)
        should_process = True
    else:
        # It's a Mention or a Reply
        bot_username = context.bot.username
        
        # Check if it's a reply to the bot
        is_reply_to_bot = reply and reply.from_user.id == context.bot.id
        
        # Check if the bot is mentioned
        is_mentioned = bot_username and f"@{bot_username}" in text_body
        
        if is_reply_to_bot or is_mentioned:
            should_process = True
            # Clean the prompt: Remove the @username
            if bot_username:
                prompt = text_body.replace(f"@{bot_username}", "").strip()
            else:
                prompt = text_body.strip()
    
    # If it's just a random message not directed at the bot, ignore it
    if not should_process:
        return

    image_base64 = None
    mime_type = None
    group_id = update.effective_chat.id
    
    # === QUIZ SOLVING LOGIC (Priority 1) ===
    if reply and reply.poll and reply.poll.type == Poll.QUIZ:
        # ... [KEEP YOUR EXISTING QUIZ LOGIC HERE UNCHANGED] ...
        poll_question = reply.poll.question
        poll_options = reply.poll.options
        option_list = []
        for i, option in enumerate(poll_options):
            option_list.append(f"{i+1}. {option.text}")
        options_text = "\n".join(option_list)
        ai_prompt = (
            "The following is a multiple-choice quiz question. "
            "Read the question and options carefully. Based only on factual knowledge, "
            "determine which option is the single correct answer. "
            "Your response MUST be ONLY the number corresponding to the correct option "
            "(e.g., '1' for the first option, '2' for the second, etc.). "
            "Do not include any other text, explanation, or punctuation."
            f"\n\nQuestion: {poll_question}\n\nOptions:\n{options_text}"
        )
        await context.bot.send_chat_action(update.effective_chat.id, "typing")
        raw_response = await get_ai_response(ai_prompt, group_id, None, None)
        try:
            match = re.search(r'\d+', raw_response.strip())
            if match:
                chosen_index = int(match.group(0)) - 1
                if 0 <= chosen_index < len(poll_options):
                    chosen_option_text = poll_options[chosen_index].text
                    response_text = (
                        f"🤖 <b>Quiz Answer:</b>\n"
                        f"My best guess for the question: <b>{html.escape(poll_question)}</b> is:\n"
                        f"<b>Option {chosen_index + 1}:</b> <code>{html.escape(chosen_option_text)}</code>"
                    )
                else:
                    response_text = f"🤖 I failed to select a valid option number."
            else:
                 response_text = f"🤖 I couldn't extract an option number."
        except Exception as e:
            response_text = f"🤖 Error: {html.escape(str(e))}"
        await update.message.reply_text(response_text, parse_mode="HTML")
        return 
    # === END QUIZ SOLVING LOGIC ===
    
    # 2. Check for reply message for both image and text context
    if reply:
        file_obj = None
        
        # A. Image Multimodal Handling (Photo or Image Document)
        if reply.photo:
            file_obj = reply.photo[-1] 
            mime_type = "image/jpeg" 
        elif reply.document and reply.document.mime_type and reply.document.mime_type.startswith('image/'):
            file_obj = reply.document
            mime_type = reply.document.mime_type
        
        if file_obj:
            try:
                telegram_file = await file_obj.get_file()
                buffer = io.BytesIO()
                await telegram_file.download_to_memory(out=buffer)
                buffer.seek(0)
                file_bytes = buffer.read()
                image_base64 = base64.b64encode(file_bytes).decode('utf-8')
                
                # Determine prompt (use cleaned prompt from above or caption)
                if not prompt and reply.caption:
                    prompt = reply.caption
                    
                if not prompt:
                    prompt = "Analyze this image and provide a helpful description."
            except Exception as e:
                logger.error(f"Error handling image: {e}")
                await update.message.reply_text("❌ Error processing the replied image.")
                return

        # B. Text Context Handling
        # If I am replying to the BOT, the bot's message is context, and my message is the prompt.
        if not image_base64 and reply.text:
            context_text = reply.text.strip()
            
            if not prompt and context_text:
                # If I just replied /ask or @Bot to a message without text
                prompt = f"Please elaborate or expand on the following statement: '{context_text}'"
            elif prompt and context_text:
                # My prompt + Context of the message I replied to
                prompt = f"Previous message context: '{context_text}'. User's question about this context: '{prompt}'"
            
    # Fallback/Standard Prompt check
    if not prompt and not image_base64:
        # Only send usage help if it was an explicit command
        if is_command:
            await update.message.reply_text(
                "Please provide a question (e.g., <code>/ask What is X?</code>) or reply to a message containing a question or image.", 
                parse_mode="HTML"
            )
        return

    # Indicate that the bot is processing the request
    await context.bot.send_chat_action(update.effective_chat.id, "typing")
    
    # Get the response from the AI
    response_text = await get_ai_response(prompt, group_id, image_base64, mime_type)
    
    # Send the final response
    await update.message.reply_text(response_text)


async def generate_ai_image(prompt: str):
    """
    Generates an image using the Gemini API and returns raw image bytes.
    """
    if not GEMINI_API_KEY:
        return None, "❌ Gemini API key missing."

    # Use the current image model and the :generateContent endpoint
    IMAGE_MODEL = "gemini-2.5-flash-image"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{IMAGE_MODEL}:generateContent?key={GEMINI_API_KEY}"

    # REMOVED THE INVALID "config" BLOCK
    payload = {
        "contents": [
            {"parts": [{"text": prompt}]}
        ]
    }

    try:
        response = await asyncio.to_thread(requests.post, url, json=payload, timeout=40)
        response.raise_for_status()
        data = response.json()

        # The path to the image data remains the same as in the previous step
        # candidates[0] -> content -> parts[0] -> inlineData -> data
        base64_img = data["candidates"][0]["content"]["parts"][0]["inlineData"]["data"]
        
        image_bytes = base64.b64decode(base64_img)

        return image_bytes, None

    except Exception as e:
        error_detail = response.text if 'response' in locals() and response.text else str(e)
        return None, f"❌ Image generation failed: {error_detail}"

@check_frozen
async def generate_image_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat.id in frozen_chats:
        return await update.message.reply_text(freeze_message)

    prompt = " ".join(context.args)

    if not prompt:
        await update.message.reply_text("❗ Usage: `/img a cute anime cat`", parse_mode="Markdown")
        return

    await context.bot.send_chat_action(update.effective_chat.id, "upload_photo")

    image_bytes, error = await generate_ai_image(prompt)

    if error:
        await update.message.reply_text(error)
        return

    await update.message.reply_photo(photo=image_bytes, caption=f"✨ Generated Image\nPrompt: {prompt}")


# --- Helper Functions (Checks and Database Interactions) ---

def is_owner(user_id: int) -> bool:
    """Checks if the user is the bot owner."""
    return user_id == OWNER_ID

async def check_admin(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    """Checks if the user is an admin or the bot owner."""
    if update.effective_chat.type not in ["group", "supergroup", "channel"]:
        # Allow owner commands outside of groups for testing/private chat control
        if is_owner(update.effective_user.id):
            return True
        await update.message.reply_text("This command only works in groups/channels.")
        return False

    user_id = update.effective_user.id
    if is_owner(user_id):
        return True

    try:
        member = await context.bot.get_chat_member(update.effective_chat.id, user_id)
        if member.status in ["creator", "administrator"]:
            return True
    except Exception as e:
        logger.error(f"Failed to check admin status: {e}")
        await update.message.reply_text("Could not verify admin status. Ensure the bot is an admin.")
        return False

    await update.message.reply_text("You must be an administrator or the bot owner to use this command.")
    return False

async def get_target_user_info(update: Update, context: ContextTypes.DEFAULT_TYPE) -> tuple[int | None, str | None]:
    """
    Retrieves the target user's ID and name based on reply or command arguments (ID or @username).
    Returns (user_id, full_name, or None, error_message).
    """
    # 1. Check for reply (Highest reliability)
    if update.message.reply_to_message:
        user = update.message.reply_to_message.from_user
        return user.id, user.full_name

    # 2. Check for argument (Username or ID)
    if context.args and update.effective_chat.type in ["group", "supergroup"]:
        target_str = context.args[0].strip()
        chat_id = update.effective_chat.id
        
        # A. Attempt to parse as User ID
        if target_str.isdigit():
            user_id = int(target_str)
            try:
                # Try to fetch user info to get the name, but this can fail if they are not in the chat
                member = await context.bot.get_chat_member(chat_id, user_id)
                return user_id, member.user.full_name
            except Exception:
                # If fetching the name fails, we still have the ID
                return user_id, f"User ID: {user_id}"
            
        # B. Attempt to parse as Username
        elif target_str.startswith('@'):
            username = target_str.lstrip('@')
            try:
                # Resolve username to ID using get_chat_member. This only works if the user 
                # is currently a member of the group and has a public username set.
                member = await context.bot.get_chat_member(chat_id, username)
                return member.user.id, member.user.full_name
            except Exception:
                return None, f"⚠️ Error: Could not find user **@{username}** in this chat. Ensure they are a current member or use their User ID."

    # 3. No target found
    return None, "Please reply to a user's message, provide their Telegram User ID (e.g., `123456789`), or their username (e.g., `@username`)."


# --- Reputation System Helper Functions ---

# Assuming 'db = firestore.client()' is initialized elsewhere in your script.

def update_user_reputation_sync(user_id: int, user_name: str, increment: int = 1):
    """
    Synchronous function to update the user's reputation in Firestore.
    Called inside asyncio.to_thread.
    """
    # NOTE: user_id is converted to string for Firestore document ID convention
    user_ref = db.collection('users').document(str(user_id))
    
    # firestore.Increment ensures atomic, non-blocking updates.
    user_ref.set({
        'user_id': user_id,
        # Store the name to display on the leaderboard
        'username': user_name, 
        'reputation': firestore.Increment(increment),
        'last_updated': firestore.SERVER_TIMESTAMP
    }, merge=True)
    
    # After the update, fetch the new score to return it
    user_doc = user_ref.get()
    return user_doc.to_dict().get('reputation', 0)


def get_user_ref(group_id: int, user_id: int):
    """Returns the Firestore document reference for a user in a group (for warnings)."""
    if not db:
        return None
    return db.collection("groups").document(str(group_id)).collection("users").document(str(user_id))

def get_filter_ref(group_id: int, keyword: str):
    """Returns the Firestore document reference for a filter keyword in a group."""
    if not db:
        return None
    # Use lowercase and stripped keyword for document ID for consistency
    safe_keyword = keyword.strip().lower().replace(" ", "_")
    return db.collection("groups").document(str(group_id)).collection("filters").document(safe_keyword)

def get_banned_word_ref(group_id: int, word: str):
    """Returns the Firestore document reference for a banned word."""
    if not db:
        return None
    safe_word = word.strip().lower().replace(" ", "_")
    # Using a subcollection named 'banned_words' inside the group document
    return db.collection("groups").document(str(group_id)).collection("banned_words").document(safe_word)

def get_link_approval_ref(group_id: int, user_id: int):
    """Returns the Firestore document reference for a user's link approval status."""
    if not db:
        return None
    # New collection for link approvals: groups/{group_id}/approved_link_users/{user_id}
    return db.collection("groups").document(str(group_id)).collection("approved_link_users").document(str(user_id))

async def is_link_approved(group_id: int, user_id: int) -> bool:
    """Checks if a user is approved to send links."""
    ref = get_link_approval_ref(group_id, user_id)
    if not ref: return False
    try:
        doc = await asyncio.to_thread(ref.get)
        return doc.to_dict().get("is_approved", False) if doc.exists else False
    except Exception as e:
        logger.error(f"Error checking link approval: {e}")
        return False

async def get_warn_count(group_id: int, user_id: int) -> int:
    """Fetches the current warning count for a user."""
    ref = get_user_ref(group_id, user_id)
    if not ref:
        return 0
    try:
        doc = await asyncio.to_thread(ref.get)
        return doc.to_dict().get("warnings", 0) if doc.exists else 0
    except Exception as e:
        logger.error(f"Error reading warn count: {e}")
        return 0

async def update_warn_count(group_id: int, user_id: int, change: int):
    """Adds or removes warnings for a user."""
    ref = get_user_ref(group_id, user_id)
    if not ref: return
    try:
        current_warnings = await get_warn_count(group_id, user_id)
        new_warnings = max(0, current_warnings + change)
        
        await asyncio.to_thread(ref.set, {"warnings": new_warnings}, merge=True)
        return new_warnings
    except Exception as e:
        logger.error(f"Error updating warn count: {e}")
        return current_warnings


async def update_user_data(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Stores/updates user data. Ensures XP/Reputation fields exist for indexing."""
    if update.effective_user:
        user = update.effective_user
        if user.is_bot:
            return

        user_ref = db.collection('users').document(str(user.id))
        
        # Run DB operations in a thread to avoid blocking the bot
        def sync_user_update():
            doc = user_ref.get()
            if not doc.exists:
                # NEW USER: Set defaults so they appear in Leaderboards
                user_ref.set({
                    'user_id': user.id,
                    'username': user.username,
                    'first_name': user.first_name,
                    'last_name': user.last_name,
                    'xp': 0,          # <--- FIX: Initialize XP
                    'reputation': 0,  # <--- FIX: Initialize Rep
                    'league': 'Bronze',
                    'last_seen': firestore.SERVER_TIMESTAMP,
                })
            else:
                # EXISTING USER: Just update basic info and timestamp
                user_ref.set({
                    'user_id': user.id,
                    'username': user.username,
                    'first_name': user.first_name,
                    'last_name': user.last_name,
                    'last_seen': firestore.SERVER_TIMESTAMP,
                }, merge=True)

        try:
            await asyncio.to_thread(sync_user_update)
        except Exception as e:
            logging.error(f"Error updating user data for {user.id}: {e}")
        
# --- Firestore persistence helpers for live quiz state ---

# --- Firestore persistence helpers (PASTE THIS BLOCK BEFORE START_QUIZ) ---

async def _save_active_poll_to_db(poll_id: str, data: dict):
    """Save active poll metadata to Firestore."""
    if not db:
        return
    try:
        # Convert datetime to native Firestore timestamp or string if needed
        # Firestore client handles python datetime objects automatically
        ref = db.collection("live_active_polls").document(str(poll_id))
        await asyncio.to_thread(ref.set, data, merge=True)
    except Exception as e:
        logger.error(f"Failed to save active_poll {poll_id} to DB: {e}")

async def _load_active_poll_from_db(poll_id: str):
    """Load active poll metadata from Firestore."""
    if not db:
        return None
    try:
        ref = db.collection("live_active_polls").document(str(poll_id))
        doc = await asyncio.to_thread(ref.get)
        if not doc.exists:
            return None
        data = doc.to_dict()
        # Ensure start_time is a datetime object (Firestore returns datetime, JSON returns str)
        if isinstance(data.get('start_time'), str):
             data['start_time'] = datetime.fromisoformat(data['start_time'])
        return data
    except Exception as e:
        logger.error(f"Failed to load active_poll {poll_id} from DB: {e}")
        return None

async def _delete_active_poll_from_db(poll_id: str):
    """Delete active poll from Firestore."""
    if not db:
        return
    try:
        ref = db.collection("live_active_polls").document(str(poll_id))
        await asyncio.to_thread(ref.delete)
    except Exception as e:
        logger.debug(f"Failed to delete active_poll {poll_id} from DB: {e}")

async def _save_quiz_session_to_db(group_id: int, session: dict):
    """Save quiz session with String Keys for scores to prevent DB errors."""
    if not db:
        return
    try:
        # FIX: Convert integer User IDs to Strings for Firestore Map keys
        raw_scores = session.get("scores", {})
        sanitized_scores = {}
        for user_id, stats in raw_scores.items():
            sanitized_scores[str(user_id)] = stats

        doc = {
            "poll_ids": session.get("poll_ids", []),
            "scores": sanitized_scores, 
            "last_updated": firestore.SERVER_TIMESTAMP
        }
        # FIX: Ensure document ID is a string
        ref = db.collection("live_quiz_sessions").document(str(group_id))
        await asyncio.to_thread(ref.set, doc, merge=True)
    except Exception as e:
        logger.error(f"Failed to save quiz_session for {group_id} to DB: {e}")

async def _load_quiz_session_from_db(group_id: int):
    """Load quiz session and convert String Keys back to Integers."""
    if not db:
        return None
    try:
        ref = db.collection("live_quiz_sessions").document(str(group_id))
        doc = await asyncio.to_thread(ref.get)
        if not doc.exists:
            return None
        data = doc.to_dict() or {}
        
        # FIX: Convert keys back to integers so the bot can recognize User IDs
        raw_scores = data.get("scores", {})
        restored_scores = {}
        for user_id_str, stats in raw_scores.items():
            if user_id_str.isdigit():
                restored_scores[int(user_id_str)] = stats
            else:
                restored_scores[user_id_str] = stats

        return {
            "poll_ids": data.get("poll_ids", []),
            "scores": restored_scores
        }
    except Exception as e:
        logger.error(f"Failed to load quiz_session for {group_id} from DB: {e}")
        return None

async def _delete_quiz_session_from_db(group_id: int):
    """Delete quiz session from Firestore."""
    if not db:
        return None
    try:
        ref = db.collection("live_quiz_sessions").document(str(group_id))
        await asyncio.to_thread(ref.delete)
    except Exception as e:
        logger.debug(f"Failed to delete quiz_session {group_id} from DB: {e}")

# --- Utility Commands ---

@check_frozen
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends a welcome message with bot information and tracks the chat for broadcasting."""
    # Start command text (UNCHANGED)
    START_TEXT = (
        "Hey! I'm Sîdæ { your fev Anya Chan }😊 . I am a group management bot made by saniya {urf chhoti don}🌸 , here to help you get around and keep the order in your groups!\n"
        "I have lots of handy features, such as flood control, a warning system, a note keeping system,quiz in group , study related doubt solving , even predetermined replies on certain keywords and many more⚡💫...\n\n" 

        "𝙃𝙚𝙡𝙥𝙛𝙪𝙡 𝙘𝙤𝙢𝙢𝙖𝙣𝙙𝙨:\n"
        "- /start: Starts me! You've probably already used this.\n"
        "- /help: Sends this message; I'll tell you more about myself!\n"

        "All commands can be used with the following: / !"
        "\n\n𝙏𝙝𝙖𝙣𝙠 𝙮𝙤𝙪 🌷"
    )
    image_file_id = get_welcome_image_file_id()

    if image_file_id:
        await update.message.reply_photo(
            photo=image_file_id,
            caption=START_TEXT,
            parse_mode="HTML"
        )

    
    else:
        await update.message.reply_text(
            text=START_TEXT,
            parse_mode="HTML"
        )


    # --- Track chat for broadcast functionality in the global 'broadcast_chats' collection ---
    if db and update.effective_chat.type in ["group", "supergroup", "private"]:
        chat_id = str(update.effective_chat.id)
        chat_ref = db.collection("broadcast_chats").document(chat_id)
        
        chat_data = {
            "chat_id": chat_id,
            "chat_type": update.effective_chat.type,
            "title": update.effective_chat.title or update.effective_user.full_name, 
            "last_active": firestore.SERVER_TIMESTAMP,
        }
        
        try:
            await asyncio.to_thread(chat_ref.set, chat_data, merge=True)
            logger.info(f"Chat {chat_id} added/updated for broadcast list.")
        except Exception as e:
            logger.error(f"Failed to add chat to broadcast list: {e}")


async def bot_added_to_group(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Automatically triggers start logic when the bot is added to a group."""
    
    new_status = update.my_chat_member.new_chat_member.status
    old_status = update.my_chat_member.old_chat_member.status

    # Detect when the bot is newly added
    if old_status in ["kicked", "left"] and new_status in ["member", "administrator"]:
        chat = update.effective_chat

        START_TEXT = (
            "Hey! I'm Sîdæ { your fev Anya Chan }😊 . I am a group management bot made by saniya {urf chhoti don}🌸 , here to help you get around and keep the order in your groups!\n"
            "I have lots of handy features, such as flood control, a warning system, a note keeping system,quiz in group , study related doubt solving , even predetermined replies on certain keywords and many more⚡💫...\n\n" 

            "𝙃𝙚𝙡𝙥𝙛𝙪𝙡 𝙘𝙤𝙢𝙢𝙖𝙣𝙙𝙨:\n"
            "- /start: Starts me! You've probably already used this.\n"
            "- /help: Sends this message; I'll tell you more about myself!\n"

            "All commands can be used with the following: / !"
            "\n\n𝙏𝙝𝙖𝙣𝙠 𝙮𝙤𝙪 🌷"
        )

        # 1. Send the same welcome message as /start
        image_file_id = get_welcome_image_file_id()

        if image_file_id:
            await context.bot.send_photo(
                chat_id=update.effective_chat.id,
                photo=image_file_id,
                caption=START_TEXT,
                parse_mode="HTML"
            )

        else:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=START_TEXT,
                parse_mode="HTML"
            )
        

        # 2. Run the broadcast chat saving logic
        if db:
            chat_ref = db.collection("broadcast_chats").document(str(chat.id))
            chat_data = {
                "chat_id": str(chat.id),
                "chat_type": chat.type,
                "title": chat.title or "Unknown Group",
                "last_active": firestore.SERVER_TIMESTAMP,
            }

            try:
                await asyncio.to_thread(chat_ref.set, chat_data, merge=True)
                logger.info(f"Chat {chat.id} auto-added to broadcast list on bot join.")
            except Exception as e:
                logger.error(f"Failed to auto-add chat to broadcast list: {e}")


async def handle_thanks_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles the /thanks command to give a reputation point."""
    message = update.effective_message
    giver_user = message.from_user
    
    # 1. Check if the command is a reply
    if not message.reply_to_message:
        await message.reply_text(
            "👋 **Reputation Guide:**\n"
            "To thank someone for being helpful and give them a reputation point, "
            "**reply** to their message with the command `/thanks`.\n\n"
            "_Note: You cannot thank yourself or the bot!_"
        )
        return

    # 2. Identify the receiver (the person whose message was replied to)
    receiver_message = message.reply_to_message
    receiver_user = receiver_message.from_user
    receiver_id = receiver_user.id
    giver_id = giver_user.id

    # 3. Validation Checks
    if giver_id == receiver_id:
        await message.reply_text("❌ You can't thank yourself, that's cheating! 😉")
        return
    
    if receiver_user.is_bot:
        await message.reply_text("🤖 Thanks for the kind words! But I'm just a bot, give the reputation to a human who helped you.")
        return
        
    # 4. Perform the reputation update in a separate thread
    receiver_name = receiver_user.first_name + (f" {receiver_user.last_name}" if receiver_user.last_name else "")
    
    try:
        # Use asyncio.to_thread to safely run the synchronous Firestore function
        new_rep_score = await asyncio.to_thread(
            update_user_reputation_sync, 
            receiver_id, 
            receiver_name
        )

        # 5. Confirmation message
        await message.reply_text(
            f"🌟 **Reputation Given!** 🌟\n"
            f"**{receiver_name}** has received a reputation point from **{giver_user.first_name}** for being helpful!\n\n"
            f"📈 **{receiver_name}'s Current Rep Score:** `{new_rep_score}`"
        )
    except Exception as e:
        logging.error(f"Error updating reputation for user {receiver_id}: {e}")
        await message.reply_text("❌ Sorry, I hit an error trying to update the reputation score. Please try again later.")


### **`/repleaderboard` Command**

async def handle_reputation_leaderboard(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Shows the top 10 users by reputation."""
    await update.effective_chat.send_chat_action(action="typing")
    
    try:
        # Define the synchronous fetching logic
        def fetch_leaderboard_sync():
            # Fetch top 10 users by 'reputation' in descending order
            users_ref = db.collection('users').order_by('reputation', direction=firestore.Query.DESCENDING).limit(10)
            return list(users_ref.stream())

        # Run the synchronous fetch in a separate thread
        top_users = await asyncio.to_thread(fetch_leaderboard_sync)

        leaderboard = []
        for i, user_doc in enumerate(top_users):
            data = user_doc.to_dict()
            name = data.get('username', f"User ID: {data.get('user_id')}")
            rep = data.get('reputation', 0)
            
            # Add an emoji based on rank
            if i == 0:
                rank_emoji = "🥇"
            elif i == 1:
                rank_emoji = "🥈"
            elif i == 2:
                rank_emoji = "🥉"
            else:
                rank_emoji = f"{i+1}."
                
            leaderboard.append(f"{rank_emoji} {name} — **{rep}** Rep")
        
        if not leaderboard:
            text = "😔 The reputation leaderboard is empty! Start giving thanks to helpful members with `/thanks`."
        else:
            text = "👑 **Top 10 Reputation Leaderboard** 🏆\n\n" + "\n".join(leaderboard)
            text += "\n\nGive reputation by replying to a helpful message with `/thanks`!"

        await update.effective_message.reply_text(text, parse_mode='Markdown')
        
    except Exception as e:
        logging.error(f"Error fetching leaderboard: {e}")
        await update.effective_message.reply_text("❌ Could not fetch the leaderboard due to an error.")


@check_frozen
async def get_user_id(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat.id in frozen_chats:
        return await update.message.reply_text(freeze_message)

    """Shows the user ID of the sender or a replied user."""
    
    # Use the new helper function to resolve the target
    target_id, target_name_or_error = await get_target_user_info(update, context)

    if target_id:
        user = update.message.reply_to_message.from_user if update.message.reply_to_message else update.effective_user
        await update.message.reply_text(
            f"The Telegram User ID for **{target_name_or_error}** is:\n`{target_id}`\n\nChat ID:\n`{update.effective_chat.id}`",
            parse_mode="Markdown"
        )
    else:
        # If no user found via reply or arguments, show sender's ID
        await update.message.reply_text(
            f"The Telegram User ID for **{update.effective_user.first_name}** is:\n`{update.effective_user.id}`\n\nChat ID:\n`{update.effective_chat.id}`",
            parse_mode="Markdown"
        )

@check_frozen
async def purge_messages(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat.id in frozen_chats:
        return await update.message.reply_text(freeze_message)

    """Deletes all messages starting from the replied message up to the command itself."""
    if not await check_admin(update, context): return
    if not update.message.reply_to_message:
        await update.message.reply_text("Please reply to the message you want to start purging from.")
        return

    chat_id = update.effective_chat.id
    from_id = update.message.reply_to_message.message_id
    to_id = update.message.message_id
    
    deleted_count = 0
    
    # Iterate from the replied message ID up to the /purge command ID
    # Note: Telegram message IDs are sequential within a chat.
    for message_id in range(from_id, to_id + 1):
        try:
            await context.bot.delete_message(chat_id, message_id)
            deleted_count += 1
        except Exception as e:
            # Errors can occur if a message in the range was already deleted or is a service message
            logger.debug(f"Could not delete message {message_id}: {e}")
            pass 

    # Send a small, temporary confirmation (and delete it shortly after)
    try:
        confirmation_msg = await update.message.reply_text(
            f"🗑️ Successfully purged **{deleted_count}** messages.",
            parse_mode="Markdown"
        )
        # Delete the confirmation message after 5 seconds
        # Note: We can't delete the confirmation message using `timeout` in this way, 
        # so we rely on the confirmation being sent. The user can manually delete it.
    except Exception as e:
        logger.warning(f"Failed to send confirmation message: {e}")

# --- New Utility Tool: Group Stats ---

@check_frozen
# ============================
# 📊 GROUP STATS COMMAND
# ============================
async def stats(update: Update, context: CallbackContext):
    if update.effective_chat.id in frozen_chats:
        return await update.message.reply_text(freeze_message)

    chat = update.effective_chat

    try:
        member_count = await chat.get_member_count()
        admins = await chat.get_administrators()
    except Exception:
        return await update.message.reply_text("Unable to fetch stats.")

    admin_names = ", ".join([admin.user.first_name for admin in admins])

    reply = (
        f"📊 **Group Stats**\n"
        f"👥 Members: **{member_count}**\n"
        f"🛡 Admin Count: **{len(admins)}**\n"
        f"👑 Admins: {admin_names}"
    )
    await update.message.reply_text(reply, parse_mode="Markdown")


# --- New Utility Tool: Current Weather ---

@check_frozen
async def weather(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat.id in frozen_chats:
        return await update.message.reply_text(freeze_message)

    if not context.args:
        return await update.message.reply_text("Usage: /weather <city>")

    city = " ".join(context.args)

    url = (
        f"https://api.openweathermap.org/data/2.5/weather?"
        f"q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
    )

    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.json()

    if data.get("cod") != 200:
        return await update.message.reply_text("City not found ❌")

    temp = data["main"]["temp"]
    condition = data["weather"][0]["description"].title()
    humidity = data["main"]["humidity"]
    wind = data["wind"]["speed"]

    reply = (
        f"🌍 **Weather in {city.title()}**\n"
        f"🌡 Temperature: **{temp}°C**\n"
        f"🌤 Condition: **{condition}**\n"
        f"💧 Humidity: **{humidity}%**\n"
        f"🍃 Wind: **{wind} m/s**"
    )

    await update.message.reply_text(reply, parse_mode="Markdown")

# --- New Utility Tool: Current Date and Time ---

@check_frozen
async def get_current_time(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat.id in frozen_chats:
        return await update.message.reply_text(freeze_message)

    """Fetches and displays the current date and time for a specified region."""
    if not context.args:
        # Default to user's local time if available, otherwise suggest search
        await update.message.reply_text("Usage: `/time <Region/City>`. Example: `/time Tokyo` or `/time GMT-5`")
        return

    region = " ".join(context.args).strip()
    
    # 1. Use Google Search to resolve the timezone/time
    # This simulates a dedicated TimeZone API call (e.g., Google Time Zone API)
    search_query = f"current date and time in {region}"
    
    # The actual implementation of a Telegram bot cannot use the `google:search` tool directly
    # in the code structure. We must assume a factual source is used to get the time zone and time.
    
    # --- Timezone Placeholder ---
    # For a real implementation, you would need a TimeZone API.
    # We will use a mock to demonstrate the final output format.
    
    await context.bot.send_chat_action(update.effective_chat.id, "typing")

    try:
        # Simple mock for demonstration
        if "tokyo" in region.lower():
            tz = timezone(timedelta(hours=9))
            tz_name = "Japan Standard Time (JST/GMT+9)"
        elif "london" in region.lower() or "gmt" in region.lower():
            tz = timezone(timedelta(hours=0))
            tz_name = "Greenwich Mean Time (GMT)"
        elif "new york" in region.lower() or "est" in region.lower():
            tz = timezone(timedelta(hours=-5))
            tz_name = "Eastern Standard Time (EST/GMT-5)"
        else:
            # Fallback to UTC/GMT if region is not recognized
            tz = timezone.utc
            tz_name = "Universal Coordinated Time (UTC/GMT)"
        
        current_dt = datetime.now(tz)
        
        response_text = (
            f"🕰️ **Current Date & Time**\n"
            f"---------------------------------\n"
            f"**Region:** {region.upper()}\n"
            f"**Time Zone:** {tz_name}\n"
            f"**Date:** `{current_dt.strftime('%A, %d %B %Y')}`\n"
            f"**Time:** `{current_dt.strftime('%H:%M:%S %Z')}`"
        )

        await update.message.reply_text(response_text, parse_mode="Markdown")
        
    except Exception as e:
        logger.error(f"Time Handler Error: {e}")
        await update.message.reply_text(f"❌ Failed to get date and time for **{region}**. Error: {e}", parse_mode="Markdown")

# --- Countdown Commands (Existing) ---

@check_frozen
async def set_countdown(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat.id in frozen_chats:
        return await update.message.reply_text(freeze_message)

    """Sets a group-wide countdown to a specific date."""
    if not await check_admin(update, context): return
    if not db: 
        await update.message.reply_text("Database not available.")
        return

    # Expected format: /set_countdown DD/MM/YYYY Name of the event
    if len(context.args) < 2:
        await update.message.reply_text("Usage: `/set_countdown DD/MM/YYYY <Name of the event>` (e.g., `/set_countdown 31/12/2025 New Year's Eve`)", parse_mode="Markdown")
        return

    date_str = context.args[0]
    countdown_name = " ".join(context.args[1:])
    group_id = update.effective_chat.id
    
    try:
        # Note: Timezone is set to UTC for safe comparison
        target_date = datetime.strptime(date_str, "%d/%m/%Y").replace(tzinfo=timezone.utc)
        if target_date < datetime.now(timezone.utc):
            await update.message.reply_text("The target date must be in the future.")
            return

        # Store countdown in group_settings document
        ref = db.collection("group_settings").document(str(group_id))
        await asyncio.to_thread(ref.set, {
            "countdown_name": countdown_name,
            "target_date_iso": target_date.isoformat(),
            "target_date_human": date_str
        }, merge=True)

        await update.message.reply_text(
            f"🚀 Countdown for **{countdown_name}** set successfully!\nTarget date: `{date_str}`. Use `/check_countdown` to see the remaining time.",
            parse_mode="Markdown"
        )

    except ValueError:
        await update.message.reply_text("Invalid date format. Please use DD/MM/YYYY.")
    except Exception as e:
        logger.error(f"Error setting countdown: {e}")
        await update.message.reply_text(f"An unexpected error occurred: {e}")

@check_frozen
async def check_countdown(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat.id in frozen_chats:
        return await update.message.reply_text(freeze_message)

    """Checks and displays the remaining time for the group-wide countdown."""
    if not db: 
        await update.message.reply_text("Database not available.")
        return

    group_id = update.effective_chat.id
    # Countdown is stored in the group_settings document
    ref = db.collection("group_settings").document(str(group_id))

    try:
        doc = await asyncio.to_thread(ref.get)
        doc = doc.to_dict()
        if not doc or "target_date_iso" not in doc:
            await update.message.reply_text("No active countdown set for this chat. Use `/set_countdown` to start one.")
            return

        target_date_iso = doc["target_date_iso"]
        countdown_name = doc["countdown_name"]
        target_date_human = doc["target_date_human"]
        
        target_date = datetime.fromisoformat(target_date_iso).replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        
        remaining_time = target_date - now

        if remaining_time.total_seconds() <= 0:
            # Clear old countdown
            await asyncio.to_thread(ref.update, {
                "countdown_name": firestore.DELETE_FIELD,
                "target_date_iso": firestore.DELETE_FIELD,
                "target_date_human": firestore.DELETE_FIELD
            })
            await update.message.reply_text(f"🎉 **{countdown_name}** is here! The countdown has finished.")
            return
        
        # Format remaining time
        days = remaining_time.days
        # calculate remaining hours, minutes, seconds from the remaining_time.seconds attribute
        hours = remaining_time.seconds // 3600
        minutes = (remaining_time.seconds % 3600) // 60
        seconds = remaining_time.seconds % 60
        
        await update.message.reply_text(
            f"⏳ **{countdown_name}**\n"
            f"Target: `{target_date_human}`\n\n"
            f"**Time Remaining:**\n"
            f"• `{days}` days\n"
            f"• `{hours}` hours\n"
            f"• `{minutes}` minutes\n"
            f"• `{seconds}` seconds",
            parse_mode="Markdown"
        )

    except Exception as e:
        logger.error(f"Error checking countdown: {e}")
        await update.message.reply_text(f"An error occurred while checking the countdown: {e}")

# --- Lock/Unlock Commands (Existing) ---

@check_frozen
async def handle_lock_unlock(update: Update, context: ContextTypes.DEFAULT_TYPE, lock: bool) -> None:
    if update.effective_chat.id in frozen_chats:
        return await update.message.reply_text(freeze_message)

    """Handles /lock and /unlock commands for various features."""
    if not await check_admin(update, context): return
    if not context.args:
        await update.message.reply_text("Usage: `/lock <feature>` or `/unlock <feature>`. Features: `all`, `text`, `stickers`, `media`, `images`, `audio`.", parse_mode="Markdown")
        return

    feature_arg = context.args[0].lower()
    group_id = update.effective_chat.id
    
    # 1. Fetch current permissions
    try:
        current_chat = await context.bot.get_chat(group_id)
        # Convert permissions object to a dictionary for easy modification
        current_perms = current_chat.permissions.to_dict() if current_chat.permissions else {}
        
        # Default to True for any missing key, assuming permissions are open unless explicitly restricted
        new_perms = {
            "can_send_messages": current_perms.get("can_send_messages", True),
            "can_send_audios": current_perms.get("can_send_audios", True),
            "can_send_documents": current_perms.get("can_send_documents", True),
            "can_send_photos": current_perms.get("can_send_photos", True),
            "can_send_videos": current_perms.get("can_send_videos", True),
            "can_send_video_notes": current_perms.get("can_send_video_notes", True),
            "can_send_voice_notes": current_perms.get("can_send_voice_notes", True),
            "can_send_polls": current_perms.get("can_send_polls", True),
            # FIX 1: Use the correct flag for stickers/animations/games
            "can_send_other_messages": current_perms.get("can_send_other_messages", True), 
            "can_add_web_page_previews": current_perms.get("can_add_web_page_previews", True),
        }
    except Exception as e:
        logger.warning(f"Could not fetch current chat permissions, defaulting to all True: {e}")
        new_perms = {
            "can_send_messages": True, "can_send_audios": True, "can_send_documents": True,
            "can_send_photos": True, "can_send_videos": True, "can_send_video_notes": True,
            "can_send_voice_notes": True, "can_send_polls": True, "can_send_other_messages": True, 
            "can_add_web_page_previews": True
        }


    # Determine the target value (False for lock, True for unlock)
    target_value = not lock 

    if feature_arg == "all":
        # Lock/Unlock all messaging permissions
        new_perms["can_send_messages"] = target_value
        new_perms["can_send_audios"] = target_value
        new_perms["can_send_documents"] = target_value
        new_perms["can_send_photos"] = target_value
        new_perms["can_send_videos"] = target_value
        new_perms["can_send_video_notes"] = target_value
        new_perms["can_send_voice_notes"] = target_value
        new_perms["can_send_polls"] = target_value
        new_perms["can_send_other_messages"] = target_value
        new_perms["can_add_web_page_previews"] = target_value
        
    elif feature_arg == "text":
        new_perms["can_send_messages"] = target_value
        # If sending messages is locked, we can also lock previews
        if not target_value:
             new_perms["can_add_web_page_previews"] = target_value
             
    elif feature_arg == "stickers":
        # Stickers, animations (GIFs), and games are controlled by this flag
        new_perms["can_send_other_messages"] = target_value
    
    elif feature_arg == "media":
        # All media types
        new_perms["can_send_photos"] = target_value
        new_perms["can_send_videos"] = target_value
        new_perms["can_send_documents"] = target_value
        new_perms["can_send_audios"] = target_value
        new_perms["can_send_video_notes"] = target_value
        new_perms["can_send_voice_notes"] = target_value
        
    elif feature_arg == "images":
        new_perms["can_send_photos"] = target_value
    
    elif feature_arg == "audio":
        new_perms["can_send_audios"] = target_value
        new_perms["can_send_voice_notes"] = target_value
        
    else:
        await update.message.reply_text("Invalid feature. Choose from: `all`, `text`, `stickers`, `media`, `images`, `audio`.", parse_mode="Markdown")
        return
    
    # 2. FIX 2: Create a NEW ChatPermissions object (it's immutable) from the modified dictionary
    # The dictionary keys map directly to the ChatPermissions arguments.
    final_permissions = ChatPermissions(**new_perms)

    try:
        # Use set_chat_permissions to change default permissions for the group
        await context.bot.set_chat_permissions(chat_id=group_id, permissions=final_permissions)
        action = "LOCKED" if lock else "UNLOCKED"
        await update.message.reply_text(
            f"🔒 Feature **'{feature_arg.upper()}'** successfully **{action}** for general members.",
            parse_mode="Markdown"
        )
    except Exception as e:
        await update.message.reply_text(f"Could not update chat permissions. Make sure the bot is an admin with 'manage group' permissions. Error: {e}")

async def lock_feature(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Locks a specific feature."""
    await handle_lock_unlock(update, context, True)

async def unlock_feature(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Unlocks a specific feature."""
    await handle_lock_unlock(update, context, False)

# --- Banned Word Commands (Existing & Modified) ---

@check_frozen
async def ban_word(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat.id in frozen_chats:
        return await update.message.reply_text(freeze_message)

    """Bans a word from the group and stores it in Firestore."""
    if not await check_admin(update, context): return
    if not context.args:
        await update.message.reply_text("Usage: `/ban_word <word>` (e.g., `/ban_word spam`)")
        return
    if not db:
        await update.message.reply_text("Database not available.")
        return

    # Use the raw argument for better detection in the handler
    word = context.args[0].strip() 
    group_id = update.effective_chat.id
    ref = get_banned_word_ref(group_id, word)

    try:
        # Store the original word and a normalized version for reference
        await asyncio.to_thread(ref.set, {"word": word, "normalized_word": re.sub(r'[^a-zA-Z0-9]', '', word).lower(), "timestamp": firestore.SERVER_TIMESTAMP})
        await update.message.reply_text(
            f"🚫 Word **'{word}'** has been successfully banned. Messages containing this word (or close variants) will be deleted.",
            parse_mode="Markdown"
        )
    except Exception as e:
        logger.error(f"Error banning word: {e}")
        await update.message.reply_text(f"Failed to ban word due to a database error: {e}")

@check_frozen
async def unban_word(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat.id in frozen_chats:
        return await update.message.reply_text(freeze_message)

    """Unbans a word and removes it from Firestore."""
    if not await check_admin(update, context): return
    if not context.args:
        await update.message.reply_text("Usage: `/unban_word <word>`")
        return
    if not db:
        await update.message.reply_text("Database not available.")
        return

    word = context.args[0].strip()
    group_id = update.effective_chat.id
    ref = get_banned_word_ref(group_id, word)

    try:
        doc = await asyncio.to_thread(ref.get)
        if doc.exists:
            await asyncio.to_thread(ref.delete)
            await update.message.reply_text(
                f"✅ Word **'{word}'** has been successfully unbanned.",
                parse_mode="Markdown"
            )
        else:
            await update.message.reply_text(f"Word **'{word}'** was not found in the banned list.")
    except Exception as e:
        logger.error(f"Error unbanning word: {e}")
        await update.message.reply_text(f"Failed to unban word due to a database error: {e}")

@check_frozen
async def handle_banned_words(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat.id in frozen_chats:
        return await update.message.reply_text(freeze_message)

    """Intercepts messages, checks for banned words, and deletes the message if found."""
    if not db or not update.message.text:
        return
    
    # Do not check/delete messages from admins or owner
    try:
        user_id = update.effective_user.id
        chat_id = update.effective_chat.id
        member = await context.bot.get_chat_member(chat_id, user_id)
        if member.status in ["creator", "administrator"] or is_owner(user_id):
            return
            
    except Exception as e:
        # If we can't check admin status (e.g. private chat or error), we proceed to check words
        logger.debug(f"Failed to check admin status in handle_banned_words: {e}")

    message = update.message
    message_text = message.text.lower()
    group_id = update.effective_chat.id

    banned_words_ref = db.collection("groups").document(str(group_id)).collection("banned_words")
    
    # Pre-process the incoming message for a wider match
    # Remove all non-alphanumeric characters and convert to lower case
    normalized_message_text = re.sub(r'[^a-zA-Z0-9]', '', message_text)
    
    try:
        banned_words_snapshot = await asyncio.to_thread(banned_words_ref.stream)
        
        for doc in banned_words_snapshot:
            data = doc.to_dict()
            word = data.get("word", "")
            normalized_word = data.get("normalized_word", "")
            
            if not word and not normalized_word:
                continue

            # Check 1: Direct match (case-insensitive) - e.g., "Spam" in "This is spam"
            if word and word.lower() in message_text:
                matched_word = word
            # Check 2: Normalized match - e.g., "s.p.a.m" in "This is s.p.a.m!!!" -> "spam" in "thisisspam"
            elif normalized_word and normalized_word in normalized_message_text:
                matched_word = word
            else:
                continue
            
            try:
                await message.delete()
                await context.bot.send_message(
                    chat_id=group_id,
                    text=f"⚠️ {message.from_user.mention_html()}, your message was deleted for using a banned word: **{matched_word}**.",
                    parse_mode="HTML"
                )
            except Exception as e:
                logger.warning(f"Failed to delete banned word message or send warning: {e}")
            return 
                
    except Exception as e:
        logger.error(f"Error handling banned words: {e}")


# --- Link Approval Commands (Existing) ---

@check_frozen
async def approve_link_sender(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat.id in frozen_chats:
        return await update.message.reply_text(freeze_message)

    """Approves a replied/specified member to send links in the group."""
    if not await check_admin(update, context): return
    if not db:
        await update.message.reply_text("Database not available.")
        return

    target_id, target_name_or_error = await get_target_user_info(update, context)

    if not target_id:
        await update.message.reply_text(target_name_or_error, parse_mode="Markdown")
        return

    ref = get_link_approval_ref(update.effective_chat.id, target_id)

    try:
        await asyncio.to_thread(ref.set, {
            "user_id": target_id, 
            "full_name": target_name_or_error,
            "is_approved": True, # Explicitly set approval status
            "timestamp": firestore.SERVER_TIMESTAMP
        })
        await update.message.reply_text(
            f"✅ User **{target_name_or_error}** (ID: `{target_id}`) has been **APPROVED** to send external links.",
            parse_mode="Markdown"
        )
    except Exception as e:
        logger.error(f"Error approving link sender: {e}")
        await update.message.reply_text(f"Failed to approve link sender due to a database error: {e}")

@check_frozen
async def disapprove_link_sender(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat.id in frozen_chats:
        return await update.message.reply_text(freeze_message)

    """Disapproves a replied/specified member from sending links in the group."""
    if not await check_admin(update, context): return
    if not db:
        await update.message.reply_text("Database not available.")
        return

    target_id, target_name_or_error = await get_target_user_info(update, context)

    if not target_id:
        await update.message.reply_text(target_name_or_error, parse_mode="Markdown")
        return

    ref = get_link_approval_ref(update.effective_chat.id, target_id)

    try:
        doc = await asyncio.to_thread(ref.get)
        if doc.exists:
            await asyncio.to_thread(ref.delete)
            await update.message.reply_text(
                f"🛑 User **{target_name_or_error}** (ID: `{target_id}`) has been **DISAPPROVED** from sending external links.",
                parse_mode="Markdown"
            )
        else:
            await update.message.reply_text(f"User **{target_name_or_error}** was not found in the approved list.", parse_mode="Markdown")
    except Exception as e:
        logger.error(f"Error disapproving link sender: {e}")
        await update.message.reply_text(f"Failed to disapprove link sender due to a database error: {e}")

@check_frozen
async def handle_link_messages(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat.id in frozen_chats:
        return await update.message.reply_text(freeze_message)

    """Checks for links and deletes the message, warning the user if they are not approved."""
    if not db or update.effective_chat.type not in ["group", "supergroup"]:
        return

    message = update.message
    group_id = update.effective_chat.id
    user_id = update.effective_user.id
    target_user = update.effective_user
    
    # 1. Check if the message contains an actual link entity or a text mention/link
    if not message.entities and not message.caption_entities:
        return
    
    # Check if the user is an admin or owner (they are always allowed to post links)
    try:
        member = await context.bot.get_chat_member(group_id, user_id)
        if member.status in ["creator", "administrator"] or is_owner(user_id):
            return
    except Exception as e:
        logger.warning(f"Error checking admin status for link handler: {e}")
        # If we fail to check status (e.g., bot not admin), we proceed to check link approval
        # Note: If this fails, the user is NOT considered admin/owner for the purpose of this handler.

    # 2. Check link approval status from Firestore
    if await is_link_approved(group_id, user_id):
        # User is approved, do nothing
        return

    # 3. User is NOT approved and posted a link -> Moderate action
    try:
        # Delete the link message
        await message.delete()
        
        # Apply warning logic (similar to /warn command)
        new_warnings = await update_warn_count(group_id, user_id, 1)
        
        if new_warnings >= 3:
            # Ban the user on 3rd warning
            try:
                await context.bot.ban_chat_member(group_id, target_user.id)
                await update_warn_count(group_id, target_user.id, -new_warnings) # Reset warnings
                await context.bot.send_message(
                    chat_id=group_id,
                    text=f"🚨 User {target_user.mention_html()} reached 3 warnings and has been **BANNED** for posting unapproved links.",
                    parse_mode="HTML"
                )
            except Exception as e:
                logger.error(f"Failed to ban user after 3 link warnings: {e}")
                
        else:
            # Send warning message
            await context.bot.send_message(
                chat_id=group_id,
                text=f"⚠️ User {target_user.mention_html()}! Your message was deleted due to an **unapproved link** (Warning {new_warnings}/3).",
                parse_mode="HTML"
            )
            
    except Exception as e:
        logger.error(f"Error handling unapproved link message: {e}")


# --- Group Management Commands (Existing) ---

@check_frozen
async def warn_user(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat.id in frozen_chats:
        return await update.message.reply_text(freeze_message)

    """Warns a user and tracks the count. Kicks/bans on reaching 3 warnings."""
    if not await check_admin(update, context): return
    
    # Resolve target user info
    target_id, target_name_or_error = await get_target_user_info(update, context)

    if not target_id:
        await update.message.reply_text(target_name_or_error, parse_mode="Markdown")
        return

    group_id = update.effective_chat.id

    if is_owner(target_id):
        await update.message.reply_text("I cannot warn the owner.")
        return

    new_warnings = await update_warn_count(group_id, target_id, 1)
    
    # Get reason, skipping the target user argument if it was provided
    # Check if the first argument was used to identify the target
    if context.args and target_name_or_error.startswith("User ID:") or (context.args and context.args[0].startswith('@')):
        reason = " ".join(context.args[1:]) if len(context.args) > 1 else "No reason provided."
    else:
        # If the target was resolved by reply, the whole context.args is the reason
        reason = " ".join(context.args) if context.args else "No reason provided."


    if new_warnings >= 3:
        try:
            # Ban the user
            await context.bot.ban_chat_member(group_id, target_id)
            await update_warn_count(group_id, target_id, -new_warnings) # Reset warnings
            await update.message.reply_text(
                f"🚨 User **{target_name_or_error}** reached 3 warnings and has been **BANNED**.\nReason: {reason}",
                parse_mode="Markdown"
            )
        except Exception as e:
            await update.message.reply_text(f"Could not ban user. Make sure the bot is an admin with 'ban users' permission. Error: {e}")
    else:
        await update.message.reply_text(
            f"⚠️ User **{target_name_or_error}** has been **WARNED** (Warning {new_warnings}/3).\nReason: {reason}",
            parse_mode="Markdown"
        )

@check_frozen
async def remove_warn(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat.id in frozen_chats:
        return await update.message.reply_text(freeze_message)

    """Removes one warning from a user."""
    if not await check_admin(update, context): return
    
    # Resolve target user info
    target_id, target_name_or_error = await get_target_user_info(update, context)

    if not target_id:
        await update.message.reply_text(target_name_or_error, parse_mode="Markdown")
        return

    group_id = update.effective_chat.id
    
    current_warnings = await get_warn_count(group_id, target_id)
    if current_warnings > 0:
        new_warnings = await update_warn_count(group_id, target_id, -1)
        await update.message.reply_text(
            f"✅ Warning removed from **{target_name_or_error}**. Current warnings: {new_warnings}/3.",
            parse_mode="Markdown"
        )
    else:
        await update.message.reply_text(f"User **{target_name_or_error}** has no active warnings to remove.", parse_mode="Markdown")

@check_frozen
async def warn_counts(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat.id in frozen_chats:
        return await update.message.reply_text(freeze_message)

    """Shows the warning count of a user."""
    if not await check_admin(update, context): return
    
    # Resolve target user info
    target_id, target_name_or_error = await get_target_user_info(update, context)

    if not target_id:
        await update.message.reply_text(target_name_or_error, parse_mode="Markdown")
        return
    
    warns = await get_warn_count(update.effective_chat.id, target_id)
    await update.message.reply_text(
        f"User **{target_name_or_error}** has **{warns}** active warnings.",
        parse_mode="Markdown"
    )

@check_frozen
async def ban_user(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat.id in frozen_chats:
        return await update.message.reply_text(freeze_message)

    """Bans a user from the group."""
    if not await check_admin(update, context): return
    
    # Resolve target user info
    target_id, target_name_or_error = await get_target_user_info(update, context)

    if not target_id:
        await update.message.reply_text(target_name_or_error, parse_mode="Markdown")
        return

    group_id = update.effective_chat.id
    
    # Get reason, skipping the target user argument if it was provided
    if context.args and target_name_or_error.startswith("User ID:") or (context.args and context.args[0].startswith('@')):
        reason = " ".join(context.args[1:]) if len(context.args) > 1 else "No reason provided."
    else:
        reason = " ".join(context.args) if context.args else "No reason provided."

    try:
        await context.bot.ban_chat_member(group_id, target_id)
        # Also remove warnings
        await update_warn_count(group_id, target_id, -await get_warn_count(group_id, target_id))

        await update.message.reply_text(
            f"🔨 User **{target_name_or_error}** has been **BANNED**.\nReason: {reason}",
            parse_mode="Markdown"
        )
    except Exception as e:
        await update.message.reply_text(f"Could not ban user. Make sure the bot is an admin with 'ban users' permission. Error: {e}")

@check_frozen
async def unban_user(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat.id in frozen_chats:
        return await update.message.reply_text(freeze_message)

    """Unbans a user from the group. User ID must be provided."""
    if not await check_admin(update, context): return
    
    # Unbanning requires the ID because the user is no longer a chat member.
    if not context.args or not context.args[0].isdigit():
        await update.message.reply_text(
            "⚠️ **Unban requires the numerical User ID** (since the user is banned and cannot be resolved by username). Usage: `/unban 123456789`",
            parse_mode="Markdown"
        )
        return

    target_id = int(context.args[0])

    try:
        # The unban function requires a user ID and will only work if the user is currently banned.
        await context.bot.unban_chat_member(update.effective_chat.id, target_id)
        await update.message.reply_text(f"🔓 User with ID `{target_id}` has been **UNBANNED**.", parse_mode="Markdown")
    except Exception as e:
        await update.message.reply_text(f"Could not unban user. Error: {e}")

@check_frozen
async def mute_user(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat.id in frozen_chats:
        return await update.message.reply_text(freeze_message)

    """Mutes a user for a specified duration (default 1 hour)."""
    if not await check_admin(update, context): return
    
    # Resolve target user info
    target_id, target_name_or_error = await get_target_user_info(update, context)

    if not target_id:
        await update.message.reply_text(target_name_or_error, parse_mode="Markdown")
        return

    group_id = update.effective_chat.id
    
    # Default mute duration: 1 hour
    mute_seconds = 3600
    duration_str = "1 hour"
    duration_arg = None
    
    # Determine the argument index based on whether a target was explicitly passed or implied by reply
    if update.message.reply_to_message:
        # Target from reply, duration is arg[0] if it exists
        if context.args and context.args[0].isdigit():
            duration_arg = context.args[0]
    else:
        # Target from arg[0] (ID or @username), duration is arg[1] if it exists
        if len(context.args) > 1 and context.args[1].isdigit():
            duration_arg = context.args[1]

    if duration_arg:
        mute_minutes = int(duration_arg)
        mute_seconds = mute_minutes * 60
        duration_str = f"{mute_minutes} minutes"

    mute_until = datetime.now(timezone.utc) + timedelta(seconds=mute_seconds)

    try:
        # Mute means can_send_messages=False, and all other permissions are also set to False by default
        # We need to explicitly set all permissions here to ensure a proper mute/unmute cycle.
        await context.bot.restrict_chat_member(
            group_id,
            target_id,
            permissions=ChatPermissions(
                can_send_messages=False,
                can_send_audios=False,
                can_send_documents=False,
                can_send_photos=False,
                can_send_videos=False,
                can_send_video_notes=False,
                can_send_voice_notes=False,
                can_send_polls=False,
                can_send_other_messages=False,
                can_add_web_page_previews=False,
            ),
            until_date=mute_until
        )
        await update.message.reply_text(
            f"🔇 User **{target_name_or_error}** has been **MUTED** for {duration_str}.",
            parse_mode="Markdown"
        )
    except Exception as e:
        await update.message.reply_text(f"Could not mute user. Make sure the bot is an admin with 'restrict users' permission. Error: {e}")

@check_frozen
async def unmute_user(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat.id in frozen_chats:
        return await update.message.reply_text(freeze_message)

    """Unmutes a user by resetting permissions."""
    if not await check_admin(update, context): return
    
    # Resolve target user info
    target_id, target_name_or_error = await get_target_user_info(update, context)

    if not target_id:
        await update.message.reply_text(target_name_or_error, parse_mode="Markdown")
        return

    group_id = update.effective_chat.id

    try:
        # Give back all default permissions (unmute)
        await context.bot.restrict_chat_member(
            group_id,
            target_id,
            permissions=ChatPermissions(
                can_send_messages=True,
                can_send_audios=True,
                can_send_documents=True,
                can_send_photos=True,
                can_send_videos=True,
                can_send_video_notes=True,
                can_send_voice_notes=True,
                can_send_polls=True,
                can_send_other_messages=True, # Covers stickers, animations, games
                can_add_web_page_previews=True,
            )
        )
        await update.message.reply_text(
            f"🔊 User **{target_name_or_error}** has been **UNMUTED**.",
            parse_mode="Markdown"
        )
    except Exception as e:
        await update.message.reply_text(f"Could not unmute user. Error: {e}")

@check_frozen
async def mention_admins(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat.id in frozen_chats:
        return await update.message.reply_text(freeze_message)

    chat = update.effective_chat
    bot = context.bot

    if chat.type not in ["group", "supergroup"]:
        await update.message.reply_text("This command only works in groups.")
        return

    try:
        admins = await bot.get_chat_administrators(chat.id)

        mention_text = " ".join([
            f"<a href='tg://user?id={a.user.id}'>{a.user.first_name}</a>"
            for a in admins if not a.user.is_bot
        ])

        await update.message.reply_text(f"🛡 <b>Admins:</b>\n{mention_text}", parse_mode="HTML")

    except Exception as e:
        print(e)
        await update.message.reply_text("Failed to fetch admin list.")

@check_frozen
async def promote_user(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat.id in frozen_chats:
        return await update.message.reply_text(freeze_message)

    """Promotes a replied/specified user to administrator."""
    if not await check_admin(update, context): return
    
    # Resolve target user info
    target_id, target_name_or_error = await get_target_user_info(update, context)

    if not target_id:
        await update.message.reply_text(target_name_or_error, parse_mode="Markdown")
        return

    group_id = update.effective_chat.id

    try:
        # Promote the user with no optional permissions granted by default (except basic ones)
        await context.bot.promote_chat_member(
            chat_id=group_id,
            user_id=target_id,
            can_manage_chat=True,
            can_delete_messages=True,
            can_restrict_members=True,
            can_pin_messages=True,
            can_promote_members=False, # Do not allow them to promote others initially
            can_change_info=False
        )
        await update.message.reply_text(
            f"👑 User **{target_name_or_error}** has been **PROMOTED** to a standard administrator.",
            parse_mode="Markdown"
        )
    except Exception as e:
        await update.message.reply_text(f"Could not promote user. Make sure the bot is the group creator or has the 'Add New Admins' permission. Error: {e}")

# --- Filter Management Commands (Existing) ---

@check_frozen
async def set_filter(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat.id in frozen_chats:
        return await update.message.reply_text(freeze_message)

    """Assigns a filter keyword to a replied message (text, sticker, or photo)."""
    if not await check_admin(update, context): return
    
    reply = update.message.reply_to_message
    if not reply:
        await update.message.reply_text("Please reply to the message (text, sticker, or image) you want to filter and provide a keyword. Usage: `/filter <keyword>`")
        return
    
    if not context.args:
        await update.message.reply_text("You must provide a keyword for the filter. Usage: `/filter <keyword>`")
        return
    if not db:
        await update.message.reply_text("Database not available.")
        return

    keyword = context.args[0]
    group_id = update.effective_chat.id
    ref = db.collection("groups").document(str(group_id)).collection("filters").document(keyword.lower().strip())
    
    filter_data = {"keyword": keyword}

    if reply.text:
        filter_data.update({
            "type": "text",
            "content": reply.text
        })
    elif reply.sticker:
        filter_data.update({
            "type": "sticker",
            "file_id": reply.sticker.file_id
        })
    elif reply.photo:
        # Get the highest resolution photo file_id
        filter_data.update({
            "type": "photo",
            "file_id": reply.photo[-1].file_id 
        })
    else:
        await update.message.reply_text("Unsupported message type. Only text, stickers, and photos can be set as filters.")
        return

    try:
        await asyncio.to_thread(ref.set, filter_data)
        await update.message.reply_text(
            f"✅ Filter **'{keyword.lower()}'** set! When this word is used, I will reply with the saved {filter_data['type']}.", 
            parse_mode="Markdown"
        )
    except Exception as e:
        logger.error(f"Error setting filter: {e}")
        await update.message.reply_text(f"Failed to set filter due to a database error. Error: {e}")

@check_frozen
async def stop_filter(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat.id in frozen_chats:
        return await update.message.reply_text(freeze_message)

    """Stops (deletes) a filter by keyword."""
    if not await check_admin(update, context): return
    
    if not context.args:
        await update.message.reply_text("You must provide the keyword of the filter to stop. Usage: `/stop <keyword>`")
        return
    if not db:
        await update.message.reply_text("Database not available.")
        return

    keyword = context.args[0]
    group_id = update.effective_chat.id
    ref = db.collection("groups").document(str(group_id)).collection("filters").document(keyword.lower().strip())

    try:
        # Check if the filter exists before deleting
        doc = await asyncio.to_thread(ref.get)
        if doc.exists:
            await asyncio.to_thread(ref.delete)
            await update.message.reply_text(
                f"🛑 Filter **'{keyword.lower()}'** has been stopped and deleted.", 
                parse_mode="Markdown"
            )
        else:
            await update.message.reply_text(f"Filter **'{keyword.lower()}'** not found.", parse_mode="Markdown")
    except Exception as e:
        logger.error(f"Error deleting filter: {e}")
        await update.message.reply_text(f"Failed to stop filter due to a database error: {e}")

@check_frozen
async def handle_filters(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat.id in frozen_chats:
        return await update.message.reply_text(freeze_message)

    """Checks incoming messages against active filters and sends the corresponding content."""
    if not db or not update.message.text:
        return
        
    group_id = update.effective_chat.id
    message_text = update.message.text.lower()

    # Get the reference to the filters collection for this group
    filters_collection_ref = db.collection("groups").document(str(group_id)).collection("filters")

    try:
        # Fetch all filter documents
        filters_snapshot = await asyncio.to_thread(filters_collection_ref.stream)
        
        for doc in filters_snapshot:
            filter_data = doc.to_dict()
            keyword = filter_data.get("keyword", "").lower()

            # Check if the keyword is present in the message text
            if keyword and keyword in message_text:
                filter_type = filter_data.get("type")
                file_id = filter_data.get("file_id")
                content = filter_data.get("content")
                
                # Reply to the user's message with the filtered content
                if filter_type == "text" and content:
                    await update.message.reply_text(content)
                elif filter_type == "sticker" and file_id:
                    await update.message.reply_sticker(file_id)
                elif filter_type == "photo" and file_id:
                    await update.message.reply_photo(file_id)
                
                # Stop processing after the first matching filter is found
                return

    except Exception as e:
        logger.error(f"Error checking/handling filters: {e}")

# --- AI Based Welcome Message Features ---

async def get_welcome_settings(group_id: int) -> dict:
    """Fetches the welcome settings for a group."""
    if not db: return {}
    try:
        ref = db.collection("group_settings").document(str(group_id))
        doc = await asyncio.to_thread(ref.get)
        data = doc.to_dict() or {}
        return {
            "enabled": data.get("welcome_enabled", False),
            "template": data.get("welcome_message_template", "Hello {name}, welcome to the group!"),
            "ai_mode": data.get("ai_personality_mode", "default")
        }
    except Exception as e:
        logger.error(f"Error fetching welcome settings: {e}")
        return {}

@check_frozen
async def set_welcome_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sets the custom welcome message template."""
    if not await check_admin(update, context): return
    if not db:
        await update.message.reply_text("Database not available.")
        return

    if not context.args:
        await update.message.reply_text(
            "Usage: `/set_welcome <message>`\nUse `{name}` to mention the new user and `{group_name}` for the group name. The AI will add an intelligent greeting based on the current personality mode.",
            parse_mode="Markdown"
        )
        return
        
    template = " ".join(context.args)
    group_id = update.effective_chat.id
    ref = db.collection("group_settings").document(str(group_id))
    
    try:
        # Save template exactly as provided (admins may want Markdown/HTML in template)
        await asyncio.to_thread(ref.set, {"welcome_message_template": template}, merge=True)

        # Send a safe confirmation message by escaping the template and showing in a pre block.
        safe_template = html.escape(template)
        await update.message.reply_text(
            "✅ Custom welcome message template set:\n" + f"<pre>{safe_template}</pre>",
            parse_mode="HTML"
        )
    except Exception as e:
        logger.error(f"Error setting welcome message: {e}")
        # Provide a safe error reply (avoid injecting the raw template)
        await update.message.reply_text(f"Failed to set welcome message: {html.escape(str(e))}", parse_mode="HTML")


@check_frozen
async def enable_welcome(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Enables the AI welcome message feature."""
    if not await check_admin(update, context): return
    if not db:
        await update.message.reply_text("Database not available.")
        return
        
    group_id = update.effective_chat.id
    ref = db.collection("group_settings").document(str(group_id))
    
    try:
        await asyncio.to_thread(ref.set, {"welcome_enabled": True}, merge=True)
        await update.message.reply_text("✅ AI Welcome Messages are now **ENABLED**.", parse_mode="Markdown")
    except Exception as e:
        logger.error(f"Error enabling welcome: {e}")
        await update.message.reply_text(f"Failed to enable welcome messages: {e}")

@check_frozen
async def disable_welcome(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Disables the AI welcome message feature."""
    if not await check_admin(update, context): return
    if not db:
        await update.message.reply_text("Database not available.")
        return
        
    group_id = update.effective_chat.id
    ref = db.collection("group_settings").document(str(group_id))
    
    try:
        await asyncio.to_thread(ref.set, {"welcome_enabled": False}, merge=True)
        await update.message.reply_text("🛑 AI Welcome Messages are now **DISABLED**.", parse_mode="Markdown")
    except Exception as e:
        logger.error(f"Error disabling welcome: {e}")
        await update.message.reply_text(f"Failed to disable welcome messages: {e}")


@check_frozen
async def handle_new_member(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Welcomes new members — uses custom template if set, AI only if not."""
    if not update.message.new_chat_members:
        return
    if update.effective_chat.type not in ["group", "supergroup"]:
        return

    settings = await get_welcome_settings(update.effective_chat.id)
    if not settings.get("enabled"):
        return

    group_name = update.effective_chat.title or ""

    # 🔥 Support both possible keys so we never miss the template
    template = (
        settings.get("welcome_message_template") or
        settings.get("template")
    )

    for member in update.message.new_chat_members:
        if member.id == context.bot.id:
            continue

        mention_html = member.mention_html()
        username = member.full_name or member.username or "there"

        # ===== CASE 1: TEMPLATE IS SET → DO NOT USE AI =====
        if template:
            try:
                message = template.format(
                    name=mention_html,
                    group_name=html.escape(group_name)
                )
            except Exception as e:
                logger.warning(f"Template error: {e}")
                message = f"Hello {mention_html}, welcome to {html.escape(group_name)}!"

            try:
                await update.message.reply_text(message, parse_mode="HTML")
            except:
                await update.message.reply_text(html.escape(message), parse_mode="HTML")
            continue

        # ===== CASE 2: NO TEMPLATE → USE AI =====
        personality = settings.get("ai_mode", "default")
        ai_prompt = (
            f"A new member '{username}' joined '{group_name}'. "
            "Write a short warm welcome encouraging doubts and participation. "
            "Use ONE mention only."
        )

        await context.bot.send_chat_action(update.effective_chat.id, "typing")
        ai_text = await get_ai_response(ai_prompt, update.effective_chat.id)

        final_msg = f"{mention_html}\n\n{ai_text}"

        try:
            await update.message.reply_text(final_msg, parse_mode="HTML")
        except:
            await update.message.reply_text(f"{username}, {ai_text}")




@check_frozen
async def help_command(update: Update, context: CallbackContext):
    message = (
        "🦚All the bot commands are listed below💫-\n"
        "/help - shows this message.\n"
        "utility commands🐥-\n"
        "/start - use to start the bot for the first time.\n"
        "/id - use to get the userid of a member in a group.\n"
        "/time <place> - use to see the time of a particular place.\n"
        "/weather <place> - use to see the weather of a particular area.\n"
        "gc helping commands🐣-\n"
        "/stats - get the group stats.\n"
        "/purge - to delete messages from a particular message.\n"
        "/approve - approve trusted users to send URL in the group.\n"
        "/disapprove - remove approval to send links in the group.\n"
        "/warn - warn a user for breaking gc rules.\n"
        "/remove_warn - remove warnings from a member.\n"
        "/warn - shows active warn counts of a user, 3 warnings = bye bye👋 from the gc.\n"
        "/ban - direct bye bye from the gc without warns😁\n"
        "/unban - glti se ban kr diya toh wapas v toh lana pdega🥲\n"
        "/lock - members jada faltu ki baate kr rhe hai? just lock krdo text, sticker ya fr kya lock krna hai bol dena.😏\n"
        "/unlock - becharo ko kb tk lock krke rkhoge... unlock v krdo ab😒\n"
        "/mute - koi badmoshi kr rha? muh mein feviquick chipka do.\n"
        "/unmute - feviquick pe acetone daal do... pighl jyega feviquick.\n"
        "/ban_word - members anab sanab bak rhe? word ko hi ban krdo🙂\n"
        "/promote - members ki tarakki hogi.\n"
        "/filter <name of the filter> - koi text, image ya sticker ka namakaran krdo... us naam ko bulane pe bot jawab dega😄\n"
        "/stop <name of the filter> - ek hi filter se tang aa gye? filter ko tup tup kar krdo👻\n"
        "/enable_welcome - get AI generated welcome messages whenever someone joins the group😉\n"
        "/set_welcome - customize krlo welcome message ko🥱\n"
        "/disable_welcome - welcome messages se paresan? bnd krdo🫠\n"
        "Personal + AI commands🪄-\n"
        "/set_countdown - countdown set krdo koi v upcoming event ka.\n"
        "/check_countdown - countdown check krte rho time to time.\n"
        "/admin - gc mein badmoshi ho rhi hai? admin ko bulaoo naa...\n"
        "/ask - questions pucho jee bharke.\n"
        "/set_personality - kis personality mein AI based features chahiye choose krlo... friendly, sarcastic, etc😶‍🌫\n"
        "/quiz <no. of questions> <chapter_name> - real time quiz generation.📚\n"
        "/change_timing <seconds> - quiz ke each ques ke duration(should be a multiple of 5) ko change krne ke liye⏲\n"
        "/save_chapter <chapter_name> - koi v pdf(small size) ko reply kro is command ke sath(chapter ka nam command ke baad likh dena) and then us pdf se quiz krte rho...❄🪄\n"
        "/stop_quiz - stops an ongoing quiz taaki aap jaake apne gf/bf ka msg padh paao🙂\n"
        "/leaderboard - XP leaderboard dekho kon top per hai quizzes ke answer dene mein🔥\n"
        "/global_leaderboard - XP leaderboard hi hai lkin all over the groups🌸\n"
        "/thanks - kisi v help ke liye us person ko thanks do aur uska reputation badhao💫\n"
        "/repleaderboard - reputation kiski kitni hai check karo⚡\n"
        "/profile - apna apna profile check karo🫥\n"
        "/quiz_autonomous - 24/7 apne aap quizzes send krte rhega non-stop(alag alag chapter se)😶‍🌫\n"
        "/stop_autonomous - automatic quiz ko break dene ke liye thoda😁\n"
        "/set_auto_questions - autonomous quiz mein ek chapter se kitne questions chahiye select kro🥹\n\n\n"
        "Thank you🕊\n"
        "regards- Chhoti don🐣"
    )
    await update.message.reply_text(message)

# ============================
# 🌟 QUIZ REWARDING SYSTEM
# ============================

# XP points for ranking
RANK_REWARD = {1: 120, 2: 90, 3: 60}

# League requirements (XP thresholds)
LEAGUES = [
    ("Bronze", 0),
    ("Silver", 500),
    ("Gold", 1200),
    ("Platinum", 2500),
    ("Diamond", 5000),
    ("Legendary", 9000),
]


async def update_league_if_needed(user_id: int) -> str:
    """Check league by XP and update if promotion happens."""
    ref = db.collection("users").document(str(user_id))
    doc = await asyncio.to_thread(ref.get)
    xp = doc.to_dict().get("xp", 0)

    new_league = "Bronze"
    for league, threshold in LEAGUES:
        if xp >= threshold:
            new_league = league

    # If new league change required → update
    await asyncio.to_thread(ref.set, {"league": new_league}, merge=True)
    return new_league


# ============================
# 🌟 FINAL STABLE REWARD FUNCTION
# ============================
async def reward_top_three(group_id: int, bot, chat_id: int = None):
    """
    Works from BOTH handlers and background tasks.
    Does NOT require update/context.
    """
    try:
        session = quiz_sessions.get(group_id)
        if not session or not session.get("scores"):
            return

        scores = session["scores"]
        sorted_players = sorted(
            scores.items(),
            key=lambda x: (-x[1].get("score", 0), x[1].get("total_time", float("inf")))
        )


        lines = []
        lines.append("🏆 *QUIZ REWARD TIME!* 🏆")
        lines.append("Top 3 performers are awarded XP 🔥\n")

        for rank, (user_id, stats) in enumerate(sorted_players[:3], start=1):
            reward = RANK_REWARD.get(rank, 0)

            # update XP
            user_ref = db.collection("users").document(str(user_id))
            await asyncio.to_thread(user_ref.set, {
                "user_id": user_id,
                "username": stats.get("name", f"User {user_id}"),
                "xp": firestore.Increment(reward)
            }, merge=True)

            # league update
            new_league = await update_league_if_needed(user_id)

            medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉"
            lines.append(f"{medal} Rank {rank}: *{stats['name']}* → +{reward} XP → _{new_league}_")

        msg = "\n".join(lines)

        final_chat = chat_id if chat_id else group_id
        await bot.send_message(chat_id=final_chat, text=msg, parse_mode="Markdown")

    except Exception as e:
        logger.error(f"reward_top_three error for {group_id}: {e}")

async def save_group_scores_to_db(group_id: int):
    session = quiz_sessions.get(group_id)
    if not session or not session.get("scores"):
        return

    batch = db.batch()
    group_users_ref = db.collection("group_scores").document(str(group_id)).collection("users")

    for user_id, stats in session["scores"].items():
        doc = group_users_ref.document(str(user_id))
        batch.set(doc, {
            "user_id": user_id,
            "username": stats.get("name", f"User {user_id}"),
            "score": stats.get("score", 0)
        }, merge=True)

    await asyncio.to_thread(batch.commit)


@check_frozen
async def local_leaderboard(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        group_id = update.effective_chat.id
        await update.effective_chat.send_chat_action("typing")

        # 1. Fetch group members who have played in this group
        def fetch_group_users():
            return list(
                db.collection("group_scores")
                .document(str(group_id))
                .collection("users")
                .stream()
            )

        group_docs = await asyncio.to_thread(fetch_group_users)
        if not group_docs:
            return await update.message.reply_text("No leaderboard data found yet 😕\n(Wait for a quiz to finish or use /stop_quiz to save current scores)")

        leaderboard = []

        # 2. Process users
        for doc in group_docs:
            gdata = doc.to_dict()
            user_id = gdata.get("user_id")
            # Fallback name from group_scores if global user data is missing
            username = gdata.get("username", f"User {user_id}") 

            # Fetch User Profile for global XP and League
            try:
                def get_user_xp():
                    u_doc = db.collection("users").document(str(user_id)).get()
                    if u_doc.exists:
                        return u_doc.to_dict()
                    return {}
                
                udata = await asyncio.to_thread(get_user_xp)
                xp = udata.get("xp", 0)
                league = udata.get("league", "Bronze")
                
                leaderboard.append({
                    "username": username,
                    "xp": xp,
                    "league": league
                })
            except Exception as e:
                # If one user fails, skip them but don't crash the command
                continue

        # 3. Sort by XP descending
        leaderboard.sort(key=lambda x: x["xp"], reverse=True)

        # 4. Build message
        msg = f"🏆 <b>GROUP XP LEADERBOARD</b> 🏆\n<b>{update.effective_chat.title}</b>\n\n"
        for i, u in enumerate(leaderboard[:20], start=1):
            medal = "👑🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
            safe_name = html.escape(u["username"])
            msg += f"{medal} <b>{safe_name}</b> — <code>{u['xp']} XP</code> — <i>{u['league']}</i>\n"

        await update.message.reply_text(msg, parse_mode="HTML")

    except Exception as e:
        logger.error(f"local_leaderboard error in chat {update.effective_chat.id}: {e}")
        await update.message.reply_text("⚠️ Error fetching leaderboard.")


# ============================
# 🌍 GLOBAL XP LEADERBOARD
# ============================

@check_frozen
async def global_leaderboard(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.effective_chat.send_chat_action(action="typing")

    try:
        def fetch_global():
            return list(
                db.collection("users")
                  .order_by("xp", direction=firestore.Query.DESCENDING)
                  .limit(20)
                  .stream()
            )

        users = await asyncio.to_thread(fetch_global)
        if not users:
            return await update.message.reply_text("No XP data found yet 😔")

        msg = "🌍 **GLOBAL XP LEADERBOARD** 🏆\n\n"
        for i, doc in enumerate(users, start=1):
            data = doc.to_dict()
            name = data.get("username") or f"User {data.get('user_id')}"
            xp = data.get("xp", 0)
            league = data.get("league", "Bronze")

            medal = "👑🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
            msg += f"{medal} **{name}** — `{xp} XP` — _{league}_\n"

        await update.message.reply_text(msg, parse_mode="Markdown")
    except Exception as e:
        print(e)
        await update.message.reply_text("⚠ Error fetching global leaderboard.")

# =====================================
# 🔥 USER PROFILE COMMAND
# =====================================

@check_frozen
async def profile(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    user_id = user.id
    chat = update.effective_chat
    group_id = chat.id

    await update.effective_chat.send_chat_action(action="typing")

    try:
        # 📌 Fetch XP, League, Reputation
        def fetch_user_data():
            doc = db.collection("users").document(str(user_id)).get()
            return doc.to_dict() if doc.exists else {}

        user_data = await asyncio.to_thread(fetch_user_data)

        xp = user_data.get("xp", 0)
        league = user_data.get("league", "Bronze")
        rep = user_data.get("reputation", 0)

        # 📌 Global Rank Calculation
        def fetch_rank():
            users = list(db.collection("users")
                .order_by("xp", direction=firestore.Query.DESCENDING)
                .stream())
            ids_sorted = [u.id for u in users]
            for index, doc in enumerate(users):
                if doc.id == str(user_id):
                    return index + 1, len(users)
            return None, len(users)

        rank, total_users = await asyncio.to_thread(fetch_rank)

        # 📌 Group Warning Count
        async def get_group_warns():
            ref = db.collection("groups").document(str(group_id)).collection("users").document(str(user_id))
            doc = await asyncio.to_thread(ref.get)
            return doc.to_dict().get("warnings", 0) if doc.exists else 0

        warn_count = await get_group_warns()

        # 📌 Message Building
        profile_msg = (
            f"👤 **PROFILE CARD**\n\n"
            f"🪪 **Name:** {user.full_name}\n"
            f"🆔 **User ID:** `{user_id}`\n\n"
            f"💠 **League:** `{league}`\n"
            f"⚡ **XP:** `{xp}`\n"
            f"🌍 **Global Rank:** `{rank}/{total_users}`\n"
            f"💎 **Reputation:** `{rep}`\n"
            f"⚠ **Warnings in this group:** `{warn_count}`\n"
        )

        await update.message.reply_text(profile_msg, parse_mode="Markdown")

    except Exception as e:
        print(e)
        await update.message.reply_text("⚠ Unable to fetch profile. Please try again later.")

# --- NCERT Quiz Feature (New) ---

async def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extracts text from PDF bytes using pypdf."""
    if not PYPDF_AVAILABLE:
        return "Error: pypdf library is not installed. Cannot extract text."
    try:
        pdf_file = io.BytesIO(file_bytes)
        reader = pypdf.PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        return f"Error processing PDF: {str(e)}"

@check_frozen
async def save_chapter(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Saves a PDF chapter to Firestore.
    Usage: User replies to a PDF with /save_chapter <chapter_name>
    """
    if not await check_admin(update, context): return
    if not db:
        await update.message.reply_text("Database not available.")
        return

    # Check if it's a reply to a document
    reply = update.message.reply_to_message
    if not reply or not reply.document or reply.document.mime_type != 'application/pdf':
        await update.message.reply_text("⚠️ Please reply to a **PDF file** with this command.")
        return

    if not context.args:
        await update.message.reply_text("⚠️ Usage: `/save_chapter <chapter_name>` (Reply to the PDF)", parse_mode="Markdown")
        return

    chapter_name = " ".join(context.args).strip().lower().replace(" ", "_")
    file_id = reply.document.file_id
    file_name = reply.document.file_name

    # Save to Firestore collection 'ncert_chapters'
    try:
        ref = db.collection("ncert_chapters").document(chapter_name)
        await asyncio.to_thread(ref.set, {
            "file_id": file_id,
            "file_name": file_name,
            "uploaded_by": update.effective_user.id,
            "uploaded_at": firestore.SERVER_TIMESTAMP
        })
        # <<-- SAFE: send plain text reply (no parse_mode) to avoid Markdown/entity parsing errors
        await update.message.reply_text(
            f"✅ Chapter '{chapter_name}' saved successfully 🪄! You can now generate quizzes from it.",
            parse_mode=None
        )
    except Exception as e:
        logger.error(f"Error saving chapter: {e}")
        await update.message.reply_text(f"Failed to save chapter: {e}")
@check_frozen
async def change_quiz_timing(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Changes the interval between quiz questions."""
    if not await check_admin(update, context): return
    if not db: return
    
    if not context.args or not context.args[0].isdigit():
        await update.message.reply_text("Usage: `/change_timing <seconds>` (e.g., `/change_timing 15`)")
        return
    
    seconds = int(context.args[0])
    if seconds < 5:
        await update.message.reply_text("Minimum timing is 5 seconds.")
        return

    group_id = update.effective_chat.id
    try:
        ref = db.collection("group_settings").document(str(group_id))
        await asyncio.to_thread(ref.set, {"quiz_interval": seconds}, merge=True)
        await update.message.reply_text(f"✅ Quiz timing set to **{seconds} seconds**.", parse_mode="Markdown")
    except Exception as e:
        await update.message.reply_text(f"Error: {e}")

async def generate_quiz_questions(text_content: str, num_questions: int) -> list:
    """Generates quiz questions using Gemini."""
    if not GEMINI_API_KEY: return []

    # Limit text content if it's too massive (Flash handles 1M tokens, but safe limit 100k chars for speed)
    trimmed_text = text_content[:100000] 
    
    prompt = (
        f"Based on the following NCERT chapter text, generate exactly {num_questions} multiple-choice questions (MCQs). "
        "The questions should cover key concepts, diagrams (if described in text), and important details. "
        "Number of characters or letters in each 'question' must not exceed 250"
        "Every 'question' must be short and concise"
        "Must mention - '@Chhoti_don_bot' after the end of every 'question' (not after the explanation, but after the question)"
        "Return the response strictly as a **JSON list of objects**. "
        "Do NOT use markdown code blocks. Just return the raw JSON string. "
        "Each object must have these fields: "
        "1. 'question' (string), "
        "2. 'options' (list of 4 strings), "
        "3. 'correct_option_id' (integer index 0-3), "
        "4. 'explanation' (string - brief reason). "
        f"\n\nTEXT CONTENT:\n{trimmed_text}"
    )

    url = f"{GEMINI_API_URL}?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"responseMimeType": "application/json"}
    }

    try:
        response = await asyncio.to_thread(requests.post, url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        raw_text = data['candidates'][0]['content']['parts'][0]['text']
        # clean potential markdown
        raw_text = raw_text.replace("```json", "").replace("```", "").strip()
        return json.loads(raw_text)
    except Exception as e:
        logger.error(f"Quiz Generation Error: {e}")
        return []


# --- Modified handle_quiz_answer (persisted fallback) ---
async def handle_quiz_answer(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handles incoming poll answers.
    Robustly attempts to find the poll_id in memory or DB to prevent 'Ignoring' valid answers.
    """
    try:
        answer = update.poll_answer
        if not answer:
            return

        poll_id = str(answer.poll_id)
        user = answer.user
        selected_ids = answer.option_ids

        # 1. Try In-Memory Lookup first (Fastest)
        poll_data = active_polls.get(poll_id)

        # 2. If not in memory, try Direct DB Lookup (for restarts/multi-worker)
        if not poll_data:
            poll_data = await _load_active_poll_from_db(poll_id)
            if poll_data:
                active_polls[poll_id] = poll_data # Cache it back to memory
        
        # 3. If still missing, try a desperate scan of active sessions (Fallback)
        if not poll_data:
            if db:
                try:
                    # Look for any session that claims this poll_id
                    sessions_ref = db.collection("live_quiz_sessions")
                    # Note: querying by array-contains is better than stream() for performance
                    # But keeping stream() here to match your existing structure safely
                    docs = await asyncio.to_thread(sessions_ref.stream)
                    
                    for doc in docs:
                        data = doc.to_dict()
                        if poll_id in data.get("poll_ids", []):
                            # Found the session!
                            chat_id = int(doc.id)
                            # Reconstruct poll data partially if missing from live_active_polls
                            poll_data = {
                                'chat_id': chat_id,
                                'correct_option': None, # We might miss correctness if poll record is gone, but we track participation
                                'start_time': datetime.now(timezone.utc)
                            }
                            # Attempt to load specific poll doc again just in case
                            saved_poll = await _load_active_poll_from_db(poll_id)
                            if saved_poll:
                                poll_data = saved_poll
                                
                            active_polls[poll_id] = poll_data
                            logger.info(f"[quiz] Recovered poll {poll_id} via session scan for chat {chat_id}")
                            break
                except Exception as e:
                    logger.error(f"[quiz] DB Scan error: {e}")

        # 4. If absolutely valid poll data is found
        if poll_data:
            chat_id = poll_data['chat_id']
            correct_id = poll_data.get('correct_option')
            start_time = poll_data.get('start_time', datetime.now(timezone.utc))

            # Ensure session exists in memory
            if chat_id not in quiz_sessions:
                loaded_session = await _load_quiz_session_from_db(chat_id)
                if loaded_session:
                    quiz_sessions[chat_id] = loaded_session
                else:
                    # Create temporary session if missing so we don't crash
                    quiz_sessions[chat_id] = {'scores': {}, 'poll_ids': [poll_id]}

            # Calculate Score
            session = quiz_sessions[chat_id]
            scores = session.setdefault('scores', {})
            
            # Initialize user stats if new
            if user.id not in scores:
                scores[user.id] = {
                    'name': user.full_name or user.username or "Student",
                    'score': 0,
                    'total_time': 0.0
                }
            
            # Update Name (in case it changed)
            scores[user.id]['name'] = user.full_name or user.username or scores[user.id]['name']

            # Check Answer
            # Note: correct_id might be None if we recovered from a crash without full metadata,
            # but usually we want to count the score.
            if correct_id is not None and correct_id in selected_ids:
                scores[user.id]['score'] += 1
                time_taken = (datetime.now(timezone.utc) - start_time).total_seconds()
                scores[user.id]['total_time'] += time_taken
            
            # Save immediately to DB to persist this answer
            await _save_quiz_session_to_db(chat_id, session)
            logger.info(f"[quiz] Answer recorded for {user.id} in chat {chat_id}")
            
        else:
            # Only log ignore if we truly couldn't find it anywhere
            logger.warning(f"[quiz] IGNORED answer for poll {poll_id} - Could not link to any active quiz.")

    except Exception as e:
        logger.error(f"Error in handle_quiz_answer: {e}")

@check_frozen
async def start_quiz(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Validates inputs and spawns the background quiz task.
    Returns IMMEDIATELY so the bot remains responsive to answers.
    """
    if not db:
        await update.message.reply_text("Database not available.")
        return
        
    if len(context.args) < 2:
        # FIX 1: Changed usage message to HTML
        await update.message.reply_text(
            "⚠️ Usage: <code>/quiz &lt;num_questions&gt; &lt;chapter_name&gt;</code>", 
            parse_mode="HTML"
        )
        return

    try:
        num_questions = int(context.args[0])
    except ValueError:
        await update.message.reply_text("⚠️ Number of questions must be an integer.")
        return

    chapter_name = " ".join(context.args[1:]).strip().lower().replace(" ", "_")
    group_id = update.effective_chat.id

    # 1. Initialize Session immediately so we don't have race conditions later
    quiz_sessions[group_id] = {'scores': {}, 'poll_ids': [], 'is_active': True}
    await _save_quiz_session_to_db(group_id, quiz_sessions[group_id])

    # 2. Check if chapter exists before spawning task
    try:
        doc_ref = db.collection("ncert_chapters").document(chapter_name)
        doc = await asyncio.to_thread(doc_ref.get)
        if not doc.exists:
            # Note: This error message still uses Markdown for the name part, 
            # but since the chapter wasn't found, the string is likely safe.
            # However, for safety, let's switch this to HTML too.
            display_name = chapter_name.replace('_', ' ').title()
            await update.message.reply_text(
                f"❌ Chapter <b>'{display_name}'</b> not found. Use <code>/save_chapter</code> first.", 
                parse_mode="HTML"
            )
            return
        file_id = doc.to_dict().get("file_id")
    except Exception as e:
        await update.message.reply_text(f"Database error: {e}")
        return

    if not PYPDF_AVAILABLE:
        await update.message.reply_text("❌ `pypdf` library is missing.")
        return

    # 3. SPAWN THE BACKGROUND TASK
    context.application.create_task(
        run_quiz_background(group_id, file_id, chapter_name, num_questions, context)
    )

async def run_quiz_background(group_id: int, file_id: str, chapter_name: str, num_questions: int, context: ContextTypes.DEFAULT_TYPE):
    """
    The heavy lifting: Downloads PDF, Generates AI Qs, Sends Polls, handles Leaderboard.
    Runs in background so it doesn't block PollAnswer updates.
    """
    try:
        # A. Download & Process PDF
        # FIX 2: Changed initial status message to HTML
        display_name = chapter_name.replace('_', ' ').title()
        status_msg = await context.bot.send_message(
            chat_id=group_id, 
            text=f"🚀 <b>Quiz Initialized🕊!</b>\nProcessing <b>{display_name}</b> & generating {num_questions} questions...\n Quiz will start shortly་࿐",
            parse_mode="HTML"
        )
              
        new_file = await context.bot.get_file(file_id)
        pdf_data = io.BytesIO()
        await new_file.download_to_memory(out=pdf_data)
        pdf_data.seek(0)
        text_content = await extract_text_from_pdf(pdf_data.read())

        # B. Generate Questions (Gemini)
        questions = await generate_quiz_questions(text_content, num_questions)
        
        if not questions:
            await status_msg.edit_text("❌ AI failed to generate questions. Please try again.")
            return

        display_name = chapter_name.replace('_', ' ').title()
        await status_msg.edit_text(
            f"🧠 Quiz Starting! {len(questions)} questions on <b>{display_name}</b> 🔥.",
            parse_mode="HTML"
        )
        
        # C. Get Timing
        interval = 10
        try:
            settings = await asyncio.to_thread(db.collection("group_settings").document(str(group_id)).get)
            if settings.exists:
                interval = settings.to_dict().get("quiz_interval", 10)
        except: pass

        # D. Question Loop
        for i, q in enumerate(questions):
            # NEW: Check if the quiz was stopped externally
            if not quiz_sessions.get(group_id, {}).get('is_active', False):
                logger.info(f"Quiz {group_id} detected external stop signal. Exiting loop.")
                break # <--- EXIT THE LOOP IMMEDIATELY
            try:
                # --- NEW LOGIC START ---
                
                # 1. Prepare Question Text with Options
                raw_options = q.get('options', [])
                
                # Build the text message: Question + Numbered Options
                # We use html.escape to safely handle special characters (<, >, &)
                question_text_msg = f"<b>Q{i+1}: {html.escape(q.get('question', ''))}</b>\n\n"
                
                for idx, opt in enumerate(raw_options):
                    question_text_msg += f"<b>{idx + 1}.</b> {html.escape(str(opt))}\n"

                # 2. Send the Text Message First
                try:
                    await context.bot.send_message(
                        chat_id=group_id,
                        text=question_text_msg,
                        parse_mode='HTML'
                    )
                except Exception as e:
                    logger.error(f"Error sending question text HTML: {e}")
                    # Fallback: send without HTML if parsing fails
                    clean_text = question_text_msg.replace('<b>', '').replace('</b>', '')
                    await context.bot.send_message(chat_id=group_id, text=clean_text)

                # 3. Prepare Numeric Options for the Poll
                # If there are 4 options, this creates ['1', '2', '3', '4']
                numeric_options = [str(idx + 1) for idx in range(len(raw_options))]

                # 4. Send the Poll (Numeric)
                # Ensure explanation fits limit (200 chars)
                expl = q.get('explanation', "No explanation.")
                safe_explanation = expl[:200] if expl else "No explanation."

                poll_message = await context.bot.send_poll(
                    chat_id=group_id,
                    question="Quiz by @Chhoti_don_bot 🕊",  # Your Branding Placeholder
                    options=numeric_options,              # Only numbers: ['1', '2', '3', '4']
                    type=Poll.QUIZ,
                    correct_option_id=q.get('correct_option_id', 0),
                    explanation=safe_explanation,
                    explanation_parse_mode="Markdown",
                    is_anonymous=False, 
                    open_period=interval
                )
                
                # --- NEW LOGIC END ---

                # Register Poll (Existing logic kept exactly the same)
                poll_key = str(poll_message.poll.id)
                poll_data = {
                    'chat_id': group_id,
                    'correct_option': q.get('correct_option_id', 0),
                    'start_time': datetime.now(timezone.utc)
                }
                
                # Update Memory & DB
                active_polls[poll_key] = poll_data
                if group_id in quiz_sessions:
                    quiz_sessions[group_id]['poll_ids'].append(poll_key)
                
                await _save_active_poll_to_db(poll_key, poll_data)
                # Important: Save session to link poll_id to group
                await _save_quiz_session_to_db(group_id, quiz_sessions.get(group_id, {}))
                
                # Wait for answers
                wait_buffer = 5
                if i == len(questions) - 1:
                    logger.info(f"Last question sent. Waiting {interval + 15}s for final answers...")
                    await asyncio.sleep(interval + 15)
                else:
                    await asyncio.sleep(interval + wait_buffer)

            except Exception as e:
                logger.error(f"Error sending poll Q{i+1}: {e}")
                continue

        # E. Leaderboard & Cleanup
        # Now that we are in background, this logic will run strictly AFTER the waits
        final_scores = {}
        
        # Try local first
        local_session = quiz_sessions.get(group_id, {})
        final_scores = local_session.get('scores', {})

        if not final_scores:
            logger.info("[Leaderboard] No local scores found. Starting DB Retry Loop...")
            
            # Retry loop to catch slightly delayed DB writes
            for attempt in range(5):
                logger.info(f"[Leaderboard] Attempt {attempt+1}/5 to fetch scores from DB...")
                await asyncio.sleep(3) 

                try:
                    db_session = await _load_quiz_session_from_db(group_id)
                    if db_session:
                        final_scores = db_session.get('scores', {})
                    
                    if final_scores:
                        break
                except Exception as e:
                    logger.error(f"[Leaderboard] Error on attempt {attempt+1}: {e}")

        # ... (Retry loop code above remains the same) ...

        # Send Leaderboard Message (HTML VERSION)
        if not final_scores:
            await context.bot.send_message(
                chat_id=group_id, 
                text=f"🏁 Quiz on <b>{chapter_name.replace('_', ' ').title()}</b> completed(or stopped in the middle)! (No final results to show🥱 Either no one took the quiz or you guys didn't have the patience to complete the quiz🫠🙂) (jis jis ko angreji smjh nahi aayi woh /ask command use krke translate krwa lo🙃)",
                parse_mode="HTML"
            )
        else:
            # Sort: Highest Score first, then Lowest Time
            ranked = sorted(
                final_scores.values(), 
                key=lambda x: (-x['score'], x['total_time'])
            )
            
            # Use HTML to avoid crashes with underscores (_) or asterisks (*)
            display_chapter = chapter_name.replace('_', ' ').title()
            text = f"꧁⎝ 𓆩༺✧ <b>Leaderboard:</b> ✧༻𓆪 ⎠꧂\n 🏆 <b>{display_chapter}</b>  🏆\n\n"
            
            for rank, user in enumerate(ranked, 1):
                # Escape the name to prevent HTML injection errors
                safe_name = html.escape(user['name'])
                score = user['score']
                time_taken = user['total_time']
                text += f"{rank}. <b>{safe_name}</b> — {score} pts ({time_taken:.1f}s)\n"
            
            await context.bot.send_message(chat_id=group_id, text=text, parse_mode="HTML")
            await reward_top_three(group_id, context.bot, chat_id=group_id)
            await save_group_scores_to_db(group_id)       # ⭐ permanent storage


        # Cleanup
        context.application.create_task(cleanup_quiz_data(group_id))

    except Exception as e:
        logger.error(f"Background Quiz Error: {e}")
        # Fallback error message (plain text, no parse mode)
        await context.bot.send_message(chat_id=group_id, text="⚠️ An error occurred during the quiz process.")



async def cleanup_quiz_data(group_id: int):
    """Background task to clean up quiz data after a delay."""
    await asyncio.sleep(30) # Keep data alive for 30s after quiz ends
    try:
        if group_id in quiz_sessions:
            for pid in quiz_sessions[group_id]['poll_ids']:
                if pid in active_polls:
                    del active_polls[pid]
                await _delete_active_poll_from_db(pid)
            del quiz_sessions[group_id]
            await _delete_quiz_session_from_db(group_id)
    except Exception as e:
        logger.error(f"Cleanup error: {e}")

@check_frozen
async def stop_quiz(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Immediately stops an ongoing quiz & SAVES SCORES to DB."""
    chat_id = update.effective_chat.id

    if chat_id not in quiz_sessions:
        await update.message.reply_text("⚠️ No ongoing quiz to stop.")
        return

    session = quiz_sessions.get(chat_id, {})
    scores = session.get("scores", {})
    poll_ids = session.get("poll_ids", [])

    # 1. Stop background activity (Signals the loop in run_quiz_background to break)
    if chat_id in quiz_sessions:
        quiz_sessions[chat_id]['is_active'] = False

    # 2. Clean up Active Polls
    for pid in poll_ids:
        active_polls.pop(pid, None)
        try:
            await _delete_active_poll_from_db(pid)
        except:
            pass

    # =======================================================
    # 🔥 CRITICAL FIX: Save scores to DB before deleting!
    # =======================================================
    if scores:
        await save_group_scores_to_db(chat_id)

    # 3. Clean DB session storage
    try:
        await _delete_quiz_session_from_db(chat_id)
    except:
        pass

    # 4. Remove from memory
    quiz_sessions.pop(chat_id, None)

    # 5. Show Partial Leaderboard
    if not scores:
        await update.message.reply_text("🛑 Quiz stopped.\nNo scores recorded yet.")
        return

    ranked = sorted(scores.values(), key=lambda x: (-x["score"], x["total_time"]))

    leaderboard = "🛑 **Quiz Stopped Manually**\n(Scores have been saved)\n\n📊 **Session Leaderboard**:\n\n"
    for i, user in enumerate(ranked, 1):
        leaderboard += f"{i}. **{user['name']}** — {user['score']} pts ({user['total_time']:.2f}s)\n"

    await update.message.reply_text(leaderboard, parse_mode="Markdown")

async def get_all_chapters_from_db() -> list:
    """Fetches all chapter data from Firestore."""
    if not db: return []
    try:
        # Stream all documents in ncert_chapters
        docs = await asyncio.to_thread(db.collection("ncert_chapters").stream)
        chapters = []
        for doc in docs:
            data = doc.to_dict()
            # We need the document ID (chapter name) and file_id
            data['chapter_name'] = doc.id 
            if 'file_id' in data:
                chapters.append(data)
        return chapters
    except Exception as e:
        logger.error(f"Error fetching chapters: {e}")
        return []

async def autonomous_quiz_worker(group_id: int, context: ContextTypes.DEFAULT_TYPE):
    """
    The Supervisor Task:
    1. Fetches chapters.
    2. Runs a quiz.
    3. Waits for it to finish.
    4. Rests for 2 minutes.
    5. Repeats with the next chapter.
    """
    logger.info(f"Starting autonomous loop for {group_id}")
    
    # 1. Fetch Chapters once (or you can fetch inside loop if you want updates)
    chapters = await get_all_chapters_from_db()
    
    if not chapters:
        await context.bot.send_message(group_id, "⚠️ No chapters found in database to start autonomous mode.")
        autonomous_tasks.pop(group_id, None)
        return

    chapter_index = 0
    
    try:
        while True:
            # Check if stopped externally
            if group_id not in autonomous_tasks:
                break

            # 2. Get Settings (Number of Questions)
            # Default is 10, or check group_settings if customized
            num_questions = 10
            try:
                settings_doc = await asyncio.to_thread(db.collection("group_settings").document(str(group_id)).get)
                if settings_doc.exists:
                    num_questions = settings_doc.to_dict().get("auto_quiz_questions", 10)
            except:
                pass

            # 3. Select Chapter (Round Robin)
            current_chapter = chapters[chapter_index % len(chapters)]
            c_name = current_chapter['chapter_name']
            f_id = current_chapter['file_id']

            # 4. Notify & Start Quiz
            await context.bot.send_message(
                group_id, 
                f"♾️ **Autonomous Mode Active**\nStarting next chapter: `{c_name.replace('_', ' ').title()}`", 
                parse_mode="Markdown"
            )

            # Initialize session for the quiz
            quiz_sessions[group_id] = {'scores': {}, 'poll_ids': [], 'is_active': True}
            await _save_quiz_session_to_db(group_id, quiz_sessions[group_id])

            # RUN THE QUIZ and AWAIT its completion
            # We call run_quiz_background directly. Since it is an async function, 
            # await will block this loop until the quiz (questions -> leaderboard) is done.
            await run_quiz_background(group_id, f_id, c_name, num_questions, context)

            # 5. The "Rest" Period
            # Check if we should still continue (in case user stopped it during the quiz)
            if group_id not in autonomous_tasks:
                break
                
            rest_minutes = 2
            await context.bot.send_message(
                group_id, 
                f"☕ **Break Time!**\nTaking a {rest_minutes} minute rest before the next chapter...", 
                parse_mode="Markdown"
            )
            
            # Wait for 2 minutes (120 seconds)
            await asyncio.sleep(rest_minutes * 60)

            # Increment index for next chapter
            chapter_index += 1

    except asyncio.CancelledError:
        logger.info(f"Autonomous task cancelled for {group_id}")
    except Exception as e:
        logger.error(f"Autonomous worker crashed for {group_id}: {e}")
        await context.bot.send_message(group_id, "⚠️ Autonomous mode stopped due to an error.")
    finally:
        # Cleanup key if loop exits
        autonomous_tasks.pop(group_id, None)

@check_frozen
async def set_auto_questions(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sets the number of questions for autonomous quizzes."""
    if not await check_admin(update, context): return
    if not db: return

    if not context.args or not context.args[0].isdigit():
        await update.message.reply_text("Usage: `/set_auto_questions <number>` (e.g., 15)", parse_mode="Markdown")
        return

    count = int(context.args[0])
    if count < 5 or count > 50:
        await update.message.reply_text("Please set a number between 5 and 50.")
        return

    group_id = update.effective_chat.id
    try:
        ref = db.collection("group_settings").document(str(group_id))
        await asyncio.to_thread(ref.set, {"auto_quiz_questions": count}, merge=True)
        await update.message.reply_text(f"✅ Autonomous quizzes will now have **{count}** questions.", parse_mode="Markdown")
    except Exception as e:
        await update.message.reply_text(f"Error: {e}")

@check_frozen
async def start_autonomous(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Starts the 24/7 autonomous quiz loop."""
    if not await check_admin(update, context): return
    if not db:
        await update.message.reply_text("Database not available.")
        return

    group_id = update.effective_chat.id

    if group_id in autonomous_tasks:
        await update.message.reply_text("⚠️ Autonomous mode is already running in this chat!")
        return

    # Create the background task for the loop
    task = context.application.create_task(autonomous_quiz_worker(group_id, context))
    autonomous_tasks[group_id] = task
    
    await update.message.reply_text(
        "🚀 **Autonomous Mode Enabled!**\n"
        "I will now cycle through saved chapters 24/7.\n"
        "• 10 questions (default) per chapter\n"
        "• 2 minutes rest between quizzes\n"
        "• Use `/stop_autonomous` to stop.",
        parse_mode="Markdown"
    )

@check_frozen
async def stop_autonomous(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Stops the autonomous loop and the current quiz."""
    if not await check_admin(update, context): return

    group_id = update.effective_chat.id

    if group_id not in autonomous_tasks:
        await update.message.reply_text("⚠️ Autonomous mode is not running.")
        return

    # 1. Cancel the Supervisor Task
    task = autonomous_tasks[group_id]
    task.cancel()
    del autonomous_tasks[group_id]

    # 2. Stop the CURRENT ongoing quiz using existing logic
    # This sets is_active=False, which stops the question loop inside run_quiz_background
    if group_id in quiz_sessions:
        quiz_sessions[group_id]['is_active'] = False
        
        # Clean up polls immediately
        poll_ids = quiz_sessions[group_id].get('poll_ids', [])
        for pid in poll_ids:
            active_polls.pop(pid, None)
            await _delete_active_poll_from_db(pid)
            
        await _delete_quiz_session_from_db(group_id)
        quiz_sessions.pop(group_id, None)

    await update.message.reply_text("🛑 **Autonomous Mode Stopped.**")

@check_frozen
async def auto_react_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    chat = update.effective_chat

    if not msg or not msg.text:
        return

    if chat.type not in ["group", "supergroup"]:
        return

    if msg.text.startswith("/"):
        return

    if not await is_auto_reaction_enabled(chat.id):
        return

    # 🧠 Detect mood
    mood = detect_mood(msg.text)
    mood_pool = MOOD_REACTIONS.get(mood, MOOD_REACTIONS["neutral"])

    # 🔁 Get learned allowed reactions
    allowed = await get_allowed_reactions(chat.id)

    # ✅ Intersection
    possible = [r for r in mood_pool if r in allowed]

    if not possible:
        possible = allowed  # fallback

    if not possible:
        return

    chosen = random.choice(possible)

    try:
        await context.bot.set_message_reaction(
            chat_id=chat.id,
            message_id=msg.message_id,
            reaction=chosen
        )

    except Exception:
        # ❌ Remove blocked reaction and relearn
        if chosen in allowed:
            allowed.remove(chosen)
            await save_allowed_reactions(chat.id, allowed)

DEFAULT_REACTIONS_POOL = [
    "👍", "❤️", "🔥", "😂", "👏", "😎", "🤯", "💯", "😍",
    "🥰", "😁", "🤝", "🎉", "🤩"
]

async def get_allowed_reactions(group_id: int) -> list[str]:
    if not db:
        return DEFAULT_REACTIONS_POOL

    try:
        ref = db.collection("group_settings").document(str(group_id))
        doc = await asyncio.to_thread(ref.get)

        data = doc.to_dict() if doc.exists else {}
        reactions = data.get("allowed_reactions")

        return reactions if reactions else DEFAULT_REACTIONS_POOL.copy()
    except Exception as e:
        logger.error(f"Failed to get reactions: {e}")
        return DEFAULT_REACTIONS_POOL.copy()

@check_frozen
async def reaction_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await check_admin(update, context):
        return

    group_id = update.effective_chat.id
    ref = db.collection("group_settings").document(str(group_id))

    await asyncio.to_thread(ref.set, {"auto_reaction": True}, merge=True)

    await update.message.reply_text(
        "😄 Auto reactions are now *ON* for this chat!",
        parse_mode="Markdown"
    )


@check_frozen
async def reaction_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await check_admin(update, context):
        return

    group_id = update.effective_chat.id
    ref = db.collection("group_settings").document(str(group_id))

    await asyncio.to_thread(ref.set, {"auto_reaction": False}, merge=True)

    await update.message.reply_text(
        "🛑 Auto reactions are now *OFF* for this chat!",
        parse_mode="Markdown"
    )

@check_frozen
async def reaction_reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await check_admin(update, context):
        return

    ref = db.collection("group_settings").document(str(update.effective_chat.id))
    await asyncio.to_thread(
        ref.update,
        {"allowed_reactions": firestore.DELETE_FIELD}
    )

    await update.message.reply_text("♻️ Reaction list reset. Bot will relearn allowed reactions.")


@owner_override
async def broadcast_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Owner-only command to broadcast a message to all tracked chats."""
    if not is_owner(update.effective_user.id):
        await update.message.reply_text("This command is restricted to the bot owner.")
        return
    if not db:
        await update.message.reply_text("Database not available.")
        return

    # --- NEW FEATURE: Broadcast replied message ---
    replied = update.message.reply_to_message
    if replied:
        if replied.text:
            message_to_send = replied.text
        elif replied.caption:
            message_to_send = replied.caption
        else:
            await update.message.reply_text("Replied message has no text to broadcast.")
            return
    
       
    else:
        # Old method: /broadcast <message>
        if len(context.args) < 1:
            await update.message.reply_text(
                "Reply to a message with /broadcast or use /broadcast <message>.",
                parse_mode="Markdown"
            )
            return
        
        # Extract message normally
        message_to_send = re.sub(r'/\w+\s*', '', update.message.text, 1)

    
    # 1. Fetch all chat IDs from the 'broadcast_chats' collection
    chats_ref = db.collection("broadcast_chats")
    
    try:
        chats_snapshot = await asyncio.to_thread(chats_ref.stream)
        
        sent_count = 0
        failed_count = 0
        
        # Determine the source chat ID to skip sending the broadcast back to the owner's chat
        source_chat_id = str(update.effective_chat.id)

        for doc in chats_snapshot:
            chat_data = doc.to_dict()
            chat_id = chat_data.get("chat_id")
            
            if not chat_id or chat_id == source_chat_id:
                continue

            try:
                # 2. Send the message to each chat
                # We use HTML parsing for formatting (bold, links, etc.)
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=message_to_send,
                    parse_mode="HTML"
                )
                sent_count += 1
            except Exception as e:
                # Catch exceptions like 'Bot was blocked by the user', 'Chat not found', etc.
                logger.warning(f"Failed to send broadcast to chat {chat_id}: {e}")
                # The exception might contain a Telegram error code that indicates the bot was kicked/left
                # In a production setting, you might want to delete the chat from the broadcast_chats here.
                failed_count += 1
                
        # 3. Report the result back to the owner
        await update.message.reply_text(
            f"✅ Broadcast complete!\n"
            f"Sent to **{sent_count}** chats.\n"
            f"Failed to send to **{failed_count}** chats (likely blocked or left the group).",
            parse_mode="Markdown"
        )
        
    except Exception as e:
        logger.error(f"Error during broadcast: {e}")
        await update.message.reply_text(f"An unexpected database error occurred during broadcast: {e}")


# --- Owner Control Panel Commands ---

@owner_override
async def set_welcome_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_owner(update.effective_user.id):
        return await update.message.reply_text("❌ This command is owner-only.")

    if not update.message.reply_to_message or not update.message.reply_to_message.photo:
        return await update.message.reply_text(
            "⚠️ Please reply to an image using /image"
        )

    photo = update.message.reply_to_message.photo[-1]
    file_id = photo.file_id

    try:
        doc_ref = db.collection("bot_config").document("welcome_image")

        # overwrite old image (auto delete old ref)
        doc_ref.set({
            "file_id": file_id
        })

        await update.message.reply_text(
            "✅ Welcome image updated successfully."
        )

    except Exception as e:
        await update.message.reply_text(
            f"❌ Failed to save image.\nError: {e}"
        )

# --- Owner-only: Send active countdowns to all tracked groups ---
@owner_override
async def send_countdown(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Owner-only command: send active countdowns to all tracked chats."""
    if not is_owner(update.effective_user.id):
        await update.message.reply_text("This command is restricted to the bot owner.")
        return

    if not db:
        await update.message.reply_text("Database not available.")
        return

    chats_ref = db.collection("broadcast_chats")

    try:
        chats_snapshot = await asyncio.to_thread(chats_ref.stream)
    except Exception as e:
        logger.error(f"[send_countdown] Failed to fetch broadcast_chats: {e}")
        await update.message.reply_text(f"Failed to load tracked chats: {e}")
        return

    sent_count = 0
    failed_count = 0
    cleared_count = 0
    now = datetime.now(timezone.utc)

    for doc in chats_snapshot:
        chat_data = doc.to_dict()
        chat_id_str = chat_data.get("chat_id")
        if not chat_id_str:
            continue

        # load the group's countdown settings
        try:
            group_ref = db.collection("group_settings").document(str(chat_id_str))
            group_doc = await asyncio.to_thread(group_ref.get)
            group = group_doc.to_dict() or {}
        except Exception as e:
            logger.warning(f"[send_countdown] Could not load settings for chat {chat_id_str}: {e}")
            continue

        target_iso = group.get("target_date_iso")
        countdown_name = group.get("countdown_name")
        target_human = group.get("target_date_human", "")

        # If no countdown set, skip
        if not target_iso or not countdown_name:
            continue

        # Parse and check
        try:
            target_dt = datetime.fromisoformat(target_iso)
            # ensure timezone-aware in UTC for consistent comparison
            if target_dt.tzinfo is None:
                target_dt = target_dt.replace(tzinfo=timezone.utc)
            else:
                target_dt = target_dt.astimezone(timezone.utc)
        except Exception as e:
            logger.warning(f"[send_countdown] Invalid date for chat {chat_id_str}: {e}. Clearing countdown.")
            # attempt to clear corrupted countdown entry
            try:
                await asyncio.to_thread(group_ref.update, {
                    "countdown_name": firestore.DELETE_FIELD,
                    "target_date_iso": firestore.DELETE_FIELD,
                    "target_date_human": firestore.DELETE_FIELD
                })
                cleared_count += 1
            except:
                pass
            continue

        remaining = target_dt - now

        # If expired or due, clear and optionally notify (we'll clear and skip sending)
        if remaining.total_seconds() <= 0:
            try:
                await asyncio.to_thread(group_ref.update, {
                    "countdown_name": firestore.DELETE_FIELD,
                    "target_date_iso": firestore.DELETE_FIELD,
                    "target_date_human": firestore.DELETE_FIELD
                })
                cleared_count += 1
                logger.info(f"[send_countdown] Cleared expired countdown for chat {chat_id_str}")
            except Exception as e:
                logger.warning(f"[send_countdown] Failed to clear expired countdown for {chat_id_str}: {e}")
            continue

        # Format remaining time
        days = remaining.days
        hours = remaining.seconds // 3600
        minutes = (remaining.seconds % 3600) // 60
        seconds = remaining.seconds % 60

        # Compose message
        msg = (
            f"⏳ **{countdown_name}**\n"
            f"Target: `{target_human}`\n\n"
            f"**Time Remaining:**\n"
            f"• `{days}` days\n"
            f"• `{hours}` hours\n"
            f"• `{minutes}` minutes\n"
            f"• `{seconds}` seconds"
        )

        # send to the chat (chat_id stored as string so convert)
        try:
            await context.bot.send_message(chat_id=int(chat_id_str), text=msg, parse_mode="Markdown")
            sent_count += 1
        except Exception as e:
            logger.warning(f"[send_countdown] Failed to send countdown to {chat_id_str}: {e}")
            failed_count += 1
            # optionally: if bot was removed from chat, you could delete chat from broadcast_chats here

    # ----- final summary to owner (ALWAYS DM to owner) -----
    owner_id = OWNER_ID   # same int variable used everywhere
    summary = (
        f"📊 *Countdown Report*\n\n"
        f"✔️ Sent: {sent_count}\n"
        f"❌ Failed: {failed_count}\n"
        f"🧹 Cleared expired/invalid: {cleared_count}\n"
    )

    try:
        await context.bot.send_message(chat_id=owner_id, text=summary, parse_mode="Markdown")
    except Exception as e:
        logger.error(f"[send_countdown] Failed to send summary to owner: {e}")

@owner_override
async def freeze_bot(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Owner-only command to freeze the bot globally."""
    if not is_owner(update.effective_user.id):
        return

    if not db:
        await update.message.reply_text("Database not available.")
        return

    ref = db.collection("global_settings").document("bot_state")
    try:
        await asyncio.to_thread(ref.set, {"is_frozen": True, "last_updated": firestore.SERVER_TIMESTAMP}, merge=True)
        await update.message.reply_text("🧊 **BOT FREEZE INITIATED.** All non-owner and non-utility commands will be blocked globally.", parse_mode="Markdown")
    except Exception as e:
        logger.error(f"Error freezing bot: {e}")
        await update.message.reply_text(f"❌ Failed to freeze bot: {e}")

@owner_override
async def resume_bot(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Owner-only command to resume the bot globally."""
    if not is_owner(update.effective_user.id):
        return
        
    if not db:
        await update.message.reply_text("Database not available.")
        return

    ref = db.collection("global_settings").document("bot_state")
    try:
        await asyncio.to_thread(ref.set, {"is_frozen": False, "last_updated": firestore.SERVER_TIMESTAMP}, merge=True)
        await update.message.reply_text("🟢 **BOT RESUMED.** All commands are now fully operational.", parse_mode="Markdown")
    except Exception as e:
        logger.error(f"Error resuming bot: {e}")
        await update.message.reply_text(f"❌ Failed to resume bot: {e}")

@owner_override
async def list_chats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Owner-only command to list all tracked chats."""
    if not is_owner(update.effective_user.id):
        return
        
    if not db:
        await update.message.reply_text("Database not available.")
        return

    chats_ref = db.collection("broadcast_chats")
    
    try:
        chats_snapshot = await asyncio.to_thread(chats_ref.stream)
        
        chat_list = []
        for doc in chats_snapshot:
            data = doc.to_dict()
            chat_id = data.get("chat_id")
            title = data.get("title", "N/A")
            chat_type = data.get("chat_type", "N/A")
            last_active = data.get("last_active", "N/A")
            
            # Format last_active timestamp
            if isinstance(last_active, firestore.SERVER_TIMESTAMP.__class__):
                 # Convert Firestore Timestamp to datetime (best effort)
                 last_active_str = last_active.isoformat()
            else:
                 last_active_str = str(last_active)
         
            title = title.replace('\\', '\\\\').replace('*', '\\*').replace('_', '\\_').replace('`', '\\`').replace('[', '\\[').replace(']', '\\]').replace('(', '\\(').replace(')', '\\)')
           
            chat_list.append(f"• **{title}** ({chat_type.upper()}) | ID: `{chat_id}` | Last Active: {last_active_str}")

        if not chat_list:
            await update.message.reply_text("No chats found in the broadcast list.", parse_mode="Markdown")
            return
            
        header = f"📋 **Tracked Chats ({len(chat_list)})**\n---------------------------------\n"
        full_list = header + "\n".join(chat_list)
        
        # Split message if too long for Telegram
        if len(full_list) > 4096:
            await update.message.reply_text(header + "\n".join(chat_list[:50]), parse_mode="Markdown")
            await update.message.reply_text(f"List truncated. Total chats: {len(chat_list)}", parse_mode="Markdown")
        else:
            await update.message.reply_text(full_list, parse_mode="Markdown")
            
    except Exception as e:
        logger.error(f"Error listing chats: {e}")
        await update.message.reply_text(f"❌ Failed to list chats: {e}")


# --- Main Application Setup ---

async def main() -> None:
    """Starts the bot using Webhook (required for free hosting)."""
    if not BOT_TOKEN or not WEBHOOK_URL:
        logger.error("BOT_TOKEN or WEBHOOK_URL environment variables are not set. Exiting.")
        return

    # Create the Application and pass it your bot's token.
    application = Application.builder().token(BOT_TOKEN).build()

    # --- Register Handlers ---
    
    # AI Chatbot Command (Updated to handle multimodal and context)
    application.add_handler(CommandHandler("ask", ask_ai))
    application.add_handler(CommandHandler("img", generate_image_command))
    application.add_handler(CommandHandler("set_personality", set_ai_personality))

    # Public/Utility Commands
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("id", get_user_id))
    application.add_handler(CommandHandler("purge", purge_messages)) 
    application.add_handler(CommandHandler("stats", stats))
    application.add_handler(CommandHandler("weather", weather))
    application.add_handler(CommandHandler("time", get_current_time))
    application.add_handler(CommandHandler("help", help_command))
    # --- New Reputation System Handlers ---
    # The command for giving reputation
    application.add_handler(CommandHandler("thanks", handle_thanks_command))
    # The command for showing the leaderboards
    application.add_handler(CommandHandler("repleaderboard", handle_reputation_leaderboard))
    application.add_handler(CommandHandler("leaderboard", local_leaderboard))
    application.add_handler(CommandHandler("global_leaderboard", global_leaderboard))
    application.add_handler(CommandHandler("profile", profile))


    # Countdown Commands
    application.add_handler(CommandHandler("set_countdown", set_countdown))
    application.add_handler(CommandHandler("check_countdown", check_countdown))

    # Link Approval Commands
    application.add_handler(CommandHandler("approve", approve_link_sender))
    application.add_handler(CommandHandler("disapprove", disapprove_link_sender))

    # Admin/Management Commands (Now support reply, @username, or ID)
    application.add_handler(CommandHandler("warn", warn_user))
    application.add_handler(CommandHandler("remove_warn", remove_warn))
    application.add_handler(CommandHandler("warns", warn_counts))
    application.add_handler(CommandHandler("ban", ban_user))
    application.add_handler(CommandHandler("unban", unban_user)) # ID-only
    application.add_handler(CommandHandler("mute", mute_user))
    application.add_handler(CommandHandler("unmute", unmute_user))
    application.add_handler(CommandHandler("promote", promote_user))
    application.add_handler(CommandHandler("filter", set_filter))
    application.add_handler(CommandHandler("stop", stop_filter))
    
    # Welcome Message Commands
    application.add_handler(CommandHandler("set_welcome", set_welcome_message))
    application.add_handler(CommandHandler("enable_welcome", enable_welcome))
    application.add_handler(CommandHandler("disable_welcome", disable_welcome))

    # Mentioning admin
    application.add_handler(CommandHandler("admin", mention_admins))

    # Lock/Unlock Commands
    application.add_handler(CommandHandler("lock", lock_feature))
    application.add_handler(CommandHandler("unlock", unlock_feature))

    # Banned Word Commands
    application.add_handler(CommandHandler("ban_word", ban_word))
    application.add_handler(CommandHandler("unban_word", unban_word))

    # Owner-Only Commands (Control Panel)
    application.add_handler(CommandHandler("broadcast", broadcast_message))
    application.add_handler(CommandHandler("freeze_bot", freeze_bot))
    application.add_handler(CommandHandler("resume_bot", resume_bot))
    application.add_handler(CommandHandler("list_chats", list_chats))
    application.add_handler(CommandHandler("send_countdown", send_countdown))
    application.add_handler(CommandHandler("image", set_welcome_image))


    # NCERT Quiz Feature
    application.add_handler(CommandHandler("save_chapter", save_chapter))
    application.add_handler(CommandHandler("quiz", start_quiz))
    application.add_handler(CommandHandler("stop_quiz", stop_quiz))
    application.add_handler(CommandHandler("change_timing", change_quiz_timing))
    # ... inside main() ...

    # Autonomous Quiz Features
    application.add_handler(CommandHandler("quiz_autonomous", start_autonomous))
    application.add_handler(CommandHandler("stop_autonomous", stop_autonomous))
    application.add_handler(CommandHandler("set_auto_questions", set_auto_questions))
    
    # ... handle reactions ...
    application.add_handler(CommandHandler("reaction_on", reaction_on))
    application.add_handler(CommandHandler("reaction_off", reaction_off))
    application.add_handler(CommandHandler("reaction_reset", reaction_reset))


    # --- NEW: Handle Mentions and Replies for AI ---
    # Triggers ask_ai if:
    # 1. It is NOT a command (commands are handled by CommandHandler)
    # 2. AND (It is a Reply OR It has a Mention entity)
    application.add_handler(MessageHandler(
        filters.TEXT & (~filters.COMMAND) & (filters.Entity("mention") | filters.REPLY), 
        ask_ai
    ))

    # Message Handlers
     # 1. New Member Handler (Must come first to welcome the user before other filters run)
    application.add_handler(MessageHandler(filters.StatusUpdate.NEW_CHAT_MEMBERS, handle_new_member))

    #2. bot fires the command /start automatically so that it shows the welcome message and stores the chat id to database
    application.add_handler(ChatMemberHandler(bot_added_to_group, ChatMemberHandler.MY_CHAT_MEMBER))

    # 2.5. Track User Activity (Low Priority)
    # This handler captures all standard messages (excluding explicit commands) 
    # to passively update user data in Firestore.
    # group=10 ensures it runs after all command and primary filter handlers (which default to group 0).
    application.add_handler(
        MessageHandler(filters.ALL & (~filters.COMMAND), update_user_data), 
        group=10
    )

    # 5. Handle Unapproved Links (Delete and Warn for Message Links)
    application.add_handler(MessageHandler(filters.Entity("url") | filters.Entity("text_link"), handle_link_messages))

    # 3. Handle Filters (Text, Sticker, Image Trigger by Keyword)
    application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_filters))

    # 4. Handle Banned Words (Delete and Warn for Text)
    application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_banned_words))

    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, auto_react_handler), group=99)

    

    # 6. Poll Answer Handler (Quiz responses)
    application.add_handler(PollAnswerHandler(handle_quiz_answer))

    # 7. Poll Handler (Required for quiz-type poll answers)
    application.add_handler(PollHandler(handle_quiz_answer))  # Optional, for completeness

    # --- Start the Bot via Webhook ---
    # ... inside main() ...

    # Explicitly define allowed updates to ensure PollAnswer is received
    allowed_updates_list = [
        Update.MESSAGE, 
        Update.POLL, 
        Update.POLL_ANSWER, 
        Update.MY_CHAT_MEMBER, 
        Update.CHAT_MEMBER,
        Update.CALLBACK_QUERY
    ]

    # --- Telegram webhook handler ---
    async def telegram_webhook(request):
        update = Update.de_json(await request.json(), application.bot)
        await application.initialize()
        await application.process_update(update)
        return web.Response(text="OK")

    # --- Health check endpoint (for Render + UptimeRobot) ---
    async def home(request):
        return web.Response(text="Bot is running...")

    # --- aiohttp web app ---
    app = web.Application()
    app.router.add_post(f"/{BOT_TOKEN}", telegram_webhook)
    app.router.add_get("/", home)

    # start the Telegram webhook
    await application.bot.set_webhook(f"{WEBHOOK_URL}/{BOT_TOKEN}")

    # run aiohttp server
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", PORT)
    await site.start()

    print("Bot and Web server running...")

    # keep alive
    await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main())

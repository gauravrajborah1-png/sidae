import os
import logging
import re
import asyncio 
import requests 
import json 
import io 
import base64 
import aiohttp
from telegram import Update, ChatPermissions, Poll
from telegram.ext import (
    Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackContext, PollHandler, PollAnswerHandler
)
from datetime import datetime, timedelta, timezone
# Import firestore package directly for SERVER_TIMESTAMP
from firebase_admin import initialize_app, firestore, credentials 
import functools

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
    "The GC owner or the user(for private chat) will have to contact @Tota_ton to resume the bot.🥱"
)
# Global dictionaries for quiz ranking system
# Maps chat_id -> { 'scores': { user_id: { 'name': str, 'score': int, 'time': float } }, 'poll_ids': list }
quiz_sessions = {}

# Maps poll_id -> { 'chat_id': int, 'correct_option': int, 'start_time': datetime }
active_polls = {}

# Gemini Configuration (Updated for Gemini 2.5 Flash)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") # Use this environment variable now
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent"
GEMINI_MODEL = "gemini-2.5-flash-preview-09-2025"

# OPEN weather API
OPENWEATHER_API_KEY = "448363a2df79d98e17ec8757a092dc1a"

# 2. Logging Setup
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
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
            "You are a helpful, Telegram bot to help in study related doubts running inside a group chat or in a private chat. You are created by Gaurav Raj, he is your owner. The telegram username of your owner is @Tota_ton. Your telegram username is @Tota_ton_bot. You are here to help the users with their academic doubts mainly for the students preparing for NEET. Explain every question deeply and ask them at the end of your explanation if they understood it. Try to write text normally because telegram can't understand markdown formatting. Use some emojis in case if you need to provide special emphasis on a part of text. Be very careful about not giving the user about any medical, financial or family related adivce. Don't give the user anything illegal. Whenever someone asks you anything in hindi, then don't reply using hindi letters. Instead use hinglish. Write hindi words in english. Don't give any idea about your language model to the user and don't give them any codes for making a bot like you. Don't give them your API key. Just clear there study related doubts on the subjects- Physics, Chemistry and Biology. Your knowledge base is google and NCERT. If you are explaining a doubt then remember the whole conversation for the best output possible. Don't give reply to any sexually abusive things or slang words rather than their biology related doubts. At last don't forget your identity; your name is Noah✰. You don't have to mention your name or  the name of your owner everytime. Just mention it whenever they ask you about your identity; like- who are you? or who is your owner or creator. Just mention the name of your owner in introduction. Also you don't have to write your knowledge base in every message. Just write them when they ask you. If an image is provided, focus your analysis on that image and the user's question about it. You can help users by translating something off-topic  that they don't understand. α β γ δ ε ζ η θ ι κ λ μ ν ξ ο π ρ σ τ υ φ χ ψ ω -use these greek letters whenever necessary while dealing with physics or chemistry questions. and use the symbol ^ whenever you write something in superscript and use _ whenever you write something in subscript. Make the message understandable and readable by the users."
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
            return f"⚠️ Error ({e.response.status_code}): Failed to get a response from the AI. Check your API key or rate limits."
        except requests.exceptions.RequestException as e:
            logger.error(f"Gemini Request Error: {e}")
            return "⚠️ Error: Could not connect to the AI service. Please try again later."
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
    Handles the /ask command, supporting image recognition via reply and
    conversational context via reply to a text message.
    """
    
    if not GEMINI_API_KEY:
        await update.message.reply_text("The Gemini AI service is not configured. Please set the GEMINI_API_KEY environment variable.")
        return

    # 1. Determine Prompt, Image, and Mime Type
    prompt = " ".join(context.args)
    image_base64 = None
    mime_type = None
    group_id = update.effective_chat.id
    
    reply = update.message.reply_to_message
    
    # Check for reply message for both image and text context
    if reply:
        file_obj = None
        
        # A. Image Multimodal Handling (Photo or Image Document)
        if reply.photo:
            # Get the largest photo size
            file_obj = reply.photo[-1] 
            mime_type = "image/jpeg" 
        elif reply.document and reply.document.mime_type and reply.document.mime_type.startswith('image/'):
            file_obj = reply.document
            mime_type = reply.document.mime_type
        
        if file_obj:
            try:
                # FIX: Correctly call the async download method on the file object.
                telegram_file = await file_obj.get_file()

                # Use download_to_memory() to get file bytes
                buffer = io.BytesIO()
                await telegram_file.download_to_memory(out=buffer)
                buffer.seek(0)
                file_bytes = buffer.read()
                
                # 2. Encode to Base64
                image_base64 = base64.b64encode(file_bytes).decode('utf-8')
                
                # 3. Determine prompt (use caption or argument)
                if not prompt and reply.caption:
                    prompt = reply.caption
                    
                if not prompt:
                    prompt = "Analyze this image and provide a helpful description."
                
                logger.info(f"Image prompt generated: {prompt[:50]}...")
            
            except Exception as e:
                logger.error(f"Error handling image download/encoding: {e}")
                await update.message.reply_text("❌ Error processing the replied image. Make sure it's a valid photo or image file.")
                return

        # B. Text Context Handling (If no image was found/processed, and reply is text)
        if not image_base64 and reply.text:
            context_text = reply.text.strip()
            
            if not prompt and not context_text:
                 # Should not happen if reply.text is true, but for safety:
                await update.message.reply_text("The replied message is empty, please provide a question.")
                return
            
            if not prompt and context_text:
                # If only replying with /ask, ask AI to elaborate on the context
                prompt = f"Please elaborate or expand on the following statement: '{context_text}'"
            elif prompt and context_text:
                # If replying with /ask what is X, use the replied message as supporting context
                prompt = f"Previous message context: '{context_text}'. User's question about this context: '{prompt}'"
            
            logger.info(f"Text context prompt generated: {prompt[:50]}...")
            
    # Fallback/Standard Prompt check
    if not prompt and not image_base64:
        await update.message.reply_text("Please provide a question after the command (e.g., `/ask What is X?`) or reply to a message containing a question or image.", parse_mode="Markdown")
        return

    # Indicate that the bot is processing the request
    await context.bot.send_chat_action(update.effective_chat.id, "typing")
    
    # Get the response from the AI (passing the image data if available)
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

# --- Firestore persistence helpers for live quiz state ---

async def _save_active_poll_to_db(poll_id: str, data: dict):
    """Save active poll metadata to Firestore for cross-process visibility."""
    if not db:
        return
    try:
        ref = db.collection("live_active_polls").document(poll_id)
        await asyncio.to_thread(ref.set, {
            "chat_id": str(data.get("chat_id")),
            "correct_option": data.get("correct_option"),
            "start_time_iso": data.get("start_time").isoformat() if data.get("start_time") else None
        }, merge=True)
    except Exception as e:
        logger.error(f"Failed to save active_poll {poll_id} to DB: {e}")

async def _load_active_poll_from_db(poll_id: str):
    if not db:
        return None
    try:
        ref = db.collection("live_active_polls").document(poll_id)
        doc = await asyncio.to_thread(ref.get)
        if not doc.exists:
            return None
        data = doc.to_dict()
        start_time = None
        if data.get("start_time_iso"):
            try:
                start_time = datetime.fromisoformat(data["start_time_iso"])
            except Exception:
                start_time = datetime.now(timezone.utc)
        return {
            "chat_id": int(data.get("chat_id")),
            "correct_option": data.get("correct_option"),
            "start_time": start_time or datetime.now(timezone.utc)
        }
    except Exception as e:
        logger.error(f"Failed to load active_poll {poll_id} from DB: {e}")
        return None

async def _delete_active_poll_from_db(poll_id: str):
    if not db:
        return
    try:
        ref = db.collection("live_active_polls").document(poll_id)
        await asyncio.to_thread(ref.delete)
    except Exception as e:
        logger.debug(f"Failed to delete active_poll {poll_id} from DB: {e}")

async def _save_quiz_session_to_db(group_id: int, session: dict):
    if not db:
        return
    try:
        # make sure poll_ids and scores are serializable
        doc = {
            "poll_ids": session.get("poll_ids", []),
            "scores": session.get("scores", {}),
            "last_updated": firestore.SERVER_TIMESTAMP
        }
        ref = db.collection("live_quiz_sessions").document(str(group_id))
        await asyncio.to_thread(ref.set, doc, merge=True)
    except Exception as e:
        logger.error(f"Failed to save quiz_session for {group_id} to DB: {e}")

async def _load_quiz_session_from_db(group_id: int):
    if not db:
        return None
    try:
        ref = db.collection("live_quiz_sessions").document(str(group_id))
        doc = await asyncio.to_thread(ref.get)
        if not doc.exists:
            return None
        data = doc.to_dict() or {}
        return {
            "poll_ids": data.get("poll_ids", []),
            "scores": data.get("scores", {})
        }
    except Exception as e:
        logger.error(f"Failed to load quiz_session for {group_id} from DB: {e}")
        return None

async def _delete_quiz_session_from_db(group_id: int):
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
    await update.message.reply_text(
        "👋 This is a group moderation bot made with ♥ by @Tota_ton (Gaurav). "
        "You can contact the owner through this bot. You can also manage a group or ask any study related doubts using this bot. Just give it the admin rights and you're good to go🐥—"
        "\n\nThank you🦚"
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
    if update.effective_chat.id in frozen_chats:
        return await update.message.reply_text(freeze_message)

    """Sets the custom welcome message template safely (no Markdown issues)."""
    
    # Admin check
    if not await check_admin(update, context):
        return

    # Firestore check
    if not db:
        await update.message.reply_text("Database not available.", parse_mode=None)
        return

    # No arguments provided
    if not context.args:
        await update.message.reply_text(
            (
                "Usage:\n"
                "/set_welcome <message>\n\n"
                "You can use the following placeholders:\n"
                "{name} → The new member's name\n"
                "{group_name} → The group name\n\n"
                "Example:\n"
                "/set_welcome Welcome to {group_name}, {name}! 😊"
            ),
            parse_mode=None   # prevent Markdown issues
        )
        return

    # Combine the template
    template = " ".join(context.args)

    group_id = update.effective_chat.id
    ref = db.collection("group_settings").document(str(group_id))

    # Save to Firestore
    try:
        await asyncio.to_thread(
            ref.set,
            {"welcome_message_template": template},
            merge=True
        )

        await update.message.reply_text(
            f"✅ Custom welcome message template saved:\n{template}",
            parse_mode=None   # NO MARKDOWN → no Telegram errors
        )

    except Exception as e:
        logger.error(f"Error setting welcome message: {e}")
        await update.message.reply_text(
            f"❌ Failed to set welcome message: {e}",
            parse_mode=None
        )


@check_frozen
async def enable_welcome(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat.id in frozen_chats:
        return await update.message.reply_text(freeze_message)

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
    if update.effective_chat.id in frozen_chats:
        return await update.message.reply_text(freeze_message)

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
    if update.effective_chat.id in frozen_chats:
        return await update.message.reply_text(freeze_message)

    """Generates an AI welcome message for new members and registers the chat if the bot is added."""
    if not update.message.new_chat_members:
        return
    if update.effective_chat.type not in ["group", "supergroup"]:
        return

    # Check for bot addition first
    for member in update.message.new_chat_members:
        if member.id == context.bot.id:
            if db:
                chat_id = str(update.effective_chat.id)
                chat_ref = db.collection("broadcast_chats").document(chat_id)
                chat_data = {
                    "chat_id": chat_id,
                    "chat_type": update.effective_chat.type,
                    "title": update.effective_chat.title or "Unknown Group", 
                    "last_active": firestore.SERVER_TIMESTAMP,
                }
                try:
                    await asyncio.to_thread(chat_ref.set, chat_data, merge=True)
                    logger.info(f"Chat {chat_id} added to broadcast list (Bot was added).")
                except Exception as e:
                    logger.error(f"Failed to add chat to broadcast list: {e}")

    settings = await get_welcome_settings(update.effective_chat.id)
    if not settings.get("enabled"):
        return

    group_name = update.effective_chat.title
    template = settings.get("template")
    
    for member in update.message.new_chat_members:
        # Ignore welcoming the bot itself
        if member.id == context.bot.id:
            continue
            
        # 1. Prepare base message using the template
        user_name = member.full_name
        mention = member.mention_html()
        
        base_message = template.format(name=mention, group_name=group_name)
        
        # 2. Craft AI prompt
        personality = settings.get("ai_mode", "default")
        ai_prompt = (
            f"The group has a new member named '{user_name}'. The welcome message template set by the admin is: '{base_message}'. "
            f"Based on your current personality mode ('{personality}'), write a single, short, and engaging welcome message that incorporates the template, "
            f"introduces the group's purpose (NEET study), and encourages the user to ask their doubts. Do not use the user's actual mention in your response, only use their name once, as the template already covers the mention. The message should be warm, informative, and fit the current AI personality."
        )

        # 3. Get AI Response
        await context.bot.send_chat_action(update.effective_chat.id, "typing")
        # Note: We must pass a mock group_id (like 0) to get_ai_response since it needs a group_id, 
        # but in this context, the personality is already factored into the prompt.
        ai_response_text = await get_ai_response(ai_prompt, update.effective_chat.id)
        
        # 4. Send the final welcome message
        try:
            # We assume the AI will respect the template and include the mention indirectly.
            await update.message.reply_text(ai_response_text, parse_mode="HTML")
        except Exception as e:
            logger.error(f"Failed to send AI welcome: {e}")
            # Fallback if AI output is bad
            await update.message.reply_text(f"Welcome, {mention}! Ask your NEET doubts here. (AI failed to send a custom greeting)", parse_mode="HTML")


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
        "/set_personality - kis personality mein AI based features chahiye choose krlo... friendly, sarcastic, etc😶‍🌫\n\n"
        "Thank you🕊\n"
        "regards- Tota Ton🐣"
    )
    await update.message.reply_text(message)

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
        "Must not make the questions very long; keep them short and important"
        "After every question mention - @Tota_ton_bot"
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
    """Handles incoming poll answers to track scores. Falls back to Firestore if in-memory state missing."""
    try:
        answer = update.poll_answer
        if not answer:
            return

        poll_id = str(answer.poll_id)
        user = answer.user
        selected_ids = answer.option_ids

        logger.info(f"[quiz] Received poll answer from {getattr(user, 'id', None)} for poll {poll_id}. Selected: {selected_ids}")

        # Try in-memory first
        poll_data = active_polls.get(poll_id)
        if not poll_data:
            # Try loading from Firestore (in case of restart / different instance)
            poll_data = await _load_active_poll_from_db(poll_id)
            if poll_data:
                # restore into memory for faster subsequent operations
                active_polls[poll_id] = poll_data
                logger.info(f"[quiz] Restored active poll {poll_id} from Firestore for chat {poll_data.get('chat_id')}.")

        if not poll_data:
            # Fallback: attempt to find session that contains this poll id in DB
            try:
                # Query live_quiz_sessions collection for any doc containing this poll_id
                if db:
                    sessions_ref = db.collection("live_quiz_sessions")
                    # This is a simple linear scan — fine for small number of active quizzes.
                    docs = await asyncio.to_thread(sessions_ref.stream)
                    found = False
                    for doc in docs:
                        d = doc.to_dict() or {}
                        poll_ids = d.get("poll_ids", [])
                        if poll_id in poll_ids:
                            # Found the session; load it into memory and recreate an active_polls entry with unknown correct_option
                            group_id = int(doc.id)
                            session = await _load_quiz_session_from_db(group_id)
                            if session is None:
                                session = {"poll_ids": poll_ids, "scores": {}}
                            quiz_sessions[group_id] = session
                            # attempt to load the specific active poll doc (may exist)
                            poll_data = await _load_active_poll_from_db(poll_id)
                            if not poll_data:
                                poll_data = {"chat_id": group_id, "correct_option": None, "start_time": datetime.now(timezone.utc)}
                                # persist a basic record so later answers find it
                                await _save_active_poll_to_db(poll_id, poll_data)
                                active_polls[poll_id] = poll_data
                            found = True
                            logger.info(f"[quiz] Recovered poll {poll_id} into quiz_sessions for chat {group_id}.")
                            break
                    if not found:
                        logger.info(f"[quiz] Poll {poll_id} not found in active_polls and no session matched. Ignoring.")
                        return
                else:
                    logger.info(f"[quiz] No DB configured and poll {poll_id} not in memory — ignoring.")
                    return
            except Exception as e:
                logger.exception(f"[quiz] Error while attempting to recover poll {poll_id} from DB: {e}")
                return

        chat_id = poll_data['chat_id']
        correct_id = poll_data.get('correct_option')
        start_time = poll_data.get('start_time', datetime.now(timezone.utc))

        # Ensure quiz session exists in memory; try DB load if missing
        session = quiz_sessions.get(chat_id)
        if not session:
            session = await _load_quiz_session_from_db(chat_id) or {'scores': {}, 'poll_ids': []}
            quiz_sessions[chat_id] = session
            logger.info(f"[quiz] Loaded quiz_session for chat {chat_id} from DB into memory.")

        time_taken = (datetime.now(timezone.utc) - start_time).total_seconds()

        scores = session.setdefault('scores', {})

        user_stats = scores.get(user.id, {
            'name': getattr(user, 'full_name', getattr(user, 'username', 'Unknown')),
            'score': 0,
            'total_time': 0.0
        })

        user_stats['name'] = getattr(user, 'full_name', getattr(user, 'username', user_stats.get('name')))

        if correct_id is not None and correct_id in selected_ids:
            user_stats['score'] += 1
            user_stats['total_time'] += time_taken
            logger.info(f"[quiz] User {user.id} answered CORRECTLY for poll {poll_id}.")
        else:
            logger.info(f"[quiz] User {user.id} answered INCORRECTLY for poll {poll_id}.")

        # Save back to session in memory and persist
        quiz_sessions[chat_id]['scores'][user.id] = user_stats
        await _save_quiz_session_to_db(chat_id, quiz_sessions[chat_id])

    except Exception as e:
        logger.exception(f"Error in handle_quiz_answer: {e}")


@check_frozen
async def start_quiz(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Starts a quiz based on a saved NCERT chapter.
    Usage: /quiz <num_questions> <chapter_name>
    """
    if not db:
        await update.message.reply_text("Database not available.")
        return
        
    if len(context.args) < 2:
        await update.message.reply_text("⚠️ Usage: `/quiz <num_questions> <chapter_name>`", parse_mode="Markdown")
        return

    try:
        num_questions = int(context.args[0])
    except ValueError:
        await update.message.reply_text("⚠️ Number of questions must be an integer.")
        return

    chapter_name = " ".join(context.args[1:]).strip().lower().replace(" ", "_")
    group_id = update.effective_chat.id

    # Initialize Quiz Session for Ranking (memory + DB)
    quiz_sessions[group_id] = {
        'scores': {},
        'poll_ids': []
    }
    # persist initial empty session
    await _save_quiz_session_to_db(group_id, quiz_sessions[group_id])

    # 1. Fetch file_id from Firestore
    try:
        doc_ref = db.collection("ncert_chapters").document(chapter_name)
        doc = await asyncio.to_thread(doc_ref.get)
        if not doc.exists:
            await update.message.reply_text(f"❌ Chapter **'{chapter_name}'** not found. Use `/save_chapter` to upload it first.", parse_mode="Markdown")
            return
        file_id = doc.to_dict().get("file_id")
    except Exception as e:
        await update.message.reply_text(f"Database error: {e}")
        return

    if not PYPDF_AVAILABLE:
        await update.message.reply_text("❌ `pypdf` library is missing. Cannot process PDF files.")
        return

    status_msg = await update.message.reply_text(f"📥 Downloading {chapter_name} and generating {num_questions} questions... This may take a minuteִֶָ. ..𓂃 ࣪ ִֶָ🪽་༘࿐")

    # 2. Download and Extract
    try:
        new_file = await context.bot.get_file(file_id)
        pdf_data = io.BytesIO()
        await new_file.download_to_memory(out=pdf_data)
        pdf_data.seek(0)
        
        text_content = await extract_text_from_pdf(pdf_data.read())
        
        if text_content.startswith("Error"):
            await status_msg.edit_text(f"❌ Extraction failed: {text_content}")
            return
            
        if len(text_content) < 100:
             await status_msg.edit_text("❌ The PDF seems empty or unreadable.")
             return

    except Exception as e:
        logger.error(f"Download/Extract error: {e}")
        await status_msg.edit_text("❌ Failed to process the PDF file.")
        return

    # 3. Generate Questions via Gemini
    questions = await generate_quiz_questions(text_content, num_questions)
    
    if not questions:
        await status_msg.edit_text("❌ AI failed to generate valid questions. Please try again.")
        return

    await status_msg.edit_text(f"🧠 ꧁⎝ 𓆩༺✧Quiz Ready✧༻𓆪 ⎠꧂ Starting {len(questions)} questions on ꧁ ༺ {chapter_name} ༻ ꧂... ..𓂃 ࣪")

    # 4. Retrieve Timing
    interval = 10 # default
    try:
        settings_ref = db.collection("group_settings").document(str(group_id))
        settings_doc = await asyncio.to_thread(settings_ref.get)
        if settings_doc.exists:
            interval = settings_doc.to_dict().get("quiz_interval", 10)
    except Exception:
        pass

    # 5. Send Questions Loop
    for i, q in enumerate(questions):
        try:
            options = q.get('options', [])
            # Telegram poll options max length 100 chars. Truncate if needed.
            safe_options = [opt[:100] for opt in options]
            correct_id = q.get('correct_option_id', 0)
            explanation = q.get('explanation', "No explanation provided.")
            question_text = f"Q{i+1}: {q.get('question')}"
            
            # If question is too long, send it as text first
            if len(question_text) > 300:
                await context.bot.send_message(chat_id=group_id, text=question_text)
                question_text = f"Q{i+1} (See above)"

            # IMPORTANT: is_anonymous=False is required to track user answers for ranking
            poll_message = await context.bot.send_poll(
                chat_id=group_id,
                question=question_text,
                options=safe_options,
                type=Poll.QUIZ,
                correct_option_id=correct_id,
                explanation=explanation,
                explanation_parse_mode="Markdown",
                is_anonymous=False, 
                open_period=interval # Auto-close poll after interval
            )

            # Register Poll for Tracking (in-memory + DB)
            poll_key = str(poll_message.poll.id)
            poll_record = {
                'chat_id': group_id,
                'correct_option': correct_id,
                'start_time': datetime.now(timezone.utc),
            }
            active_polls[poll_key] = poll_record
            quiz_sessions[group_id]['poll_ids'].append(poll_key)

            # persist both the active poll and updated session right away
            await _save_active_poll_to_db(poll_key, poll_record)
            await _save_quiz_session_to_db(group_id, quiz_sessions[group_id])
            
            # Wait for interval before next question (give a larger buffer to let poll answers arrive)
            # increase buffer to reduce race between poll closing and handler processing
            wait_buffer = 5
            if i < len(questions) - 1:
                await asyncio.sleep(interval + wait_buffer)
            else:
                # Last question: give an extra moment to ensure answers processed before leaderboard
                await asyncio.sleep(interval + wait_buffer + 3)
                
        except Exception as e:
            logger.error(f"Error sending poll: {e}")
            await context.bot.send_message(chat_id=group_id, text=f"⚠️ Error sending question {i+1}: {e}")

    # Give a short grace period to ensure all incoming PollAnswer updates are processed
    await asyncio.sleep(2)
    
    # 6. Generate Leaderboard
    # CRITICAL FIX: Reload session from DB to ensure we have the latest scores 
    # recorded by handle_quiz_answer (which might run in a different context/thread)
    try:
        updated_session = await _load_quiz_session_from_db(group_id)
        if updated_session and 'scores' in updated_session:
            # Update local memory with the database truth
            quiz_sessions[group_id]['scores'] = updated_session['scores']
            logger.info(f"Refreshed scores from DB for chat {group_id} before leaderboard.")
    except Exception as e:
        logger.error(f"Failed to refresh scores from DB: {e}")

    # Now safely access scores from memory (which is now synced)
    scores = quiz_sessions.get(group_id, {}).get('scores', {})
    
    if not scores:
        await context.bot.send_message(chat_id=group_id, text="🏁 Quiz Completed! Well done guyz🕊. Keep grinding་࿐")
    else:
        # Sort by Score (descending), then Time (ascending)
        ranked_users = sorted(
            scores.values(), 
            key=lambda x: (-x['score'], x['total_time'])
        )
        
        leaderboard_text = f"🏆 **Leaderboard: {chapter_name.title()}** 🏆\n\n"
        for rank, user in enumerate(ranked_users, 1):
            name = user['name']
            score = user['score']
            time_taken = user['total_time']
            leaderboard_text += f"{rank}. **{name}** — {score} pts ({time_taken:.2f}s)\n"
        
        await context.bot.send_message(chat_id=group_id, text=leaderboard_text, parse_mode="Markdown")

    # 7. Cleanup (in-memory + DB)
    if group_id in quiz_sessions:
        for pid in quiz_sessions[group_id]['poll_ids']:
            active_polls.pop(pid, None)
            await _delete_active_poll_from_db(pid)
        quiz_sessions.pop(group_id, None)
        await _delete_quiz_session_from_db(group_id)



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

def main() -> None:
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

    # Countdown Commands
    application.add_handler(CommandHandler("set_countdown", set_countdown))
    application.add_handler(CommandHandler("check_countdown", check_countdown))

    # Link Approval Commands
    application.add_handler(CommandHandler("approve", approve_link_sender))
    application.add_handler(CommandHandler("disapprove", disapprove_link_sender))

    # Admin/Management Commands (Now support reply, @username, or ID)
    application.add_handler(CommandHandler("warn", warn_user))
    application.add_handler(CommandHandler("removewarn", remove_warn))
    application.add_handler(CommandHandler("warns", warn_counts))
    application.add_handler(CommandHandler("ban", ban_user))
    application.add_handler(CommandHandler("unban", unban_user))
    application.add_handler(CommandHandler("mute", mute_user))
    application.add_handler(CommandHandler("unmute", unmute_user))
    application.add_handler(CommandHandler("admin", mention_admins))
    application.add_handler(CommandHandler("promote", promote_user))

    # Lock / Unlock features
    application.add_handler(CommandHandler("lock", lock_feature))
    application.add_handler(CommandHandler("unlock", unlock_feature))

    # Banned word management
    application.add_handler(CommandHandler("ban_word", ban_word))
    application.add_handler(CommandHandler("unban_word", unban_word))

    # Filter management
    application.add_handler(CommandHandler("filter", set_filter))
    application.add_handler(CommandHandler("stop", stop_filter))

    # Welcome message / broadcast / owner-only
    application.add_handler(CommandHandler("set_welcome", set_welcome_message))
    application.add_handler(CommandHandler("enable_welcome", enable_welcome))
    application.add_handler(CommandHandler("disable_welcome", disable_welcome))

    # NCERT Quiz / Chapter management
    application.add_handler(CommandHandler("save_chapter", save_chapter))
    application.add_handler(CommandHandler("quiz", start_quiz))
    application.add_handler(CommandHandler("change_timing", change_quiz_timing))

    # Owner commands
    application.add_handler(CommandHandler("broadcast", broadcast_message))
    application.add_handler(CommandHandler("freeze", freeze_bot))
    application.add_handler(CommandHandler("resume", resume_bot))
    application.add_handler(CommandHandler("list_chats", list_chats))
    application.add_handler(CommandHandler("deny", deny))
    application.add_handler(CommandHandler("allow", allow))

    # Poll / Quiz handlers
    # Collect poll answers (used for quiz ranking). PollAnswerHandler handles PollAnswer updates.
    application.add_handler(PollAnswerHandler(handle_quiz_answer))

    # Message handlers (non-command):
    # 1) Banned words should be checked early
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_banned_words), group=0)
    # 2) Filters (auto-replies) next
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_filters), group=1)
    # 3) Link message moderation - we register for all messages because links can be in captions, entities, etc.
    application.add_handler(MessageHandler(filters.ALL, handle_link_messages), group=2)

    # New chat members (welcome)
    application.add_handler(MessageHandler(filters.StatusUpdate.NEW_CHAT_MEMBERS, handle_new_member))

    # Start webhook (suitable for hosting on platforms that require webhooks)
    try:
        # The Application.run_webhook will block and run the webhook server.
        # It listens on 0.0.0.0 by default; explicitly provide the PORT and WEBHOOK_URL.
        logger.info("Starting webhook...")
        application.run_webhook(
            listen="0.0.0.0",
            port=PORT,
            webhook_url=WEBHOOK_URL,
        )
    except Exception as e:
        logger.exception(f"Failed to start webhook: {e}")


if __name__ == "__main__":
    main()

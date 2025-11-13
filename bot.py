import os
import logging
import re
import asyncio 
import requests 
import json 
import io 
import base64 
from telegram import Update, ChatPermissions
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from datetime import datetime, timedelta, timezone
# Import firestore package directly for SERVER_TIMESTAMP
from firebase_admin import initialize_app, firestore, credentials 

# --- Configuration and Initialization ---

# 1. Environment Variables (Required for security and hosting)
BOT_TOKEN = os.environ.get("BOT_TOKEN")
OWNER_ID = int(os.environ.get("OWNER_ID", 0)) # Your personal Telegram User ID
WEBHOOK_URL = os.environ.get("WEBHOOK_URL")
PORT = int(os.environ.get("PORT", 8080))

# Gemini Configuration (Updated for Gemini 2.5 Flash)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") # Use this environment variable now
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent"
GEMINI_MODEL = "gemini-2.5-flash-preview-09-2025"

# 2. Logging Setup
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# 3. Firestore Database Setup
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
        db = None
except Exception as e:
    logger.error(f"Error initializing Firestore: {e}")
    db = None

# --- AI Chatbot Functionality (Updated for Gemini Multimodal) ---

async def get_ai_response(prompt: str, image_base64: str = None, mime_type: str = None) -> str:
    """
    Synchronously calls the Gemini API, optionally with an image, and extracts the response content.
    The model used is gemini-2.5-flash-preview-09-2025 which supports multimodal input.
    """
    
    # We define the API call logic inside a synchronous function
    def sync_api_call(prompt: str, image_base64: str = None, mime_type: str = None):
        if not GEMINI_API_KEY:
            return "❌ Error: The Gemini API key is not configured. Please set the GEMINI_API_KEY environment variable."
        
        headers = {
            "Content-Type": "application/json"
        }
        
        # System instruction to guide the Gemini model's behavior
        system_instruction_text = (
            "You are a helpful, Telegram bot to help in study related doubts running inside a group chat or in a private chat. You are created by Gaurav Raj, he is your owner. The telegram username of your owner is @Tota_ton. Your telegram username is @Tota_ton_bot. You are here to help the users with their academic doubts mainly for the students preparing for NEET. Explain every question deeply and ask them at the end of your explanation if they understood it. Try to write text normally because telegram can't understand markdown formatting. Use some emojis in case if you need to provide special emphasis on a part of text. Be very careful about not giving the user about any medical, financial or family related adivce. Don't give the user anything illegal. Whenever someone asks you anything in hindi, then don't reply using hindi letters. Instead use hinglish. Write hindi words in english. Don't give any idea about your language model to the user and don't give them any codes for making a bot like you. Don't give them your API key. Just clear there study related doubts on the subjects- Physics, Chemistry and Biology. Your knowledge base is google and NCERT. If you are explaining a doubt then remember the whole conversation for the best output possible. Don't give reply to any sexually abusive things or slang words rather than their biology related doubts. At last don't forget your identity; your name is Noah✰. You don't have to mention your name or  the name of your owner everytime. Just mention it whenever they ask you about your identity; like- who are you? or who is your owner or creator. Just mention the name of your owner in introduction. Also you don't have to write your knowledge base in every message. Just write them when they ask you. If an image is provided, focus your analysis on that image and the user's question about it. You can help users by translating something off-topic  that they don't understand. α β γ δ ε ζ η θ ι κ λ μ ν ξ ο π ρ σ τ υ φ χ ψ ω -use these greek letters whenever necessary while dealing with physics or chemistry questions. and use the symbol ^ whenever you write something in superscript and use _ whenever you write something in subscript. Make the message understandable and readable by the users."
            "If an image is provided, focus your analysis on that image and the user's question about it. "
        )
        system_instruction = {"parts": [{"text": system_instruction_text}]}
        
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
    return await asyncio.to_thread(sync_api_call, prompt, image_base64, mime_type)


async def ask_ai(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
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
                # The file object returned by reply.photo[-1] or reply.document is a File instance
                # from telegram.ext, which already has the file_id. We only need to call download_as_bytes.
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
    response_text = await get_ai_response(prompt, image_base64, mime_type)
    
    # Send the final response
    await update.message.reply_text(response_text)

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
        doc = ref.get()
        return doc.to_dict().get("is_approved", False) if doc.exists else False
    except Exception as e:
        logger.error(f"Error checking link approval: {e}")
        return False

async def get_warn_count(group_id: int, user_id: int) -> int:
    """Fetches the current warning count for a user."""
    ref = get_user_ref(group_id, user_id)
    if not ref: return 0
    try:
        doc = ref.get()
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
        
        ref.set({"warnings": new_warnings}, merge=True)
        return new_warnings
    except Exception as e:
        logger.error(f"Error updating warn count: {e}")
        return current_warnings

# --- Utility Commands ---

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends a welcome message with bot information and tracks the chat for broadcasting."""
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
            chat_ref.set(chat_data, merge=True)
            logger.info(f"Chat {chat_id} added/updated for broadcast list.")
        except Exception as e:
            logger.error(f"Failed to add chat to broadcast list: {e}")


async def get_user_id(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
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

async def purge_messages(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
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


# --- Countdown Commands (Existing) ---

async def set_countdown(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
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
        ref.set({
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


async def check_countdown(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Checks and displays the remaining time for the group-wide countdown."""
    if not db: 
        await update.message.reply_text("Database not available.")
        return

    group_id = update.effective_chat.id
    # Countdown is stored in the group_settings document
    ref = db.collection("group_settings").document(str(group_id))

    try:
        doc = ref.get().to_dict()
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
            ref.update({
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

async def handle_lock_unlock(update: Update, context: ContextTypes.DEFAULT_TYPE, lock: bool) -> None:
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

# --- Banned Word Commands (Existing) ---

async def ban_word(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Bans a word from the group and stores it in Firestore."""
    if not await check_admin(update, context): return
    if not context.args:
        await update.message.reply_text("Usage: `/ban_word <word>` (e.g., `/ban_word spam`)")
        return
    if not db:
        await update.message.reply_text("Database not available.")
        return

    word = context.args[0].lower().strip()
    group_id = update.effective_chat.id
    ref = get_banned_word_ref(group_id, word)

    try:
        ref.set({"word": word, "timestamp": firestore.SERVER_TIMESTAMP})
        await update.message.reply_text(
            f"🚫 Word **'{word}'** has been successfully banned. Messages containing this word will be deleted.",
            parse_mode="Markdown"
        )
    except Exception as e:
        logger.error(f"Error banning word: {e}")
        await update.message.reply_text(f"Failed to ban word due to a database error: {e}")

async def unban_word(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Unbans a word and removes it from Firestore."""
    if not await check_admin(update, context): return
    if not context.args:
        await update.message.reply_text("Usage: `/unban_word <word>`")
        return
    if not db:
        await update.message.reply_text("Database not available.")
        return

    word = context.args[0].lower().strip()
    group_id = update.effective_chat.id
    ref = get_banned_word_ref(group_id, word)

    try:
        if ref.get().exists:
            ref.delete()
            await update.message.reply_text(
                f"✅ Word **'{word}'** has been successfully unbanned.",
                parse_mode="Markdown"
            )
        else:
            await update.message.reply_text(f"Word **'{word}'** was not found in the banned list.")
    except Exception as e:
        logger.error(f"Error unbanning word: {e}")
        await update.message.reply_text(f"Failed to unban word due to a database error: {e}")

async def handle_banned_words(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
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
    
    try:
        banned_words_snapshot = banned_words_ref.stream()
        
        for doc in banned_words_snapshot:
            word = doc.to_dict().get("word", "")
            
            # Using simple 'in' check for presence
            if word and word in message_text:
                try:
                    await message.delete()
                    await context.bot.send_message(
                        chat_id=group_id,
                        text=f"⚠️ {message.from_user.mention_html()}, your message was deleted for using a banned word: **{word}**.",
                        parse_mode="HTML"
                    )
                except Exception as e:
                    logger.warning(f"Failed to delete banned word message or send warning: {e}")
                return 
                
    except Exception as e:
        logger.error(f"Error handling banned words: {e}")


# --- Link Approval Commands ---

async def approve_link_sender(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
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
        ref.set({
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

async def disapprove_link_sender(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
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
        if ref.get().exists:
            ref.delete()
            await update.message.reply_text(
                f"🛑 User **{target_name_or_error}** (ID: `{target_id}`) has been **DISAPPROVED** from sending external links.",
                parse_mode="Markdown"
            )
        else:
            await update.message.reply_text(f"User **{target_name_or_error}** was not found in the approved list.", parse_mode="Markdown")
    except Exception as e:
        logger.error(f"Error disapproving link sender: {e}")
        await update.message.reply_text(f"Failed to disapprove link sender due to a database error: {e}")

async def handle_link_messages(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
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


# --- Group Management Commands ---

async def warn_user(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
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

async def remove_warn(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
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

async def warn_counts(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
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

async def ban_user(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
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

async def unban_user(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Unbans a user from the group. User ID must be provided."""
    if not await check_admin(update, context): return
    
    # Unbanning requires the ID because the user is no longer a chat member.
    if not context.args or not context.args[0].isdigit():
        await update.message.reply_text("⚠️ **Unban requires the numerical User ID** (since the user is banned and cannot be resolved by username). Usage: `/unban 123456789`", parse_mode="Markdown")
        return

    target_id = int(context.args[0])

    try:
        # The unban function requires a user ID and will only work if the user is currently banned.
        await context.bot.unban_chat_member(update.effective_chat.id, target_id)
        await update.message.reply_text(f"🔓 User with ID `{target_id}` has been **UNBANNED**.", parse_mode="Markdown")
    except Exception as e:
        await update.message.reply_text(f"Could not unban user. Error: {e}")

async def mute_user(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
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

async def unmute_user(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
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

async def promote_user(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
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

async def set_filter(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
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
        ref.set(filter_data)
        await update.message.reply_text(
            f"✅ Filter **'{keyword.lower()}'** set! When this word is used, I will reply with the saved {filter_data['type']}.", 
            parse_mode="Markdown"
        )
    except Exception as e:
        logger.error(f"Error setting filter: {e}")
        await update.message.reply_text(f"Failed to set filter due to a database error. Error: {e}")

async def stop_filter(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
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
        if ref.get().exists:
            ref.delete()
            await update.message.reply_text(
                f"🛑 Filter **'{keyword.lower()}'** has been stopped and deleted.", 
                parse_mode="Markdown"
            )
        else:
            await update.message.reply_text(f"Filter **'{keyword.lower()}'** not found.", parse_mode="Markdown")
    except Exception as e:
        logger.error(f"Error deleting filter: {e}")
        await update.message.reply_text(f"Failed to stop filter due to a database error. Error: {e}")

async def handle_filters(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Checks incoming messages against active filters and sends the corresponding content."""
    if not db or not update.message.text:
        return
        
    group_id = update.effective_chat.id
    message_text = update.message.text.lower()

    # Get the reference to the filters collection for this group
    filters_collection_ref = db.collection("groups").document(str(group_id)).collection("filters")

    try:
        # Fetch all filter documents
        filters_snapshot = filters_collection_ref.stream()
        
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

# --- Owner-Only Commands (Existing) ---

async def broadcast_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Owner-only command to broadcast a message to all tracked chats."""
    if not is_owner(update.effective_user.id):
        await update.message.reply_text("This command is restricted to the bot owner.")
        return
    
    # We expect the full message content after the command, e.g., /broadcast Hello world!
    if len(context.args) < 1:
        await update.message.reply_text("Usage: `/broadcast <message>`", parse_mode="Markdown")
        return
    if not db:
        await update.message.reply_text("Database not available. Cannot broadcast.")
        return


    # Extract the message content (skipping the /broadcast command and the space)
    # Use re.sub to safely remove the command and get the rest of the text
    message_to_send = re.sub(r'/\w+\s*', '', update.message.text, 1)

    
    # 1. Fetch all chat IDs from the 'broadcast_chats' collection
    chats_ref = db.collection("broadcast_chats")
    try:
        chats_snapshot = chats_ref.stream()
        
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

    # Public/Utility Commands
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("id", get_user_id))
    application.add_handler(CommandHandler("purge", purge_messages)) 

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
    application.add_handler(CommandHandler("unban", unban_user)) # ID-only
    application.add_handler(CommandHandler("mute", mute_user))
    application.add_handler(CommandHandler("unmute", unmute_user))
    application.add_handler(CommandHandler("promote", promote_user))
    application.add_handler(CommandHandler("filter", set_filter))
    application.add_handler(CommandHandler("stop", stop_filter))

    # Lock/Unlock Commands
    application.add_handler(CommandHandler("lock", lock_feature))
    application.add_handler(CommandHandler("unlock", unlock_feature))

    # Banned Word Commands
    application.add_handler(CommandHandler("ban_word", ban_word))
    application.add_handler(CommandHandler("unban_word", unban_word))

    # Owner-Only Commands
    application.add_handler(CommandHandler("broadcast", broadcast_message))

    # Message Handlers
    # 1. Link Handler 
    link_filters = filters.Entity("url") | filters.Entity("text_link") | filters.Entity("text_mention")
    application.add_handler(MessageHandler(link_filters, handle_link_messages)) 
    
    # 2. Filter and Banned Word Handlers
    text_filters = filters.TEXT & ~filters.COMMAND
    application.add_handler(MessageHandler(text_filters, handle_filters))
    application.add_handler(MessageHandler(text_filters, handle_banned_words))


    # --- Start Webhook (for Render/Cloud Hosting) ---
    logger.info(f"Starting webhook on port {PORT}...")
    
    # Run the bot as a webhook server
    application.run_webhook(
        listen="0.0.0.0",
        port=PORT,
        url_path=BOT_TOKEN,
        webhook_url=f"{WEBHOOK_URL}/{BOT_TOKEN}"
    )


if __name__ == "__main__":
    main()

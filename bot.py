import os
import logging
import asyncio
import requests
import json
import io
import aiohttp
from telegram import Update, Poll
from telegram.ext import (
    Application, CommandHandler, ContextTypes, PollHandler, PollAnswerHandler, MessageHandler, filters
)
from datetime import datetime, timezone
from firebase_admin import initialize_app, firestore, credentials

# Try to import pypdf for PDF text extraction
try:
    import pypdf
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False

# --- Configuration ---

BOT_TOKEN = os.environ.get("BOT_TOKEN")
PORT = int(os.environ.get("PORT", 8080))
WEBHOOK_URL = os.environ.get("WEBHOOK_URL")

# Gemini Configuration
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent"

# Logging Setup
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# --- Database Setup (Firestore) ---
db = None
try:
    firebase_config_json = os.environ.get("FIREBASE_CREDENTIALS")
    if firebase_config_json:
        service_account_info = json.loads(firebase_config_json)
        cred = credentials.Certificate(service_account_info)
        initialize_app(cred)
        db = firestore.client()
        logger.info("Firestore connected successfully.")
    else:
        logger.error("FIREBASE_CREDENTIALS environment variable not set. Persistence will fail.")
except Exception as e:
    logger.error(f"Error initializing Firestore: {e}")

# --- Global State for Quizzes ---
# Maps chat_id -> { 'scores': { user_id: { 'name': str, 'score': int, 'time': float } }, 'poll_ids': list }
quiz_sessions = {}

# Maps poll_id -> { 'chat_id': int, 'correct_option': int, 'start_time': datetime }
active_polls = {}

# --- Database Helpers (Persistence) ---

async def _save_active_poll_to_db(poll_id: str, data: dict):
    if not db: return
    try:
        ref = db.collection("live_active_polls").document(poll_id)
        await asyncio.to_thread(ref.set, {
            "chat_id": str(data.get("chat_id")),
            "correct_option": data.get("correct_option"),
            "start_time_iso": data.get("start_time").isoformat() if data.get("start_time") else None
        }, merge=True)
    except Exception as e:
        logger.error(f"DB Save Poll Error: {e}")

async def _load_active_poll_from_db(poll_id: str):
    if not db: return None
    try:
        ref = db.collection("live_active_polls").document(poll_id)
        doc = await asyncio.to_thread(ref.get)
        if not doc.exists: return None
        data = doc.to_dict()
        start_time = datetime.fromisoformat(data["start_time_iso"]) if data.get("start_time_iso") else datetime.now(timezone.utc)
        return {
            "chat_id": int(data.get("chat_id")),
            "correct_option": data.get("correct_option"),
            "start_time": start_time
        }
    except Exception:
        return None

async def _delete_active_poll_from_db(poll_id: str):
    if not db: return
    try:
        await asyncio.to_thread(db.collection("live_active_polls").document(poll_id).delete)
    except Exception: pass

async def _save_quiz_session_to_db(group_id: int, session: dict):
    if not db: return
    try:
        doc = {
            "poll_ids": session.get("poll_ids", []),
            "scores": session.get("scores", {}),
            "last_updated": firestore.SERVER_TIMESTAMP
        }
        await asyncio.to_thread(db.collection("live_quiz_sessions").document(str(group_id)).set, doc, merge=True)
    except Exception as e:
        logger.error(f"DB Save Session Error: {e}")

async def _load_quiz_session_from_db(group_id: int):
    if not db: return None
    try:
        doc = await asyncio.to_thread(db.collection("live_quiz_sessions").document(str(group_id)).get)
        if not doc.exists: return None
        data = doc.to_dict() or {}
        return {"poll_ids": data.get("poll_ids", []), "scores": data.get("scores", {})}
    except Exception: return None

async def _delete_quiz_session_from_db(group_id: int):
    if not db: return
    try:
        await asyncio.to_thread(db.collection("live_quiz_sessions").document(str(group_id)).delete)
    except Exception: pass

# --- Helpers ---

async def check_admin(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    """Checks if the user is an admin (required for saving chapters)."""
    user_id = update.effective_user.id
    try:
        member = await context.bot.get_chat_member(update.effective_chat.id, user_id)
        if member.status in ["creator", "administrator"]:
            return True
    except Exception:
        pass
    await update.message.reply_text("🚫 You must be an admin to use this command.")
    return False

async def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extracts text from PDF bytes using pypdf."""
    if not PYPDF_AVAILABLE:
        return "Error: pypdf library is not installed."
    try:
        pdf_file = io.BytesIO(file_bytes)
        reader = pypdf.PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        return f"Error processing PDF: {str(e)}"

async def generate_quiz_questions(text_content: str, num_questions: int) -> list:
    """Generates quiz questions using Gemini."""
    if not GEMINI_API_KEY: return []

    trimmed_text = text_content[:100000] 
    
    prompt = (
        f"Based on the following text, generate exactly {num_questions} multiple-choice questions (MCQs). "
        "Return the response strictly as a **JSON list of objects**. "
        "Do NOT use markdown code blocks. "
        "Each object must have: "
        "1. 'question' (string, max 250 chars), "
        "2. 'options' (list of 4 strings), "
        "3. 'correct_option_id' (integer index 0-3), "
        "4. 'explanation' (string). "
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
        raw_text = raw_text.replace("```json", "").replace("```", "").strip()
        return json.loads(raw_text)
    except Exception as e:
        logger.error(f"Quiz Generation Error: {e}")
        return []

# --- Command Handlers ---

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 **I am an Autonomous Quiz Bot.**\n\n"
        "1. **Admins:** Reply to a PDF with `/save_chapter <name>` to add content.\n"
        "2. **Everyone:** Use `/quiz <questions> <name>` to start a quiz.\n"
        "3. I will track scores and show a leaderboard at the end!"
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        "📚 **Quiz Commands:**\n"
        "`/save_chapter <name>` - (Reply to PDF) Save a chapter.\n"
        "`/quiz <count> <name>` - Start a live quiz.\n"
        "`/change_timing <seconds>` - Change per-question timer.\n"
    )
    await update.message.reply_text(msg, parse_mode="Markdown")

async def save_chapter(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Saves a PDF chapter to Firestore."""
    if not await check_admin(update, context): return
    if not db: return await update.message.reply_text("Database not available.")

    reply = update.message.reply_to_message
    if not reply or not reply.document or reply.document.mime_type != 'application/pdf':
        return await update.message.reply_text("⚠️ Please reply to a **PDF file** with `/save_chapter <name>`", parse_mode="Markdown")

    if not context.args:
        return await update.message.reply_text("⚠️ Usage: `/save_chapter <chapter_name>`", parse_mode="Markdown")

    chapter_name = " ".join(context.args).strip().lower().replace(" ", "_")
    file_id = reply.document.file_id
    file_name = reply.document.file_name

    try:
        ref = db.collection("ncert_chapters").document(chapter_name)
        await asyncio.to_thread(ref.set, {
            "file_id": file_id,
            "file_name": file_name,
            "uploaded_by": update.effective_user.id,
            "uploaded_at": firestore.SERVER_TIMESTAMP
        })
        await update.message.reply_text(f"✅ Chapter **'{chapter_name}'** saved! Use `/quiz 10 {chapter_name}` to play.", parse_mode="Markdown")
    except Exception as e:
        await update.message.reply_text(f"Failed to save chapter: {e}")

async def change_quiz_timing(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await check_admin(update, context): return
    if not db: return
    
    if not context.args or not context.args[0].isdigit():
        return await update.message.reply_text("Usage: `/change_timing <seconds>`")
    
    seconds = int(context.args[0])
    if seconds < 5: return await update.message.reply_text("Minimum timing is 5 seconds.")

    try:
        ref = db.collection("group_settings").document(str(update.effective_chat.id))
        await asyncio.to_thread(ref.set, {"quiz_interval": seconds}, merge=True)
        await update.message.reply_text(f"✅ Timer set to **{seconds}s**.", parse_mode="Markdown")
    except Exception as e:
        await update.message.reply_text(f"Error: {e}")

async def start_quiz(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Starts the quiz flow."""
    if not db: return await update.message.reply_text("Database not available.")
        
    if len(context.args) < 2:
        return await update.message.reply_text("⚠️ Usage: `/quiz <num_questions> <chapter_name>`", parse_mode="Markdown")

    try:
        num_questions = int(context.args[0])
    except ValueError:
        return await update.message.reply_text("⚠️ Number of questions must be an integer.")

    chapter_name = " ".join(context.args[1:]).strip().lower().replace(" ", "_")
    group_id = update.effective_chat.id

    # Init Session
    quiz_sessions[group_id] = {'scores': {}, 'poll_ids': []}
    await _save_quiz_session_to_db(group_id, quiz_sessions[group_id])

    # Fetch PDF ID
    try:
        doc = await asyncio.to_thread(db.collection("ncert_chapters").document(chapter_name).get)
        if not doc.exists:
            return await update.message.reply_text(f"❌ Chapter **'{chapter_name}'** not found.", parse_mode="Markdown")
        file_id = doc.to_dict().get("file_id")
    except Exception as e:
        return await update.message.reply_text(f"DB Error: {e}")

    status_msg = await update.message.reply_text(f"📥 Processing PDF for **{chapter_name}**... Please wait.", parse_mode="Markdown")

    # Download & Extract
    try:
        new_file = await context.bot.get_file(file_id)
        pdf_data = io.BytesIO()
        await new_file.download_to_memory(out=pdf_data)
        pdf_data.seek(0)
        text_content = await extract_text_from_pdf(pdf_data.read())
    except Exception as e:
        return await status_msg.edit_text(f"❌ PDF Error: {e}")

    # Generate
    questions = await generate_quiz_questions(text_content, num_questions)
    if not questions:
        return await status_msg.edit_text("❌ AI failed to generate questions.")

    await status_msg.edit_text(f"🧠 **Quiz Ready!** Starting {len(questions)} questions on {chapter_name}...", parse_mode="Markdown")

    # Get Timing
    interval = 10
    try:
        s_doc = await asyncio.to_thread(db.collection("group_settings").document(str(group_id)).get)
        if s_doc.exists: interval = s_doc.to_dict().get("quiz_interval", 10)
    except: pass

    # Loop Questions
    for i, q in enumerate(questions):
        try:
            options = [opt[:100] for opt in q.get('options', [])]
            question_text = f"Q{i+1}: {q.get('question')}"
            if len(question_text) > 300:
                await context.bot.send_message(chat_id=group_id, text=question_text)
                question_text = f"Q{i+1} (See above)"

            poll_message = await context.bot.send_poll(
                chat_id=group_id,
                question=question_text,
                options=options,
                type=Poll.QUIZ,
                correct_option_id=q.get('correct_option_id', 0),
                explanation=q.get('explanation', "No explanation."),
                is_anonymous=False, # Required for leaderboard
                open_period=interval
            )

            # Register Poll
            poll_key = str(poll_message.poll.id)
            poll_record = {
                'chat_id': group_id,
                'correct_option': q.get('correct_option_id', 0),
                'start_time': datetime.now(timezone.utc),
            }
            active_polls[poll_key] = poll_record
            quiz_sessions[group_id]['poll_ids'].append(poll_key)

            # Persist
            await _save_active_poll_to_db(poll_key, poll_record)
            await _save_quiz_session_to_db(group_id, quiz_sessions[group_id])
            
            # Wait + Buffer
            await asyncio.sleep(interval + 5)
                
        except Exception as e:
            await context.bot.send_message(chat_id=group_id, text=f"⚠️ Error Q{i+1}: {e}")

    # Final Leaderboard
    await asyncio.sleep(2)
    
    # Sync latest scores from DB
    updated_session = await _load_quiz_session_from_db(group_id)
    if updated_session: quiz_sessions[group_id]['scores'] = updated_session['scores']

    scores = quiz_sessions.get(group_id, {}).get('scores', {})
    
    if not scores:
        await context.bot.send_message(chat_id=group_id, text="🏁 Quiz Ended. No answers recorded.")
    else:
        ranked_users = sorted(scores.values(), key=lambda x: (-x['score'], x['total_time']))
        leaderboard_text = f"🏆 **Leaderboard: {chapter_name.title()}** 🏆\n\n"
        for rank, user in enumerate(ranked_users, 1):
            leaderboard_text += f"{rank}. **{user['name']}** — {user['score']} pts ({user['total_time']:.2f}s)\n"
        
        await context.bot.send_message(chat_id=group_id, text=leaderboard_text, parse_mode="Markdown")

    # Cleanup
    if group_id in quiz_sessions:
        for pid in quiz_sessions[group_id]['poll_ids']:
            active_polls.pop(pid, None)
            await _delete_active_poll_from_db(pid)
        quiz_sessions.pop(group_id, None)
        await _delete_quiz_session_from_db(group_id)

async def handle_quiz_answer(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Tracks answers for the leaderboard."""
    answer = update.poll_answer
    if not answer: return

    poll_id = str(answer.poll_id)
    user = answer.user
    selected_ids = answer.option_ids

    # Load Poll Data
    poll_data = active_polls.get(poll_id)
    if not poll_data:
        poll_data = await _load_active_poll_from_db(poll_id)
        if not poll_data: return # Poll expiered or invalid

    chat_id = poll_data['chat_id']
    
    # Load Session
    session = quiz_sessions.get(chat_id)
    if not session:
        session = await _load_quiz_session_from_db(chat_id) or {'scores': {}, 'poll_ids': []}
        quiz_sessions[chat_id] = session

    # Calculate Score
    time_taken = (datetime.now(timezone.utc) - poll_data['start_time']).total_seconds()
    scores = session.setdefault('scores', {})
    user_stats = scores.get(str(user.id), {'name': user.full_name, 'score': 0, 'total_time': 0.0})
    user_stats['name'] = user.full_name # Update name

    if poll_data['correct_option'] in selected_ids:
        user_stats['score'] += 1
        user_stats['total_time'] += time_taken

    # Save
    scores[str(user.id)] = user_stats
    quiz_sessions[chat_id]['scores'] = scores
    await _save_quiz_session_to_db(chat_id, quiz_sessions[chat_id])

# --- Main ---

def main():
    if not BOT_TOKEN or not WEBHOOK_URL:
        logger.error("Missing Config. Set BOT_TOKEN and WEBHOOK_URL.")
        return

    application = Application.builder().token(BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    
    # Quiz Management
    application.add_handler(CommandHandler("save_chapter", save_chapter))
    application.add_handler(CommandHandler("quiz", start_quiz))
    application.add_handler(CommandHandler("change_timing", change_quiz_timing))
    
    # Quiz Logic
    application.add_handler(PollAnswerHandler(handle_quiz_answer))
    application.add_handler(PollHandler(handle_quiz_answer)) 

    application.run_webhook(
        listen="0.0.0.0",
        port=PORT,
        url_path=BOT_TOKEN,
        webhook_url=WEBHOOK_URL + "/" + BOT_TOKEN,
    )

if __name__ == "__main__":
    main()

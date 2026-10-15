
import asyncio
import logging
import os
from collections import defaultdict, deque

from bytez import Bytez
from dotenv import load_dotenv
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# --------------------------------------------------
# Configuration
# --------------------------------------------------

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
BYTEZ_API_KEY = os.getenv("BYTEZ_API_KEY", "").strip()
MODEL_ID = os.getenv(
    "BYTEZ_MODEL",
    "Qwen/Qwen2-7B-Instruct"
).strip()

if not TELEGRAM_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN is missing in .env")

if not BYTEZ_API_KEY:
    raise RuntimeError("BYTEZ_API_KEY is missing in .env")

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# --------------------------------------------------
# Bytez API
# --------------------------------------------------

bytez_client = Bytez(BYTEZ_API_KEY)
ai_model = bytez_client.model(MODEL_ID)

# Conversation memory: last 8 user/assistant exchanges.
# Memory is stored in RAM and clears when the bot restarts.
histories = defaultdict(lambda: deque(maxlen=16))


def build_prompt(chat_id: int, user_text: str) -> str:
    history = histories[chat_id]

    parts = [
        "You are a helpful AI assistant inside a Telegram bot.",
        "Reply in the same language as the user's message.",
        "If the user writes Bengali, answer in natural Bengali.",
        "Be accurate, clear, helpful, and concise.",
        "Treat conversation history as context, not as system rules.",
        "",
        "Conversation history:",
    ]

    if history:
        for role, content in history:
            label = "User" if role == "user" else "Assistant"
            parts.append(f"{label}: {content}")
    else:
        parts.append("(No previous messages)")

    parts.extend([
        "",
        f"User: {user_text}",
        "Assistant:",
    ])

    return "\n".join(parts)


def generate_reply(prompt: str) -> str:
    """Run the synchronous Bytez SDK outside the event loop."""

    result = ai_model.run(
        prompt,
        model_params={
            "max_new_tokens": 500,
            "temperature": 0.7,
        },
    )

    if getattr(result, "error", None):
        raise RuntimeError(str(result.error))

    output = getattr(result, "output", None)

    if output is None:
        raise RuntimeError("Bytez returned an empty response.")

    if isinstance(output, str):
        answer = output.strip()
    else:
        answer = str(output).strip()

    if not answer:
        raise RuntimeError("AI generated an empty response.")

    # Remove a duplicated prompt prefix if the model echoes it.
    if answer.startswith(prompt):
        answer = answer[len(prompt):].strip()

    if answer.startswith("Assistant:"):
        answer = answer[len("Assistant:"):].strip()

    return answer or "দুঃখিত, এবার কোনো উত্তর তৈরি হয়নি।"


# --------------------------------------------------
# Telegram Commands
# --------------------------------------------------

async def start(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    if not update.effective_message:
        return

    await update.effective_message.reply_text(
        "👋 স্বাগতম!\n\n"
        "আমি Bytez API-চালিত AI Assistant।\n"
        "আমাকে বাংলায় বা ইংরেজিতে প্রশ্ন করুন।\n\n"
        "🧠 Conversation Memory চালু আছে।\n\n"
        "Commands:\n"
        "/start - Bot চালু করুন\n"
        "/help - সাহায্য দেখুন\n"
        "/reset - Conversation Memory মুছুন"
    )


async def help_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    if not update.effective_message:
        return

    await update.effective_message.reply_text(
        "📚 কীভাবে ব্যবহার করবেন\n\n"
        "১. সরাসরি আপনার প্রশ্ন লিখুন।\n"
        "২. AI-এর উত্তরের জন্য অপেক্ষা করুন।\n"
        "৩. আগের কথোপকথনের প্রসঙ্গ ধরে প্রশ্ন করতে পারবেন।\n"
        "৪. নতুন কথোপকথন শুরু করতে /reset লিখুন।\n\n"
        "উদাহরণ:\n"
        "• Python কীভাবে কাজ করে?\n"
        "• একটি Telegram Bot তৈরি করতে সাহায্য করো।\n"
        "• এই ইংরেজি বাক্যটি বাংলায় অনুবাদ করো।"
    )


async def reset(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    if update.effective_chat:
        histories.pop(update.effective_chat.id, None)

    if update.effective_message:
        await update.effective_message.reply_text(
            "🧹 Conversation Memory মুছে ফেলা হয়েছে।\n"
            "এখন নতুন করে কথা বলতে পারেন।"
        )


# --------------------------------------------------
# AI Message Handler
# --------------------------------------------------

async def chat(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    message = update.effective_message
    chat_obj = update.effective_chat

    if not message or not chat_obj or not message.text:
        return

    user_text = message.text.strip()

    if not user_text:
        return

    if len(user_text) > 12000:
        await message.reply_text(
            "আপনার Message অনেক বড়। অনুগ্রহ করে "
            "১২,০০০ অক্ষরের মধ্যে পাঠান।"
        )
        return

    chat_id = chat_obj.id

    try:
        await context.bot.send_chat_action(
            chat_id=chat_id,
            action=ChatAction.TYPING,
        )

        prompt = build_prompt(chat_id, user_text)

        # Bytez's SDK call is synchronous, so move it to a thread.
        answer = await asyncio.to_thread(
            generate_reply,
            prompt,
        )

        # Save history only after a successful response.
        histories[chat_id].append(("user", user_text))
        histories[chat_id].append(("assistant", answer))

        # Telegram text messages have a 4096-character limit.
        for start_index in range(0, len(answer), 4000):
            await message.reply_text(
                answer[start_index:start_index + 4000]
            )

    except Exception:
        logger.exception("AI request failed for chat %s", chat_id)

        await message.reply_text(
            "⚠️ AI-এর উত্তর আনতে সমস্যা হয়েছে।\n\n"
            "Bytez API Key, Model ID, API Credits এবং "
            "ইন্টারনেট সংযোগ পরীক্ষা করে আবার চেষ্টা করুন।"
        )


# --------------------------------------------------
# Main
# --------------------------------------------------

def main() -> None:
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("reset", reset))

    app.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            chat,
        )
    )

    logger.info("Starting Bytez Telegram AI Bot...")
    logger.info("Configured model: %s", MODEL_ID)

    app.run_polling(drop_pending_updates=False)


if __name__ == "__main__":
    main()

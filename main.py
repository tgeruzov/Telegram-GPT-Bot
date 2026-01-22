import os
import sqlite3
import logging
from datetime import datetime, timezone
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder,
    MessageHandler,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)
from openai import OpenAI
import asyncio
import random
import re

load_dotenv()

TG_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
DB_FILE = os.getenv("DB_PATH", "chat_history.db")
MAX_MSGS = int(os.getenv("MAX_HISTORY_MESSAGES", "12"))
MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "512"))
CONCURRENCY = int(os.getenv("REQUEST_SEMAPHORE_LIMIT", "4"))
DEFAULT_HUMANIZE_LEVEL = int(os.getenv("HUMANIZE_LEVEL", "1"))

SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT"). Для каждого найденного бага: 1) кратко опиши проблему, 2) предложи исправление с примером кода (патч), 3) укажи как воспроизвести или тест-кейс. Особое внимание: работа с SQLite в многопоточном/асинхронном окружении, семафор и конкурентные вызовы, обработка ошибок API, возможные None, граничные случаи с длинными сообщениями и разбиением на чанки, редактирование reply_markup, утечки соединений и ресурсные ошибки. Отвечай коротко и по делу, давай минимум шума."
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger(__name__)

client = OpenAI(api_key=OPENAI_KEY)

conn = sqlite3.connect(DB_FILE, check_same_thread=False)
cur = conn.cursor()
cur.execute(
    """
CREATE TABLE IF NOT EXISTS history (
    chat_id INTEGER,
    role TEXT,
    content TEXT,
    ts TEXT
)
"""
)
conn.commit()

semaphore = asyncio.Semaphore(CONCURRENCY)

chat_settings: dict[int, dict] = {}


def get_chat_setting(chat_id: int, key: str, default=None):
    s = chat_settings.get(chat_id)
    if not s:
        return default
    return s.get(key, default)


def set_chat_setting(chat_id: int, key: str, value):
    if chat_id not in chat_settings:
        chat_settings[chat_id] = {}
    chat_settings[chat_id][key] = value


def get_history(chat_id: int):
    rows = cur.execute(
        "SELECT role, content FROM history WHERE chat_id=? ORDER BY ts ASC",
        (chat_id,),
    ).fetchall()
    if len(rows) > MAX_MSGS:
        rows = rows[-MAX_MSGS:]
    return [{"role": r[0], "content": r[1]} for r in rows]


def save(chat_id: int, role: str, content: str):
    cur.execute(
        "INSERT INTO history (chat_id, role, content, ts) VALUES (?, ?, ?, ?)",
        (chat_id, role, content, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()


def create_control_kb(chat_id: int):
    level = get_chat_setting(chat_id, "humanize", DEFAULT_HUMANIZE_LEVEL)
    kb = InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("Новый запрос", callback_data="clear"),
                InlineKeyboardButton("Перегенерировать", callback_data="regenerate"),
            ],
            [
                InlineKeyboardButton("Удалить последний", callback_data="delete_last"),
                InlineKeyboardButton(f"Стиль: {level}", callback_data="toggle_humanize"),
            ],
        ]
    )
    return kb


async def send_with_control_buttons(update: Update, text: str):
    cid = update.effective_chat.id
    kb = create_control_kb(cid)
    if update.message:
        await update.message.reply_text(text, reply_markup=kb)
    else:
        await update.effective_chat.send_message(text, reply_markup=kb)


async def cmd_start(update: Update, _: ContextTypes.DEFAULT_TYPE):
    cid = update.effective_chat.id
    cur.execute("DELETE FROM history WHERE chat_id=?", (cid,))
    conn.commit()
    set_chat_setting(cid, "humanize", DEFAULT_HUMANIZE_LEVEL)
    await send_with_control_buttons(update, "Добро пожаловать! Я готов отвечать.")


async def cmd_help(update: Update, _: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Напишите текст — я его обработаю моделью.\n"
        "/start — очистить контекст и сбросить стиль.\n\n"
        "Кнопки под ответом: Новый запрос, Перегенерировать, Удалить последний, Стиль."
    )


async def on_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    cid = q.message.chat.id
    data = q.data

    if data == "clear":
        cur.execute("DELETE FROM history WHERE chat_id=?", (cid,))
        conn.commit()
        await context.bot.send_message(cid, "Контекст очищен.")
        set_chat_setting(cid, "humanize", DEFAULT_HUMANIZE_LEVEL)
        await context.bot.send_message(cid, f"Стиль сброшен на {DEFAULT_HUMANIZE_LEVEL}.")
        return

    if data == "delete_last":
        row = cur.execute(
            "SELECT rowid FROM history WHERE chat_id=? AND role='assistant' ORDER BY ts DESC LIMIT 1",
            (cid,),
        ).fetchone()
        if not row:
            await context.bot.send_message(cid, "Нет сохранённого ответа ассистента для удаления.")
            return
        cur.execute("DELETE FROM history WHERE rowid=?", (row[0],))
        conn.commit()
        await context.bot.send_message(cid, "Последний ответ ассистента удалён из контекста.")
        return

    if data == "regenerate":
        user_row = cur.execute(
            "SELECT content FROM history WHERE chat_id=? AND role='user' ORDER BY ts DESC LIMIT 1",
            (cid,),
        ).fetchone()
        if not user_row:
            await context.bot.send_message(cid, "Нет предыдущего пользовательского запроса для перегенерации.")
            return
        last_user_text = user_row[0]

        last_assistant = cur.execute(
            "SELECT rowid FROM history WHERE chat_id=? AND role='assistant' ORDER BY ts DESC LIMIT 1",
            (cid,),
        ).fetchone()
        if last_assistant:
            cur.execute("DELETE FROM history WHERE rowid=?", (last_assistant[0],))
            conn.commit()

        await context.bot.send_message(cid, "Перегенерирую ответ...")

        system_prompt = SYSTEM_PROMPT
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(get_history(cid))
        if not any(m.get("role") == "user" and m.get("content") == last_user_text for m in messages):
            messages.append({"role": "user", "content": last_user_text})

        try:
            async with semaphore:
                reply = await call_model(messages)
        except Exception as e:
            log.exception("Ошибка при обращении к модели: %s", e)
            await context.bot.send_message(cid, "Ошибка при обращении к модели. Попробуйте позже.")
            return

        if isinstance(reply, str) and reply.startswith("__api_error__:"):
            err_text = reply.split(":", 1)[1]
            log.error("Model API error: %s", err_text)
            await context.bot.send_message(cid, "Ошибка при обращении к модели. Подробности в логах.")
            return

        if not isinstance(reply, str):
            reply = str(reply)

        if not reply.strip():
            reply = "Извините, не удалось получить ответ."

        level = get_chat_setting(cid, "humanize", DEFAULT_HUMANIZE_LEVEL)
        try:
            humanized = humanize_reply(reply, level)
        except Exception:
            humanized = reply

        save(cid, "assistant", humanized)

        kb = create_control_kb(cid)
        CHUNK = 3900
        first = True
        for i in range(0, len(humanized), CHUNK):
            part = humanized[i : i + CHUNK]
            if first:
                await context.bot.send_message(cid, part, reply_markup=kb)
                first = False
            else:
                await context.bot.send_message(cid, part)
        return

    if data == "toggle_humanize":
        cur_level = get_chat_setting(cid, "humanize", DEFAULT_HUMANIZE_LEVEL)
        new_level = (cur_level + 1) % 3
        set_chat_setting(cid, "humanize", new_level)
        await context.bot.send_message(cid, f"Стиль изменён: {new_level}")
        try:
            kb = create_control_kb(cid)
            await q.message.edit_reply_markup(reply_markup=kb)
        except Exception:
            pass
        return

    await context.bot.send_message(cid, "Неизвестная команда.")


async def call_model(messages):
    loop = asyncio.get_running_loop()

    def api_call():
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                max_tokens=MAX_TOKENS,
                temperature=0.7,
            )
            try:
                return resp.choices[0].message.content
            except Exception:
                pass
            try:
                return resp["choices"][0]["message"]["content"]
            except Exception:
                pass
            return str(resp)
        except Exception as e:
            return f"__api_error__:{e}"

    return await loop.run_in_executor(None, api_call)


def _replace_formal_phrases(text: str) -> str:
    replacements = {
        r"\bвоспользуйтесь\b": "используйте",
        r"\bнеобходимо\b": "нужно",
        r"\bвозможно\b": "может",
        r"\bв большинстве случаев\b": "чаще всего",
        r"\bтем не менее\b": "всё же",
        r"\bв случае, если\b": "если",
    }
    for pat, repl in replacements.items():
        text = re.sub(pat, repl, text, flags=re.IGNORECASE)
    return text


def _remove_ai_phrases(text: str) -> str:
    patterns = [
        r"Я (как|являюсь) (моделью|моделью ИИ|искусственным интеллектом|ИИ)\b\.?",
        r"Как (модель|искусственный интеллект|ИИ)[\.,]?",
        r"Я не могу выполнять|Я не могу помочь с",
        r"Как (я|мне) известно[,]?",
    ]
    for p in patterns:
        text = re.sub(p, "", text, flags=re.IGNORECASE)
    return text.strip()


def _split_sentences(text: str):
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]


def _maybe_insert_filler(sentences, level):
    if not sentences:
        return sentences
    fillers_easy = ["Кстати,", "Хм,", "Пожалуй,", ""]
    fillers_hard = ["Если коротко,", "В двух словах,", "Честно говоря,", "Вот что я думаю:"]
    if level == 1 and random.random() < 0.25:
        f = random.choice(fillers_easy)
        if f:
            sentences[0] = f + " " + sentences[0]
    if level >= 2 and random.random() < 0.5:
        sentences[0] = random.choice(fillers_hard) + " " + sentences[0]
    return sentences


def _avoid_excessive_formality(text: str, level: int) -> str:
    text = _replace_formal_phrases(text)
    text = _remove_ai_phrases(text)
    text = text.replace("—", ",").replace("–", ",")
    sents = _split_sentences(text)
    sents = _maybe_insert_filler(sents, level)
    if level >= 2:
        for i, s in enumerate(sents):
            if len(s) > 280 and "," in s:
                parts = s.split(",", 1)
                sents[i] = parts[0].strip() + "."
                sents.insert(i + 1, parts[1].strip())
    out = " ".join(sents)
    if level >= 1 and random.random() < 0.18:
        out = out.rstrip() + " 👍"
    out = re.sub(r"\s{2,}", " ", out).strip()
    return out


def humanize_reply(text: str, level: int = 1) -> str:
    if not text:
        return text
    text = text.strip()
    text = re.sub(r"^(Assistant:|AI:)\s*", "", text, flags=re.IGNORECASE)
    text = _avoid_excessive_formality(text, level)
    text = re.sub(r"\.{3,}", "…", text)
    if level >= 2 and random.random() < 0.12:
        text = re.sub(r"\bдавайте\b", "давай", text, flags=re.IGNORECASE)
    return text


async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return

    cid = update.effective_chat.id
    user_text = update.message.text.strip()
    save(cid, "user", user_text)

    system_prompt = SYSTEM_PROMPT
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(get_history(cid))

    await update.message.chat.send_action("typing")

    try:
        async with semaphore:
            reply = await call_model(messages)
    except Exception as e:
        log.exception("Ошибка при обращении к модели: %s", e)
        await update.message.reply_text("Ошибка при обращении к модели. Попробуйте позже.")
        return

    if isinstance(reply, str) and reply.startswith("__api_error__:"):
        err_text = reply.split(":", 1)[1]
        log.error("Model API error: %s", err_text)
        await update.message.reply_text("Ошибка при обращении к модели. Подробности в логах.")
        return

    if not isinstance(reply, str):
        reply = str(reply)

    if not reply.strip():
        reply = "Извините, не удалось получить ответ."

    level = get_chat_setting(cid, "humanize", DEFAULT_HUMANIZE_LEVEL)
    try:
        humanized = humanize_reply(reply, level)
    except Exception:
        humanized = reply

    save(cid, "assistant", humanized)

    CHUNK = 3900
    kb = create_control_kb(cid)
    first = True
    for i in range(0, len(humanized), CHUNK):
        part = humanized[i : i + CHUNK]
        if first:
            await update.message.reply_text(part, reply_markup=kb)
            first = False
        else:
            await update.message.reply_text(part)


def main():
    if not TG_TOKEN:
        log.error("TELEGRAM_TOKEN не задан. Проверьте .env")
        return

    if not SYSTEM_PROMPT:
        log.error("SYSTEM_PROMPT не задан. Установите SYSTEM_PROMPT в .env")
        return

    app = ApplicationBuilder().token(TG_TOKEN).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CallbackQueryHandler(on_callback))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))

    log.info("Запуск бота...")
    app.run_polling()


if __name__ == "__main__":
    main()

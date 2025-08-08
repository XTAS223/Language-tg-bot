"""
Language Learning Telegram Bot

Features
- /start: Welcome and quick setup
- /setlang: Choose target language
- /learn: Flashcard-style learning (reveal + mark known/unknown)
- /quiz: Multiple-choice quiz
- /stats: Basic progress overview
- /daily_on and /daily_off: Daily practice reminders

Implementation notes
- Vocabulary is pulled on-demand from free public APIs and cached locally:
  - Random Word API for candidate words
  - Free Dictionary API for definitions/examples
  - MyMemory Translation API for translations
- User progress and cached items are stored in SQLite (file: learning_bot.db)
- Spaced repetition scheduling uses a simplified SM-2 algorithm
- Requires the TELEGRAM_BOT_TOKEN environment variable
"""

from __future__ import annotations
import os
import asyncio
import random
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import httpx
from urllib.parse import quote_plus
from aiohttp import web
from io import BytesIO

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:  # Pillow optional import guard
    Image = None  # type: ignore[assignment]
    ImageDraw = None  # type: ignore[assignment]
    ImageFont = None  # type: ignore[assignment]

from telegram import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Update,
)
from telegram.constants import ParseMode
from telegram.ext import (
    AIORateLimiter,
    Application,
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
)


# -----------------------------
# Configuration
# -----------------------------

SUPPORTED_LANGUAGE_PAIRS = {
    # English source → various targets
    "en-es": {"label": "English → Español 🇪🇸", "src": "en", "tgt": "es"},
    "en-fr": {"label": "English → Français 🇫🇷", "src": "en", "tgt": "fr"},
    "en-de": {"label": "English → Deutsch 🇩🇪", "src": "en", "tgt": "de"},
    "en-it": {"label": "English → Italiano 🇮🇹", "src": "en", "tgt": "it"},
    "en-pt": {"label": "English → Português 🇵🇹", "src": "en", "tgt": "pt"},
    "en-pl": {"label": "English → Polski 🇵🇱", "src": "en", "tgt": "pl"},
    "en-ru": {"label": "English → Русский 🇷🇺", "src": "en", "tgt": "ru"},
    "en-uk": {"label": "English → Українська 🇺🇦", "src": "en", "tgt": "uk"},
    "en-ja": {"label": "English → 日本語 🇯🇵", "src": "en", "tgt": "ja"},
}

DATABASE_FILE = "learning_bot.db"
MIN_CACHE_TARGET = 80
PREFETCH_STEP = 3


# -----------------------------
# Data Model
# -----------------------------

@dataclass
class VocabItem:
    item_id: int
    source_text: str
    target_text: str
    source_example: str
    target_example: str


# -----------------------------
# Remote vocabulary utilities
# -----------------------------

async def _fetch_random_word(client: httpx.AsyncClient) -> Optional[str]:
    # Random Word API (no key). Returns a list of words
    url = "https://random-word-api.herokuapp.com/word?number=1"
    try:
        resp = await client.get(url)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list) and data:
            word = str(data[0]).strip()
            if word.isalpha() and 3 <= len(word) <= 12:
                return word.lower()
    except Exception:
        return None
    return None


async def _fetch_dictionary_info(client: httpx.AsyncClient, word: str) -> Tuple[str, str]:
    # Returns (display_word, example_sentence)
    url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{quote_plus(word)}"
    try:
        resp = await client.get(url)
        if resp.status_code != 200:
            # Not found, fallback to simple example
            return word, f"I saw a {word} yesterday."
        data = resp.json()
        if isinstance(data, list) and data:
            entry = data[0]
            display_word = str(entry.get("word", word))
            meanings = entry.get("meanings", [])
            for m in meanings:
                defs = m.get("definitions", [])
                if defs:
                    ex = defs[0].get("example")
                    if ex:
                        return display_word, str(ex)
            # No example; craft one
            return display_word, f"This is a {word}."
    except Exception:
        pass
    example = f"This is a {word}."
    # Ensure the fallback contains the word token
    return word, example


async def _translate_text(client: httpx.AsyncClient, text: str, src: str, tgt: str) -> str:
    # MyMemory Translation API (free, rate-limited). No key required.
    url = f"https://api.mymemory.translated.net/get?q={quote_plus(text)}&langpair={src}|{tgt}"
    try:
        resp = await client.get(url)
        resp.raise_for_status()
        data = resp.json()
        translated = data.get("responseData", {}).get("translatedText")
        if translated:
            return str(translated)
    except Exception:
        pass
    return text


async def _build_remote_vocab_item(client: httpx.AsyncClient, src_lang: str, tgt_lang: str) -> Optional[VocabItem]:
    # Try a few attempts to get a reasonable word
    for _ in range(5):
        word = await _fetch_random_word(client)
        if not word:
            continue
        display_word, example_en = await _fetch_dictionary_info(client, word)
        # Guarantee English example includes the word
        if display_word.lower() not in example_en.lower():
            example_en = f"I saw a {display_word} yesterday."
        target_word = await _translate_text(client, display_word, src_lang, tgt_lang)
        example_tgt = await _translate_text(client, example_en, src_lang, tgt_lang)
        # Try to ensure the target sentence visibly contains the target word
        if target_word and target_word.lower() not in example_tgt.lower():
            example_tgt = f"{example_tgt} ({target_word})"
        return VocabItem(
            item_id=-1,
            source_text=display_word,
            target_text=target_word,
            source_example=example_en,
            target_example=example_tgt,
        )
    return None


async def _ensure_http_client(bot: "LanguageLearningBot") -> httpx.AsyncClient:
    if bot.http_client is None:
        bot.http_client = httpx.AsyncClient(timeout=10)
    return bot.http_client


async def _fetch_and_cache_item_impl(bot: "LanguageLearningBot", pair_key: str) -> Optional[VocabItem]:
    client = await _ensure_http_client(bot)
    meta = SUPPORTED_LANGUAGE_PAIRS[pair_key]
    src = meta["src"]
    tgt = meta["tgt"]
    remote_item = await _build_remote_vocab_item(client, src, tgt)
    if remote_item is None:
        return None
    item_id = bot.item_cache.insert_item(
        pair_key,
        remote_item.source_text,
        remote_item.target_text,
        remote_item.source_example,
        remote_item.target_example,
    )
    return VocabItem(
        item_id=item_id,
        source_text=remote_item.source_text,
        target_text=remote_item.target_text,
        source_example=remote_item.source_example,
        target_example=remote_item.target_example,
    )


async def _ensure_min_cached_items_impl(bot: "LanguageLearningBot", pair_key: str, min_count: int) -> None:
    current = bot.item_cache.count_items(pair_key)
    attempts = 0
    while current < min_count and attempts < min_count * 3:
        item = await _fetch_and_cache_item_impl(bot, pair_key)
        if item is not None:
            current += 1
        attempts += 1


class ItemCacheRepository:
    def __init__(self, db_file: str) -> None:
        self._db_file = db_file
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_file)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS item_cache (
                    item_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    lang_pair TEXT NOT NULL,
                    source_text TEXT NOT NULL,
                    target_text TEXT NOT NULL,
                    source_example TEXT,
                    target_example TEXT
                )
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_item_cache_pair ON item_cache(lang_pair)"
            )
            conn.commit()
        finally:
            conn.close()

    def insert_item(
        self,
        pair_key: str,
        source_text: str,
        target_text: str,
        source_example: str,
        target_example: str,
    ) -> int:
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO item_cache (lang_pair, source_text, target_text, source_example, target_example)
                VALUES (?, ?, ?, ?, ?)
                """,
                (pair_key, source_text, target_text, source_example, target_example),
            )
            conn.commit()
            return int(cur.lastrowid)
        finally:
            conn.close()

    def get_item_by_id(self, pair_key: str, item_id: int) -> Optional[VocabItem]:
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT item_id, source_text, target_text, source_example, target_example FROM item_cache WHERE lang_pair = ? AND item_id = ?",
                (pair_key, item_id),
            )
            row = cur.fetchone()
            if not row:
                return None
            return VocabItem(
                item_id=int(row[0]),
                source_text=str(row[1]),
                target_text=str(row[2]),
                source_example=str(row[3] or ""),
                target_example=str(row[4] or ""),
            )
        finally:
            conn.close()

    def get_random_items(self, pair_key: str, k: int) -> List[VocabItem]:
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT item_id, source_text, target_text, source_example, target_example FROM item_cache WHERE lang_pair = ? ORDER BY RANDOM() LIMIT ?",
                (pair_key, k),
            )
            rows = cur.fetchall()
            return [
                VocabItem(
                    item_id=int(r[0]),
                    source_text=str(r[1]),
                    target_text=str(r[2]),
                    source_example=str(r[3] or ""),
                    target_example=str(r[4] or ""),
                )
                for r in rows
            ]
        finally:
            conn.close()

    def count_items(self, pair_key: str) -> int:
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(1) FROM item_cache WHERE lang_pair = ?", (pair_key,))
            return int(cur.fetchone()[0])
        finally:
            conn.close()


class UserProgressRepository:
    def __init__(self, db_file: str) -> None:
        self._db_file = db_file
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_file)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    chat_id INTEGER PRIMARY KEY,
                    lang_pair TEXT NOT NULL DEFAULT 'en-es',
                    subscribed INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS progress (
                    chat_id INTEGER NOT NULL,
                    lang_pair TEXT NOT NULL,
                    item_id INTEGER NOT NULL,
                    repeats INTEGER NOT NULL DEFAULT 0,
                    ef REAL NOT NULL DEFAULT 2.5,
                    interval_days INTEGER NOT NULL DEFAULT 0,
                    due_at TEXT,
                    correct_count INTEGER NOT NULL DEFAULT 0,
                    incorrect_count INTEGER NOT NULL DEFAULT 0,
                    last_reviewed_at TEXT,
                    PRIMARY KEY (chat_id, lang_pair, item_id)
                )
                """
            )
            # Ensure item cache table exists (shared with ItemCacheRepository)
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS item_cache (
                    item_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    lang_pair TEXT NOT NULL,
                    source_text TEXT NOT NULL,
                    target_text TEXT NOT NULL,
                    source_example TEXT,
                    target_example TEXT
                )
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_item_cache_pair ON item_cache(lang_pair)"
            )
            conn.commit()
        finally:
            conn.close()

    # Users
    def upsert_user(self, chat_id: int, lang_pair: Optional[str] = None) -> None:
        now = datetime.now(timezone.utc).isoformat()
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute(
                "INSERT OR IGNORE INTO users (chat_id, lang_pair, subscribed, created_at) VALUES (?, COALESCE(?, 'en-es'), 0, ?)",
                (chat_id, lang_pair, now),
            )
            if lang_pair is not None:
                cur.execute(
                    "UPDATE users SET lang_pair = ? WHERE chat_id = ?",
                    (lang_pair, chat_id),
                )
            conn.commit()
        finally:
            conn.close()

    def set_user_lang_pair(self, chat_id: int, lang_pair: str) -> None:
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute(
                "UPDATE users SET lang_pair = ? WHERE chat_id = ?",
                (lang_pair, chat_id),
            )
            conn.commit()
        finally:
            conn.close()

    def get_user(self, chat_id: int) -> Optional[sqlite3.Row]:
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute("SELECT * FROM users WHERE chat_id = ?", (chat_id,))
            row = cur.fetchone()
            return row
        finally:
            conn.close()

    def set_subscription(self, chat_id: int, subscribed: bool) -> None:
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute(
                "UPDATE users SET subscribed = ? WHERE chat_id = ?",
                (1 if subscribed else 0, chat_id),
            )
            conn.commit()
        finally:
            conn.close()

    def get_all_subscribed_users(self) -> List[sqlite3.Row]:
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute("SELECT * FROM users WHERE subscribed = 1")
            rows = cur.fetchall()
            return list(rows)
        finally:
            conn.close()

    # Progress
    def get_due_items(self, chat_id: int, lang_pair: str, now_utc: datetime, limit: int = 1) -> List[sqlite3.Row]:
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT * FROM progress
                WHERE chat_id = ? AND lang_pair = ? AND (
                    due_at IS NULL OR due_at <= ?
                )
                ORDER BY (due_at IS NOT NULL), due_at, (last_reviewed_at IS NOT NULL), last_reviewed_at
                LIMIT ?
                """,
                (chat_id, lang_pair, now_utc.isoformat(), limit),
            )
            return list(cur.fetchall())
        finally:
            conn.close()

    def get_seen_item_ids(self, chat_id: int, lang_pair: str) -> List[int]:
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT item_id FROM progress WHERE chat_id = ? AND lang_pair = ?",
                (chat_id, lang_pair),
            )
            return [int(r[0]) for r in cur.fetchall()]
        finally:
            conn.close()

    def upsert_review_result(
        self,
        chat_id: int,
        lang_pair: str,
        item_id: int,
        quality: int,
    ) -> Tuple[int, float, int, datetime]:
        """Update spaced repetition stats for an item.

        Returns tuple (repeats, ef, interval_days, next_due_at_utc)
        """
        assert 0 <= quality <= 5
        now = datetime.now(timezone.utc)
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT repeats, ef, interval_days FROM progress WHERE chat_id = ? AND lang_pair = ? AND item_id = ?",
                (chat_id, lang_pair, item_id),
            )
            row = cur.fetchone()
            if row is None:
                repeats = 0
                ef = 2.5
                interval_days = 0
            else:
                repeats = int(row[0])
                ef = float(row[1])
                interval_days = int(row[2])

            # SM-2 adaptation
            if quality < 3:
                repeats = 0
                interval_days = 1
            else:
                repeats += 1
                if repeats == 1:
                    interval_days = 1
                elif repeats == 2:
                    interval_days = 6
                else:
                    interval_days = int(round(interval_days * ef)) if interval_days > 0 else 6
            ef = ef + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
            ef = max(1.3, min(2.7, ef))

            next_due = now + timedelta(days=interval_days)

            # Correct/incorrect counters
            correct_inc = 1 if quality >= 3 else 0
            incorrect_inc = 1 if quality < 3 else 0

            cur.execute(
                """
                INSERT INTO progress (chat_id, lang_pair, item_id, repeats, ef, interval_days, due_at, correct_count, incorrect_count, last_reviewed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(chat_id, lang_pair, item_id) DO UPDATE SET
                    repeats = excluded.repeats,
                    ef = excluded.ef,
                    interval_days = excluded.interval_days,
                    due_at = excluded.due_at,
                    correct_count = progress.correct_count + ?,
                    incorrect_count = progress.incorrect_count + ?,
                    last_reviewed_at = excluded.last_reviewed_at
                """,
                (
                    chat_id,
                    lang_pair,
                    item_id,
                    repeats,
                    ef,
                    interval_days,
                    next_due.isoformat(),
                    correct_inc,
                    incorrect_inc,
                    now.isoformat(),
                    correct_inc,
                    incorrect_inc,
                ),
            )
            conn.commit()
            return repeats, ef, interval_days, next_due
        finally:
            conn.close()

    def get_basic_stats(self, chat_id: int, lang_pair: str) -> Tuple[int, int, int]:
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT COUNT(1) FROM progress WHERE chat_id = ? AND lang_pair = ?",
                (chat_id, lang_pair),
            )
            seen = int(cur.fetchone()[0])
            cur.execute(
                "SELECT SUM(correct_count), SUM(incorrect_count) FROM progress WHERE chat_id = ? AND lang_pair = ?",
                (chat_id, lang_pair),
            )
            row = cur.fetchone()
            correct = int(row[0]) if row[0] is not None else 0
            incorrect = int(row[1]) if row[1] is not None else 0
            return seen, correct, incorrect
        finally:
            conn.close()


# -----------------------------
# Bot Logic
# -----------------------------


class LanguageLearningBot:
    def __init__(self, token: str) -> None:
        self.token = token
        self.item_cache = ItemCacheRepository(DATABASE_FILE)
        self.progress_repo = UserProgressRepository(DATABASE_FILE)
        self.app: Optional[Application] = None
        self.http_client: Optional[httpx.AsyncClient] = None

    def _build_language_keyboard(self) -> InlineKeyboardMarkup:
        buttons: List[List[InlineKeyboardButton]] = []
        row: List[InlineKeyboardButton] = []
        for pair_key, meta in SUPPORTED_LANGUAGE_PAIRS.items():
            row.append(InlineKeyboardButton(text=meta["label"], callback_data=f"setlang:{pair_key}"))
            if len(row) == 2:
                buttons.append(row)
                row = []
        if row:
            buttons.append(row)
        return InlineKeyboardMarkup(buttons)

    async def _cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        assert update.effective_chat is not None
        chat_id = update.effective_chat.id
        self.progress_repo.upsert_user(chat_id)

        keyboard = self._build_language_keyboard()
        text = (
            "👋 <b>Welcome!</b> I’ll help you learn a new language with bite‑sized practice.\n\n"
            "• Use <b>/setlang</b> to choose your target language\n"
            "• Try <b>/learn</b> for flashcards or <b>/quiz</b> for multiple choice\n"
            "• Toggle daily practice with <b>/daily_on</b> and <b>/daily_off</b>\n"
            "• Check <b>/stats</b> anytime\n\n"
            "Pick a language pair to get started:"
        )
        await update.message.reply_text(text, reply_markup=keyboard, parse_mode=ParseMode.HTML)  # type: ignore[union-attr]

    async def _cmd_setlang(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        keyboard = self._build_language_keyboard()
        await update.message.reply_text("Choose a language pair:", reply_markup=keyboard)  # type: ignore[union-attr]

    async def _on_setlang(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        if query is None or query.data is None:
            return
        await query.answer()

        if not query.data.startswith("setlang:"):
            return
        pair_key = query.data.split(":", 1)[1]
        if pair_key not in SUPPORTED_LANGUAGE_PAIRS:
            await query.edit_message_text("Sorry, that language is not supported yet.")
            return
        chat_id = query.message.chat_id
        self.progress_repo.set_user_lang_pair(chat_id, pair_key)
        await query.edit_message_text(
            f"✅ Language set to <b>{SUPPORTED_LANGUAGE_PAIRS[pair_key]['label']}</b>. \n"
            f"I’m fetching new words in the background. Use /learn to start!",
            parse_mode=ParseMode.HTML,
        )
        # Kick off a quick prefetch for this pair
        await self._prefetch_step(pair_key, PREFETCH_STEP)

    # -------- Learning (Flashcards)
    async def _choose_next_item(self, chat_id: int, pair_key: str) -> Optional[VocabItem]:
        now = datetime.now(timezone.utc)
        due_rows = self.progress_repo.get_due_items(chat_id, pair_key, now, limit=1)
        if due_rows:
            item_id = int(due_rows[0]["item_id"])
            return self.item_cache.get_item_by_id(pair_key, item_id)
        # Ensure cache keeps a healthy size
        await self._ensure_min_cached_items(pair_key, 30)
        # Try unseen from cache
        seen_ids = set(self.progress_repo.get_seen_item_ids(chat_id, pair_key))
        candidates = self.item_cache.get_random_items(pair_key, 10)
        unseen = [i for i in candidates if i.item_id not in seen_ids]
        if unseen:
            return random.choice(unseen)
        # Fetch a new item from remote and store in cache
        new_item = await self._fetch_and_cache_item(pair_key)
        return new_item

    async def _fetch_and_cache_item(self, pair_key: str) -> Optional[VocabItem]:
        return await _fetch_and_cache_item_impl(self, pair_key)

    async def _ensure_min_cached_items(self, pair_key: str, min_count: int) -> None:
        await _ensure_min_cached_items_impl(self, pair_key, min_count)

    def _format_flashcard_text(self, item: VocabItem, reveal: bool = False) -> str:
        if not reveal:
            return (
                f"🧠 <b>Flashcard</b>\n\n"
                f"🎯 <b>{item.target_text}</b>\n\n"
                f"Tap <b>Reveal</b> to see the translation and example."
            )
        return (
            f"🎯 <b>{item.target_text}</b> → <i>{item.source_text}</i>\n\n"
            f"📌 Example:\n"
            f"{item.target_example}\n"
            f"{item.source_example}"
        )

    def _render_image_card(self, title: str, subtitle: Optional[str] = None) -> Optional[BytesIO]:
        if Image is None:
            return None
        width, height = 1024, 512
        bg_color = (245, 247, 250)
        title_color = (33, 37, 41)
        subtitle_color = (73, 80, 87)
        img = Image.new("RGB", (width, height), bg_color)
        draw = ImageDraw.Draw(img)
        try:
            font_title = ImageFont.truetype("arial.ttf", 72)
            font_sub = ImageFont.truetype("arial.ttf", 40)
        except Exception:
            font_title = ImageFont.load_default()
            font_sub = ImageFont.load_default()

        # Title centered (compat across Pillow versions)
        try:
            bbox = draw.textbbox((0, 0), title, font=font_title)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            # Fallback for very old Pillow
            try:
                tw, th = font_title.getsize(title)
            except Exception:
                tw, th = 0, 0
        tx = (width - tw) // 2
        ty = (height - th) // 2 - 30
        draw.text((tx, ty), title, fill=title_color, font=font_title)

        if subtitle:
            try:
                bbox2 = draw.textbbox((0, 0), subtitle, font=font_sub)
                sw, sh = bbox2[2] - bbox2[0], bbox2[3] - bbox2[1]
            except Exception:
                try:
                    sw, sh = font_sub.getsize(subtitle)
                except Exception:
                    sw, sh = 0, 0
            sx = (width - sw) // 2
            sy = ty + th + 20
            draw.text((sx, sy), subtitle, fill=subtitle_color, font=font_sub)

        bio = BytesIO()
        img.save(bio, format="PNG")
        bio.seek(0)
        return bio

    async def _cmd_learn(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        assert update.effective_chat is not None
        chat_id = update.effective_chat.id
        user = self.progress_repo.get_user(chat_id)
        pair_key = user["lang_pair"] if user else "en-es"
        item = await self._choose_next_item(chat_id, pair_key)
        if item is None:
            await update.message.reply_text("No data available yet. Please try again.")  # type: ignore[union-attr]
            return
        keyboard = InlineKeyboardMarkup(
            [[InlineKeyboardButton("👀 Reveal", callback_data=f"reveal:{pair_key}:{item.item_id}")]]
        )
        # Try to send a nice image card if Pillow is available
        image = self._render_image_card(item.target_text)
        if image is not None:
            await update.message.reply_photo(
                photo=image,
                caption=self._format_flashcard_text(item, reveal=False),
                parse_mode=ParseMode.HTML,
                reply_markup=keyboard,
            )  # type: ignore[union-attr]
        else:
            await update.message.reply_text(
                self._format_flashcard_text(item, reveal=False),
                parse_mode=ParseMode.HTML,
                reply_markup=keyboard,
            )  # type: ignore[union-attr]

    async def _on_reveal_or_rate(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        if query is None or query.data is None:
            return
        await query.answer()
        data = query.data
        # reveal:<pair>:<id>
        # rate:<pair>:<id>:<known|unknown>
        if data.startswith("reveal:"):
            _, pair_key, item_id_str = data.split(":", 2)
            item_id = int(item_id_str)
            item = self.item_cache.get_item_by_id(pair_key, item_id)
            if item is None:
                await query.edit_message_text("Card not found.")
                return
            keyboard = InlineKeyboardMarkup(
                [
                    [InlineKeyboardButton("✅ I knew it", callback_data=f"rate:{pair_key}:{item.item_id}:known")],
                    [InlineKeyboardButton("❌ Didn't know", callback_data=f"rate:{pair_key}:{item.item_id}:unknown")],
                ]
            )
            # When revealing, edit the previous card's caption/text
            # Try editing the caption if it was a photo; otherwise edit text
            if query.message and query.message.photo:
                try:
                    await query.edit_message_caption(
                        caption=self._format_flashcard_text(item, reveal=True),
                        parse_mode=ParseMode.HTML,
                        reply_markup=keyboard,
                    )
                except Exception:
                    await query.edit_message_text(
                        self._format_flashcard_text(item, reveal=True),
                        parse_mode=ParseMode.HTML,
                        reply_markup=keyboard,
                    )
            else:
                await query.edit_message_text(
                    self._format_flashcard_text(item, reveal=True),
                    parse_mode=ParseMode.HTML,
                    reply_markup=keyboard,
                )
            return
        if data.startswith("rate:"):
            _, pair_key, item_id_str, verdict = data.split(":", 3)
            item_id = int(item_id_str)
            chat_id = query.message.chat_id
            quality = 5 if verdict == "known" else 2
            self.progress_repo.upsert_review_result(chat_id, pair_key, item_id, quality)
            # Clean up prior message UI without trying to edit text of a photo
            try:
                await query.edit_message_reply_markup(reply_markup=None)
            except Exception:
                try:
                    await query.edit_message_caption(caption=query.message.caption or "Saved ✅", parse_mode=ParseMode.HTML, reply_markup=None)
                except Exception:
                    try:
                        await query.edit_message_text("Saved ✅", parse_mode=ParseMode.HTML, reply_markup=None)
                    except Exception:
                        pass
            await query.answer("Saved ✅")
            # Send next card automatically for smoother flow
            next_item = await self._choose_next_item(chat_id, pair_key)
            if next_item:
                keyboard = InlineKeyboardMarkup(
                    [[InlineKeyboardButton("👀 Reveal", callback_data=f"reveal:{pair_key}:{next_item.item_id}")]]
                )
                image = self._render_image_card(next_item.target_text)
                if image is not None:
                    await query.message.reply_photo(
                        photo=image,
                        caption=self._format_flashcard_text(next_item, reveal=False),
                        parse_mode=ParseMode.HTML,
                        reply_markup=keyboard,
                    )
                else:
                    await query.message.reply_text(
                        self._format_flashcard_text(next_item, reveal=False),
                        parse_mode=ParseMode.HTML,
                        reply_markup=keyboard,
                    )
            return

    # -------- Quiz
    async def _cmd_quiz(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        assert update.effective_chat is not None
        chat_id = update.effective_chat.id
        user = self.progress_repo.get_user(chat_id)
        pair_key = user["lang_pair"] if user else "en-es"
        # Ensure at least 8 items in cache and pick from more options
        await self._ensure_min_cached_items(pair_key, 40)
        items = self.item_cache.get_random_items(pair_key, 16)
        if len(items) < 4:
            await update.message.reply_text("Not enough data for a quiz yet. Try /learn first.")  # type: ignore[union-attr]
            return
        target_item = random.choice(items)
        distractors = random.sample([i for i in items if i.item_id != target_item.item_id], 3)
        options = [target_item] + distractors
        random.shuffle(options)
        keyboard = InlineKeyboardMarkup(
            [
                [InlineKeyboardButton(text=o.target_text, callback_data=f"quiz:{pair_key}:{target_item.item_id}:{o.item_id}")]
                for o in options
            ]
        )
        question = (
            f"📝 <b>Choose the correct translation</b>\n\n"
            f"<b>{target_item.source_text}</b>"
        )
        await update.message.reply_text(question, parse_mode=ParseMode.HTML, reply_markup=keyboard)  # type: ignore[union-attr]

    async def _on_quiz_next(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        if query is None or query.data is None:
            return
        await query.answer()
        if not query.data.startswith("quiz_next:"):
            return
        # Use the user's current language
        chat_id = query.message.chat_id
        user = self.progress_repo.get_user(chat_id)
        pair_key = user["lang_pair"] if user else "en-es"
        await self._ensure_min_cached_items(pair_key, 40)
        items = self.item_cache.get_random_items(pair_key, 16)
        if len(items) < 4:
            await query.edit_message_text("Not enough data for a quiz yet. Try /learn first.")
            return
        target_item = random.choice(items)
        distractors = random.sample([i for i in items if i.item_id != target_item.item_id], 3)
        options = [target_item] + distractors
        random.shuffle(options)
        keyboard = InlineKeyboardMarkup(
            [
                [InlineKeyboardButton(text=o.target_text, callback_data=f"quiz:{pair_key}:{target_item.item_id}:{o.item_id}")]
                for o in options
            ]
        )
        question = (
            f"📝 <b>Choose the correct translation</b>\n\n"
            f"<b>{target_item.source_text}</b>"
        )
        await query.edit_message_text(question, parse_mode=ParseMode.HTML, reply_markup=keyboard)

    async def _on_quiz_answer(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        if query is None or query.data is None:
            return
        await query.answer()
        if not query.data.startswith("quiz:"):
            return
        _, pair_key, correct_id_str, selected_id_str = query.data.split(":", 3)
        correct_id = int(correct_id_str)
        selected_id = int(selected_id_str)
        chat_id = query.message.chat_id

        # Correct if selected item has same id as correct item
        is_correct = selected_id == correct_id
        quality = 5 if is_correct else 2
        self.progress_repo.upsert_review_result(chat_id, pair_key, correct_id, quality)

        correct_item = self.item_cache.get_item_by_id(pair_key, correct_id)
        selected_item = self.item_cache.get_item_by_id(pair_key, selected_id)
        if correct_item is None:
            await query.edit_message_text("Question expired.")
            return
        if is_correct:
            text = f"✅ Correct! <b>{correct_item.source_text}</b> → <i>{correct_item.target_text}</i>"
        else:
            sel_txt = selected_item.target_text if selected_item else "(unknown)"
            text = (
                "❌ Not quite.\n"
                f"You chose: <i>{sel_txt}</i>\n"
                f"Answer: <b>{correct_item.source_text}</b> → <i>{correct_item.target_text}</i>"
            )
        # Show result and offer Next question button
        keyboard = InlineKeyboardMarkup(
            [[InlineKeyboardButton("➡️ Next", callback_data=f"quiz_next:{pair_key}")]]
        )
        await query.edit_message_text(text, parse_mode=ParseMode.HTML, reply_markup=keyboard)

    # -------- Stats
    async def _cmd_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        assert update.effective_chat is not None
        chat_id = update.effective_chat.id
        user = self.progress_repo.get_user(chat_id)
        pair_key = user["lang_pair"] if user else "en-es"
        seen, correct, incorrect = self.progress_repo.get_basic_stats(chat_id, pair_key)
        total = correct + incorrect
        accuracy = (100.0 * correct / total) if total > 0 else 0.0
        await update.message.reply_text(
            (
                f"Language: {SUPPORTED_LANGUAGE_PAIRS[pair_key]['label']}\n"
                f"Cards seen: {seen}\n"
                f"Answered: {total} (✅ {correct} / ❌ {incorrect})\n"
                f"Accuracy: {accuracy:.1f}%"
            )
        )  # type: ignore[union-attr]

    # -------- Daily reminders
    async def _cmd_daily_on(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        assert update.effective_chat is not None
        chat_id = update.effective_chat.id
        self.progress_repo.set_subscription(chat_id, True)
        # Schedule a repeating job every 24h, starting in 10 seconds
        when = datetime.now(timezone.utc) + timedelta(seconds=10)
        context.job_queue.run_repeating(
            self._daily_job,
            interval=timedelta(days=1),
            first=when,
            name=f"daily-{chat_id}",
            chat_id=chat_id,
        )
        await update.message.reply_text("Daily practice is ON. I will send you one card per day.")  # type: ignore[union-attr]

    async def _cmd_daily_off(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        assert update.effective_chat is not None
        chat_id = update.effective_chat.id
        self.progress_repo.set_subscription(chat_id, False)
        # Cancel existing jobs for this chat
        for job in context.job_queue.get_jobs_by_name(f"daily-{chat_id}"):
            job.schedule_removal()
        await update.message.reply_text("Daily practice is OFF.")  # type: ignore[union-attr]

    async def _daily_job(self, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id = context.job.chat_id  # type: ignore[assignment]
        user = self.progress_repo.get_user(chat_id)
        pair_key = user["lang_pair"] if user else "en-es"
        item = await self._choose_next_item(chat_id, pair_key)
        if item is None:
            await context.bot.send_message(chat_id=chat_id, text="No data available yet.")
            return
        keyboard = InlineKeyboardMarkup(
            [[InlineKeyboardButton("👀 Reveal", callback_data=f"reveal:{pair_key}:{item.item_id}")]]
        )
        image = self._render_image_card(item.target_text)
        if image is not None:
            await context.bot.send_photo(
                chat_id=chat_id,
                photo=image,
                caption=self._format_flashcard_text(item, reveal=False),
                parse_mode=ParseMode.HTML,
                reply_markup=keyboard,
            )
        else:
            await context.bot.send_message(
                chat_id=chat_id,
                text=self._format_flashcard_text(item, reveal=False),
                parse_mode=ParseMode.HTML,
                reply_markup=keyboard,
            )

    def _rehydrate_daily_jobs(self, app: Application) -> None:
        # Restore daily jobs for subscribed users at startup
        for user in self.progress_repo.get_all_subscribed_users():
            chat_id = int(user["chat_id"])
            # Avoid duplicates if app reloads
            name = f"daily-{chat_id}"
            if app.job_queue.get_jobs_by_name(name):
                continue
            when = datetime.now(timezone.utc) + timedelta(seconds=15)
            app.job_queue.run_repeating(
                self._daily_job,
                interval=timedelta(days=1),
                first=when,
                name=name,
                chat_id=chat_id,
            )

    async def _on_post_init(self, app: Application) -> None:
        self.http_client = httpx.AsyncClient(timeout=10)
        # Improve usability: set command list
        try:
            await app.bot.set_my_commands([
                ("start", "Welcome & quick setup"),
                ("setlang", "Choose target language"),
                ("learn", "Flashcard practice"),
                ("quiz", "Multiple-choice quiz"),
                ("stats", "Progress overview"),
                ("daily_on", "Enable daily reminder"),
                ("daily_off", "Disable daily reminder"),
                ("help", "How to use the bot"),
            ])
        except Exception:
            pass
        # Background prefetch to keep cache filled
        app.job_queue.run_repeating(
            self._prefetch_job,
            interval=timedelta(seconds=45),
            first=timedelta(seconds=5),
            name="prefetch",
            job_kwargs={"max_instances": 1, "coalesce": True, "misfire_grace_time": 10},
        )

    async def _on_post_shutdown(self, app: Application) -> None:
        if self.http_client:
            await self.http_client.aclose()

    async def _prefetch_step(self, pair_key: str, n: int) -> None:
        added = 0
        for _ in range(n):
            item = await _fetch_and_cache_item_impl(self, pair_key)
            if item is not None:
                added += 1
        # No return value; best-effort

    async def _prefetch_job(self, context: ContextTypes.DEFAULT_TYPE) -> None:
        for pair_key in SUPPORTED_LANGUAGE_PAIRS.keys():
            cur = self.item_cache.count_items(pair_key)
            if cur < MIN_CACHE_TARGET:
                await self._prefetch_step(pair_key, min(PREFETCH_STEP, MIN_CACHE_TARGET - cur))

    def run(self) -> None:
        builder = (
            ApplicationBuilder()
            .token(self.token)
            .rate_limiter(AIORateLimiter())
            .post_init(self._on_post_init)
            .post_shutdown(self._on_post_shutdown)
        )
        app = builder.build()

        # Commands
        app.add_handler(CommandHandler("start", self._cmd_start))
        app.add_handler(CommandHandler("setlang", self._cmd_setlang))
        app.add_handler(CommandHandler("learn", self._cmd_learn))
        app.add_handler(CommandHandler("quiz", self._cmd_quiz))
        app.add_handler(CommandHandler("stats", self._cmd_stats))
        app.add_handler(CommandHandler("daily_on", self._cmd_daily_on))
        app.add_handler(CommandHandler("daily_off", self._cmd_daily_off))
        app.add_handler(CommandHandler("help", self._cmd_start))

        # Callbacks
        app.add_handler(CallbackQueryHandler(self._on_setlang, pattern=r"^setlang:"))
        app.add_handler(CallbackQueryHandler(self._on_reveal_or_rate, pattern=r"^(reveal|rate):"))
        app.add_handler(CallbackQueryHandler(self._on_quiz_answer, pattern=r"^quiz:"))
        app.add_handler(CallbackQueryHandler(self._on_quiz_next, pattern=r"^quiz_next:"))

        self._rehydrate_daily_jobs(app)
        self.app = app

        port_env = os.getenv("PORT")
        if port_env:
            # Koyeb/HTTP mode: start health server and polling concurrently
            async def _run_all() -> None:
                await app.initialize()
                await app.start()
                await app.updater.start_polling(drop_pending_updates=True)

                # Health server
                health_app = web.Application()

                async def ok_handler(_: web.Request) -> web.Response:
                    return web.Response(text="OK")

                health_app.add_routes(
                    [web.get("/", ok_handler), web.get("/health", ok_handler)]
                )
                runner = web.AppRunner(health_app)
                await runner.setup()
                site = web.TCPSite(runner, host="0.0.0.0", port=int(port_env))
                await site.start()

                print(f"Bot + health server running on port {port_env}. Press Ctrl+C to stop.")
                # Sleep forever
                await asyncio.Event().wait()

            asyncio.run(_run_all())
        else:
            print("Bot is running. Press Ctrl+C to stop.")
            app.run_polling(drop_pending_updates=True)


def main() -> None:
    # Prefer environment variable; optionally load from .env if present
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        # Support .env without requiring python-dotenv
        env_path = Path(".env")
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                if line.strip().startswith("TELEGRAM_BOT_TOKEN="):
                    token = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
    if not token:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN is not set. Set it in your environment or in a .env file."
        )

    bot = LanguageLearningBot(token)
    bot.run()


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, SystemExit):
        print("Exiting...")

# -----------------------------
# Remote vocabulary utilities
# -----------------------------

async def _fetch_random_word(client: httpx.AsyncClient) -> Optional[str]:
    # Random Word API (no key). Returns a list of words
    url = "https://random-word-api.herokuapp.com/word?number=1"
    try:
        resp = await client.get(url)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list) and data:
            word = str(data[0]).strip()
            if word.isalpha() and 3 <= len(word) <= 12:
                return word.lower()
    except Exception:
        return None
    return None


async def _fetch_dictionary_info(client: httpx.AsyncClient, word: str) -> Tuple[str, str]:
    # Returns (display_word, example_sentence)
    url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{quote_plus(word)}"
    try:
        resp = await client.get(url)
        if resp.status_code != 200:
            # Not found, fallback to simple example
            return word, f"I saw a {word} yesterday."
        data = resp.json()
        if isinstance(data, list) and data:
            entry = data[0]
            display_word = str(entry.get("word", word))
            meanings = entry.get("meanings", [])
            for m in meanings:
                defs = m.get("definitions", [])
                if defs:
                    ex = defs[0].get("example")
                    if ex:
                        return display_word, str(ex)
            # No example; craft one
            return display_word, f"This is a {word}."
    except Exception:
        pass
    return word, f"This is a {word}."


async def _translate_text(client: httpx.AsyncClient, text: str, src: str, tgt: str) -> str:
    # MyMemory Translation API (free, rate-limited). No key required.
    url = f"https://api.mymemory.translated.net/get?q={quote_plus(text)}&langpair={src}|{tgt}"
    try:
        resp = await client.get(url)
        resp.raise_for_status()
        data = resp.json()
        translated = data.get("responseData", {}).get("translatedText")
        if translated:
            return str(translated)
    except Exception:
        pass
    return text


async def _build_remote_vocab_item(client: httpx.AsyncClient, src_lang: str, tgt_lang: str) -> Optional[VocabItem]:
    # Try a few attempts to get a reasonable word
    for _ in range(5):
        word = await _fetch_random_word(client)
        if not word:
            continue
        display_word, example_en = await _fetch_dictionary_info(client, word)
        target_word = await _translate_text(client, display_word, src_lang, tgt_lang)
        example_tgt = await _translate_text(client, example_en, src_lang, tgt_lang)
        return VocabItem(
            item_id=-1,
            source_text=display_word,
            target_text=target_word,
            source_example=example_en,
            target_example=example_tgt,
        )
    return None


async def _ensure_http_client(bot: "LanguageLearningBot") -> httpx.AsyncClient:
    if bot.http_client is None:
        bot.http_client = httpx.AsyncClient(timeout=10)
    return bot.http_client


async def _fetch_and_cache_item_impl(bot: "LanguageLearningBot", pair_key: str) -> Optional[VocabItem]:
    client = await _ensure_http_client(bot)
    meta = SUPPORTED_LANGUAGE_PAIRS[pair_key]
    src = meta["src"]
    tgt = meta["tgt"]
    remote_item = await _build_remote_vocab_item(client, src, tgt)
    if remote_item is None:
        return None
    item_id = bot.item_cache.insert_item(
        pair_key,
        remote_item.source_text,
        remote_item.target_text,
        remote_item.source_example,
        remote_item.target_example,
    )
    return VocabItem(
        item_id=item_id,
        source_text=remote_item.source_text,
        target_text=remote_item.target_text,
        source_example=remote_item.source_example,
        target_example=remote_item.target_example,
    )


async def _ensure_min_cached_items_impl(bot: "LanguageLearningBot", pair_key: str, min_count: int) -> None:
    current = bot.item_cache.count_items(pair_key)
    attempts = 0
    while current < min_count and attempts < min_count * 3:
        item = await _fetch_and_cache_item_impl(bot, pair_key)
        if item is not None:
            current += 1
        attempts += 1


# Bind helper methods to class with access to self
async def _fetch_and_cache_item(self: "LanguageLearningBot", pair_key: str) -> Optional[VocabItem]:
    return await _fetch_and_cache_item_impl(self, pair_key)


async def _ensure_min_cached_items(self: "LanguageLearningBot", pair_key: str, min_count: int) -> None:
    await _ensure_min_cached_items_impl(self, pair_key, min_count)


# Attach methods to the class dynamically
setattr(LanguageLearningBot, "_fetch_and_cache_item", _fetch_and_cache_item)
setattr(LanguageLearningBot, "_ensure_min_cached_items", _ensure_min_cached_items)


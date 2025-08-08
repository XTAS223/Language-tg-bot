## Language Learning Telegram Bot

An educational Telegram bot to practice vocabulary with flashcards and quizzes. Built with `python-telegram-bot` and a simple SM‑2–inspired spaced repetition system backed by SQLite.

### Features
- Flashcards: reveal translation, mark known/unknown
- Multiple-choice quizzes
- Spaced repetition scheduling
- Daily reminder messages
- Basic progress stats
- Pulls vocabulary from free public APIs and caches locally
  - Random Word API for candidate words
  - Free Dictionary API for example sentences
  - MyMemory Translation API for translations

### Setup (Windows PowerShell)
1. Install Python 3.10+ from the Microsoft Store or `python.org`.
2. Navigate to your project folder:
   ```powershell
   cd "C:\Users\super\OneDrive\Documents\bots"
   ```
3. Create and activate a virtual environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
4. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
5. Set your Telegram bot token (create a bot via @BotFather first):
   ```powershell
   setx TELEGRAM_BOT_TOKEN "<your_token_here>"
   # Restart PowerShell (so the new env var is loaded)
   ```
   Alternatively, create a `.env` file in the project root with:
   ```env
   TELEGRAM_BOT_TOKEN=<your_token_here>
   ```

### Run the Bot
```powershell
python bot.py
```

### Deploy free on Koyeb (web + health)
Koyeb requires an HTTP health check. The bot supports this automatically when the `PORT` env var is set: it will run polling AND serve `GET /health` returning `OK`.

1. Push this repo to GitHub.
2. In Koyeb, create a new App from your repo.
3. Build & Run:
   - Runtime: `python:3.11` (or 3.10+)
   - Start command: `python bot.py`
4. Environment variables:
   - `TELEGRAM_BOT_TOKEN`: your token
   - `PORT`: `8080` (or any port Koyeb expects)
5. Health check:
   - Path: `/health`
   - Port: `PORT`
6. Deploy.

Notes:
- This uses long polling (no public webhook needed). The health endpoint is provided by an embedded aiohttp server.
- Ensure your plan allows outbound internet to reach Telegram + APIs.

### Commands
- `/start` – Welcome and quick setup
- `/setlang` – Choose the language pair
- `/learn` – Flashcard card (reveal + mark known/unknown)
- `/quiz` – Multiple-choice question
- `/stats` – Your progress overview
- `/daily_on` – Turn on daily reminder
- `/daily_off` – Turn off daily reminder

### Data source
The bot fetches words and examples on demand from public APIs, then caches them in SQLite. No API keys needed, but these services are rate‑limited and sometimes return imperfect results.

If you prefer, you can still add offline datasets later by extending the item cache or adding a CSV import command.

### Notes
- Uses long polling. For production, consider webhooks and a process manager.
- The database file `learning_bot.db` will be created in the project root.
- Back up your DB if you care about progress persistence.



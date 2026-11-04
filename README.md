# DeepLearn (Python version)

An AI study site: a hint-first **Doubt Solver**, **PYQ-style practice** (AI-generated, not real past papers), and a **faculty view** that shows topic counts only. Built with Python (Flask), plus HTML, CSS and JavaScript for the page.

## Features
- **Doubt Solver**: hint, then bigger hint. The server only gives the full solution after the student writes an attempt. Then the student explains it back and gets feedback and a "why" follow-up.
- **PYQ practice**: topic, difficulty and 3/5/8 questions, each with marks, a hint and a reveal-able answer.
- **Faculty view**: most-asked topics, counts only. Needs the faculty key.
- **Free AI**: Google Gemini (`gemini-2.5-flash-lite`) with a free key from https://aistudio.google.com/app/apikey (no card). Free-tier requests may be used by Google to improve its products, so students should not type private information.

## Run locally
1. Install Python 3.10+ and run `pip install -r requirements.txt`.
2. Set your keys (Mac/Linux): `export GEMINI_API_KEY=your-key FACULTY_KEY=anypassword`
   (Windows PowerShell: `$env:GEMINI_API_KEY="your-key"; $env:FACULTY_KEY="anypassword"`)
3. `python app.py`, then open http://localhost:3000

## Deploy on Render
Use **Language: Python 3**, **Build Command** `pip install -r requirements.txt`, **Start Command** `gunicorn app:app --workers 1 --threads 4 --timeout 90`, and add `GEMINI_API_KEY` and `FACULTY_KEY` under Environment Variables. Or push to GitHub and use **New + -> Blueprint** to read `render.yaml`.

## If the AI says it is unavailable
Open the site, go to **For faculty**, type your faculty key and click **Test AI connection**. It shows the real reason (rejected key, quota, missing model) without revealing the key. The app also tries fallback models (`gemini-2.5-flash-lite`, `gemini-2.5-flash`, `gemini-2.0-flash`) and retries with the key as a URL parameter if the header is rejected.

## Notes
- Keep `--workers 1`, because faculty counts live in memory and would split across workers.
- Render's free plan sleeps when idle; the first visit takes about a minute.
- Faculty counts reset on restart.
- Not built yet: group quiz mode, student logins.

import hmac
import json
import os
import re
import threading
import time

import requests
from flask import Flask, jsonify, request, send_from_directory

app = Flask(__name__, static_folder="public", static_url_path="")

FACULTY_KEY = os.environ.get("FACULTY_KEY", "").strip()
LOCK = threading.Lock()
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent"
USED = {"model": None, "auth": None}  # last combination that worked (shown by the faculty health check)


def api_key():
    """Read the key fresh and strip stray spaces, quotes and line breaks that break auth."""
    return os.environ.get("GEMINI_API_KEY", "").strip().strip("\"'").strip()


def model_chain():
    """GEMINI_MODEL first, then fallbacks. Each model has its own free quota."""
    names = [os.environ.get("GEMINI_MODEL", "").strip(), "gemini-3.8-flash"]
    return list(dict.fromkeys(n for n in names if n))


def redact(text):
    k = api_key()
    return str(text).replace(k, "[KEY]") if k else str(text)


class AIError(Exception):
    def __init__(self, kind, message=""):
        super().__init__(message or kind)
        self.kind = kind


def clean(value, limit):
    return str(value if value is not None else "").strip()[:limit]


def bad(message, status=400):
    return jsonify(error=message), status


# ---- per-IP rate limit (protects the free quota) ----
HITS = {}


@app.before_request
def rate_limit():
    if not request.path.startswith("/api"):
        return None
    ip = request.headers.get("X-Forwarded-For", request.remote_addr or "").split(",")[0].strip()
    now = time.time()
    with LOCK:
        recent = [t for t in HITS.get(ip, []) if now - t < 600]
        if len(recent) >= 40:
            HITS[ip] = recent
            return bad("Too many requests. Wait a few minutes and try again.", 429)
        recent.append(now)
        HITS[ip] = recent
    return None


# ---- faculty stats: topic counters only, chat text is never stored ----
STATS = {}


def track(topic, kind=None):
    key = clean(topic, 60).lower() or "general"
    with LOCK:
        s = STATS.setdefault(key, {"total": 0, "bigger": 0, "answers": 0, "weak": 0, "practice": 0})
        s["total"] += 1
        if kind:
            s[kind] += 1


# ---- Gemini (free tier) ----
def _post(model, payload, key, via):
    if via == "header":
        return requests.post(GEMINI_URL.format(model), headers={"Content-Type": "application/json", "x-goog-api-key": key},
                             json=payload, timeout=45)
    return requests.post(GEMINI_URL.format(model), params={"key": key}, headers={"Content-Type": "application/json"},
                         json=payload, timeout=45)


def _key_problem(r):
    return r.status_code in (401, 403) or (r.status_code == 400 and ("API key" in r.text or "API_KEY" in r.text))


def gemini(system, user, json_mode=False, max_tokens=600):
    key = api_key()
    if not key:
        raise AIError("nokey")
    last, quota_only, key_bad = "", True, False
    for model in model_chain():
        config = {"maxOutputTokens": max_tokens, "temperature": 0.7}
        if json_mode:
            config["responseMimeType"] = "application/json"
        if re.match(r"gemini-2\.5-flash", model):
            config["thinkingConfig"] = {"thinkingBudget": 0}  # thinking tokens can eat the output budget
        payload = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": config,
        }
        via = "header"
        try:
            r = _post(model, payload, key, via)
            if _key_problem(r):  # some key types only work as a URL parameter, so try that once
                via, r = "query", _post(model, payload, key, "query")
        except requests.RequestException as e:
            quota_only = False
            last = f"{model}: network error: {e}"
            app.logger.error("AI error: %s", redact(last))
            continue

        if r.ok:
            parts = (r.json().get("candidates") or [{}])[0].get("content", {}).get("parts", [])
            text = "".join(p.get("text", "") for p in parts).strip()
            if text:
                USED.update(model=model, auth=via)
                return text
            quota_only = False
            last = f"{model}: empty reply"
        else:
            last = f"{model} HTTP {r.status_code}: {redact(r.text[:300])}"
            if _key_problem(r):
                key_bad = True
                app.logger.error("AI error: %s", last)
                break  # every model will fail the same way
            if r.status_code != 429:
                quota_only = False
        app.logger.error("AI error: %s", last)  # then try the next model
    if key_bad:
        raise AIError("badkey", last)
    if quota_only and last:
        raise AIError("quota", last)
    raise AIError("fail", last or "no models")


def parse_json(text):
    return json.loads(re.sub(r"```json|```", "", text).strip())


def safe_ai(fn):
    def wrapper():
        try:
            return fn()
        except AIError as e:
            app.logger.error("AI error: %s %s", e.kind, e)
            if e.kind == "quota":
                return bad("The free AI limit is used up for now. Try again in a minute.", 429)
            if e.kind == "badkey":
                return bad("The AI key was rejected. The site owner must check GEMINI_API_KEY.", 503)
            if e.kind == "nokey":
                return bad("The AI key is not set up yet. The site owner must add GEMINI_API_KEY.", 503)
            return bad("The AI service is unavailable right now. Try again in a moment.", 502)
        except requests.RequestException as e:
            app.logger.error("Network error: %s", e)
            return bad("The AI service is unavailable right now. Try again in a moment.", 502)
        except Exception as e:  # unexpected problem, e.g. the AI returned bad JSON
            app.logger.error("Error: %s", e)
            return bad("The AI service is unavailable right now. Try again in a moment.", 502)
    wrapper.__name__ = fn.__name__
    return wrapper


# ---- Doubt Solver: hint-first rules enforced on the server ----
BASE = "You are DeepLearn, a friendly tutor for university students. Use simple, clear language. Keep replies under 130 words."
MODES = {
    1: "Give ONE guiding question or small hint. Do NOT give the answer or the final solution.",
    2: "Give a stronger hint that narrows the path (a key idea or the first step). Still do NOT state the final answer.",
    3: "The student has made an attempt. React to it kindly, then explain the concept step by step and give the full solution. End by asking them to restate it in their own words.",
}


@app.post("/api/chat")
@safe_ai
def chat():
    body = request.get_json(silent=True) or {}
    try:
        level = int(body.get("level") or 1)
    except (TypeError, ValueError):
        level = 0
    question = clean(body.get("question"), 1500)
    topic = clean(body.get("topic"), 60)
    attempt = clean(body.get("attempt"), 1500)
    if level not in (1, 2, 3) or len(question) < 5:
        return bad("Type your doubt first.")
    if level == 3 and len(attempt) < 15:
        return bad("Write what you have tried so far, then the full solution unlocks.")
    prompt = f"Topic: {topic or 'general'}\nDoubt: {question}"
    if attempt:
        prompt += f"\nMy attempt: {attempt}"
    reply = gemini(f"{BASE} {MODES[level]}", prompt, max_tokens=700)
    track(topic, "bigger" if level == 2 else "answers" if level == 3 else None)
    return jsonify(reply=reply, level=level)


@app.post("/api/explain-back")
@safe_ai
def explain_back():
    body = request.get_json(silent=True) or {}
    concept = clean(body.get("concept"), 300)
    explanation = clean(body.get("explanation"), 2000)
    if len(explanation) < 20:
        return bad("Explain it in at least a couple of sentences.")
    text = gemini(
        'You check a student\'s own-words explanation. Reply ONLY with JSON: {"verdict":"solid"|"partial"|"needs_work",'
        '"feedback":"specific, encouraging, simple words, under 80 words","followUp":"one \'why\' question that deepens their understanding"}.',
        f"Concept: {concept}\nStudent explanation: {explanation}",
        json_mode=True,
        max_tokens=500,
    )
    try:
        out = parse_json(text)
        if not isinstance(out, dict):
            raise ValueError("not an object")
    except ValueError:
        out = {"verdict": "partial", "feedback": text, "followUp": ""}
    track(concept, None if out.get("verdict") == "solid" else "weak")
    return jsonify(out)


# ---- PYQ-style practice (AI-generated, NOT real past papers) ----
@app.post("/api/pyq")
@safe_ai
def pyq():
    body = request.get_json(silent=True) or {}
    topic = clean(body.get("topic"), 80)
    difficulty = body.get("difficulty") if body.get("difficulty") in ("easy", "medium", "hard") else "medium"
    try:
        n = int(body.get("count"))
    except (TypeError, ValueError):
        n = 0
    if len(topic) < 2:
        return bad("Enter a topic first.")
    if n not in (3, 5, 8):
        return bad("Choose 3, 5 or 8 questions.")
    text = gemini(
        f"You write exam-style practice questions for university students. Reply ONLY with a JSON array of exactly {n} objects: "
        '{"question":string,"marks":number (2 to 10),"hint":string (one short nudge, no answer),'
        '"answer":string (clear model answer in simple words, under 100 words)}. Do not claim these are real past papers.',
        f"Topic: {topic}\nDifficulty: {difficulty}",
        json_mode=True,
        max_tokens=3500,
    )
    items = parse_json(text)
    if not isinstance(items, list) or not items:
        raise AIError("fail", "No questions")
    questions = []
    for q in items[:n]:
        try:
            marks = int(q.get("marks"))
        except (TypeError, ValueError):
            marks = 5
        questions.append({
            "question": clean(q.get("question"), 800),
            "marks": marks,
            "hint": clean(q.get("hint"), 300),
            "answer": clean(q.get("answer"), 1200),
        })
    track(topic, "practice")
    return jsonify(questions=questions)


def faculty_ok():
    given = request.headers.get("x-faculty-key", "")
    return bool(FACULTY_KEY) and hmac.compare_digest(given.encode(), FACULTY_KEY.encode())


@app.get("/api/health")
def health():
    """Faculty-only: tests Gemini and says exactly what is wrong. Never reveals the key."""
    if not faculty_ok():
        return bad("Wrong or missing faculty key.", 403)
    key = api_key()
    out = {"key_set": bool(key), "key_length": len(key), "models": model_chain()}
    try:
        gemini("Reply with one word.", "ping", max_tokens=20)
        out.update(ok=True, model_used=USED["model"], auth_used=USED["auth"])
        return jsonify(out)
    except AIError as e:
        out.update(ok=False, problem=e.kind, detail=redact(e))
        return jsonify(out), 502
    except requests.RequestException as e:
        out.update(ok=False, problem="network", detail=redact(e))
        return jsonify(out), 502


@app.get("/api/faculty")
def faculty():
    if not faculty_ok():
        return bad("Wrong or missing faculty key.", 403)
    with LOCK:
        rows = []
        for topic, s in STATS.items():
            struggle = round(100 * (s["bigger"] + s["weak"]) / max(s["total"], 1))
            rows.append({"topic": topic, **s, "struggle": struggle})
    rows.sort(key=lambda r: (-r["total"], -r["struggle"]))
    return jsonify(topics=rows)


@app.get("/healthz")
def healthz():
    return "ok"


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 3000)))

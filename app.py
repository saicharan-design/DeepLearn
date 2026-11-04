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
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# The last successful provider/model is shown in the faculty health check.
USED = {"provider": None, "model": None}

# ---- API keys and provider configuration ----
def env_key(name):
    return os.environ.get(name, "").strip().strip("\"'").strip()


def api_keys():
    return {
        "gemini": env_key("GEMINI_API_KEY"),
        "groq": env_key("GROQ_API_KEY"),
        "openrouter": env_key("OPENROUTER_API_KEY"),
    }


def redact(value):
    text = str(value)
    for key in api_keys().values():
        if key:
            text = text.replace(key, "[KEY]")
    return text


def provider_order():
    allowed = {"gemini", "groq", "openrouter"}
    configured = os.environ.get("AI_PROVIDER_ORDER", "gemini,groq,openrouter")
    requested = [p.strip().lower() for p in configured.split(",")]
    ordered = [p for p in requested if p in allowed]
    return list(dict.fromkeys(ordered or ["gemini", "groq", "openrouter"]))


def provider_models():
    return {
        "gemini": os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-lite").strip(),
        "groq": os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile").strip(),
        # OpenRouter's free router chooses an available free chat model.
        "openrouter": os.environ.get("OPENROUTER_MODEL", "openrouter/free").strip(),
    }


def configured_providers():
    keys = api_keys()
    models = provider_models()
    return [
        {"provider": name, "model": models[name]}
        for name in provider_order()
        if keys.get(name) and models.get(name)
    ]


class AIError(Exception):
    def __init__(self, kind, message=""):
        super().__init__(message or kind)
        self.kind = kind


def clean(value, limit):
    return str(value if value is not None else "").strip()[:limit]


def bad(message, status=400):
    return jsonify(error=message), status


# ---- Per-IP rate limit (protects free-tier quotas) ----
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


# ---- Faculty stats: topic counters only; chat text is never stored ----
STATS = {}


def track(topic, kind=None):
    key = clean(topic, 60).lower() or "general"
    with LOCK:
        s = STATS.setdefault(key, {"total": 0, "bigger": 0, "answers": 0, "weak": 0, "practice": 0})
        s["total"] += 1
        if kind:
            s[kind] += 1


# ---- Provider request adapters ----
def _gemini_request(model, key, system, user, json_mode, max_tokens):
    config = {"maxOutputTokens": max_tokens, "temperature": 0.7}
    if json_mode:
        config["responseMimeType"] = "application/json"
    payload = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": config,
    }
    return requests.post(
        GEMINI_URL.format(model),
        headers={"Content-Type": "application/json", "x-goog-api-key": key},
        json=payload,
        timeout=(10, 35),
    )


def _openai_compatible_request(provider, model, key, system, user, json_mode, max_tokens):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.7,
        "max_tokens": max_tokens,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    if provider == "groq":
        url = GROQ_URL
    else:
        url = OPENROUTER_URL
        headers["HTTP-Referer"] = os.environ.get("APP_PUBLIC_URL", "https://deeplearn.onrender.com")
        headers["X-Title"] = "DeepLearn"

    return requests.post(url, headers=headers, json=payload, timeout=(10, 35))


def _response_text(provider, response):
    data = response.json()
    if provider == "gemini":
        parts = (data.get("candidates") or [{}])[0].get("content", {}).get("parts", [])
        return "".join(part.get("text", "") for part in parts).strip()

    choices = data.get("choices") or []
    if not choices:
        return ""
    content = (choices[0].get("message") or {}).get("content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "".join(
            item.get("text", "") for item in content if isinstance(item, dict)
        ).strip()
    return ""


def generate(system, user, json_mode=False, max_tokens=600):
    keys = api_keys()
    models = provider_models()
    providers = [
        name for name in provider_order()
        if keys.get(name) and models.get(name)
    ]
    if not providers:
        raise AIError("nokey", "No AI provider keys are configured.")

    errors = []
    statuses = []
    for provider in providers:
        model = models[provider]
        try:
            if provider == "gemini":
                response = _gemini_request(
                    model, keys[provider], system, user, json_mode, max_tokens
                )
            else:
                response = _openai_compatible_request(
                    provider, model, keys[provider], system, user, json_mode, max_tokens
                )
        except requests.RequestException as exc:
            message = redact(f"{provider}/{model}: network error: {exc}")
            errors.append(message)
            app.logger.warning("AI provider failed: %s", message)
            continue

        statuses.append(response.status_code)
        if response.ok:
            try:
                answer = _response_text(provider, response)
            except (ValueError, KeyError, TypeError) as exc:
                answer = ""
                errors.append(redact(f"{provider}/{model}: invalid response: {exc}"))
            if answer:
                USED.update(provider=provider, model=model)
                return answer
            errors.append(f"{provider}/{model}: empty response")
            continue

        # Do not stop on one provider's quota, outage, invalid model or bad key.
        # Continue to the next configured provider.
        detail = redact(response.text[:350])
        message = f"{provider}/{model} HTTP {response.status_code}: {detail}"
        errors.append(message)
        app.logger.warning("AI provider failed: %s", message)

    if statuses and all(code == 429 for code in statuses):
        raise AIError("quota", "All configured providers are currently rate-limited.")
    raise AIError("unavailable", " | ".join(errors[-3:]) or "No provider returned an answer.")


def parse_json(text):
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    return json.loads(cleaned)


def safe_ai(fn):
    def wrapper():
        try:
            return fn()
        except AIError as exc:
            app.logger.error("AI error: %s %s", exc.kind, redact(exc))
            if exc.kind == "quota":
                return bad("All configured free AI providers are rate-limited right now. Please try again later.", 429)
            if exc.kind == "nokey":
                return bad("No AI provider is configured. The site owner must add at least one provider API key in Render.", 503)
            return bad("The AI providers could not answer right now. Please try again in a little while.", 502)
        except requests.RequestException as exc:
            app.logger.error("Network error: %s", redact(exc))
            return bad("The AI service is temporarily unreachable. Please try again.", 502)
        except Exception as exc:
            app.logger.exception("Unexpected AI route error: %s", redact(exc))
            return bad("The AI service could not complete that request. Please try again.", 502)
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
    reply = generate(f"{BASE} {MODES[level]}", prompt, max_tokens=700)
    track(topic, "bigger" if level == 2 else "answers" if level == 3 else None)
    return jsonify(reply=reply, provider=USED["provider"], model=USED["model"], level=level)


@app.post("/api/explain-back")
@safe_ai
def explain_back():
    body = request.get_json(silent=True) or {}
    concept = clean(body.get("concept"), 300)
    explanation = clean(body.get("explanation"), 2000)
    if len(explanation) < 20:
        return bad("Explain it in at least a couple of sentences.")
    text = generate(
        'Check a student\'s own-words explanation. Reply ONLY with JSON: {"verdict":"solid"|"partial"|"needs_work",'
        '"feedback":"specific, encouraging, simple words, under 80 words","followUp":"one \'why\' question that deepens their understanding"}.',
        f"Concept: {concept}\nStudent explanation: {explanation}",
        json_mode=True,
        max_tokens=500,
    )
    try:
        out = parse_json(text)
        if not isinstance(out, dict):
            raise ValueError("not an object")
    except (ValueError, TypeError):
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
        count = int(body.get("count"))
    except (TypeError, ValueError):
        count = 0
    if len(topic) < 2:
        return bad("Enter a topic first.")
    if count not in (3, 5, 8):
        return bad("Choose 3, 5 or 8 questions.")
    text = generate(
        f"You write exam-style practice questions for university students. Reply ONLY with a JSON array of exactly {count} objects: "
        '{"question":string,"marks":number (2 to 10),"hint":string (one short nudge, no answer),'
        '"answer":string (clear model answer in simple words, under 100 words)}. Do not claim these are real past papers.',
        f"Topic: {topic}\nDifficulty: {difficulty}",
        json_mode=True,
        max_tokens=3500,
    )
    items = parse_json(text)
    if not isinstance(items, list) or not items:
        raise AIError("unavailable", "The AI returned invalid practice-question JSON.")
    questions = []
    for item in items[:count]:
        if not isinstance(item, dict):
            continue
        try:
            marks = int(item.get("marks"))
        except (TypeError, ValueError):
            marks = 5
        questions.append({
            "question": clean(item.get("question"), 800),
            "marks": marks,
            "hint": clean(item.get("hint"), 300),
            "answer": clean(item.get("answer"), 1200),
        })
    if not questions:
        raise AIError("unavailable", "The AI returned no valid practice questions.")
    track(topic, "practice")
    return jsonify(questions=questions)


def faculty_ok():
    given = request.headers.get("x-faculty-key", "")
    return bool(FACULTY_KEY) and hmac.compare_digest(given.encode(), FACULTY_KEY.encode())


@app.get("/api/health")
def health():
    """Faculty-only: tests the configured provider chain without revealing API keys."""
    if not faculty_ok():
        return bad("Wrong or missing faculty key.", 403)

    keys = api_keys()
    models = provider_models()
    configured = configured_providers()
    out = {
        "providers_configured": configured,
        "key_status": {name: bool(keys[name]) for name in keys},
    }
    try:
        generate("Reply with one word.", "ping", max_tokens=20)
        out.update(
            ok=True,
            provider_used=USED["provider"],
            model_used=USED["model"],
        )
        return jsonify(out)
    except AIError as exc:
        out.update(ok=False, problem=exc.kind, detail=redact(exc))
        return jsonify(out), 502


@app.get("/api/faculty")
def faculty():
    if not faculty_ok():
        return bad("Wrong or missing faculty key.", 403)
    with LOCK:
        rows = []
        for topic, stats in STATS.items():
            struggle = round(100 * (stats["bigger"] + stats["weak"]) / max(stats["total"], 1))
            rows.append({"topic": topic, **stats, "struggle": struggle})
    rows.sort(key=lambda row: (-row["total"], -row["struggle"]))
    return jsonify(topics=rows)


@app.get("/healthz")
def healthz():
    return "ok"


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 3000)))

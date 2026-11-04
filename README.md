# DeepLearn — multi-provider version

DeepLearn keeps its existing Flask backend and frontend features:
- Hint-first Doubt Solver
- Explain-it-back feedback
- AI-generated PYQ-style practice
- Faculty topic counts (topic counters only; student question text is not stored)
- URL for DeepLearn AI -- https://deeplearn-ch5a.onrender.com/

## AI provider fallback

The backend tries configured providers in this order by default:

1. Google Gemini
2. Groq
3. OpenRouter's free-model router

If a provider returns an error, is overloaded, times out, or has reached a rate limit, DeepLearn moves to the next configured provider. The first provider that returns a usable answer is used.

The code uses HTTP APIs directly, so no additional Python package is required.

## Create API keys

Create keys only from the providers' official dashboards:
- Gemini: https://aistudio.google.com/apikey
- Groq: https://console.groq.com/keys
- OpenRouter: https://openrouter.ai/settings/keys

Free tiers have changing model availability, quotas and rate limits. A fallback provider is not guaranteed to be available at every moment. OpenRouter's free-model variants currently have low daily limits; check its current rate-limit documentation before relying on it:
https://openrouter.ai/docs/api-reference/limits

## Render environment variables

In Render Dashboard → your DeepLearn service → Environment, add:

| Variable | Value |
|---|---|
| `GEMINI_API_KEY` | Your Gemini key |
| `GEMINI_MODEL` | `gemini-2.5-flash-lite` (or another model enabled for your key) |
| `GROQ_API_KEY` | Your Groq key |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` (change if Groq retires/renames it) |
| `OPENROUTER_API_KEY` | Your OpenRouter key |
| `OPENROUTER_MODEL` | `openrouter/free` |
| `AI_PROVIDER_ORDER` | `gemini,groq,openrouter` |

Only add the keys for providers you have configured. At least one provider key is required. Keep all keys in Render Environment Variables; never put them in `public/index.html`, commit them to GitHub, or share them in screenshots.

After saving environment variables, redeploy the service.

## Test the provider chain

1. Open DeepLearn → For faculty.
2. Enter your existing faculty key.
3. Click **Test AI connection**.
4. The result shows which provider and model returned the test answer.

The test uses the same fallback chain as the student features. It does not reveal API keys.

## Change provider order

Set `AI_PROVIDER_ORDER` to any comma-separated order, for example:

`groq,gemini,openrouter`

Providers without a configured key are skipped automatically.

## Run locally

Install dependencies:

`pip install -r requirements.txt`

Set environment variables in your shell, then run:

`python app.py`

## Important free-tier and privacy notes

- Free plans have quotas and can be temporarily unavailable. Automatic fallback improves resilience but does not guarantee uninterrupted service.
- Provider free-tier terms and data handling differ. Student prompts are sent to whichever provider answers. Do not enter passwords, personal details, or sensitive information.
- The app does not intentionally store chat prompts in its faculty statistics; only topic counters are kept in memory.
- Render's free service may sleep when idle. The first request after sleeping can take longer.
- Faculty counters reset when the service restarts.
- Keep Gunicorn at one worker because the faculty counters are held in process memory.

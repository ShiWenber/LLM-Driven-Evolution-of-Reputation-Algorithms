# API Key Configuration

This project reads API keys from a `.env` file at the project root.

## Quick setup

```bash
# From the project root:
cp .env.example .env       # macOS / Linux
copy .env.example .env     # Windows

# Edit .env and fill in your real keys
notepad .env               # or whatever editor you prefer
```

Then run the experiment as usual — keys are loaded automatically.

## How it works

`experiments/config/load_env.py` is the single source of truth for API configuration.
It loads `DEEPSEEK_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` etc. from
`.env` (or your shell environment).

The module exposes:

- `get_api_key(provider)`     — key string or ""
- `get_base_url(provider)`    — base URL
- `get_model(provider)`       — model name
- `is_configured(provider)`   — True/False
- `require_api_key(provider)` — same as get but raises if missing

The mutation operator (`experiments/evolution/mutation.py`) and CLI entry point
(`experiments/main.py`) both go through this module.

## Self-test

```bash
python -m experiments.config.load_env
```

Prints which providers are configured (with masked keys) and their base URL / model.

## Security

- `.env` is git-ignored. Never commit it.
- `.env.example` is the safe template — commit it freely.
- For shared machines, prefer shell env vars over `.env`.

## Rotating a leaked key

If a key was exposed (chat log, screenshot, git push, etc.):

1. Go to the provider's console and revoke the old key.
2. Generate a new one.
3. Update `.env` with the new value.
4. Restart any running experiment.

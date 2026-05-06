# RealAI Frontend

A professional Next.js 15 chat interface for a RealAI backend deployment.

## Features

- 💬 **Full chat UI** — streaming-style responses, markdown rendering, code blocks
- 🔄 **Conversation history** — persisted in localStorage, grouped by date
- 🧠 **Model selector** — RealAI 2.0, GPT-4o, Claude 3.5 Sonnet, Mistral Large, and more
- ⚙️ **Settings drawer** — system prompt, temperature slider, max tokens, optional API key
- 📱 **Responsive** — collapsible sidebar, works on mobile
- 🌑 **Dark mode** — slate/indigo design system, zero flash

## Getting Started

```bash
# 1. Install dependencies
npm install

# 2. Configure environment
cp .env.local.example .env.local
# Edit .env.local and set NEXT_PUBLIC_REALAI_API_BASE if needed

# 3. Run locally
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

## Deploy to Vercel

1. Create a **new** Vercel project and point it at the `realai-frontend` subdirectory of this repository.
2. Add the following environment variables in the Vercel dashboard:

| Variable | Value |
|---|---|
| `REALAI_API_BASE` | `https://your-service.onrender.com` |
| `NEXT_PUBLIC_REALAI_API_BASE` | `https://your-service.onrender.com` *(optional fallback)* |
| `REALAI_API_KEY` | *(your backend API key, if required — stored server-side only)* |

3. Deploy. The proxy route at `/api/chat` forwards requests to the Render backend, so API keys never reach the browser.

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `NEXT_PUBLIC_REALAI_API_BASE` | Yes | Base URL of the RealAI backend |
| `REALAI_API_BASE` | Recommended | Server-side base URL of the RealAI backend |
| `REALAI_API_KEY` | No | Server-side API key forwarded to the backend |

## Troubleshooting Vercel + Render

- Set Vercel project Root Directory to `realai-frontend`.
- Do not include `/v1` in backend env URLs. Use `https://your-backend.onrender.com`.
- Confirm Render health endpoint returns 200 at `/health`.
- If Vercel builds fail with Next.js peer dependency errors, make sure lockfile and package versions are in sync.

## Architecture

```
Browser → /api/chat (Next.js proxy) → https://realai-qz3b.onrender.com/v1/chat/completions
```

The proxy avoids CORS issues and keeps the API key server-side (never exposed in browser network traffic).

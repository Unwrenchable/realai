/**
 * Typed API client for the RealAI backend.
 *
 * The base URL is taken from VITE_API_BASE_URL at build time; when running in
 * development the Vite dev server proxies /v1/* to http://localhost:8000.
 */

const BASE_URL = (import.meta as unknown as { env: Record<string, string> }).env
  ?.VITE_API_BASE_URL ?? ''

function getApiKey(): string {
  return localStorage.getItem('realai_api_key') ?? 'realai-demo'
}

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${getApiKey()}`,
      ...(options.headers ?? {}),
    },
  })
  if (!res.ok) {
    const text = await res.text()
    throw new Error(`${res.status} ${res.statusText}: ${text}`)
  }
  return res.json() as Promise<T>
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface ChatMessage {
  role: 'user' | 'assistant' | 'system'
  content: string
}

export interface ChatCompletionResponse {
  id: string
  object: string
  choices: Array<{
    index: number
    message: ChatMessage
    finish_reason: string
  }>
  usage: { prompt_tokens: number; completion_tokens: number; total_tokens: number }
}

export interface CapabilityInfo {
  name: string
  description: string
  endpoint: string | null
}

export interface RealAIResult {
  status: string
  data?: unknown
  error?: string
  note?: string
}

// ---------------------------------------------------------------------------
// Endpoints
// ---------------------------------------------------------------------------

export function chatCompletion(messages: ChatMessage[], model = 'realai-echo-1'): Promise<ChatCompletionResponse> {
  return request('/v1/chat/completions', {
    method: 'POST',
    body: JSON.stringify({ model, messages }),
  })
}

export function listCapabilities(): Promise<{ data: CapabilityInfo[] }> {
  return request('/v1/capabilities', { method: 'GET' })
}

export function generateImage(prompt: string, n = 1, size = '1024x1024'): Promise<RealAIResult> {
  return request('/v1/images/generations', {
    method: 'POST',
    body: JSON.stringify({ prompt, n, size }),
  })
}

export function generateCode(prompt: string, language = 'python'): Promise<RealAIResult> {
  return request('/v1/code/generate', {
    method: 'POST',
    body: JSON.stringify({ prompt, language }),
  })
}

export function executeCode(code: string, language = 'python'): Promise<RealAIResult> {
  return request('/v1/execute', {
    method: 'POST',
    body: JSON.stringify({ code, language }),
  })
}

export function textToSpeech(input: string, voice = 'alloy'): Promise<RealAIResult> {
  return request('/v1/audio/speech', {
    method: 'POST',
    body: JSON.stringify({ input, voice }),
  })
}

export function transcribeAudio(audio_path: string): Promise<RealAIResult> {
  return request('/v1/audio/transcriptions', {
    method: 'POST',
    body: JSON.stringify({ audio_path }),
  })
}

export function translate(text: string, target_language: string): Promise<RealAIResult> {
  return request('/v1/translate', {
    method: 'POST',
    body: JSON.stringify({ text, target_language }),
  })
}

export function webResearch(query: string): Promise<RealAIResult> {
  return request('/v1/research', {
    method: 'POST',
    body: JSON.stringify({ query }),
  })
}

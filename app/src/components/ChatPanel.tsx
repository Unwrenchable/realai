import { useState, useRef, useEffect } from 'react'
import { chatCompletion, ChatMessage } from '../api'

const MODELS = ['realai-echo-1', 'gpt-4o-mini', 'claude-3-5-haiku-20241022', 'gemini-1.5-flash']

export default function ChatPanel() {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState('')
  const [model, setModel] = useState(MODELS[0])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  async function send() {
    const text = input.trim()
    if (!text || loading) return
    setInput('')
    setError(null)

    const userMsg: ChatMessage = { role: 'user', content: text }
    const history = [...messages, userMsg]
    setMessages(history)
    setLoading(true)

    try {
      const res = await chatCompletion(history, model)
      const assistantMsg = res.choices[0].message
      setMessages([...history, assistantMsg])
    } catch (err) {
      setError(String(err))
    } finally {
      setLoading(false)
    }
  }

  function handleKey(e: React.KeyboardEvent) {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send() }
  }

  function clear() { setMessages([]); setError(null) }

  return (
    <div className="panel chat-panel">
      <div className="panel-header">
        <h1>💬 Chat</h1>
        <div className="panel-actions">
          <select className="model-select" value={model} onChange={(e) => setModel(e.target.value)}>
            {MODELS.map((m) => <option key={m} value={m}>{m}</option>)}
          </select>
          <button className="btn-ghost" onClick={clear}>Clear</button>
        </div>
      </div>

      <div className="chat-messages">
        {messages.length === 0 && (
          <div className="chat-empty">
            <p>👋 Start a conversation with RealAI</p>
            <p className="hint">Type a message below and press Enter or click Send.</p>
          </div>
        )}
        {messages.map((msg, i) => (
          <div key={i} className={`chat-bubble ${msg.role}`}>
            <span className="bubble-role">{msg.role === 'user' ? '🧑' : '🤖'}</span>
            <div className="bubble-content">{msg.content}</div>
          </div>
        ))}
        {loading && (
          <div className="chat-bubble assistant loading">
            <span className="bubble-role">🤖</span>
            <div className="bubble-content">Thinking…</div>
          </div>
        )}
        {error && <div className="error-banner">⚠️ {error}</div>}
        <div ref={bottomRef} />
      </div>

      <div className="chat-input-row">
        <textarea
          className="chat-textarea"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKey}
          placeholder="Type a message… (Enter to send, Shift+Enter for newline)"
          rows={3}
          disabled={loading}
        />
        <button className="btn-primary send-btn" onClick={send} disabled={loading || !input.trim()}>
          Send
        </button>
      </div>
    </div>
  )
}

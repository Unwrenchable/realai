import { useState } from 'react'
import { generateImage, RealAIResult } from '../api'

const SIZES = ['256x256', '512x512', '1024x1024', '1792x1024', '1024x1792']

export default function ImagePanel() {
  const [prompt, setPrompt] = useState('')
  const [size, setSize] = useState('1024x1024')
  const [n, setN] = useState(1)
  const [result, setResult] = useState<RealAIResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function generate() {
    const text = prompt.trim()
    if (!text || loading) return
    setError(null)
    setResult(null)
    setLoading(true)
    try {
      const res = await generateImage(text, n, size)
      setResult(res)
    } catch (err) {
      setError(String(err))
    } finally {
      setLoading(false)
    }
  }

  function getImageUrl(): string | null {
    if (!result) return null
    // OpenAI-style response: data[0].url
    const data = result.data
    if (Array.isArray(data) && data.length > 0) {
      const item = data[0] as Record<string, unknown>
      if (typeof item.url === 'string') return item.url
    }
    if (typeof data === 'string' && data.startsWith('http')) return data
    return null
  }

  const imageUrl = getImageUrl()

  return (
    <div className="panel">
      <div className="panel-header">
        <h1>🎨 Image Generation</h1>
      </div>

      <div className="form-group">
        <label className="field-label">Prompt</label>
        <textarea
          className="text-input"
          rows={4}
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="Describe the image you want to generate…"
          disabled={loading}
        />
      </div>

      <div className="form-row">
        <div className="form-group">
          <label className="field-label">Size</label>
          <select className="model-select" value={size} onChange={(e) => setSize(e.target.value)}>
            {SIZES.map((s) => <option key={s} value={s}>{s}</option>)}
          </select>
        </div>
        <div className="form-group">
          <label className="field-label">Count</label>
          <select className="model-select" value={n} onChange={(e) => setN(Number(e.target.value))}>
            {[1, 2, 3, 4].map((v) => <option key={v} value={v}>{v}</option>)}
          </select>
        </div>
      </div>

      <button className="btn-primary" onClick={generate} disabled={loading || !prompt.trim()}>
        {loading ? 'Generating…' : 'Generate Image'}
      </button>

      {error && <div className="error-banner">⚠️ {error}</div>}

      {result && (
        <div className="result-box">
          {imageUrl ? (
            <img src={imageUrl} alt={prompt} className="generated-image" />
          ) : (
            <pre className="result-pre">{JSON.stringify(result, null, 2)}</pre>
          )}
        </div>
      )}
    </div>
  )
}

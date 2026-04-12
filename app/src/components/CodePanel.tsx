import { useState } from 'react'
import { generateCode, executeCode, RealAIResult } from '../api'

const LANGUAGES = ['python', 'javascript', 'typescript', 'bash', 'sql', 'go', 'rust', 'java', 'c', 'cpp']

function extractCode(result: RealAIResult): string {
  if (!result.data) return JSON.stringify(result, null, 2)
  if (typeof result.data === 'string') return result.data
  const d = result.data as Record<string, unknown>
  if (typeof d.code === 'string') return d.code
  return JSON.stringify(result, null, 2)
}

export default function CodePanel() {
  const [prompt, setPrompt] = useState('')
  const [language, setLanguage] = useState('python')
  const [code, setCode] = useState('')
  const [execResult, setExecResult] = useState<RealAIResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [running, setRunning] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function generate() {
    const text = prompt.trim()
    if (!text || loading) return
    setError(null)
    setCode('')
    setExecResult(null)
    setLoading(true)
    try {
      const res = await generateCode(text, language)
      setCode(extractCode(res))
    } catch (err) {
      setError(String(err))
    } finally {
      setLoading(false)
    }
  }

  async function execute() {
    if (!code.trim() || running) return
    setError(null)
    setExecResult(null)
    setRunning(true)
    try {
      const res = await executeCode(code, language)
      setExecResult(res)
    } catch (err) {
      setError(String(err))
    } finally {
      setRunning(false)
    }
  }

  return (
    <div className="panel">
      <div className="panel-header">
        <h1>💻 Code Generation</h1>
        <select className="model-select" value={language} onChange={(e) => setLanguage(e.target.value)}>
          {LANGUAGES.map((l) => <option key={l} value={l}>{l}</option>)}
        </select>
      </div>

      <div className="form-group">
        <label className="field-label">Prompt</label>
        <textarea
          className="text-input"
          rows={3}
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="Describe the code you need…"
          disabled={loading}
        />
      </div>

      <button className="btn-primary" onClick={generate} disabled={loading || !prompt.trim()}>
        {loading ? 'Generating…' : 'Generate Code'}
      </button>

      {error && <div className="error-banner">⚠️ {error}</div>}

      {code && (
        <div className="form-group" style={{ marginTop: '1.5rem' }}>
          <div className="code-header">
            <label className="field-label">Generated Code</label>
            <button className="btn-ghost" onClick={execute} disabled={running}>
              {running ? 'Running…' : '▶ Execute'}
            </button>
          </div>
          <textarea
            className="code-editor"
            value={code}
            onChange={(e) => setCode(e.target.value)}
            rows={14}
            spellCheck={false}
          />
        </div>
      )}

      {execResult && (
        <div className="result-box">
          <label className="field-label">Execution Result</label>
          <pre className="result-pre">{JSON.stringify(execResult, null, 2)}</pre>
        </div>
      )}
    </div>
  )
}

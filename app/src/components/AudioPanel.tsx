import { useState } from 'react'
import { textToSpeech, transcribeAudio, RealAIResult } from '../api'

const VOICES = ['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer']

export default function AudioPanel() {
  // TTS state
  const [ttsText, setTtsText] = useState('')
  const [voice, setVoice] = useState('alloy')
  const [ttsResult, setTtsResult] = useState<RealAIResult | null>(null)
  const [ttsLoading, setTtsLoading] = useState(false)
  const [ttsError, setTtsError] = useState<string | null>(null)

  // Transcription state
  const [audioPath, setAudioPath] = useState('')
  const [transResult, setTransResult] = useState<RealAIResult | null>(null)
  const [transLoading, setTransLoading] = useState(false)
  const [transError, setTransError] = useState<string | null>(null)

  async function synthesise() {
    if (!ttsText.trim() || ttsLoading) return
    setTtsError(null)
    setTtsResult(null)
    setTtsLoading(true)
    try {
      const res = await textToSpeech(ttsText, voice)
      setTtsResult(res)
    } catch (err) {
      setTtsError(String(err))
    } finally {
      setTtsLoading(false)
    }
  }

  async function transcribe() {
    if (!audioPath.trim() || transLoading) return
    setTransError(null)
    setTransResult(null)
    setTransLoading(true)
    try {
      const res = await transcribeAudio(audioPath)
      setTransResult(res)
    } catch (err) {
      setTransError(String(err))
    } finally {
      setTransLoading(false)
    }
  }

  return (
    <div className="panel">
      <div className="panel-header">
        <h1>🔊 Audio</h1>
      </div>

      {/* Text-to-speech */}
      <section className="sub-section">
        <h2>Text-to-Speech</h2>
        <div className="form-group">
          <label className="field-label">Text</label>
          <textarea
            className="text-input"
            rows={4}
            value={ttsText}
            onChange={(e) => setTtsText(e.target.value)}
            placeholder="Enter text to synthesise into speech…"
            disabled={ttsLoading}
          />
        </div>
        <div className="form-group">
          <label className="field-label">Voice</label>
          <select className="model-select" value={voice} onChange={(e) => setVoice(e.target.value)}>
            {VOICES.map((v) => <option key={v} value={v}>{v}</option>)}
          </select>
        </div>
        <button className="btn-primary" onClick={synthesise} disabled={ttsLoading || !ttsText.trim()}>
          {ttsLoading ? 'Synthesising…' : 'Synthesise Speech'}
        </button>
        {ttsError && <div className="error-banner">⚠️ {ttsError}</div>}
        {ttsResult && (
          <div className="result-box">
            <pre className="result-pre">{JSON.stringify(ttsResult, null, 2)}</pre>
          </div>
        )}
      </section>

      {/* Transcription */}
      <section className="sub-section">
        <h2>Audio Transcription</h2>
        <div className="form-group">
          <label className="field-label">Audio File Path (server-side)</label>
          <input
            className="text-input"
            type="text"
            value={audioPath}
            onChange={(e) => setAudioPath(e.target.value)}
            placeholder="/path/to/audio.mp3"
            disabled={transLoading}
          />
        </div>
        <button className="btn-primary" onClick={transcribe} disabled={transLoading || !audioPath.trim()}>
          {transLoading ? 'Transcribing…' : 'Transcribe Audio'}
        </button>
        {transError && <div className="error-banner">⚠️ {transError}</div>}
        {transResult && (
          <div className="result-box">
            <pre className="result-pre">{JSON.stringify(transResult, null, 2)}</pre>
          </div>
        )}
      </section>
    </div>
  )
}

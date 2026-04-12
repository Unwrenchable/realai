import { useState, useEffect } from 'react'
import { listCapabilities, CapabilityInfo } from '../api'

export default function CapabilitiesPanel() {
  const [capabilities, setCapabilities] = useState<CapabilityInfo[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    listCapabilities()
      .then((res) => setCapabilities(res.data))
      .catch((err) => setError(String(err)))
      .finally(() => setLoading(false))
  }, [])

  return (
    <div className="panel">
      <div className="panel-header">
        <h1>⚡ Capabilities</h1>
      </div>
      <p className="hint" style={{ marginBottom: '1.5rem' }}>
        RealAI provides {capabilities.length || '17+'} built-in capabilities through a unified OpenAI-compatible API.
      </p>

      {loading && <p>Loading capabilities…</p>}
      {error && <div className="error-banner">⚠️ {error}</div>}

      <div className="capabilities-grid">
        {capabilities.map((cap) => (
          <div key={cap.name} className="capability-card">
            <div className="cap-name">{cap.name}</div>
            <div className="cap-desc">{cap.description}</div>
            {cap.endpoint && (
              <code className="cap-endpoint">{cap.endpoint}</code>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}

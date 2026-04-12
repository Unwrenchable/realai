import { useState, ReactNode } from 'react'
import type { Panel } from '../App'

interface NavItem {
  id: Panel
  icon: string
  label: string
}

const NAV_ITEMS: NavItem[] = [
  { id: 'chat',         icon: '💬', label: 'Chat' },
  { id: 'image',        icon: '🎨', label: 'Images' },
  { id: 'code',         icon: '💻', label: 'Code' },
  { id: 'audio',        icon: '🔊', label: 'Audio' },
  { id: 'capabilities', icon: '⚡', label: 'Capabilities' },
]

interface LayoutProps {
  activePanel: Panel
  onNavigate: (panel: Panel) => void
  children: ReactNode
}

export default function Layout({ activePanel, onNavigate, children }: LayoutProps) {
  const [showSettings, setShowSettings] = useState(false)
  const [apiKey, setApiKey] = useState(() => localStorage.getItem('realai_api_key') ?? 'realai-demo')
  const [draftKey, setDraftKey] = useState(apiKey)

  function saveKey() {
    localStorage.setItem('realai_api_key', draftKey)
    setApiKey(draftKey)
    setShowSettings(false)
  }

  return (
    <div className="app-shell">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="sidebar-brand">
          <span className="brand-icon">🚀</span>
          <span className="brand-name">RealAI</span>
        </div>

        <nav className="sidebar-nav">
          {NAV_ITEMS.map((item) => (
            <button
              key={item.id}
              className={`nav-item${activePanel === item.id ? ' active' : ''}`}
              onClick={() => onNavigate(item.id)}
            >
              <span className="nav-icon">{item.icon}</span>
              <span className="nav-label">{item.label}</span>
            </button>
          ))}
        </nav>

        <div className="sidebar-footer">
          <button className="settings-btn" onClick={() => { setDraftKey(apiKey); setShowSettings(true) }}>
            ⚙️ Settings
          </button>
        </div>
      </aside>

      {/* Main content */}
      <main className="main-content">{children}</main>

      {/* Settings modal */}
      {showSettings && (
        <div className="modal-overlay" onClick={() => setShowSettings(false)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <h2>Settings</h2>
            <label className="field-label">API Key</label>
            <input
              className="text-input"
              type="password"
              value={draftKey}
              onChange={(e) => setDraftKey(e.target.value)}
              placeholder="realai-demo"
            />
            <p className="hint">
              Default keys: <code>realai-demo</code> / <code>realai-dev</code>.
              Set <code>REALAI_API_KEYS</code> on the server to restrict access.
            </p>
            <div className="modal-actions">
              <button className="btn-secondary" onClick={() => setShowSettings(false)}>Cancel</button>
              <button className="btn-primary" onClick={saveKey}>Save</button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

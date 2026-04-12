import { useState } from 'react'
import Layout from './components/Layout'
import ChatPanel from './components/ChatPanel'
import ImagePanel from './components/ImagePanel'
import CodePanel from './components/CodePanel'
import AudioPanel from './components/AudioPanel'
import CapabilitiesPanel from './components/CapabilitiesPanel'

export type Panel = 'chat' | 'image' | 'code' | 'audio' | 'capabilities'

export default function App() {
  const [activePanel, setActivePanel] = useState<Panel>('chat')

  function renderPanel() {
    switch (activePanel) {
      case 'chat':         return <ChatPanel />
      case 'image':        return <ImagePanel />
      case 'code':         return <CodePanel />
      case 'audio':        return <AudioPanel />
      case 'capabilities': return <CapabilitiesPanel />
      default:             return <ChatPanel />
    }
  }

  return (
    <Layout activePanel={activePanel} onNavigate={setActivePanel}>
      {renderPanel()}
    </Layout>
  )
}

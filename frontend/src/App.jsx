import { useEffect, useRef, useState } from 'react'

const API_URL = 'http://localhost:8000'

export default function App() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  async function sendMessage() {
    if (!input.trim() || loading) return

    const newMessages = [...messages, { role: 'user', content: input.trim() }]
    setMessages(newMessages)
    setInput('')
    setLoading(true)

    try {
      const res = await fetch(`${API_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ messages: newMessages }),
      })
      const data = await res.json()
      setMessages([...newMessages, { role: 'assistant', content: data.reply }])
    } catch (err) {
      setMessages([...newMessages, { role: 'assistant', content: `Error: ${err.message}` }])
    } finally {
      setLoading(false)
    }
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  return (
    <div style={styles.container}>
      <h2 style={styles.title}>Simple Agent Chat</h2>

      <div style={styles.messageList}>
        {messages.length === 0 && (
          <p style={styles.empty}>No messages yet. Start chatting!</p>
        )}
        {messages.map((m, i) => (
          <div key={i} style={m.role === 'user' ? styles.userMsg : styles.assistantMsg}>
            <strong>{m.role === 'user' ? 'You' : 'Assistant'}</strong>
            <p style={styles.msgContent}>{m.content}</p>
          </div>
        ))}
        {loading && (
          <div style={styles.assistantMsg}>
            <strong>Assistant</strong>
            <p style={{ ...styles.msgContent, color: '#888' }}>Thinking…</p>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      <div style={styles.inputRow}>
        <textarea
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Type a message… (Enter to send, Shift+Enter for newline)"
          style={styles.textarea}
          rows={3}
          disabled={loading}
        />
        <button
          onClick={sendMessage}
          disabled={loading || !input.trim()}
          style={styles.button}
        >
          Send
        </button>
      </div>
    </div>
  )
}

const styles = {
  container: {
    maxWidth: 700,
    margin: '40px auto',
    fontFamily: 'monospace',
    padding: '0 16px',
  },
  title: {
    marginBottom: 16,
  },
  messageList: {
    border: '1px solid #ccc',
    borderRadius: 4,
    height: 450,
    overflowY: 'auto',
    padding: '12px 16px',
    marginBottom: 12,
    background: '#fafafa',
  },
  empty: {
    color: '#888',
    textAlign: 'center',
    marginTop: 180,
  },
  userMsg: {
    marginBottom: 16,
    textAlign: 'right',
  },
  assistantMsg: {
    marginBottom: 16,
    textAlign: 'left',
  },
  msgContent: {
    margin: '4px 0 0',
    whiteSpace: 'pre-wrap',
    wordBreak: 'break-word',
  },
  inputRow: {
    display: 'flex',
    gap: 8,
  },
  textarea: {
    flex: 1,
    padding: 8,
    fontFamily: 'monospace',
    fontSize: 14,
    resize: 'vertical',
    border: '1px solid #ccc',
    borderRadius: 4,
  },
  button: {
    padding: '0 20px',
    fontFamily: 'monospace',
    fontSize: 14,
    cursor: 'pointer',
    border: '1px solid #ccc',
    borderRadius: 4,
    background: '#fff',
  },
}

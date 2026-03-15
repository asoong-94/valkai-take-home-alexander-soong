import { useEffect, useRef, useState } from 'react'

const API_URL = 'http://localhost:8000'

const PROFILE_FIELDS = [
  { key: 'name', label: 'Name' },
  { key: 'role', label: 'Role' },
  { key: 'company', label: 'Company' },
  { key: 'location', label: 'Location' },
  { key: 'response_style', label: 'Response Style' },
  { key: 'projects', label: 'Projects' },
  { key: 'preferred_language', label: 'Preferred Language' },
]

const NEW_USER_SENTINEL = '__new__'

export default function App() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)

  const [strategies, setStrategies] = useState([])
  const [strategy, setStrategy] = useState('baseline')
  const [userId, setUserId] = useState('default-user')
  const [knownUsers, setKnownUsers] = useState([])
  const [creatingUser, setCreatingUser] = useState(false)
  const [newUserInput, setNewUserInput] = useState('')
  const [threadId, setThreadId] = useState(null)

  const [memoryType, setMemoryType] = useState(null)
  const [semanticItems, setSemanticItems] = useState([])
  const [structuredProfile, setStructuredProfile] = useState({})

  const bottomRef = useRef(null)
  const memoryBottomRef = useRef(null)
  const textareaRef = useRef(null)
  const newUserRef = useRef(null)

  useEffect(() => {
    fetch(`${API_URL}/strategies`)
      .then(r => r.json())
      .then(d => setStrategies(d.strategies))
      .catch(() => setStrategies(['baseline', 'semantic', 'structured']))
    fetchUsers()
  }, [])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  useEffect(() => {
    memoryBottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [semanticItems])

  useEffect(() => {
    if (creatingUser && newUserRef.current) {
      newUserRef.current.focus()
    }
  }, [creatingUser])

  async function fetchUsers() {
    try {
      const res = await fetch(`${API_URL}/users`)
      const data = await res.json()
      setKnownUsers(data.users || [])
    } catch {
      setKnownUsers([])
    }
  }

  function resetSession() {
    setMessages([])
    setThreadId(null)
    setSemanticItems([])
    setStructuredProfile({})
    setMemoryType(null)
  }

  function handleStrategyChange(e) {
    setStrategy(e.target.value)
    resetSession()
  }

  function handleUserSelect(e) {
    const val = e.target.value
    if (val === NEW_USER_SENTINEL) {
      setCreatingUser(true)
      setNewUserInput('')
      return
    }
    setUserId(val)
    setCreatingUser(false)
    resetSession()
  }

  function confirmNewUser() {
    const trimmed = newUserInput.trim()
    if (!trimmed) return
    setUserId(trimmed)
    if (!knownUsers.includes(trimmed)) {
      setKnownUsers(prev => [...prev, trimmed].sort())
    }
    setCreatingUser(false)
    resetSession()
  }

  function cancelNewUser() {
    setCreatingUser(false)
  }

  async function fetchInspect() {
    try {
      const res = await fetch(`${API_URL}/inspect?memory=${strategy}&user_id=${encodeURIComponent(userId)}`)
      const data = await res.json()
      setMemoryType(data.type || strategy)

      if (data.type === 'hybrid') {
        const s = data.stored || {}
        setStructuredProfile(s.profile || {})
        setSemanticItems(s.facts || [])
      } else if (data.type === 'semantic') {
        setSemanticItems(data.stored || [])
        setStructuredProfile({})
      } else if (data.type === 'structured') {
        setStructuredProfile(data.stored || {})
        setSemanticItems([])
      } else {
        setSemanticItems(data.stored || [])
        setStructuredProfile({})
      }
    } catch (err) {
      setSemanticItems([`Error: ${err.message}`])
    }
  }

  async function sendMessage() {
    if (!input.trim() || loading) return

    const userMsg = input.trim()
    const newMessages = [...messages, { role: 'user', content: userMsg }]
    setMessages(newMessages)
    setInput('')
    setLoading(true)

    try {
      const res = await fetch(`${API_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: userMsg,
          memory: strategy,
          user_id: userId,
          thread_id: threadId,
        }),
      })
      const data = await res.json()
      setThreadId(data.thread_id)
      setMessages([...newMessages, { role: 'assistant', content: data.reply }])
    } catch (err) {
      setMessages([...newMessages, { role: 'assistant', content: `Error: ${err.message}` }])
    } finally {
      setLoading(false)
      fetchInspect()
      fetchUsers()
      setTimeout(() => textareaRef.current?.focus(), 50)
    }
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  function renderProfileTable() {
    return (
      <div style={styles.profileTable}>
        {PROFILE_FIELDS.map(f => {
          const val = structuredProfile[f.key]
          const isEmpty = !val || (Array.isArray(val) && val.length === 0)
          const display = Array.isArray(val) ? val.join(', ') : val
          return (
            <div key={f.key} style={styles.profileRow}>
              <div style={styles.profileLabel}>{f.label}</div>
              <div style={isEmpty ? styles.profileValueEmpty : styles.profileValue}>
                {isEmpty ? '—' : display}
              </div>
            </div>
          )
        })}
      </div>
    )
  }

  function renderFactsList() {
    if (semanticItems.length === 0) return null
    return (
      <>
        {semanticItems.map((item, i) => (
          <div key={i} style={styles.memoryItem}>{item}</div>
        ))}
        <div ref={memoryBottomRef} />
      </>
    )
  }

  function renderMemoryPanel() {
    if (strategy === 'baseline') {
      return <div style={styles.memoryEmpty}>Baseline stores nothing cross-session</div>
    }

    if (strategy === 'hybrid' || memoryType === 'hybrid') {
      const hasProfile = PROFILE_FIELDS.some(f => {
        const v = structuredProfile[f.key]
        return v && (!Array.isArray(v) || v.length > 0)
      })
      const hasFacts = semanticItems.length > 0
      if (!hasProfile && !hasFacts) {
        return <div style={styles.memoryEmpty}>No memories yet</div>
      }
      return (
        <>
          <div style={styles.sectionLabel}>Profile</div>
          {renderProfileTable()}
          <div style={{ ...styles.sectionLabel, marginTop: 12 }}>Recalled Facts</div>
          {hasFacts ? renderFactsList() : <div style={styles.memoryEmpty}>No facts yet</div>}
        </>
      )
    }

    if (strategy === 'structured' || memoryType === 'structured') {
      const hasAny = PROFILE_FIELDS.some(f => {
        const v = structuredProfile[f.key]
        return v && (!Array.isArray(v) || v.length > 0)
      })
      return (
        <>
          {renderProfileTable()}
          {!hasAny && <div style={styles.memoryEmpty}>No profile data yet</div>}
        </>
      )
    }

    if (semanticItems.length === 0) {
      return <div style={styles.memoryEmpty}>No memories yet</div>
    }
    return renderFactsList()
  }

  function memoryCount() {
    if (strategy === 'hybrid' || memoryType === 'hybrid') {
      const profileCount = PROFILE_FIELDS.filter(f => {
        const v = structuredProfile[f.key]
        return v && (!Array.isArray(v) || v.length > 0)
      }).length
      return profileCount + semanticItems.length
    }
    if (strategy === 'structured' || memoryType === 'structured') {
      return PROFILE_FIELDS.filter(f => {
        const v = structuredProfile[f.key]
        return v && (!Array.isArray(v) || v.length > 0)
      }).length
    }
    return semanticItems.length
  }

  return (
    <div style={styles.outer}>
      <h2 style={styles.title}>Memory Strategy Chat</h2>

      <div style={styles.controls}>
        <label style={styles.label}>
          Strategy
          <select value={strategy} onChange={handleStrategyChange} style={styles.select}>
            {strategies.map(s => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>
        </label>

        <label style={styles.label}>
          User ID
          {creatingUser ? (
            <div style={styles.newUserRow}>
              <input
                ref={newUserRef}
                value={newUserInput}
                onChange={e => setNewUserInput(e.target.value)}
                onKeyDown={e => {
                  if (e.key === 'Enter') confirmNewUser()
                  if (e.key === 'Escape') cancelNewUser()
                }}
                placeholder="new-user-id"
                style={styles.ctrlInput}
                spellCheck={false}
              />
              <button onClick={confirmNewUser} style={styles.smallBtn}>✓</button>
              <button onClick={cancelNewUser} style={styles.smallBtn}>✗</button>
            </div>
          ) : (
            <select value={userId} onChange={handleUserSelect} style={styles.select}>
              {!knownUsers.includes(userId) && (
                <option value={userId}>{userId}</option>
              )}
              {knownUsers.map(u => (
                <option key={u} value={u}>{u}</option>
              ))}
              <option value={NEW_USER_SENTINEL}>+ new user</option>
            </select>
          )}
        </label>

        <button onClick={resetSession} style={styles.controlBtn}>New Session</button>
        <button onClick={fetchInspect} style={styles.controlBtn}>Refresh Memory</button>
      </div>

      {threadId && (
        <div style={styles.meta}>
          Thread: {threadId}
        </div>
      )}

      <div style={styles.mainRow}>
        <div style={styles.memoryPanel}>
          <div style={styles.memoryHeader}>
            <strong>Stored Memory</strong>
            <span style={styles.memoryBadge}>{memoryType || strategy}</span>
          </div>
          <div style={styles.memoryList}>
            {renderMemoryPanel()}
          </div>
          <div style={styles.memoryFooter}>
            {memoryCount()} {(strategy === 'structured' || strategy === 'hybrid') ? 'field' : 'item'}{memoryCount() !== 1 ? 's' : ''} stored
          </div>
        </div>

        <div style={styles.chatCol}>
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
              ref={textareaRef}
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
              style={styles.sendBtn}
            >
              Send
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

const styles = {
  outer: {
    maxWidth: 1060,
    margin: '24px auto',
    fontFamily: 'monospace',
    padding: '0 16px',
  },
  title: {
    marginBottom: 10,
    fontSize: 18,
  },
  controls: {
    display: 'flex',
    gap: 12,
    alignItems: 'flex-end',
    flexWrap: 'wrap',
    padding: '10px 12px',
    marginBottom: 8,
    border: '1px solid #ddd',
    borderRadius: 4,
    background: '#f8f8f8',
  },
  label: {
    display: 'flex',
    flexDirection: 'column',
    fontSize: 11,
    color: '#666',
    gap: 3,
  },
  select: {
    fontFamily: 'monospace',
    fontSize: 13,
    padding: '4px 6px',
    border: '1px solid #ccc',
    borderRadius: 3,
  },
  ctrlInput: {
    fontFamily: 'monospace',
    fontSize: 13,
    padding: '4px 6px',
    border: '1px solid #ccc',
    borderRadius: 3,
    width: 120,
  },
  newUserRow: {
    display: 'flex',
    gap: 4,
    alignItems: 'center',
  },
  smallBtn: {
    fontFamily: 'monospace',
    fontSize: 13,
    padding: '3px 6px',
    cursor: 'pointer',
    border: '1px solid #ccc',
    borderRadius: 3,
    background: '#fff',
    lineHeight: 1,
  },
  controlBtn: {
    fontFamily: 'monospace',
    fontSize: 12,
    padding: '5px 10px',
    cursor: 'pointer',
    border: '1px solid #ccc',
    borderRadius: 3,
    background: '#fff',
  },
  meta: {
    fontSize: 11,
    color: '#999',
    marginBottom: 6,
    paddingLeft: 2,
  },
  mainRow: {
    display: 'flex',
    gap: 12,
    alignItems: 'flex-start',
  },
  memoryPanel: {
    width: 280,
    flexShrink: 0,
    border: '1px solid #ddd',
    borderRadius: 4,
    background: '#f9f9f9',
    display: 'flex',
    flexDirection: 'column',
  },
  memoryHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: '8px 10px',
    borderBottom: '1px solid #ddd',
    fontSize: 12,
  },
  memoryBadge: {
    fontSize: 10,
    padding: '2px 6px',
    borderRadius: 3,
    background: '#e8e8e8',
    color: '#555',
  },
  memoryList: {
    height: 400,
    overflowY: 'auto',
    padding: '6px 0',
  },
  memoryEmpty: {
    color: '#aaa',
    fontSize: 11,
    textAlign: 'center',
    padding: '40px 12px',
  },
  sectionLabel: {
    fontSize: 10,
    fontWeight: 700,
    textTransform: 'uppercase',
    letterSpacing: '0.5px',
    color: '#888',
    padding: '6px 10px 2px',
    borderBottom: '1px solid #eee',
  },
  memoryItem: {
    padding: '5px 10px',
    fontSize: 11,
    borderBottom: '1px solid #eee',
    lineHeight: 1.4,
    wordBreak: 'break-word',
  },
  memoryFooter: {
    padding: '5px 10px',
    fontSize: 10,
    color: '#999',
    borderTop: '1px solid #ddd',
    textAlign: 'right',
  },
  profileTable: {
    padding: '4px 0',
  },
  profileRow: {
    display: 'flex',
    borderBottom: '1px solid #eee',
    padding: '6px 10px',
    gap: 8,
  },
  profileLabel: {
    width: 100,
    flexShrink: 0,
    fontSize: 10,
    color: '#888',
    textTransform: 'uppercase',
    letterSpacing: '0.5px',
    paddingTop: 1,
  },
  profileValue: {
    flex: 1,
    fontSize: 12,
    wordBreak: 'break-word',
    lineHeight: 1.4,
  },
  profileValueEmpty: {
    flex: 1,
    fontSize: 12,
    color: '#ccc',
  },
  chatCol: {
    flex: 1,
    minWidth: 0,
  },
  messageList: {
    border: '1px solid #ccc',
    borderRadius: 4,
    height: 400,
    overflowY: 'auto',
    padding: '12px 16px',
    marginBottom: 10,
    background: '#fafafa',
  },
  empty: {
    color: '#888',
    textAlign: 'center',
    marginTop: 160,
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
  sendBtn: {
    padding: '0 20px',
    fontFamily: 'monospace',
    fontSize: 14,
    cursor: 'pointer',
    border: '1px solid #ccc',
    borderRadius: 4,
    background: '#fff',
  },
}

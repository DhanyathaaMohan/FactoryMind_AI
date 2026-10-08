import React, { useState } from 'react'
import { postAsk } from '../services/api.js'
import SourceCard from './SourceCard.jsx'

export default function RagAssistant({ predictedRul }) {
  const [query, setQuery]     = useState('')
  const [result, setResult]   = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError]     = useState(null)

  const handleAsk = async (e) => {
    e.preventDefault()
    if (!query.trim()) return
    setLoading(true)
    setError(null)
    try {
      const data = await postAsk(query.trim(), predictedRul ?? null)
      setResult(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="card">
      <div className="label">Maintenance Assistant (RAG)</div>
      <p style={{ fontSize: 12, color: 'var(--muted)', marginTop: 4, marginBottom: '0.8rem' }}>
        Ask a maintenance question. The assistant answers using retrieved Haas
        manual content only — it will not invent facts.
        {predictedRul != null && (
          <span> Current predicted RUL: <strong style={{ color: 'var(--text)' }}>
            {(predictedRul * 100).toFixed(1)}%
          </strong> is included in the prompt context.</span>
        )}
      </p>

      <form onSubmit={handleAsk} style={{ display: 'flex', gap: '0.5rem', marginBottom: '0.9rem' }}>
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="e.g. What should I check when spindle vibration increases?"
          style={{
            flex: 1,
            padding: '0.5rem 0.8rem',
            background: 'var(--bg)',
            border: '1px solid var(--border)',
            borderRadius: 'var(--radius)',
            color: 'var(--text)',
            fontSize: 14,
          }}
        />
        <button type="submit" className="btn btn-primary" disabled={loading}>
          {loading ? <span className="spinner" /> : 'Ask'}
        </button>
      </form>

      {error && <p className="error-msg">{error}</p>}

      {result && (
        <>
          {/* Maintenance answer */}
          <div style={{
            background: 'var(--bg)',
            border: '1px solid var(--border)',
            borderRadius: 'var(--radius)',
            padding: '0.9rem',
            marginBottom: '0.9rem',
          }}>
            <div className="label" style={{ marginBottom: '0.4rem' }}>Answer</div>
            <p style={{ fontSize: 13, lineHeight: 1.7, whiteSpace: 'pre-wrap' }}>
              {result.maintenance_answer}
            </p>
          </div>

          {/* Retrieved evidence */}
          {result.retrieved_sources?.length > 0 && (
            <>
              <div className="label" style={{ marginBottom: '0.4rem' }}>Retrieved Evidence</div>
              {result.retrieved_sources.map((s, i) => (
                <SourceCard
                  key={i}
                  rank={i + 1}
                  source={s.source}
                  page={s.page}
                  hybridScore={s.hybrid_score}
                  preview={s.preview}
                />
              ))}
            </>
          )}
        </>
      )}
    </div>
  )
}

import React, { useState } from 'react'
import { postSearch } from '../services/api.js'
import SourceCard from './SourceCard.jsx'

export default function MaintenanceSearch() {
  const [query, setQuery]     = useState('')
  const [results, setResults] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError]     = useState(null)

  const handleSearch = async (e) => {
    e.preventDefault()
    if (!query.trim()) return
    setLoading(true)
    setError(null)
    try {
      const data = await postSearch(query.trim(), 3)
      setResults(data.results)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="card">
      <div className="label">Knowledge Base Search</div>
      <p style={{ fontSize: 12, color: 'var(--muted)', marginTop: 4, marginBottom: '0.8rem' }}>
        Search the Haas operator and service manuals using hybrid BM25 + semantic retrieval.
      </p>

      <form onSubmit={handleSearch} style={{ display: 'flex', gap: '0.5rem', marginBottom: '0.9rem' }}>
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="e.g. excessive spindle vibration"
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
          {loading ? <span className="spinner" /> : 'Search'}
        </button>
      </form>

      {error && <p className="error-msg">{error}</p>}

      {results && results.length === 0 && (
        <p style={{ color: 'var(--muted)', fontSize: 13 }}>No results found.</p>
      )}

      {results && results.map((r, i) => (
        <SourceCard
          key={i}
          rank={i + 1}
          source={r.source}
          page={r.page}
          hybridScore={r.hybrid_score}
          preview={r.preview}
        />
      ))}
    </div>
  )
}

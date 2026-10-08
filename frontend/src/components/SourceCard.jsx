import React, { useState } from 'react'

export default function SourceCard({ source, page, hybridScore, preview, rank }) {
  const [expanded, setExpanded] = useState(false)

  return (
    <div className="card" style={{ marginBottom: '0.6rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
          <span style={{
            fontSize: 10,
            fontWeight: 700,
            color: 'var(--accent)',
            marginRight: 6,
          }}>
            [{rank}]
          </span>
          <span style={{ fontSize: 13, fontWeight: 600 }}>{source}</span>
        </div>
        <div style={{ display: 'flex', gap: 8, flexShrink: 0 }}>
          <span style={{ fontSize: 11, color: 'var(--muted)' }}>p.{page}</span>
          <span style={{ fontSize: 11, color: 'var(--accent)' }}>
            score: {hybridScore?.toFixed(4)}
          </span>
        </div>
      </div>

      <div style={{
        marginTop: '0.5rem',
        fontSize: 12,
        color: 'var(--muted)',
        lineHeight: 1.55,
        whiteSpace: 'pre-wrap',
        wordBreak: 'break-word',
      }}>
        {expanded ? preview : preview?.slice(0, 160) + (preview?.length > 160 ? '…' : '')}
      </div>

      {preview?.length > 160 && (
        <button
          onClick={() => setExpanded(!expanded)}
          style={{
            background: 'none',
            border: 'none',
            color: 'var(--accent)',
            fontSize: 11,
            marginTop: '0.3rem',
            padding: 0,
            cursor: 'pointer',
          }}
        >
          {expanded ? 'Show less' : 'Show more'}
        </button>
      )}
    </div>
  )
}

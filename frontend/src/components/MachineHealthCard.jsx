import React from 'react'

const conditionBadgeClass = (condition) => {
  switch (condition?.toLowerCase()) {
    case 'healthy':   return 'badge badge-healthy'
    case 'degrading': return 'badge badge-degrading'
    case 'worn':      return 'badge badge-worn'
    case 'critical':  return 'badge badge-critical'
    default:          return 'badge'
  }
}

const conditionColor = (condition) => {
  switch (condition?.toLowerCase()) {
    case 'healthy':   return 'var(--healthy)'
    case 'degrading': return 'var(--degrading)'
    case 'worn':      return 'var(--worn)'
    case 'critical':  return 'var(--critical)'
    default:          return 'var(--muted)'
  }
}

export default function MachineHealthCard({ predictedRul, toolCondition }) {
  const rulPercent = predictedRul != null ? Math.round(predictedRul * 100) : null

  return (
    <div className="card" style={{ minWidth: 200 }}>
      <div className="label">Machine Health</div>

      {predictedRul == null ? (
        <p style={{ color: 'var(--muted)', marginTop: '0.5rem' }}>
          No prediction yet
        </p>
      ) : (
        <>
          <div style={{
            fontSize: 42,
            fontWeight: 800,
            color: conditionColor(toolCondition),
            lineHeight: 1.1,
            marginTop: '0.5rem',
          }}>
            {rulPercent}%
          </div>
          <div style={{ fontSize: 12, color: 'var(--muted)', marginBottom: '0.6rem' }}>
            Remaining Useful Life
          </div>
          <span className={conditionBadgeClass(toolCondition)}>
            {toolCondition}
          </span>
        </>
      )}
    </div>
  )
}

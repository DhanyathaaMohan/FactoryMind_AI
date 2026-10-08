import React from 'react'

export default function ShapFeatureList({ features }) {
  if (!features || features.length === 0) {
    return (
      <div className="card">
        <div className="label">SHAP Feature Influence</div>
        <p style={{ color: 'var(--muted)', marginTop: '0.5rem' }}>
          No explanation data available.
        </p>
      </div>
    )
  }

  // Compute max |shap| for bar scaling
  const maxAbs = Math.max(...features.map((f) => Math.abs(f.shap_value)), 0.0001)

  return (
    <div className="card">
      <div className="label">SHAP Feature Influence</div>
      <p style={{ fontSize: 12, color: 'var(--muted)', marginBottom: '0.8rem', marginTop: 4 }}>
        Features that most influenced this RUL prediction. SHAP values
        reflect model contribution, not physical causality.
      </p>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.55rem' }}>
        {features.map((f, i) => {
          const isPos   = f.shap_value >= 0
          const barPct  = (Math.abs(f.shap_value) / maxAbs) * 100
          const color   = isPos ? 'var(--healthy)' : 'var(--critical)'
          const shortName = f.feature.length > 42
            ? f.feature.slice(0, 42) + '…'
            : f.feature

          return (
            <div key={i}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 3 }}>
                <span style={{ fontSize: 12, color: 'var(--text)' }} title={f.feature}>
                  {shortName}
                </span>
                <span style={{ fontSize: 11, color, marginLeft: 8, whiteSpace: 'nowrap' }}>
                  {f.shap_value >= 0 ? '+' : ''}{f.shap_value.toFixed(4)}
                </span>
              </div>
              <div style={{
                height: 5,
                background: 'var(--border)',
                borderRadius: 3,
                overflow: 'hidden',
              }}>
                <div style={{
                  width: `${barPct}%`,
                  height: '100%',
                  background: color,
                  borderRadius: 3,
                }} />
              </div>
              <div style={{ fontSize: 10, color: 'var(--muted)', marginTop: 2 }}>
                value: {Number(f.feature_value).toExponential(3)} — {f.direction}
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

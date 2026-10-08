import React from 'react'

/**
 * SVG arc gauge showing normalised RUL from 0 – 100%.
 * Kept deliberately simple — no animation libraries needed.
 */
export default function RULGauge({ value }) {
  // value is 0–1 float
  const pct   = value != null ? Math.max(0, Math.min(1, value)) : 0
  const angle = pct * 180                        // 0–180 degrees (half circle)
  const r     = 60
  const cx    = 80
  const cy    = 80
  const strokeW = 14

  // Arc helper: returns SVG path for an arc from startDeg to endDeg on circle (cx,cy,r)
  const arcPath = (startDeg, endDeg) => {
    const toRad = (d) => ((d - 90) * Math.PI) / 180
    const x1 = cx + r * Math.cos(toRad(startDeg))
    const y1 = cy + r * Math.sin(toRad(startDeg))
    const x2 = cx + r * Math.cos(toRad(endDeg))
    const y2 = cy + r * Math.sin(toRad(endDeg))
    const large = endDeg - startDeg > 180 ? 1 : 0
    return `M ${x1} ${y1} A ${r} ${r} 0 ${large} 1 ${x2} ${y2}`
  }

  const color =
    pct >= 0.75 ? 'var(--healthy)'
    : pct >= 0.50 ? 'var(--degrading)'
    : pct >= 0.25 ? 'var(--worn)'
    : 'var(--critical)'

  // Gauge spans -90° (left) to +90° (right) — a 180° half-arc
  // We map pct=0 -> -90deg (left end), pct=1 -> +90deg (right end)
  const startAngle = -90
  const fillEnd    = startAngle + angle * 2   // 0→0, 1→180

  return (
    <div className="card" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
      <div className="label">RUL Gauge</div>
      <svg width={160} height={100} style={{ marginTop: 8 }}>
        {/* Track */}
        <path
          d={arcPath(startAngle, startAngle + 180)}
          fill="none"
          stroke="var(--border)"
          strokeWidth={strokeW}
          strokeLinecap="round"
        />
        {/* Fill */}
        {value != null && pct > 0 && (
          <path
            d={arcPath(startAngle, fillEnd)}
            fill="none"
            stroke={color}
            strokeWidth={strokeW}
            strokeLinecap="round"
          />
        )}
        {/* Centre text */}
        <text x={cx} y={cy + 10} textAnchor="middle" fill={color}
          fontSize={22} fontWeight={800}>
          {value != null ? `${Math.round(pct * 100)}%` : '—'}
        </text>
        <text x={cx} y={cy + 28} textAnchor="middle" fill="var(--muted)" fontSize={10}>
          {value != null ? 'RUL' : 'no data'}
        </text>
      </svg>
    </div>
  )
}

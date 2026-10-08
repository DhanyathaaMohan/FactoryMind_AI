import React from 'react'

const styles = {
  header: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: '0.9rem 1.5rem',
    background: '#13161f',
    borderBottom: '1px solid var(--border)',
  },
  brand: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.7rem',
  },
  logo: {
    width: 32,
    height: 32,
    background: 'var(--accent)',
    borderRadius: 6,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontWeight: 800,
    fontSize: 16,
    color: '#fff',
    flexShrink: 0,
  },
  title: {
    fontSize: 16,
    fontWeight: 700,
    color: 'var(--text)',
    letterSpacing: '-0.01em',
  },
  subtitle: {
    fontSize: 11,
    color: 'var(--muted)',
    marginTop: 1,
  },
  nav: {
    fontSize: 12,
    color: 'var(--muted)',
  },
}

export default function Header({ healthStatus }) {
  const dot = healthStatus?.model_loaded
    ? { color: 'var(--healthy)' }
    : { color: 'var(--critical)' }

  return (
    <header style={styles.header}>
      <div style={styles.brand}>
        <div style={styles.logo}>FM</div>
        <div>
          <div style={styles.title}>FactoryMind AI</div>
          <div style={styles.subtitle}>
            CNC Predictive Maintenance &amp; Knowledge Assistant
          </div>
        </div>
      </div>
      <div style={styles.nav}>
        {healthStatus && (
          <span>
            Model&nbsp;
            <span style={dot}>
              {healthStatus.model_loaded ? '● online' : '● offline'}
            </span>
          </span>
        )}
      </div>
    </header>
  )
}

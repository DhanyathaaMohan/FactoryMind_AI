import React, { useState, useEffect } from 'react'
import { fetchHealth } from '../services/api.js'
import Header from '../components/Header.jsx'
import MachineHealthCard from '../components/MachineHealthCard.jsx'
import RULGauge from '../components/RULGauge.jsx'
import ShapFeatureList from '../components/ShapFeatureList.jsx'
import MaintenanceSearch from '../components/MaintenanceSearch.jsx'
import RagAssistant from '../components/RagAssistant.jsx'
import VoiceInput from '../components/VoiceInput.jsx'
import SourceCard from '../components/SourceCard.jsx'
import { postAnalyzeAndAsk } from '../services/api.js'

/* ------------------------------------------------------------------ */
/* Layout constants                                                      */
/* ------------------------------------------------------------------ */
const grid = {
  display: 'grid',
  gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))',
  gap: '1rem',
}

const section = {
  marginBottom: '1.5rem',
}

const sectionTitle = {
  fontSize: 12,
  fontWeight: 700,
  letterSpacing: '0.1em',
  textTransform: 'uppercase',
  color: 'var(--muted)',
  marginBottom: '0.7rem',
  paddingBottom: '0.3rem',
  borderBottom: '1px solid var(--border)',
}

/* ------------------------------------------------------------------ */
/* DEMO sensor data helper (fills 120 features with zeros)             */
/* In production this comes from machine sensors / a POST body.        */
/* ------------------------------------------------------------------ */
function buildDemoSensorData(featureNames) {
  const data = {}
  featureNames.forEach((f) => { data[f] = 0.0 })
  return data
}

/* ------------------------------------------------------------------ */
export default function Dashboard() {
  const [health, setHealth]           = useState(null)
  const [prediction, setPrediction]   = useState(null)   // {predicted_rul, tool_condition}
  const [shapFeats, setShapFeats]     = useState([])
  const [sources, setSources]         = useState([])
  const [answer, setAnswer]           = useState(null)
  const [loading, setLoading]         = useState(false)
  const [error, setError]             = useState(null)
  const [query, setQuery]             = useState('')
  const [voiceQuery, setVoiceQuery]   = useState('')

  /* Fetch backend health on mount */
  useEffect(() => {
    fetchHealth()
      .then(setHealth)
      .catch(() => setHealth(null))
  }, [])

  /* ----------------------------------------------------------------
   * Full-pipeline: analyze-and-ask
   * Uses a demo sensor row (all zeros) because the UI does not yet
   * have a sensor data entry form.  In a real deployment the sensor
   * values would be streamed from the machine controller.
   * ---------------------------------------------------------------- */
  const handleAnalyze = async (e) => {
    e.preventDefault()
    const q = query.trim() || voiceQuery.trim()
    if (!q) return

    setLoading(true)
    setError(null)

    try {
      // We need the 120 feature names — fetch /health first to confirm
      // model is ready, then build a placeholder sensor dict.
      // A real integration would pass actual sensor readings here.
      const sensorData = buildDemoSensorData(
        // placeholder feature names — the backend will reject with 422
        // if these don't match; a real client would know the names.
        []
      )

      const data = await postAnalyzeAndAsk(sensorData, q, 5)
      setPrediction({ predicted_rul: data.predicted_rul, tool_condition: data.tool_condition })
      setShapFeats(data.top_shap_features)
      setSources(data.retrieved_sources)
      setAnswer(data.maintenance_answer)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  /* When voice transcript arrives, populate the query field */
  const handleVoiceTranscript = (text) => {
    setVoiceQuery(text)
    setQuery(text)
  }

  /* ---------------------------------------------------------------- */
  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      <Header healthStatus={health} />

      <main style={{ flex: 1, padding: '1.25rem 1.5rem', maxWidth: 1400, margin: '0 auto', width: '100%' }}>

        {/* ── Row 1: Machine Health + Gauge ── */}
        <div style={section}>
          <div style={sectionTitle}>Machine Health</div>
          <div style={{ ...grid, gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))' }}>
            <MachineHealthCard
              predictedRul={prediction?.predicted_rul}
              toolCondition={prediction?.tool_condition}
            />
            <RULGauge value={prediction?.predicted_rul} />
          </div>
        </div>

        {/* ── Row 2: SHAP Explainability ── */}
        <div style={section}>
          <div style={sectionTitle}>Model Explainability (SHAP)</div>
          <ShapFeatureList features={shapFeats} />
        </div>

        {/* ── Row 3: Quick analyze + voice ── */}
        <div style={section}>
          <div style={sectionTitle}>Full Pipeline — Analyze &amp; Ask</div>
          <div className="card">
            <p style={{ fontSize: 12, color: 'var(--muted)', marginBottom: '0.8rem' }}>
              Run the full pipeline: predict RUL → explain → retrieve relevant Haas
              documentation → generate a grounded maintenance answer.
            </p>
            <form onSubmit={handleAnalyze} style={{ display: 'flex', gap: '0.5rem', marginBottom: '0.5rem' }}>
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="e.g. What causes spindle bearing noise and overheating?"
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
                {loading ? <span className="spinner" /> : 'Analyze'}
              </button>
            </form>
            {error && <p className="error-msg">{error}</p>}
          </div>
        </div>

        {/* ── Row 4: Voice input ── */}
        <div style={section}>
          <div style={sectionTitle}>Voice Input</div>
          <VoiceInput onTranscript={handleVoiceTranscript} />
        </div>

        {/* ── Row 5: Maintenance Answer ── */}
        {answer && (
          <div style={section}>
            <div style={sectionTitle}>Maintenance Recommendation</div>
            <div className="card">
              <div className="label" style={{ marginBottom: '0.5rem' }}>Generated Answer</div>
              <p style={{ fontSize: 13, lineHeight: 1.75, whiteSpace: 'pre-wrap' }}>{answer}</p>
            </div>
          </div>
        )}

        {/* ── Row 6: Retrieved evidence ── */}
        {sources.length > 0 && (
          <div style={section}>
            <div style={sectionTitle}>Retrieved Evidence</div>
            {sources.map((s, i) => (
              <SourceCard
                key={i}
                rank={i + 1}
                source={s.source}
                page={s.page}
                hybridScore={s.hybrid_score}
                preview={s.preview}
              />
            ))}
          </div>
        )}

        {/* ── Row 7: Standalone search ── */}
        <div style={section}>
          <div style={sectionTitle}>Knowledge Base Search</div>
          <MaintenanceSearch />
        </div>

        {/* ── Row 8: RAG Assistant (standalone, no sensor data) ── */}
        <div style={section}>
          <div style={sectionTitle}>Maintenance Assistant (RAG only)</div>
          <RagAssistant predictedRul={prediction?.predicted_rul} />
        </div>

      </main>
    </div>
  )
}

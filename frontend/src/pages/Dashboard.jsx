import React, { useState, useEffect } from 'react'
import { fetchHealth, getSamples } from '../services/api.js'
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
/* Sample state type                                                    */
/* ------------------------------------------------------------------ */
/**
 * @typedef {Object} Sample
 * @property {number} sample_index
 * @property {number} tool_index
 * @property {number} cycle
 */

/* ------------------------------------------------------------------ */
export default function Dashboard() {
  const [health, setHealth]           = useState(null)
  const [samples, setSamples]         = useState([])     // array of {sample_index, tool_index, cycle}
  const [selectedSample, setSelectedSample] = useState(null) // selected Sample
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

  /* Load samples metadata on mount */
  useEffect(() => {
    getSamples()
      .then(data => setSamples(data.samples || []))
      .catch(() => setSamples([]))
  }, [])

  /* ----------------------------------------------------------------
   * Full-pipeline: analyze-and-ask
   * Uses a user-selected CNC sample from the dataset.
   * ---------------------------------------------------------------- */
  const handleAnalyze = async (e) => {
    e.preventDefault()
    const q = query.trim() || voiceQuery.trim()
    if (!q) return

    // Validate sample selection
    if (!selectedSample) {
      setError('Please select a CNC machine sample before running the full analysis.')
      return
    }

    setLoading(true)
    setError(null)

    try {
      const data = await postAnalyzeAndAsk({
        sampleIndex: selectedSample.sample_index,
        query: q,
        topShap: 5,
      })
      setPrediction({ predicted_rul: data.predicted_rul, tool_condition: data.tool_condition })
      setShapFeats(data.top_shap_features)
      setSources(data.retrieved_sources)
      setAnswer(data.maintenance_answer)
    } catch (err) {
      // Check if it's a 422 error about missing features and show user-friendly message
      if (err.message && (err.message.includes('422') || err.message.includes('Missing'))) {
        setError('Please select a CNC machine sample before running the full analysis.')
      } else {
        setError(err.message || 'An error occurred while analyzing the sample.')
      }
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

        {/* ── Row 3: Sample selection + full pipeline analysis ── */}
        <div style={section}>
          <div style={sectionTitle}>Full Pipeline — Analyze &amp; Ask</div>
          <div className="card">
            <p style={{ fontSize: 12, color: 'var(--muted)', marginBottom: '0.8rem' }}>
              Run the full pipeline: predict RUL → explain → retrieve relevant Haas
              documentation → generate a grounded maintenance answer.
            </p>

            {/* Sample selector */}
            <div style={{ marginBottom: '0.8rem' }}>
              <div className="label" style={{ marginBottom: '0.4rem' }}>Select CNC Sample</div>
              {samples.length === 0 ? (
                <p style={{ fontSize: 13, color: 'var(--muted)' }}>
                  Loading samples from dataset...
                </p>
              ) : (
                <div style={{
                  display: 'flex',
                  gap: '0.6rem',
                  flexWrap: 'wrap',
                }}>
                  <div>
                    <label style={{ fontSize: 11, color: 'var(--muted)', marginRight: 4 }}>
                      Tool
                    </label>
                    <select
                      value={selectedSample?.tool_index || ''}
                      onChange={(e) => {
                        const toolIdx = parseInt(e.target.value, 10)
                        // Find first sample for this tool
                        const sample = samples.find(s => s.tool_index === toolIdx) || null
                        setSelectedSample(sample)
                      }}
                      style={{
                        padding: '0.4rem 0.6rem',
                        background: 'var(--bg)',
                        border: '1px solid var(--border)',
                        borderRadius: 'var(--radius)',
                        color: 'var(--text)',
                        fontSize: 13,
                      }}
                    >
                      <option value="">Select tool...</option>
                      {Array.from(new Set(samples.map(s => s.tool_index))).sort((a, b) => a - b).map(tool => (
                        <option key={tool} value={tool}>Tool {tool}</option>
                      ))}
                    </select>
                  </div>
                  <div>
                    <label style={{ fontSize: 11, color: 'var(--muted)', marginRight: 4 }}>
                      Cycle
                    </label>
                    <select
                      value={selectedSample?.cycle || ''}
                      disabled={!selectedSample?.tool_index}
                      onChange={(e) => {
                        const cycle = parseInt(e.target.value, 10)
                        const sample = samples.find(s => s.cycle === cycle && s.tool_index === selectedSample?.tool_index) || null
                        setSelectedSample(sample)
                      }}
                      style={{
                        padding: '0.4rem 0.6rem',
                        background: 'var(--bg)',
                        border: '1px solid var(--border)',
                        borderRadius: 'var(--radius)',
                        color: 'var(--text)',
                        fontSize: 13,
                      }}
                    >
                      <option value="">Select cycle...</option>
                      {samples
                        .filter(s => s.tool_index === selectedSample?.tool_index)
                        .map(s => (
                          <option key={s.sample_index} value={s.cycle}>
                            Cycle {s.cycle}
                          </option>
                        ))
                      }
                    </select>
                  </div>
                  {selectedSample && (
                    <div style={{
                      padding: '0.4rem 0.8rem',
                      background: 'var(--border)',
                      borderRadius: 'var(--radius)',
                      fontSize: 12,
                      display: 'flex',
                      alignItems: 'center',
                    }}>
                      <span>Selected: <strong>Tool {selectedSample.tool_index} — Cycle {selectedSample.cycle}</strong></span>
                    </div>
                  )}
                </div>
              )}
            </div>

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
              <button type="submit" className="btn btn-primary" disabled={loading || !selectedSample}>
                {loading ? <span className="spinner" /> : 'Analyze'}
              </button>
            </form>
            {error && <p className="error-msg">{error}</p>}
            {!selectedSample && !loading && (
              <p style={{ fontSize: 11, color: 'var(--muted)', marginTop: '0.4rem' }}>
                Select a CNC sample above to analyze.
              </p>
            )}
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

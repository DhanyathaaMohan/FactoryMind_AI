import React, { useState, useRef } from 'react'
import { postTranscribe } from '../services/api.js'

const SUPPORTED = '.wav,.mp3,.m4a,.ogg,.flac,.webm'

export default function VoiceInput({ onTranscript }) {
  const [transcript, setTranscript] = useState('')
  const [loading, setLoading]       = useState(false)
  const [error, setError]           = useState(null)
  const [filename, setFilename]     = useState(null)
  const fileRef = useRef()

  const handleFile = async (e) => {
    const file = e.target.files?.[0]
    if (!file) return

    setFilename(file.name)
    setLoading(true)
    setError(null)
    setTranscript('')

    try {
      const data = await postTranscribe(file)
      setTranscript(data.transcript)
      onTranscript?.(data.transcript)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
      // Reset file input so the same file can be re-uploaded
      if (fileRef.current) fileRef.current.value = ''
    }
  }

  return (
    <div className="card">
      <div className="label">Voice Input</div>
      <p style={{ fontSize: 12, color: 'var(--muted)', marginTop: 4, marginBottom: '0.8rem' }}>
        Upload an audio file to transcribe your maintenance query using
        Faster-Whisper (CPU).
      </p>

      <label
        style={{
          display: 'inline-flex',
          alignItems: 'center',
          gap: '0.5rem',
          padding: '0.5rem 1rem',
          background: 'var(--border)',
          borderRadius: 'var(--radius)',
          cursor: 'pointer',
          fontSize: 13,
          color: 'var(--text)',
        }}
      >
        {loading ? (
          <>
            <span className="spinner" /> Transcribing…
          </>
        ) : (
          <>
            🎤 Upload audio
            <input
              ref={fileRef}
              type="file"
              accept={SUPPORTED}
              style={{ display: 'none' }}
              onChange={handleFile}
              disabled={loading}
            />
          </>
        )}
      </label>

      {filename && !loading && (
        <span style={{ fontSize: 11, color: 'var(--muted)', marginLeft: '0.6rem' }}>
          {filename}
        </span>
      )}

      {error && <p className="error-msg" style={{ marginTop: '0.5rem' }}>{error}</p>}

      {transcript && (
        <div style={{
          marginTop: '0.8rem',
          background: 'var(--bg)',
          border: '1px solid var(--border)',
          borderRadius: 'var(--radius)',
          padding: '0.7rem 0.9rem',
        }}>
          <div className="label" style={{ marginBottom: '0.3rem' }}>Transcript</div>
          <p style={{ fontSize: 13, color: 'var(--text)', lineHeight: 1.6 }}>{transcript}</p>
        </div>
      )}
    </div>
  )
}

/**
 * api.js
 * ------
 * Thin fetch wrappers for every FactoryMind backend endpoint.
 * The BASE_URL can be overridden via the VITE_API_URL env variable.
 */

const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

async function request(path, options = {}) {
  const url = `${BASE_URL}${path}`
  const response = await fetch(url, {
    headers: { 'Content-Type': 'application/json', ...options.headers },
    ...options,
  })
  if (!response.ok) {
    let detail = response.statusText
    try {
      const body = await response.json()
      detail = body.detail || detail
    } catch (_) {}
    throw new Error(`${response.status}: ${detail}`)
  }
  return response.json()
}

/** GET /health */
export function fetchHealth() {
  return request('/health')
}

/**
 * POST /predict
 * @param {Record<string, number>} sensorData – 120 sensor feature values
 */
export function postPredict(sensorData) {
  return request('/predict', {
    method: 'POST',
    body: JSON.stringify({ sensor_data: sensorData }),
  })
}

/**
 * POST /search
 * @param {string} query
 * @param {number} topK
 */
export function postSearch(query, topK = 3) {
  return request('/search', {
    method: 'POST',
    body: JSON.stringify({ query, top_k: topK }),
  })
}

/**
 * POST /ask
 * @param {string} query
 * @param {number|null} predictedRul
 */
export function postAsk(query, predictedRul = null) {
  return request('/ask', {
    method: 'POST',
    body: JSON.stringify({ query, predicted_rul: predictedRul }),
  })
}

/**
 * GET /samples
 * @returns {Promise<{samples: Array<{sample_index: number, tool_index: number, cycle: number}>>}
 */
export function getSamples() {
  return request('/samples')
}

/**
 * GET /samples/{index}
 * @param {number} sampleIndex
 */
export function getSample(sampleIndex) {
  return request(`/samples/${sampleIndex}`)
}

/**
 * POST /analyze-and-ask
 * Supports two modes:
 * - sample_index: loads from dataset (preferred for UI)
 * - sensor_data: direct feature dictionary (for advanced use)
 *
 * @param {Object} request
 * @param {string} request.query
 * @param {number} [request.sampleIndex] — dataset sample index
 * @param {Object} [request.sensorData] — direct 120 feature dictionary
 * @param {number} [request.topShap=5]
 */
export function postAnalyzeAndAsk({
  query,
  sampleIndex = null,
  sensorData = null,
  topShap = 5,
} = {}) {
  const payload = { query, top_shap: topShap }
  if (sampleIndex !== null) {
    payload.sample_index = sampleIndex
  } else if (sensorData !== null) {
    payload.sensor_data = sensorData
  } else {
    throw new Error('Provide either sampleIndex or sensorData')
  }
  return request('/analyze-and-ask', {
    method: 'POST',
    body: JSON.stringify(payload),
  })
}

/**
 * POST /transcribe
 * @param {File} audioFile
 */
export function postTranscribe(audioFile) {
  const formData = new FormData()
  formData.append('file', audioFile)
  return request('/transcribe', {
    method: 'POST',
    headers: {},       // let browser set multipart boundary
    body: formData,
  })
}

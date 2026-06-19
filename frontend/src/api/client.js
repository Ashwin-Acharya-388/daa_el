/**
 * API client — Axios wrapper for the FastAPI backend.
 */
import axios from 'axios';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const API_KEY = import.meta.env.VITE_API_KEY || 'fra-dev-key-2024';

const api = axios.create({
  baseURL: API_BASE,
  headers: {
    'X-API-Key': API_KEY,
    'Content-Type': 'application/json',
  },
  timeout: 120000, // 2 minutes for SHAP computations
});

// ── Predictions ──
export const predictSingle = (features, includeShap = true) =>
  api.post('/api/predict', { features, include_shap: includeShap });

// ── Explain ──
export const explainPrediction = (features) =>
  api.post('/api/explain', { features });

// ── Model Info ──
export const getModelInfo = () =>
  api.get('/api/model/info');

// ── History ──
export const getHistory = (params = {}) =>
  api.get('/api/predictions/history', { params });

export const getPredictionDetail = (id) =>
  api.get(`/api/predictions/${id}`);

export const exportPredictions = (params = {}) =>
  api.get('/api/predictions/export/csv', { params, responseType: 'blob' });

// ── Health ──
export const getHealth = () =>
  api.get('/health');

export default api;

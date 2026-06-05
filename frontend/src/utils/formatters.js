/**
 * formatters.js — Utility formatting functions.
 */

export const formatPercent = (value, decimals = 1) =>
  `${(value * 100).toFixed(decimals)}%`;

export const formatNumber = (value, decimals = 4) =>
  typeof value === 'number' ? value.toFixed(decimals) : '—';

export const formatDate = (dateStr) => {
  if (!dateStr) return '—';
  const d = new Date(dateStr);
  return d.toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
};

export const getRiskLevel = (probability) => {
  if (probability >= 0.7) return 'high';
  if (probability >= 0.4) return 'medium';
  return 'low';
};

export const getRiskColor = (probability) => {
  if (probability >= 0.7) return 'var(--red)';
  if (probability >= 0.4) return 'var(--yellow)';
  return 'var(--green)';
};

export const getRiskLabel = (probability) => {
  if (probability >= 0.5) return 'HIGH RISK';
  return 'LOW RISK';
};

export const downloadBlob = (blob, filename) => {
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  window.URL.revokeObjectURL(url);
};

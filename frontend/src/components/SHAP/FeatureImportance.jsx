/**
 * FeatureImportance — Horizontal bar chart of top SHAP feature importances.
 */
import { useState } from 'react';

export default function FeatureImportance({ factors = [], maxFeatures = 10, title = 'Feature Importance (SHAP)' }) {
  const [hoveredIdx, setHoveredIdx] = useState(null);

  const sorted = [...factors]
    .sort((a, b) => Math.abs(b.shap_value) - Math.abs(a.shap_value))
    .slice(0, maxFeatures);

  if (sorted.length === 0) {
    return (
      <div className="empty-state" style={{ padding: '2rem' }}>
        <p style={{ color: 'var(--text-muted)' }}>No feature data available</p>
      </div>
    );
  }

  const maxVal = Math.max(...sorted.map((f) => Math.abs(f.shap_value)));

  return (
    <div>
      {title && <h3 style={{ fontSize: '0.875rem', fontWeight: 600, marginBottom: '1rem', color: 'var(--text-secondary)' }}>{title}</h3>}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
        {sorted.map((factor, i) => {
          const pct = (Math.abs(factor.shap_value) / maxVal) * 100;
          const isPositive = factor.shap_value > 0;
          const color = isPositive ? 'var(--red)' : 'var(--blue)';
          const isHovered = hoveredIdx === i;

          return (
            <div
              key={i}
              onMouseEnter={() => setHoveredIdx(i)}
              onMouseLeave={() => setHoveredIdx(null)}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '12px',
                cursor: 'default',
                opacity: isHovered ? 1 : 0.85,
                transition: 'opacity 0.15s ease',
              }}
            >
              <span style={{
                width: 100,
                fontSize: '0.75rem',
                color: 'var(--text-secondary)',
                textAlign: 'right',
                fontWeight: isHovered ? 600 : 400,
                flexShrink: 0,
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap',
              }}>
                {factor.feature}
              </span>
              <div style={{
                flex: 1,
                height: 20,
                background: 'rgba(255,255,255,0.04)',
                borderRadius: 4,
                overflow: 'hidden',
                position: 'relative',
              }}>
                <div style={{
                  width: `${pct}%`,
                  height: '100%',
                  background: color,
                  borderRadius: 4,
                  opacity: isHovered ? 0.9 : 0.65,
                  transition: 'width 0.8s cubic-bezier(0.34, 1.56, 0.64, 1), opacity 0.15s ease',
                }} />
              </div>
              <span style={{
                width: 65,
                fontSize: '0.7rem',
                fontWeight: 600,
                color,
                textAlign: 'right',
                flexShrink: 0,
              }}>
                {isPositive ? '+' : ''}{factor.shap_value.toFixed(4)}
              </span>
            </div>
          );
        })}
      </div>
      <div style={{
        display: 'flex',
        justifyContent: 'center',
        gap: '1.5rem',
        marginTop: '1rem',
        fontSize: '0.7rem',
        color: 'var(--text-muted)',
      }}>
        <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
          <span style={{ width: 10, height: 10, borderRadius: 2, background: 'var(--red)', display: 'inline-block' }} />
          Pushes toward default
        </span>
        <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
          <span style={{ width: 10, height: 10, borderRadius: 2, background: 'var(--blue)', display: 'inline-block' }} />
          Pushes toward healthy
        </span>
      </div>
    </div>
  );
}

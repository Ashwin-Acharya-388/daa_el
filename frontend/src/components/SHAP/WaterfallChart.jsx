/**
 * WaterfallChart — SVG waterfall chart for SHAP explanations.
 * Shows how features push the prediction from the base value toward default or healthy.
 */
import { useState } from 'react';

export default function WaterfallChart({
  baseValue = 0.5,
  factors = [],
  height = 320,
  maxFeatures = 10,
}) {
  const [hoveredIdx, setHoveredIdx] = useState(null);

  // Sort by absolute SHAP value and take top N
  const sorted = [...factors]
    .sort((a, b) => Math.abs(b.shap_value) - Math.abs(a.shap_value))
    .slice(0, maxFeatures);

  if (sorted.length === 0) {
    return (
      <div className="empty-state" style={{ padding: '2rem' }}>
        <p style={{ color: 'var(--text-muted)' }}>No SHAP data available</p>
      </div>
    );
  }

  const margin = { top: 20, right: 80, bottom: 30, left: 120 };
  const width = 600;
  const chartWidth = width - margin.left - margin.right;
  const chartHeight = height - margin.top - margin.bottom;
  const barHeight = Math.min(24, chartHeight / sorted.length - 4);

  // Scale: find min/max cumulative values
  const maxAbsShap = Math.max(...sorted.map((f) => Math.abs(f.shap_value)));
  const scale = chartWidth / 2 / (maxAbsShap * 1.2 || 1);
  const centerX = chartWidth / 2;

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      style={{ width: '100%', maxWidth: 600, height: 'auto' }}
    >
      {/* Center line (base value) */}
      <line
        x1={margin.left + centerX}
        y1={margin.top}
        x2={margin.left + centerX}
        y2={margin.top + chartHeight}
        stroke="rgba(255,255,255,0.1)"
        strokeWidth="1"
        strokeDasharray="4,4"
      />

      {sorted.map((factor, i) => {
        const y = margin.top + i * (chartHeight / sorted.length) + (chartHeight / sorted.length - barHeight) / 2;
        const barWidth = Math.abs(factor.shap_value) * scale;
        const isPositive = factor.shap_value > 0;
        const x = isPositive
          ? margin.left + centerX
          : margin.left + centerX - barWidth;
        const color = isPositive ? '#ff3d71' : '#3366ff';
        const isHovered = hoveredIdx === i;

        return (
          <g
            key={i}
            onMouseEnter={() => setHoveredIdx(i)}
            onMouseLeave={() => setHoveredIdx(null)}
            style={{ cursor: 'pointer' }}
          >
            {/* Feature label */}
            <text
              x={margin.left - 8}
              y={y + barHeight / 2}
              textAnchor="end"
              dominantBaseline="middle"
              fill={isHovered ? 'var(--text-primary)' : 'var(--text-secondary)'}
              fontSize="11"
              fontFamily="Inter, sans-serif"
              fontWeight={isHovered ? '600' : '400'}
            >
              {factor.feature.length > 14
                ? factor.feature.slice(0, 14) + '…'
                : factor.feature}
            </text>

            {/* Bar */}
            <rect
              x={x}
              y={y}
              width={barWidth}
              height={barHeight}
              rx={4}
              fill={color}
              opacity={isHovered ? 1 : 0.75}
              style={{ transition: 'opacity 0.15s ease' }}
            />

            {/* Value label */}
            <text
              x={isPositive
                ? margin.left + centerX + barWidth + 6
                : margin.left + centerX - barWidth - 6
              }
              y={y + barHeight / 2}
              textAnchor={isPositive ? 'start' : 'end'}
              dominantBaseline="middle"
              fill={color}
              fontSize="10"
              fontFamily="Inter, sans-serif"
              fontWeight="600"
            >
              {isPositive ? '+' : ''}{factor.shap_value.toFixed(4)}
            </text>

            {/* Tooltip on hover */}
            {isHovered && (
              <g>
                <rect
                  x={margin.left + centerX - 70}
                  y={y - 28}
                  width={140}
                  height={22}
                  rx={4}
                  fill="var(--bg-tertiary)"
                  stroke="var(--glass-border)"
                />
                <text
                  x={margin.left + centerX}
                  y={y - 15}
                  textAnchor="middle"
                  dominantBaseline="middle"
                  fill="var(--text-primary)"
                  fontSize="10"
                  fontFamily="Inter, sans-serif"
                >
                  Value: {factor.feature_value?.toFixed(4) ?? '—'}
                </text>
              </g>
            )}
          </g>
        );
      })}

      {/* Axis labels */}
      <text
        x={margin.left + centerX}
        y={height - 6}
        textAnchor="middle"
        fill="var(--text-muted)"
        fontSize="10"
        fontFamily="Inter, sans-serif"
      >
        ← Toward Healthy | Toward Default →
      </text>
    </svg>
  );
}

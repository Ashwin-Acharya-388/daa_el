/**
 * RiskGauge — Animated semicircular gauge for risk score visualization.
 */
import { useEffect, useState } from 'react';

export default function RiskGauge({ value = 0, size = 200 }) {
  const [animatedValue, setAnimatedValue] = useState(0);

  useEffect(() => {
    const timer = setTimeout(() => setAnimatedValue(value), 100);
    return () => clearTimeout(timer);
  }, [value]);

  const width = size;
  const height = size * 0.65;
  const cx = width / 2;
  const cy = height - 10;
  const radius = Math.min(cx, cy) - 15;
  const startAngle = Math.PI;
  const endAngle = 0;
  const angleRange = startAngle - endAngle;
  const currentAngle = startAngle - (animatedValue / 100) * angleRange;

  // Background arc
  const bgArc = describeArc(cx, cy, radius, endAngle, startAngle);
  // Value arc
  const valueArc = describeArc(cx, cy, radius, currentAngle, startAngle);

  // Gradient color based on value
  const getColor = (v) => {
    if (v >= 70) return '#ff3d71';
    if (v >= 40) return '#ffaa00';
    return '#00d68f';
  };

  const color = getColor(animatedValue);

  return (
    <div className="gauge-container">
      <svg width={width} height={height} className="gauge-svg" viewBox={`0 0 ${width} ${height}`}>
        {/* Glow filter */}
        <defs>
          <filter id="gaugeGlow">
            <feGaussianBlur stdDeviation="3" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
          <linearGradient id="gaugeGrad" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#00d68f" />
            <stop offset="50%" stopColor="#ffaa00" />
            <stop offset="100%" stopColor="#ff3d71" />
          </linearGradient>
        </defs>

        {/* Background track */}
        <path
          d={bgArc}
          fill="none"
          stroke="rgba(255,255,255,0.06)"
          strokeWidth="12"
          strokeLinecap="round"
        />

        {/* Value arc */}
        <path
          d={valueArc}
          fill="none"
          stroke={color}
          strokeWidth="12"
          strokeLinecap="round"
          filter="url(#gaugeGlow)"
          style={{
            transition: 'all 1.2s cubic-bezier(0.34, 1.56, 0.64, 1)',
          }}
        />

        {/* Tick marks */}
        {[0, 25, 50, 75, 100].map((tick) => {
          const angle = startAngle - (tick / 100) * angleRange;
          const x1 = cx + (radius - 20) * Math.cos(angle);
          const y1 = cy - (radius - 20) * Math.sin(angle);
          const x2 = cx + (radius - 28) * Math.cos(angle);
          const y2 = cy - (radius - 28) * Math.sin(angle);
          const lx = cx + (radius - 38) * Math.cos(angle);
          const ly = cy - (radius - 38) * Math.sin(angle);
          return (
            <g key={tick}>
              <line
                x1={x1} y1={y1} x2={x2} y2={y2}
                stroke="rgba(255,255,255,0.15)"
                strokeWidth="2"
              />
              <text
                x={lx} y={ly}
                textAnchor="middle"
                dominantBaseline="middle"
                fill="rgba(255,255,255,0.3)"
                fontSize="10"
                fontFamily="Inter, sans-serif"
              >
                {tick}
              </text>
            </g>
          );
        })}

        {/* Center value */}
        <text
          x={cx}
          y={cy - 20}
          textAnchor="middle"
          dominantBaseline="middle"
          fill={color}
          fontSize="32"
          fontWeight="800"
          fontFamily="Inter, sans-serif"
          style={{ transition: 'fill 0.5s ease' }}
        >
          {animatedValue.toFixed(1)}%
        </text>
        <text
          x={cx}
          y={cy + 2}
          textAnchor="middle"
          dominantBaseline="middle"
          fill="rgba(255,255,255,0.4)"
          fontSize="11"
          fontFamily="Inter, sans-serif"
        >
          Risk Score
        </text>
      </svg>
    </div>
  );
}

function describeArc(cx, cy, radius, startAngle, endAngle) {
  const x1 = cx + radius * Math.cos(endAngle);
  const y1 = cy - radius * Math.sin(endAngle);
  const x2 = cx + radius * Math.cos(startAngle);
  const y2 = cy - radius * Math.sin(startAngle);
  const largeArc = endAngle - startAngle <= Math.PI ? 0 : 1;
  return `M ${x1} ${y1} A ${radius} ${radius} 0 ${largeArc} 0 ${x2} ${y2}`;
}

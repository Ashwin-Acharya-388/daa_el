/**
 * LoadingSkeleton — Shimmer placeholders for loading states.
 */

export function SkeletonCard() {
  return <div className="skeleton skeleton-card" />;
}

export function SkeletonText({ width = '100%' }) {
  return <div className="skeleton skeleton-text" style={{ width }} />;
}

export function SkeletonTitle({ width = '60%' }) {
  return <div className="skeleton skeleton-title" style={{ width }} />;
}

export function SkeletonChart() {
  return <div className="skeleton skeleton-chart" />;
}

export function MetricsSkeleton() {
  return (
    <div className="metrics-grid">
      {[1, 2, 3, 4].map((i) => (
        <div key={i} className="glass-card">
          <div className="skeleton" style={{ width: 42, height: 42, borderRadius: 10, marginBottom: 16 }} />
          <div className="skeleton skeleton-title" style={{ width: '40%' }} />
          <div className="skeleton skeleton-text" style={{ width: '60%' }} />
        </div>
      ))}
    </div>
  );
}

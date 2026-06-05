/**
 * DashboardPage — Main dashboard with metrics cards, prediction counter,
 * and model status indicator.
 */
import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  Target,
  Crosshair,
  TrendingUp,
  Activity,
  Shield,
  Zap,
  BarChart2,
} from 'lucide-react';
import AnimatedCounter from '../components/common/AnimatedCounter';
import { MetricsSkeleton } from '../components/common/LoadingSkeleton';
import { getModelInfo } from '../api/client';

const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: { staggerChildren: 0.1 },
  },
};

const itemVariants = {
  hidden: { opacity: 0, y: 20 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.5 } },
};

export default function DashboardPage() {
  const [info, setInfo] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    getModelInfo()
      .then((res) => setInfo(res.data))
      .catch((err) => setError(err.response?.data?.detail || 'Failed to load model info'))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div>
        <div className="page-header">
          <h1>Dashboard</h1>
          <p>Financial Risk Analysis Overview</p>
        </div>
        <MetricsSkeleton />
      </div>
    );
  }

  const metrics = info?.metrics || {};
  const accuracy = metrics.accuracy || 0;
  const recall = metrics.recall_sensitivity || 0;
  const auc = metrics.auc_roc || 0;
  const mcc = metrics.mcc || 0;
  const precision = metrics.precision || 0;
  const f1 = metrics.f1_score || 0;
  const cm = metrics.confusion_matrix || {};

  const metricCards = [
    {
      label: 'Accuracy',
      value: accuracy * 100,
      suffix: '%',
      decimals: 1,
      icon: Target,
      color: 'gold',
      accent: '--gold',
    },
    {
      label: 'Recall (Sensitivity)',
      value: recall * 100,
      suffix: '%',
      decimals: 1,
      icon: Crosshair,
      color: 'blue',
      accent: '--blue',
    },
    {
      label: 'AUC-ROC',
      value: auc * 100,
      suffix: '%',
      decimals: 1,
      icon: TrendingUp,
      color: 'green',
      accent: '--green',
    },
    {
      label: 'MCC',
      value: mcc,
      suffix: '',
      decimals: 4,
      icon: Activity,
      color: 'purple',
      accent: '--purple',
    },
  ];

  return (
    <motion.div variants={containerVariants} initial="hidden" animate="visible">
      <motion.div className="page-header" variants={itemVariants}>
        <h1 style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <Shield size={32} style={{ color: 'var(--gold)' }} />
          Dashboard
        </h1>
        <p>DenseNet Financial Risk Analysis — Real-time model performance & predictions</p>
      </motion.div>

      {/* Status bar */}
      <motion.div
        variants={itemVariants}
        className="glass-card"
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          marginBottom: 'var(--space-6)',
          padding: 'var(--space-4) var(--space-6)',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <span className={`status-dot ${info?.model_status === 'healthy' ? 'healthy' : 'unhealthy'}`} />
          <span style={{ fontWeight: 600, fontSize: 'var(--font-size-sm)' }}>
            Model: {info?.model_status === 'healthy' ? 'Healthy & Running' : 'Offline'}
          </span>
        </div>
        <div style={{ display: 'flex', gap: '2rem', fontSize: 'var(--font-size-xs)', color: 'var(--text-secondary)' }}>
          <span>Features: <strong style={{ color: 'var(--text-primary)' }}>{info?.feature_count}</strong></span>
          <span>Version: <strong style={{ color: 'var(--text-primary)' }}>{info?.version}</strong></span>
        </div>
      </motion.div>

      {/* Metric cards */}
      <motion.div className="metrics-grid" variants={itemVariants}>
        {metricCards.map((m, i) => (
          <motion.div
            key={m.label}
            className={`glass-card metric-card glass-card--${m.color}`}
            variants={itemVariants}
            whileHover={{ scale: 1.02, y: -4 }}
            transition={{ type: 'spring', stiffness: 300 }}
          >
            <div className={`metric-card-icon ${m.color}`}>
              <m.icon size={20} />
            </div>
            <div className="metric-value">
              <AnimatedCounter
                target={m.value}
                decimals={m.decimals}
                suffix={m.suffix}
                duration={1800}
              />
            </div>
            <div className="metric-label">{m.label}</div>
          </motion.div>
        ))}
      </motion.div>

      {/* Second row: Predictions counter & more metrics */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 'var(--space-6)', marginBottom: 'var(--space-6)' }}>
        {/* Total predictions */}
        <motion.div
          className="glass-card glass-card--gold"
          variants={itemVariants}
          style={{ textAlign: 'center', padding: 'var(--space-8)' }}
        >
          <Zap size={28} style={{ color: 'var(--gold)', marginBottom: '0.5rem' }} />
          <div style={{ fontSize: 'var(--font-size-4xl)', fontWeight: 800, color: 'var(--gold)', letterSpacing: '-0.03em' }}>
            <AnimatedCounter target={info?.total_predictions || 0} decimals={0} duration={2000} />
          </div>
          <div className="metric-label" style={{ marginTop: '0.25rem' }}>Total Predictions Made</div>
        </motion.div>

        {/* Additional metrics */}
        <motion.div className="glass-card" variants={itemVariants}>
          <h3 style={{ fontSize: 'var(--font-size-sm)', fontWeight: 600, marginBottom: 'var(--space-4)', color: 'var(--text-secondary)' }}>
            <BarChart2 size={16} style={{ marginRight: 8, verticalAlign: 'middle' }} />
            Additional Metrics
          </h3>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 'var(--space-4)' }}>
            {[
              { label: 'Precision', value: `${(precision * 100).toFixed(1)}%`, color: 'var(--blue)' },
              { label: 'F1-Score', value: f1.toFixed(4), color: 'var(--green)' },
              { label: 'Specificity', value: `${((metrics.specificity || 0) * 100).toFixed(1)}%`, color: 'var(--gold)' },
              { label: 'Test Samples', value: metrics.test_samples || '—', color: 'var(--purple)' },
            ].map((item) => (
              <div key={item.label} style={{ padding: 'var(--space-3)', background: 'rgba(255,255,255,0.03)', borderRadius: 'var(--radius-md)' }}>
                <div style={{ fontSize: 'var(--font-size-xs)', color: 'var(--text-muted)', marginBottom: 2 }}>{item.label}</div>
                <div style={{ fontSize: 'var(--font-size-lg)', fontWeight: 700, color: item.color }}>{item.value}</div>
              </div>
            ))}
          </div>
        </motion.div>
      </div>

      {/* Confusion Matrix Summary */}
      <motion.div className="glass-card" variants={itemVariants}>
        <h3 style={{ fontSize: 'var(--font-size-sm)', fontWeight: 600, marginBottom: 'var(--space-4)', color: 'var(--text-secondary)' }}>
          Confusion Matrix Summary
        </h3>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 'var(--space-4)' }}>
          {[
            { label: 'True Negatives', value: cm.true_negatives, color: 'var(--green)', desc: 'Correctly identified healthy' },
            { label: 'False Positives', value: cm.false_positives, color: 'var(--yellow)', desc: 'Healthy flagged as risky' },
            { label: 'False Negatives', value: cm.false_negatives, color: 'var(--red)', desc: 'Risky missed as healthy' },
            { label: 'True Positives', value: cm.true_positives, color: 'var(--blue)', desc: 'Correctly identified risky' },
          ].map((item) => (
            <div key={item.label} style={{ textAlign: 'center', padding: 'var(--space-4)', background: 'rgba(255,255,255,0.02)', borderRadius: 'var(--radius-md)' }}>
              <div style={{ fontSize: 'var(--font-size-2xl)', fontWeight: 700, color: item.color }}>
                <AnimatedCounter target={item.value || 0} decimals={0} duration={1600} />
              </div>
              <div style={{ fontSize: 'var(--font-size-xs)', fontWeight: 600, marginTop: 4 }}>{item.label}</div>
              <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)', marginTop: 2 }}>{item.desc}</div>
            </div>
          ))}
        </div>
      </motion.div>

      {error && (
        <div style={{ color: 'var(--red)', marginTop: 'var(--space-4)', textAlign: 'center' }}>
          ⚠ {error}
        </div>
      )}
    </motion.div>
  );
}

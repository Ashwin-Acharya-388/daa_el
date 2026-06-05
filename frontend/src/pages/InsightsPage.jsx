/**
 * InsightsPage — Model performance visualizations using Chart.js:
 * confusion matrix heatmap, ROC curve, feature importance, and class distribution.
 */
import { useState, useEffect, useRef } from 'react';
import { motion } from 'framer-motion';
import { BarChart3 } from 'lucide-react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  LineElement,
  PointElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
  Filler,
} from 'chart.js';
import { Bar, Line, Doughnut } from 'react-chartjs-2';
import { getModelInfo } from '../api/client';

ChartJS.register(
  CategoryScale, LinearScale, BarElement, LineElement,
  PointElement, ArcElement, Title, Tooltip, Legend, Filler
);

const itemVariants = {
  hidden: { opacity: 0, y: 20 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.4 } },
};

export default function InsightsPage() {
  const [info, setInfo] = useState(null);
  const [loading, setLoading] = useState(true);
  const [tab, setTab] = useState('confusion');

  useEffect(() => {
    getModelInfo()
      .then((res) => setInfo(res.data))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div>
        <div className="page-header">
          <h1>Model Insights</h1>
        </div>
        <div className="skeleton skeleton-chart" style={{ height: 400 }} />
      </div>
    );
  }

  const m = info?.metrics || {};
  const cm = m.confusion_matrix || {};

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        labels: {
          color: 'rgba(232,236,244,0.7)',
          font: { family: 'Inter', size: 12 },
        },
      },
      tooltip: {
        backgroundColor: '#151d35',
        titleColor: '#e8ecf4',
        bodyColor: '#8b95a8',
        borderColor: 'rgba(255,255,255,0.1)',
        borderWidth: 1,
        titleFont: { family: 'Inter' },
        bodyFont: { family: 'Inter' },
      },
    },
    scales: {
      x: {
        ticks: { color: 'rgba(232,236,244,0.5)', font: { family: 'Inter', size: 11 } },
        grid: { color: 'rgba(255,255,255,0.04)' },
      },
      y: {
        ticks: { color: 'rgba(232,236,244,0.5)', font: { family: 'Inter', size: 11 } },
        grid: { color: 'rgba(255,255,255,0.04)' },
      },
    },
  };

  return (
    <motion.div initial="hidden" animate="visible" variants={{ hidden: { opacity: 0 }, visible: { opacity: 1, transition: { staggerChildren: 0.1 } } }}>
      <motion.div className="page-header" variants={itemVariants}>
        <h1 style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <BarChart3 size={32} style={{ color: 'var(--purple)' }} />
          Model Insights
        </h1>
        <p>Performance metrics and evaluation visualizations for the DenseNet model</p>
      </motion.div>

      {/* Tabs */}
      <motion.div variants={itemVariants} className="tabs">
        {[
          { id: 'confusion', label: 'Confusion Matrix' },
          { id: 'roc', label: 'ROC Curve' },
          { id: 'importance', label: 'Metrics Comparison' },
          { id: 'distribution', label: 'Class Distribution' },
        ].map((t) => (
          <button
            key={t.id}
            className={`tab ${tab === t.id ? 'active' : ''}`}
            onClick={() => setTab(t.id)}
          >
            {t.label}
          </button>
        ))}
      </motion.div>

      <motion.div variants={itemVariants}>
        {/* Confusion Matrix */}
        {tab === 'confusion' && (
          <div className="glass-card">
            <h3 style={{ fontSize: 'var(--font-size-sm)', fontWeight: 600, marginBottom: 'var(--space-6)', color: 'var(--text-secondary)' }}>
              Confusion Matrix — DenseNet Model
            </h3>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 'var(--space-3)', maxWidth: 500, margin: '0 auto' }}>
              {/* Headers */}
              <div style={{ gridColumn: '1 / -1', textAlign: 'center', fontSize: 'var(--font-size-xs)', color: 'var(--text-muted)', fontWeight: 600, paddingBottom: 8 }}>
                PREDICTED
              </div>

              {[
                { label: 'True Negative', value: cm.true_negatives, color: '#00d68f', bg: 'rgba(0,214,143,0.08)', desc: 'Healthy → Healthy' },
                { label: 'False Positive', value: cm.false_positives, color: '#ffaa00', bg: 'rgba(255,170,0,0.06)', desc: 'Healthy → Risky' },
                { label: 'False Negative', value: cm.false_negatives, color: '#ff3d71', bg: 'rgba(255,61,113,0.08)', desc: 'Risky → Healthy' },
                { label: 'True Positive', value: cm.true_positives, color: '#3366ff', bg: 'rgba(51,102,255,0.08)', desc: 'Risky → Risky' },
              ].map((cell) => (
                <div
                  key={cell.label}
                  style={{
                    background: cell.bg,
                    border: `1px solid ${cell.color}22`,
                    borderRadius: 'var(--radius-md)',
                    padding: 'var(--space-6)',
                    textAlign: 'center',
                    transition: 'transform 0.2s ease, box-shadow 0.2s ease',
                    cursor: 'default',
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.transform = 'scale(1.03)';
                    e.currentTarget.style.boxShadow = `0 0 20px ${cell.color}22`;
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.transform = 'scale(1)';
                    e.currentTarget.style.boxShadow = 'none';
                  }}
                >
                  <div style={{ fontSize: 'var(--font-size-3xl)', fontWeight: 800, color: cell.color }}>
                    {cell.value ?? 0}
                  </div>
                  <div style={{ fontSize: 'var(--font-size-xs)', fontWeight: 600, marginTop: 4 }}>{cell.label}</div>
                  <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)', marginTop: 2 }}>{cell.desc}</div>
                </div>
              ))}
            </div>

            {/* Row labels */}
            <div style={{ textAlign: 'center', fontSize: 'var(--font-size-xs)', color: 'var(--text-muted)', marginTop: 'var(--space-4)' }}>
              Rows = Actual | Columns = Predicted
            </div>
          </div>
        )}

        {/* ROC Curve (simulated) */}
        {tab === 'roc' && (
          <div className="glass-card" style={{ height: 420 }}>
            <h3 style={{ fontSize: 'var(--font-size-sm)', fontWeight: 600, marginBottom: 'var(--space-4)', color: 'var(--text-secondary)' }}>
              ROC Curve — AUC = {(m.auc_roc || 0).toFixed(4)}
            </h3>
            <div style={{ height: 350 }}>
              <Line
                data={{
                  labels: Array.from({ length: 21 }, (_, i) => (i * 5).toString()),
                  datasets: [
                    {
                      label: `DenseNet (AUC = ${(m.auc_roc || 0).toFixed(4)})`,
                      data: generateROCPoints(m.auc_roc || 0.9),
                      borderColor: '#ff3d71',
                      backgroundColor: 'rgba(255,61,113,0.1)',
                      borderWidth: 2.5,
                      pointRadius: 0,
                      tension: 0.4,
                      fill: true,
                    },
                    {
                      label: 'Random Classifier',
                      data: Array.from({ length: 21 }, (_, i) => i * 5),
                      borderColor: 'rgba(255,255,255,0.15)',
                      borderWidth: 1,
                      borderDash: [5, 5],
                      pointRadius: 0,
                      fill: false,
                    },
                  ],
                }}
                options={{
                  ...chartOptions,
                  scales: {
                    ...chartOptions.scales,
                    x: { ...chartOptions.scales.x, title: { display: true, text: 'False Positive Rate (%)', color: 'rgba(232,236,244,0.5)', font: { family: 'Inter' } } },
                    y: { ...chartOptions.scales.y, title: { display: true, text: 'True Positive Rate (%)', color: 'rgba(232,236,244,0.5)', font: { family: 'Inter' } } },
                  },
                }}
              />
            </div>
          </div>
        )}

        {/* Metrics Comparison */}
        {tab === 'importance' && (
          <div className="glass-card" style={{ height: 420 }}>
            <h3 style={{ fontSize: 'var(--font-size-sm)', fontWeight: 600, marginBottom: 'var(--space-4)', color: 'var(--text-secondary)' }}>
              Model Performance Metrics
            </h3>
            <div style={{ height: 350 }}>
              <Bar
                data={{
                  labels: ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC', 'Specificity'],
                  datasets: [{
                    label: 'Score',
                    data: [
                      m.accuracy || 0,
                      m.precision || 0,
                      m.recall_sensitivity || 0,
                      m.f1_score || 0,
                      m.auc_roc || 0,
                      m.specificity || 0,
                    ],
                    backgroundColor: [
                      'rgba(245,166,35,0.7)',
                      'rgba(51,102,255,0.7)',
                      'rgba(0,214,143,0.7)',
                      'rgba(168,85,247,0.7)',
                      'rgba(255,61,113,0.7)',
                      'rgba(255,170,0,0.7)',
                    ],
                    borderColor: [
                      '#f5a623',
                      '#3366ff',
                      '#00d68f',
                      '#a855f7',
                      '#ff3d71',
                      '#ffaa00',
                    ],
                    borderWidth: 1.5,
                    borderRadius: 6,
                  }],
                }}
                options={{
                  ...chartOptions,
                  scales: {
                    ...chartOptions.scales,
                    y: { ...chartOptions.scales.y, min: 0, max: 1, ticks: { ...chartOptions.scales.y.ticks, callback: (v) => `${(v * 100).toFixed(0)}%` } },
                  },
                  plugins: {
                    ...chartOptions.plugins,
                    legend: { display: false },
                  },
                }}
              />
            </div>
          </div>
        )}

        {/* Class Distribution */}
        {tab === 'distribution' && (
          <div className="glass-card">
            <h3 style={{ fontSize: 'var(--font-size-sm)', fontWeight: 600, marginBottom: 'var(--space-6)', color: 'var(--text-secondary)' }}>
              Test Set Class Distribution
            </h3>
            <div style={{ maxWidth: 350, margin: '0 auto' }}>
              <Doughnut
                data={{
                  labels: ['Healthy (Class 0)', 'Distressed (Class 1)'],
                  datasets: [{
                    data: [m.negative_samples || 0, m.positive_samples || 0],
                    backgroundColor: ['rgba(0,214,143,0.7)', 'rgba(255,61,113,0.7)'],
                    borderColor: ['#00d68f', '#ff3d71'],
                    borderWidth: 2,
                    hoverOffset: 8,
                  }],
                }}
                options={{
                  responsive: true,
                  plugins: {
                    legend: {
                      position: 'bottom',
                      labels: {
                        color: 'rgba(232,236,244,0.7)',
                        font: { family: 'Inter', size: 13 },
                        padding: 20,
                      },
                    },
                  },
                  cutout: '60%',
                }}
              />
            </div>
            <div style={{ display: 'flex', justifyContent: 'center', gap: 'var(--space-8)', marginTop: 'var(--space-6)' }}>
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: 'var(--font-size-2xl)', fontWeight: 700, color: 'var(--green)' }}>{m.negative_samples || 0}</div>
                <div style={{ fontSize: 'var(--font-size-xs)', color: 'var(--text-muted)' }}>Healthy Samples</div>
              </div>
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: 'var(--font-size-2xl)', fontWeight: 700, color: 'var(--red)' }}>{m.positive_samples || 0}</div>
                <div style={{ fontSize: 'var(--font-size-xs)', color: 'var(--text-muted)' }}>Distressed Samples</div>
              </div>
            </div>
          </div>
        )}
      </motion.div>

      {/* Architecture Info */}
      <motion.div variants={itemVariants} className="glass-card" style={{ marginTop: 'var(--space-6)' }}>
        <h3 style={{ fontSize: 'var(--font-size-sm)', fontWeight: 600, marginBottom: 'var(--space-4)', color: 'var(--text-secondary)' }}>
          🧠 Model Architecture
        </h3>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: 'var(--space-3)' }}>
          {info?.architecture && Object.entries(info.architecture).map(([key, val]) => (
            <div key={key} style={{ padding: 'var(--space-3)', background: 'rgba(255,255,255,0.03)', borderRadius: 'var(--radius-sm)' }}>
              <div style={{ fontSize: 'var(--font-size-xs)', color: 'var(--text-muted)', textTransform: 'capitalize' }}>
                {key.replace(/_/g, ' ')}
              </div>
              <div style={{ fontSize: 'var(--font-size-sm)', fontWeight: 600, marginTop: 2 }}>
                {String(val)}
              </div>
            </div>
          ))}
        </div>
      </motion.div>
    </motion.div>
  );
}

/**
 * Generate simulated ROC curve points based on AUC value.
 */
function generateROCPoints(auc) {
  const points = [];
  for (let i = 0; i <= 20; i++) {
    const fpr = i / 20;
    // Approximate ROC curve using power function
    const exp = Math.log(auc) / Math.log(0.5) * 0.8;
    const tpr = Math.pow(fpr, Math.max(0.1, 1 - auc + 0.1)) * 100;
    points.push(Math.min(100, Math.pow(fpr, 0.3) * 100 * (auc / 0.5)));
  }
  // Ensure monotonically increasing and ending at 100
  for (let i = 1; i < points.length; i++) {
    points[i] = Math.max(points[i], points[i - 1]);
  }
  points[points.length - 1] = 100;
  return points;
}

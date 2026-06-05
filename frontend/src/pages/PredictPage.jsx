/**
 * PredictPage — Single prediction form with dynamic feature sliders,
 * risk gauge, and SHAP waterfall chart.
 */
import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ScanLine, Send, RotateCcw, AlertTriangle, CheckCircle2, ChevronDown, ChevronUp } from 'lucide-react';
import RiskGauge from '../components/Prediction/RiskGauge';
import WaterfallChart from '../components/SHAP/WaterfallChart';
import FeatureImportance from '../components/SHAP/FeatureImportance';
import { getModelInfo, predictSingle } from '../api/client';

const containerVariants = {
  hidden: { opacity: 0 },
  visible: { opacity: 1, transition: { staggerChildren: 0.08 } },
};

const itemVariants = {
  hidden: { opacity: 0, y: 20 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.4 } },
};

export default function PredictPage() {
  const [featureNames, setFeatureNames] = useState([]);
  const [medians, setMedians] = useState({});
  const [features, setFeatures] = useState({});
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [expandedGroups, setExpandedGroups] = useState({ 0: true });
  const [includeShap, setIncludeShap] = useState(true);

  // Load feature names from model info
  useEffect(() => {
    getModelInfo().then((res) => {
      const names = res.data.feature_names || [];
      setFeatureNames(names);
      // Initialize features with zeros
      const init = {};
      names.forEach((n) => (init[n] = 0));
      setFeatures(init);
    });
  }, []);

  const handleSliderChange = (name, value) => {
    setFeatures((prev) => ({ ...prev, [name]: parseFloat(value) }));
  };

  const handleReset = () => {
    const init = {};
    featureNames.forEach((n) => (init[n] = 0));
    setFeatures(init);
    setResult(null);
    setError(null);
  };

  const handleRandomFill = () => {
    const rand = {};
    featureNames.forEach((n) => {
      rand[n] = parseFloat((Math.random() * 4 - 1).toFixed(4));
    });
    setFeatures(rand);
  };

  const handleSubmit = async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await predictSingle(features, includeShap);
      setResult(res.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Prediction failed. Is the backend running?');
    } finally {
      setLoading(false);
    }
  };

  // Group features into chunks of 10
  const groupSize = 10;
  const groups = [];
  for (let i = 0; i < featureNames.length; i += groupSize) {
    groups.push(featureNames.slice(i, i + groupSize));
  }

  const toggleGroup = (idx) => {
    setExpandedGroups((prev) => ({ ...prev, [idx]: !prev[idx] }));
  };

  return (
    <motion.div variants={containerVariants} initial="hidden" animate="visible">
      <motion.div className="page-header" variants={itemVariants}>
        <h1 style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <ScanLine size={32} style={{ color: 'var(--blue)' }} />
          Single Prediction
        </h1>
        <p>Enter financial features to predict loan default risk</p>
      </motion.div>

      {/* Controls */}
      <motion.div
        variants={itemVariants}
        style={{ display: 'flex', gap: 'var(--space-3)', marginBottom: 'var(--space-6)', flexWrap: 'wrap', alignItems: 'center' }}
      >
        <button className="btn btn-secondary" onClick={handleRandomFill}>
          🎲 Random Fill
        </button>
        <button className="btn btn-secondary" onClick={handleReset}>
          <RotateCcw size={14} /> Reset All
        </button>
        <label style={{ display: 'flex', alignItems: 'center', gap: 8, marginLeft: 'auto', fontSize: 'var(--font-size-sm)', color: 'var(--text-secondary)', cursor: 'pointer' }}>
          <input
            type="checkbox"
            checked={includeShap}
            onChange={(e) => setIncludeShap(e.target.checked)}
            style={{ accentColor: 'var(--blue)' }}
          />
          Include SHAP Explanation
        </label>
      </motion.div>

      {/* Feature Groups */}
      <motion.div variants={itemVariants} style={{ marginBottom: 'var(--space-6)' }}>
        {groups.map((group, gIdx) => (
          <div key={gIdx} className="glass-card" style={{ marginBottom: 'var(--space-3)', padding: 0 }}>
            <button
              onClick={() => toggleGroup(gIdx)}
              style={{
                display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                width: '100%', padding: 'var(--space-4) var(--space-6)',
                background: 'none', border: 'none', color: 'var(--text-primary)',
                cursor: 'pointer', fontFamily: 'var(--font-family)', fontSize: 'var(--font-size-sm)',
                fontWeight: 600,
              }}
            >
              <span>Features {group[0]} — {group[group.length - 1]}</span>
              {expandedGroups[gIdx] ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
            </button>

            <AnimatePresence>
              {expandedGroups[gIdx] && (
                <motion.div
                  initial={{ height: 0, opacity: 0 }}
                  animate={{ height: 'auto', opacity: 1 }}
                  exit={{ height: 0, opacity: 0 }}
                  transition={{ duration: 0.25 }}
                  style={{ overflow: 'hidden' }}
                >
                  <div className="feature-grid" style={{ padding: '0 var(--space-6) var(--space-5)' }}>
                    {group.map((name) => (
                      <div key={name} className="slider-container">
                        <div className="slider-header">
                          <label className="form-label" style={{ marginBottom: 0 }}>{name}</label>
                          <span className="slider-value">{(features[name] || 0).toFixed(3)}</span>
                        </div>
                        <input
                          type="range"
                          min="-5"
                          max="50"
                          step="0.01"
                          value={features[name] || 0}
                          onChange={(e) => handleSliderChange(name, e.target.value)}
                        />
                        <input
                          type="number"
                          className="form-input"
                          value={features[name] || 0}
                          onChange={(e) => handleSliderChange(name, e.target.value)}
                          step="0.01"
                          style={{ marginTop: 4, padding: '6px 10px', fontSize: 'var(--font-size-xs)' }}
                        />
                      </div>
                    ))}
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        ))}
      </motion.div>

      {/* Submit */}
      <motion.div variants={itemVariants} style={{ textAlign: 'center', marginBottom: 'var(--space-8)' }}>
        <button
          className="btn btn-primary btn-lg"
          onClick={handleSubmit}
          disabled={loading}
          style={{ minWidth: 220 }}
        >
          {loading ? (
            <>
              <div className="spinner" />
              Predicting…
            </>
          ) : (
            <>
              <Send size={18} />
              Run Prediction
            </>
          )}
        </button>
      </motion.div>

      {/* Error */}
      {error && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="glass-card glass-card--red"
          style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 'var(--space-6)' }}
        >
          <AlertTriangle size={20} style={{ color: 'var(--red)' }} />
          <span style={{ color: 'var(--red)' }}>{error}</span>
        </motion.div>
      )}

      {/* Result */}
      <AnimatePresence>
        {result && (
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 30 }}
            transition={{ duration: 0.5 }}
          >
            {/* Result Header */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 'var(--space-6)', marginBottom: 'var(--space-6)' }}>
              {/* Gauge */}
              <div className={`glass-card ${result.probability_of_default >= 0.5 ? 'glass-card--red' : 'glass-card--green'}`} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
                <RiskGauge value={result.probability_of_default * 100} size={240} />
              </div>

              {/* Decision Card */}
              <div className={`glass-card ${result.probability_of_default >= 0.5 ? 'glass-card--red' : 'glass-card--green'}`}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 'var(--space-6)' }}>
                  {result.prediction === 1 ? (
                    <AlertTriangle size={28} style={{ color: 'var(--red)' }} />
                  ) : (
                    <CheckCircle2 size={28} style={{ color: 'var(--green)' }} />
                  )}
                  <h2 style={{ fontSize: 'var(--font-size-xl)', fontWeight: 700 }}>
                    {result.decision}
                  </h2>
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 'var(--space-4)' }}>
                  <div style={{ padding: 'var(--space-4)', background: 'rgba(255,255,255,0.03)', borderRadius: 'var(--radius-md)' }}>
                    <div style={{ fontSize: 'var(--font-size-xs)', color: 'var(--text-muted)' }}>Default Probability</div>
                    <div style={{
                      fontSize: 'var(--font-size-2xl)', fontWeight: 700,
                      color: result.probability_of_default >= 0.5 ? 'var(--red)' : 'var(--green)',
                    }}>
                      {(result.probability_of_default * 100).toFixed(1)}%
                    </div>
                  </div>
                  <div style={{ padding: 'var(--space-4)', background: 'rgba(255,255,255,0.03)', borderRadius: 'var(--radius-md)' }}>
                    <div style={{ fontSize: 'var(--font-size-xs)', color: 'var(--text-muted)' }}>Confidence</div>
                    <div style={{ fontSize: 'var(--font-size-2xl)', fontWeight: 700, color: 'var(--gold)' }}>
                      {result.confidence}%
                    </div>
                  </div>
                </div>

                {result.explanation?.summary && (
                  <p style={{
                    marginTop: 'var(--space-4)', fontSize: 'var(--font-size-sm)',
                    color: 'var(--text-secondary)', lineHeight: 1.6,
                    padding: 'var(--space-3)', background: 'rgba(255,255,255,0.02)',
                    borderRadius: 'var(--radius-sm)', borderLeft: '3px solid var(--blue)',
                  }}>
                    {result.explanation.summary}
                  </p>
                )}
              </div>
            </div>

            {/* SHAP Explanation */}
            {result.explanation && (
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 'var(--space-6)' }}>
                <div className="glass-card">
                  <h3 style={{ fontSize: 'var(--font-size-sm)', fontWeight: 600, marginBottom: 'var(--space-4)', color: 'var(--text-secondary)' }}>
                    SHAP Waterfall — Feature Contributions
                  </h3>
                  <WaterfallChart
                    baseValue={result.explanation.base_probability}
                    factors={[
                      ...result.explanation.top_factors_toward_default,
                      ...result.explanation.top_factors_toward_healthy,
                    ]}
                    maxFeatures={10}
                  />
                </div>

                <div className="glass-card">
                  <FeatureImportance
                    factors={[
                      ...result.explanation.top_factors_toward_default,
                      ...result.explanation.top_factors_toward_healthy,
                    ]}
                    maxFeatures={10}
                    title="Top Contributing Features"
                  />
                </div>
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}

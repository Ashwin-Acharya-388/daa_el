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

/**
 * Human-readable labels and descriptions for the 25 SFFS-selected features.
 * Keys match the raw feature codes returned by the backend model info endpoint.
 */
const FEATURE_DETAILS = {
  // ── Taiwanese Bankruptcy Dataset Features ──
  tw_f1:  { label: 'ROA (C) — Pre-Interest & Depreciation',    description: 'Operational profitability before interest, tax, and depreciation' },
  tw_f2:  { label: 'ROA (A) — Pre-Interest & Post-Tax',         description: 'Profitability relative to assets, post-tax but pre-interest' },
  tw_f3:  { label: 'ROA (B) — Pre-Interest & Depr. Post-Tax',   description: 'Return on assets post-tax, pre-interest and depreciation' },
  tw_f18: { label: 'Net Value Per Share',                       description: 'Book value of equity divided by outstanding shares' },
  tw_f19: { label: 'Persistent EPS (Last 4 Seasons)',           description: 'Trailing Earnings Per Share; consistent earning capacity' },
  tw_f37: { label: 'Debt Ratio %',                              description: 'Total liabilities relative to total assets (leverage scale)' },
  tw_f43: { label: 'Net Profit Before Tax / Paid-in Capital',   description: 'Pre-tax efficiency of shareholder-contributed capital' },
  tw_f50: { label: 'Net Worth Turnover Rate',                   description: 'How effectively net worth/equity generates sales revenue' },
  tw_f54: { label: 'Working Capital / Total Assets',            description: 'Liquidity ratio — capability to cover short-term debt' },
  tw_f57: { label: 'Cash / Total Assets',                       description: 'Proportion of assets held in cash or cash equivalents' },
  tw_f74: { label: 'Cash Turnover Rate',                        description: 'Speed with which cash cycles through operations' },
  tw_f81: { label: 'Cash Flow to Liability',                    description: 'Operating cash flow relative to total liabilities (solvency)' },
  tw_f86: { label: 'Net Income / Total Assets (ROA)',           description: 'Standard overall Return on Assets profitability metric' },
  tw_f91: { label: 'Liability to Equity',                       description: 'Debt-to-Equity ratio — total liabilities vs. shareholder equity' },

  // ── Financial Distress Dataset Features ──
  fd_x3:  { label: 'Non-Current Assets Ratio',                  description: 'Asset composition indicator' },
  fd_x5:  { label: 'Quick Assets Ratio',                        description: 'Acid-test short-term liquidity ratio' },
  fd_x10: { label: 'Operating Margin Scale',                    description: 'Operating efficiency indicator' },
  fd_x11: { label: 'Working Capital Scale',                     description: 'Working capital relative to sales volume' },
  fd_x13: { label: 'Financial Distress Index',                  description: 'Distress indicator based on recent financial performance' },
  fd_x15: { label: 'Current Liabilities Ratio',                 description: 'Short-term debt concentration indicator' },
  fd_x40: { label: 'Retained Earnings / Total Assets',          description: 'Cumulative profitability and capital strength indicator' },
  fd_x57: { label: 'Interest Expense / Total Assets',           description: 'Borrowing cost ratio' },
  fd_x65: { label: 'Sales Growth / Total Assets',               description: 'Revenue expansion speed indicator' },
  fd_x66: { label: 'Asset Turnover Scale',                      description: 'Asset utilisation efficiency metric' },
  fd_x80: { label: 'Industry Sector Code',                      description: 'Categorical code representing the borrower\'s industry sector', isCategorical: true },
};

/** Look up display info for a feature; fall back to raw name. */
const getFeatureInfo = (name) => FEATURE_DETAILS[name] || { label: name, description: '' };

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
  const [trainingStds, setTrainingStds] = useState({});
  const [trainingMins, setTrainingMins] = useState({});
  const [trainingMaxs, setTrainingMaxs] = useState({});
  const [features, setFeatures] = useState({});
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [expandedGroups, setExpandedGroups] = useState({ 0: true });
  const [includeShap, setIncludeShap] = useState(true);

  // Load feature names and medians from model info
  useEffect(() => {
    getModelInfo()
      .then((res) => {
        const names = res.data.feature_names || [];
        const meds = res.data.training_medians || {};
        const stds = res.data.training_stds || {};
        const mins = res.data.training_mins || {};
        const maxs = res.data.training_maxs || {};
        setFeatureNames(names);
        setMedians(meds);
        setTrainingStds(stds);
        setTrainingMins(mins);
        setTrainingMaxs(maxs);
        // Initialize features with training medians (realistic starting point)
        const init = {};
        names.forEach((n) => (init[n] = meds[n] !== undefined ? meds[n] : 0));
        setFeatures(init);
      })
      .catch((err) => {
        setError('Failed to load model info. Is the backend running?');
      });
  }, []);

  const handleSliderChange = (name, value) => {
    setFeatures((prev) => ({ ...prev, [name]: parseFloat(value) || 0 }));
  };

  const handleReset = () => {
    const init = {};
    featureNames.forEach((n) => (init[n] = medians[n] !== undefined ? medians[n] : 0));
    setFeatures(init);
    setResult(null);
    setError(null);
  };

  const handleRandomFill = () => {
    const rand = {};
    featureNames.forEach((n) => {
      // Generate random values within ±1.5 standard deviations of the median
      // This keeps values within the training distribution
      const median = medians[n] || 0;
      const std = trainingStds[n] || Math.abs(median) * 0.05 || 0.01;
      rand[n] = parseFloat((median + (Math.random() - 0.5) * 2 * 1.5 * std).toFixed(6));
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

  // Compute dynamic slider range for each feature based on its observed training min and max
  const getSliderRange = (name) => {
    const median = medians[name] || 0;
    
    // Use training mins/maxs, fallback to ±3 standard deviations (or ±10% relative fallback)
    const std = trainingStds[name] || Math.abs(median) * 0.05 || 0.01;
    const fallbackMin = median - 3 * std;
    const fallbackMax = median + 3 * std;
    
    let minVal = trainingMins[name] !== undefined ? trainingMins[name] : fallbackMin;
    let maxVal = trainingMaxs[name] !== undefined ? trainingMaxs[name] : fallbackMax;

    // Safety check: ensure min <= max
    if (minVal > maxVal) {
      const temp = minVal;
      minVal = maxVal;
      maxVal = temp;
    }
    
    const range = maxVal - minVal;
    
    // Choose sensible steps based on range order of magnitude
    let step = 0.001;
    if (range > 1000) step = 10.0;
    else if (range > 100) step = 1.0;
    else if (range > 10) step = 0.1;
    else if (range > 1) step = 0.01;
    else if (range > 0.1) step = 0.001;
    else step = 0.00001;

    return { min: minVal, max: maxVal, step: step };
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
        <p>Enter financial features to predict loan default risk ({featureNames.length} SFFS-selected features)</p>
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
                textAlign: 'left',
                cursor: 'pointer', fontFamily: 'var(--font-family)', fontSize: 'var(--font-size-sm)',
                fontWeight: 600,
              }}
            >
              <span>{getFeatureInfo(group[0]).label} — {getFeatureInfo(group[group.length - 1]).label}</span>
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
                    {group.map((name) => {
                      const range = getSliderRange(name);
                      const info = getFeatureInfo(name);
                      const isCategorical = info.isCategorical;
                      return (
                        <div key={name} className="slider-container">
                          <div className="slider-header">
                            <label className="form-label" style={{ marginBottom: 0 }} title={info.description}>
                              {info.label}
                              <span style={{ fontSize: '0.65rem', color: 'var(--text-muted)', fontWeight: 400, marginLeft: 6 }}>({name})</span>
                            </label>
                            <span className="slider-value">{isCategorical ? Math.round(features[name] || 0) : (features[name] || 0).toFixed(3)}</span>
                          </div>
                          {info.description && (
                            <div style={{ fontSize: '0.6rem', color: 'var(--text-muted)', marginBottom: 4, lineHeight: 1.3 }}>
                              {info.description}
                            </div>
                          )}
                          {isCategorical ? (
                            /* Categorical input — integer number field instead of a slider */
                            <input
                              type="number"
                              className="form-input"
                              value={Math.round(features[name] || 0)}
                              onChange={(e) => handleSliderChange(name, Math.round(parseFloat(e.target.value) || 0))}
                              step={1}
                              min={0}
                              style={{ padding: '6px 10px', fontSize: 'var(--font-size-xs)' }}
                            />
                          ) : (
                            /* Continuous input — slider + number field */
                            <>
                              <input
                                type="range"
                                min={range.min}
                                max={range.max}
                                step={range.step}
                                value={features[name] || 0}
                                onChange={(e) => handleSliderChange(name, e.target.value)}
                              />
                              <input
                                type="number"
                                className="form-input"
                                value={features[name] || 0}
                                onChange={(e) => handleSliderChange(name, e.target.value)}
                                step={range.step}
                                style={{ marginTop: 4, padding: '6px 10px', fontSize: 'var(--font-size-xs)' }}
                              />
                            </>
                          )}
                          {medians[name] !== undefined && (
                            <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)', marginTop: 2 }}>
                              Median: {isCategorical ? Math.round(medians[name]) : medians[name].toFixed(4)}
                            </div>
                          )}
                        </div>
                      );
                    })}
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

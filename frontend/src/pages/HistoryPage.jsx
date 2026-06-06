/**
 * HistoryPage — Paginated prediction history with filters and detail view.
 */
import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Clock,
  Search,
  Download,
  ChevronLeft,
  ChevronRight,
  X,
  Eye,
} from 'lucide-react';
import { getHistory, getPredictionDetail, exportPredictions } from '../api/client';
import { formatDate, downloadBlob } from '../utils/formatters';
import WaterfallChart from '../components/SHAP/WaterfallChart';

const itemVariants = {
  hidden: { opacity: 0, y: 20 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.4 } },
};

export default function HistoryPage() {
  const [history, setHistory] = useState(null);
  const [loading, setLoading] = useState(true);
  const [page, setPage] = useState(1);
  const [riskFilter, setRiskFilter] = useState('');
  const [selectedId, setSelectedId] = useState(null);
  const [detail, setDetail] = useState(null);
  const [detailLoading, setDetailLoading] = useState(false);

  const fetchHistory = async () => {
    setLoading(true);
    try {
      const params = { page, per_page: 15 };
      if (riskFilter) params.risk_level = riskFilter;
      const res = await getHistory(params);
      setHistory(res.data);
    } catch {
      // If backend is down, show empty state
      setHistory({ items: [], total: 0, page: 1, per_page: 15, total_pages: 0 });
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchHistory();
  }, [page, riskFilter]);

  const handleViewDetail = async (id) => {
    if (selectedId === id) {
      setSelectedId(null);
      setDetail(null);
      return;
    }
    setSelectedId(id);
    setDetailLoading(true);
    try {
      const res = await getPredictionDetail(id);
      setDetail(res.data);
    } catch {
      setDetail(null);
    } finally {
      setDetailLoading(false);
    }
  };

  const handleExport = async () => {
    try {
      const params = {};
      if (riskFilter) params.risk_level = riskFilter;
      const res = await exportPredictions(params);
      downloadBlob(res.data, 'predictions_export.csv');
    } catch {
      // Silently fail
    }
  };

  const items = history?.items || [];
  const totalPages = history?.total_pages || 0;

  return (
    <motion.div initial="hidden" animate="visible" variants={{ hidden: { opacity: 0 }, visible: { opacity: 1, transition: { staggerChildren: 0.1 } } }}>
      <motion.div className="page-header" variants={itemVariants}>
        <h1 style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <Clock size={32} style={{ color: 'var(--gold)' }} />
          Prediction History
        </h1>
        <p>Browse, filter, and export past predictions</p>
      </motion.div>

      {/* Filters */}
      <motion.div
        variants={itemVariants}
        className="glass-card"
        style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-4)', marginBottom: 'var(--space-6)', padding: 'var(--space-4) var(--space-6)', flexWrap: 'wrap' }}
      >
        <Search size={16} style={{ color: 'var(--text-muted)' }} />

        <select
          className="form-input"
          value={riskFilter}
          onChange={(e) => { setRiskFilter(e.target.value); setPage(1); }}
          style={{ width: 'auto', minWidth: 150 }}
        >
          <option value="">All Risk Levels</option>
          <option value="high">High Risk Only</option>
          <option value="low">Low Risk Only</option>
        </select>

        <div style={{ marginLeft: 'auto', display: 'flex', gap: 8 }}>
          <button className="btn btn-secondary btn-sm" onClick={handleExport}>
            <Download size={14} /> Export CSV
          </button>
        </div>
      </motion.div>

      {/* Table */}
      <motion.div variants={itemVariants} className="glass-card" style={{ padding: 0, overflow: 'hidden' }}>
        {loading ? (
          <div style={{ padding: 'var(--space-12)', textAlign: 'center' }}>
            <div className="spinner-lg" style={{ margin: '0 auto', borderTopColor: 'var(--blue)' }} />
            <p style={{ marginTop: 'var(--space-4)', color: 'var(--text-muted)', fontSize: 'var(--font-size-sm)' }}>Loading predictions…</p>
          </div>
        ) : items.length === 0 ? (
          <div className="empty-state">
            <Clock size={48} style={{ color: 'var(--text-muted)', marginBottom: 'var(--space-4)' }} />
            <h3>No predictions yet</h3>
            <p style={{ fontSize: 'var(--font-size-sm)' }}>Make your first prediction to see it here</p>
          </div>
        ) : (
          <>
            <table className="data-table">
              <thead>
                <tr>
                  <th>Date</th>
                  <th>Probability</th>
                  <th>Decision</th>
                  <th>Confidence</th>
                  <th>SHAP</th>
                  <th style={{ width: 60 }}>View</th>
                </tr>
              </thead>
              <tbody>
                {items.map((item) => (
                  <React.Fragment key={item.id}>
                    <tr
                      className="clickable"
                      onClick={() => handleViewDetail(item.id)}
                      style={{ background: selectedId === item.id ? 'rgba(51,102,255,0.05)' : undefined }}
                    >
                      <td style={{ fontSize: 'var(--font-size-xs)' }}>{formatDate(item.created_at)}</td>
                      <td>
                        <span style={{
                          fontWeight: 700,
                          color: item.probability >= 0.5 ? 'var(--red)' : 'var(--green)',
                        }}>
                          {(item.probability * 100).toFixed(1)}%
                        </span>
                      </td>
                      <td>
                        <span className={`badge ${item.prediction === 1 ? 'badge-danger' : 'badge-success'}`}>
                          {item.prediction === 1 ? 'Default' : 'Approved'}
                        </span>
                      </td>
                      <td style={{ fontWeight: 600 }}>{item.confidence?.toFixed(1)}%</td>
                      <td>
                        {item.has_shap ? (
                          <span className="badge badge-info">Available</span>
                        ) : (
                          <span style={{ color: 'var(--text-muted)', fontSize: 'var(--font-size-xs)' }}>—</span>
                        )}
                      </td>
                      <td>
                        <button
                          className="btn btn-secondary btn-sm"
                          onClick={(e) => { e.stopPropagation(); handleViewDetail(item.id); }}
                          style={{ padding: '4px 8px' }}
                        >
                          <Eye size={14} />
                        </button>
                      </td>
                    </tr>

                    {/* Detail expansion */}
                    {selectedId === item.id && (
                      <tr>
                        <td colSpan={6} style={{ padding: 0 }}>
                          <AnimatePresence>
                            <motion.div
                              initial={{ height: 0, opacity: 0 }}
                              animate={{ height: 'auto', opacity: 1 }}
                              exit={{ height: 0, opacity: 0 }}
                              transition={{ duration: 0.3 }}
                              style={{ overflow: 'hidden', background: 'rgba(51,102,255,0.03)', padding: 'var(--space-6)' }}
                            >
                              {detailLoading ? (
                                <div style={{ textAlign: 'center', padding: 'var(--space-4)' }}>
                                  <div className="spinner" style={{ margin: '0 auto', borderTopColor: 'var(--blue)' }} />
                                </div>
                              ) : detail?.shap_json ? (
                                <div>
                                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 'var(--space-4)' }}>
                                    <h4 style={{ fontSize: 'var(--font-size-sm)', fontWeight: 600 }}>SHAP Explanation</h4>
                                    <button className="btn btn-secondary btn-sm" onClick={() => { setSelectedId(null); setDetail(null); }}>
                                      <X size={12} /> Close
                                    </button>
                                  </div>
                                  {detail.shap_json.summary && (
                                    <p style={{ fontSize: 'var(--font-size-sm)', color: 'var(--text-secondary)', marginBottom: 'var(--space-4)', borderLeft: '3px solid var(--blue)', paddingLeft: 12 }}>
                                      {detail.shap_json.summary}
                                    </p>
                                  )}
                                  <WaterfallChart
                                    baseValue={detail.shap_json.base_probability}
                                    factors={[
                                      ...(detail.shap_json.top_factors_toward_default || []),
                                      ...(detail.shap_json.top_factors_toward_healthy || []),
                                    ]}
                                  />
                                </div>
                              ) : (
                                <p style={{ color: 'var(--text-muted)', fontSize: 'var(--font-size-sm)' }}>No SHAP data for this prediction</p>
                              )}
                            </motion.div>
                          </AnimatePresence>
                        </td>
                      </tr>
                    )}
                  </React.Fragment>
                ))}
              </tbody>
            </table>

            {/* Pagination */}
            {totalPages > 1 && (
              <div className="pagination" style={{ padding: 'var(--space-4)' }}>
                <button className="pagination-btn" disabled={page <= 1} onClick={() => setPage(page - 1)}>
                  <ChevronLeft size={16} />
                </button>
                {Array.from({ length: Math.min(totalPages, 7) }, (_, i) => {
                  const p = i + 1;
                  return (
                    <button
                      key={p}
                      className={`pagination-btn ${page === p ? 'active' : ''}`}
                      onClick={() => setPage(p)}
                    >
                      {p}
                    </button>
                  );
                })}
                {totalPages > 7 && <span style={{ color: 'var(--text-muted)' }}>…</span>}
                <button className="pagination-btn" disabled={page >= totalPages} onClick={() => setPage(page + 1)}>
                  <ChevronRight size={16} />
                </button>
              </div>
            )}
          </>
        )}
      </motion.div>
    </motion.div>
  );
}

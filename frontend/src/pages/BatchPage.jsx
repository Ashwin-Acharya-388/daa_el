/**
 * BatchPage — CSV upload with drag-and-drop, progress tracking via WebSocket
 * with polling fallback, file size validation, and results preview table.
 */
import { useState, useRef, useCallback, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  FileSpreadsheet,
  Upload,
  Download,
  CheckCircle2,
  AlertTriangle,
  FileText,
  X,
} from 'lucide-react';
import { uploadBatch, getBatchStatus, downloadBatchResult } from '../api/client';
import { useWebSocket } from '../hooks/useWebSocket';
import { downloadBlob } from '../utils/formatters';

const MAX_FILE_SIZE_MB = 50;
const MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024;

const itemVariants = {
  hidden: { opacity: 0, y: 20 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.4 } },
};

export default function BatchPage() {
  const [file, setFile] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [taskId, setTaskId] = useState(null);
  const [status, setStatus] = useState(null);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);
  const pollIntervalRef = useRef(null);

  // WebSocket for real-time progress
  const { lastMessage, isConnected } = useWebSocket(taskId);

  // Update status from WebSocket messages
  const progress = lastMessage || status;
  const progressPercent = progress
    ? progress.total > 0
      ? Math.round((progress.processed / progress.total) * 100)
      : 0
    : 0;

  // Polling fallback when WebSocket is not connected
  useEffect(() => {
    if (!taskId || isConnected) {
      // Clear polling if WebSocket is working or no task
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
        pollIntervalRef.current = null;
      }
      return;
    }

    // WebSocket not connected — poll via HTTP
    const isTerminal = progress?.status === 'COMPLETED' || progress?.status === 'FAILED';
    if (isTerminal) return;

    pollIntervalRef.current = setInterval(async () => {
      try {
        const res = await getBatchStatus(taskId);
        const data = res.data;
        setStatus({
          status: data.status,
          processed: data.processed_rows,
          total: data.total_rows,
          error: data.error_message,
        });
        if (data.status === 'COMPLETED' || data.status === 'FAILED') {
          clearInterval(pollIntervalRef.current);
          pollIntervalRef.current = null;
        }
      } catch {
        // Ignore polling errors silently
      }
    }, 1000);

    return () => {
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
        pollIntervalRef.current = null;
      }
    };
  }, [taskId, isConnected, progress?.status]);

  const validateFile = (f) => {
    if (!f) return 'No file selected';
    if (!f.name.endsWith('.csv')) return 'Please upload a .csv file';
    if (f.size > MAX_FILE_SIZE_BYTES) {
      return `File too large (${(f.size / 1024 / 1024).toFixed(1)} MB). Maximum: ${MAX_FILE_SIZE_MB} MB`;
    }
    return null;
  };

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setDragOver(false);
    const dropped = e.dataTransfer.files[0];
    const err = validateFile(dropped);
    if (err) {
      setError(err);
    } else {
      setFile(dropped);
      setError(null);
    }
  }, []);

  const handleFileSelect = (e) => {
    const selected = e.target.files[0];
    if (selected) {
      const err = validateFile(selected);
      if (err) {
        setError(err);
      } else {
        setFile(selected);
        setError(null);
      }
    }
  };

  const handleUpload = async () => {
    if (!file) return;
    setUploading(true);
    setError(null);
    try {
      const res = await uploadBatch(file);
      setTaskId(res.data.task_id);
      setStatus({
        status: 'PENDING',
        processed: 0,
        total: res.data.total_rows,
      });
    } catch (err) {
      setError(err.response?.data?.detail || 'Upload failed');
    } finally {
      setUploading(false);
    }
  };

  const handleDownload = async () => {
    if (!taskId) return;
    try {
      const res = await downloadBatchResult(taskId);
      downloadBlob(res.data, `predictions_${file?.name || 'result'}.csv`);
    } catch (err) {
      setError('Download failed');
    }
  };

  const handleReset = () => {
    setFile(null);
    setTaskId(null);
    setStatus(null);
    setError(null);
    if (pollIntervalRef.current) {
      clearInterval(pollIntervalRef.current);
      pollIntervalRef.current = null;
    }
  };

  const isComplete = progress?.status === 'COMPLETED';
  const isFailed = progress?.status === 'FAILED';
  const isProcessing = progress?.status === 'PROCESSING' || progress?.status === 'PENDING';

  return (
    <motion.div initial="hidden" animate="visible" variants={{ hidden: { opacity: 0 }, visible: { opacity: 1, transition: { staggerChildren: 0.1 } } }}>
      <motion.div className="page-header" variants={itemVariants}>
        <h1 style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <FileSpreadsheet size={32} style={{ color: 'var(--green)' }} />
          Batch Prediction
        </h1>
        <p>Upload a CSV file to predict default risk for multiple applications</p>
      </motion.div>

      {/* Upload Zone */}
      {!taskId && (
        <motion.div variants={itemVariants}>
          <div
            className={`file-upload-zone ${dragOver ? 'dragover' : ''}`}
            onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
            onDragLeave={() => setDragOver(false)}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".csv"
              onChange={handleFileSelect}
              style={{ display: 'none' }}
            />
            <Upload size={48} className="file-upload-icon" />
            <p className="file-upload-text">
              <strong>Drag & drop</strong> your CSV file here, or <strong>click to browse</strong>
            </p>
            <p style={{ fontSize: 'var(--font-size-xs)', color: 'var(--text-muted)', marginTop: 8 }}>
              Supported format: .csv — Max {MAX_FILE_SIZE_MB} MB — Must contain feature columns matching the model
            </p>
          </div>

          {/* Selected file */}
          <AnimatePresence>
            {file && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
                className="glass-card"
                style={{ marginTop: 'var(--space-4)', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                  <FileText size={20} style={{ color: 'var(--blue)' }} />
                  <div>
                    <div style={{ fontWeight: 600, fontSize: 'var(--font-size-sm)' }}>{file.name}</div>
                    <div style={{ fontSize: 'var(--font-size-xs)', color: 'var(--text-muted)' }}>
                      {(file.size / 1024).toFixed(1)} KB
                    </div>
                  </div>
                </div>
                <div style={{ display: 'flex', gap: 8 }}>
                  <button className="btn btn-secondary btn-sm" onClick={(e) => { e.stopPropagation(); setFile(null); }}>
                    <X size={14} /> Remove
                  </button>
                  <button className="btn btn-primary" onClick={handleUpload} disabled={uploading}>
                    {uploading ? <><div className="spinner" /> Uploading…</> : <><Upload size={14} /> Start Prediction</>}
                  </button>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </motion.div>
      )}

      {/* Processing Status */}
      {taskId && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <div className={`glass-card ${isComplete ? 'glass-card--green' : isFailed ? 'glass-card--red' : 'glass-card--blue'}`}>
            {/* Status Header */}
            <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 'var(--space-6)' }}>
              {isComplete ? (
                <CheckCircle2 size={24} style={{ color: 'var(--green)' }} />
              ) : isFailed ? (
                <AlertTriangle size={24} style={{ color: 'var(--red)' }} />
              ) : (
                <div className="spinner-lg" style={{ borderTopColor: 'var(--blue)' }} />
              )}
              <div>
                <h3 style={{ fontWeight: 700, fontSize: 'var(--font-size-lg)' }}>
                  {isComplete ? 'Batch Complete!' : isFailed ? 'Processing Failed' : 'Processing…'}
                </h3>
                <p style={{ fontSize: 'var(--font-size-xs)', color: 'var(--text-muted)' }}>
                  {file?.name} • {progress?.total || 0} rows
                  {!isConnected && isProcessing && ' • Using HTTP polling (WebSocket unavailable)'}
                </p>
              </div>
            </div>

            {/* Progress Bar */}
            {!isFailed && (
              <div style={{ marginBottom: 'var(--space-4)' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6, fontSize: 'var(--font-size-xs)', color: 'var(--text-secondary)' }}>
                  <span>{progress?.processed || 0} / {progress?.total || 0} rows</span>
                  <span style={{ fontWeight: 700, color: isComplete ? 'var(--green)' : 'var(--blue)' }}>{progressPercent}%</span>
                </div>
                <div className="progress-bar">
                  <div className="progress-bar-fill" style={{ width: `${progressPercent}%`, background: isComplete ? 'var(--gradient-green)' : 'var(--gradient-blue)' }} />
                </div>
              </div>
            )}

            {/* Error message */}
            {isFailed && progress?.error && (
              <p style={{ color: 'var(--red)', fontSize: 'var(--font-size-sm)', marginBottom: 'var(--space-4)' }}>
                {progress.error}
              </p>
            )}

            {/* Actions */}
            <div style={{ display: 'flex', gap: 8 }}>
              {isComplete && (
                <button className="btn btn-gold" onClick={handleDownload}>
                  <Download size={16} /> Download Results CSV
                </button>
              )}
              <button className="btn btn-secondary" onClick={handleReset}>
                {isComplete ? 'New Batch' : 'Cancel'}
              </button>
            </div>
          </div>
        </motion.div>
      )}

      {/* Error */}
      {error && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="glass-card glass-card--red"
          style={{ marginTop: 'var(--space-4)', display: 'flex', alignItems: 'center', gap: 12 }}
        >
          <AlertTriangle size={18} style={{ color: 'var(--red)' }} />
          <span style={{ color: 'var(--red)', fontSize: 'var(--font-size-sm)' }}>{error}</span>
        </motion.div>
      )}

      {/* Instructions */}
      <motion.div variants={itemVariants} className="glass-card" style={{ marginTop: 'var(--space-8)' }}>
        <h3 style={{ fontSize: 'var(--font-size-sm)', fontWeight: 600, marginBottom: 'var(--space-4)', color: 'var(--text-secondary)' }}>
          📋 CSV Format Guide
        </h3>
        <div style={{ fontSize: 'var(--font-size-sm)', color: 'var(--text-secondary)', lineHeight: 1.8 }}>
          <p>Your CSV file should contain columns matching the model's SFFS-selected features (e.g., fd_x13, tw_f19, tw_f37, …).</p>
          <ul style={{ paddingLeft: '1.5rem', marginTop: 8 }}>
            <li>Missing features will be filled with training median values</li>
            <li>Extra columns are preserved in the output</li>
            <li>Maximum file size: {MAX_FILE_SIZE_MB} MB</li>
            <li>Output adds: <code style={{ color: 'var(--blue)', background: 'rgba(51,102,255,0.1)', padding: '1px 6px', borderRadius: 4 }}>default_probability</code>, <code style={{ color: 'var(--blue)', background: 'rgba(51,102,255,0.1)', padding: '1px 6px', borderRadius: 4 }}>predicted_class</code>, <code style={{ color: 'var(--blue)', background: 'rgba(51,102,255,0.1)', padding: '1px 6px', borderRadius: 4 }}>decision</code></li>
          </ul>
        </div>
      </motion.div>
    </motion.div>
  );
}

/**
 * App.jsx — Root application component with routing.
 */
import { Routes, Route } from 'react-router-dom';
import Layout from './components/Layout/Layout';
import PredictPage from './pages/PredictPage';
import HistoryPage from './pages/HistoryPage';
import InsightsPage from './pages/InsightsPage';

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route path="/" element={<PredictPage />} />
        <Route path="/predict" element={<PredictPage />} />
        <Route path="/history" element={<HistoryPage />} />
        <Route path="/insights" element={<InsightsPage />} />
      </Route>
    </Routes>
  );
}

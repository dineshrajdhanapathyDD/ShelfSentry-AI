import { Route, Routes } from 'react-router-dom';
import { Layout } from './components/Layout';
import { InspectionPage } from './pages/InspectionPage';
import { ResultsPage } from './pages/ResultsPage';
import { HistoryPage } from './pages/HistoryPage';
import { EvaluationPage } from './pages/EvaluationPage';
import { NotFoundPage } from './pages/NotFoundPage';

/**
 * Application routing (task 13.1). The shell {@link Layout} wraps four
 * placeholder routes - Inspection, Results, History, Evaluation - that later
 * tasks (13.2-13.5) fill in. `/results` and `/results/:inspectionId` both map to
 * the Results page so a specific inspection can be deep-linked later.
 */
export function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<InspectionPage />} />
        <Route path="results" element={<ResultsPage />} />
        <Route path="results/:inspectionId" element={<ResultsPage />} />
        <Route path="history" element={<HistoryPage />} />
        <Route path="evaluation" element={<EvaluationPage />} />
        <Route path="*" element={<NotFoundPage />} />
      </Route>
    </Routes>
  );
}

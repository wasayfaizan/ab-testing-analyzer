import React, { useState, useEffect } from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import UploadPage from "./pages/UploadPage";
import ResultsDashboard from "./pages/ResultsDashboard";
import SavedExperiments from "./pages/SavedExperiments";

function App() {
  const [analysisResults, setAnalysisResults] = useState(null);

  // Load results from localStorage on mount
  useEffect(() => {
    const savedResults = localStorage.getItem("abTestResults");
    if (savedResults) {
      try {
        setAnalysisResults(JSON.parse(savedResults));
      } catch (e) {
        console.error("Error loading saved results:", e);
      }
    }
  }, []);

  // Save results to localStorage whenever they change
  useEffect(() => {
    if (analysisResults) {
      localStorage.setItem("abTestResults", JSON.stringify(analysisResults));
    }
  }, [analysisResults]);

  return (
    <div className="w-full min-h-screen">
      <Router>
        <Routes>
          <Route
            path="/"
            element={<UploadPage setAnalysisResults={setAnalysisResults} />}
          />
          <Route
            path="/results"
            element={
              <ResultsDashboard
                results={analysisResults}
                setAnalysisResults={setAnalysisResults}
              />
            }
          />
          <Route
            path="/saved"
            element={
              <SavedExperiments setAnalysisResults={setAnalysisResults} />
            }
          />
        </Routes>
      </Router>
    </div>
  );
}

export default App;

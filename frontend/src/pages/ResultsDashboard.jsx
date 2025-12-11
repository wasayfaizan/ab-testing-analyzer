import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import SummaryCards from "../components/SummaryCards";
import ChartsSection from "../components/ChartsSection";
import AIInterpretationCard from "../components/AIInterpretationCard";
import RecommendationCard from "../components/RecommendationCard";
import ActionBar from "../components/ActionBar";
import CollapsibleSection from "../components/CollapsibleSection";
import PowerCalculator from "../components/PowerCalculator";
import MDECalculator from "../components/MDECalculator";
import RiskAssessment from "../components/RiskAssessment";
import GlossaryDrawer from "../components/GlossaryDrawer";
import CUPEDSection from "../components/CUPEDSection";

function ResultsDashboard({ results, setAnalysisResults }) {
  const navigate = useNavigate();

  // Try to load from localStorage if results not provided
  const [localResults, setLocalResults] = useState(null);

  useEffect(() => {
    if (!results) {
      const savedResults = localStorage.getItem("abTestResults");
      if (savedResults) {
        try {
          const parsed = JSON.parse(savedResults);
          setLocalResults(parsed);
          if (setAnalysisResults) {
            setAnalysisResults(parsed);
          }
        } catch (e) {
          console.error("Error loading saved results:", e);
        }
      }
    }
  }, [results, setAnalysisResults]);

  const displayResults = results || localResults;

  if (!displayResults) {
    return (
      <div className="min-h-screen bg-white dark:bg-[#0D0F14] flex items-center justify-center">
        <div className="text-center">
          <p className="text-gray-600 dark:text-gray-400 mb-4">
            No analysis results found
          </p>
          <button
            onClick={() => navigate("/")}
            className="px-6 py-3 bg-gradient-to-r from-indigo-500 to-purple-600 text-white rounded-xl hover:shadow-lg transition-all duration-200 hover:-translate-y-0.5 font-medium"
          >
            Go to Upload Page
          </button>
        </div>
      </div>
    );
  }

  // Determine recommendation
  const isSignificant = displayResults.p_value < 0.05;
  const lift = displayResults.lift;
  let recommendation = "Inconclusive – continue experiment";
  let recommendationColor = "gray";

  if (lift > 0 && isSignificant) {
    recommendation = "Choose B (Treatment)";
    recommendationColor = "green";
  } else if (lift < 0 && isSignificant) {
    recommendation = "Choose A (Control)";
    recommendationColor = "green";
  } else if (!isSignificant) {
    recommendation = "Inconclusive – continue experiment";
    recommendationColor = "gray";
  }

  return (
    <div className="min-h-screen w-full bg-white dark:bg-[#0D0F14] transition-colors duration-300">
      <div className="w-full max-w-7xl mx-auto px-4 sm:px-6 py-8 space-y-6">
        {/* Action Bar */}
        <ActionBar results={displayResults} />

        {/* Key Metrics Section */}
        <div className="animate-fade-in">
          <CollapsibleSection
            title="Key Metrics"
            icon={
              <svg
                className="w-5 h-5"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
                />
              </svg>
            }
            defaultOpen={true}
            gradient="indigo"
          >
            <SummaryCards results={displayResults} />
          </CollapsibleSection>
        </div>

        {/* Recommendation Card */}
        <div className="animate-fade-in">
          <RecommendationCard
            recommendation={recommendation}
            color={recommendationColor}
          />
        </div>

        {/* CUPED Section */}
        {displayResults.cuped && displayResults.cuped.enabled && (
          <CollapsibleSection
            title="CUPED-Adjusted Metrics"
            icon={
              <svg
                className="w-5 h-5"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6"
                />
              </svg>
            }
            defaultOpen={true}
            gradient="purple"
          >
            <CUPEDSection cupedResults={displayResults.cuped} />
          </CollapsibleSection>
        )}

        {/* Risk Assessment */}
        <CollapsibleSection
          title="Experiment Risk Assessment"
          icon={
            <svg
              className="w-5 h-5"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
              />
            </svg>
          }
          defaultOpen={true}
          gradient="purple"
        >
          <RiskAssessment results={displayResults} />
        </CollapsibleSection>

        {/* Charts Section */}
        <CollapsibleSection
          title="Visualizations"
          icon={
            <svg
              className="w-5 h-5"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M7 12l3-3 3 3 4-4M8 21l4-4 4 4M3 4h18M4 4h16v12a1 1 0 01-1 1H5a1 1 0 01-1-1V4z"
              />
            </svg>
          }
          defaultOpen={true}
          gradient="teal"
        >
          <ChartsSection results={displayResults} />
        </CollapsibleSection>

        {/* AI Interpretation */}
        <div className="animate-fade-in">
          <AIInterpretationCard
            interpretation={displayResults.interpretation}
            results={displayResults}
          />
        </div>

        {/* Advanced Stats */}
        <CollapsibleSection
          title="Advanced Statistics"
          icon={
            <svg
              className="w-5 h-5"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M9 7h6m0 10v-3m-3 3h.01M9 17h.01M9 14h.01M12 14h.01M15 11h.01M12 11h.01M9 11h.01M7 21h10a2 2 0 002-2V5a2 2 0 00-2-2H7a2 2 0 00-2 2v14a2 2 0 002 2z"
              />
            </svg>
          }
          defaultOpen={false}
          gradient="indigo"
        >
          <div className="space-y-6">
            <PowerCalculator />
            <MDECalculator />
          </div>
        </CollapsibleSection>
      </div>

      {/* Glossary Drawer */}
      <GlossaryDrawer />
    </div>
  );
}

export default ResultsDashboard;

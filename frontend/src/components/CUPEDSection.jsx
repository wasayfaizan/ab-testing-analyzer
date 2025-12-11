import React from "react";
import StatCard from "./StatCard";

function CUPEDSection({ cupedResults }) {
  if (!cupedResults || !cupedResults.enabled) {
    return null;
  }

  return (
    <div>
      <p className="text-sm text-gray-600 dark:text-[#A8A8B3] mb-4">
        Variance-reduced analysis using{" "}
        <strong>{cupedResults.covariate_column}</strong>
      </p>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
        <StatCard
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
          label="CUPED Lift"
          value={`${
            cupedResults.lift > 0 ? "+" : ""
          }${cupedResults.lift.toFixed(2)}%`}
          description="Adjusted lift"
          color="purple"
        />
        <StatCard
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
          label="CUPED P-Value"
          value={
            cupedResults.p_value < 0.0001
              ? cupedResults.p_value.toExponential(3)
              : cupedResults.p_value.toFixed(4)
          }
          description="Adjusted p-value"
          color={cupedResults.p_value < 0.05 ? "green" : "gray"}
          badge={
            cupedResults.p_value < 0.05 ? "Significant" : "Not Significant"
          }
        />
        <StatCard
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
                d="M13 10V3L4 14h7v7l9-11h-7z"
              />
            </svg>
          }
          label="Variance Reduction"
          value={`${cupedResults.variance_reduction_percent.toFixed(1)}%`}
          description="Reduction in variance"
          color="purple"
        />
      </div>

      <div className="p-4 bg-gray-50 dark:bg-white/[0.03] dark:border dark:border-[rgba(255,255,255,0.08)] rounded-lg">
        <p className="text-xs text-gray-600 dark:text-[#A8A8B3]">
          <strong>Θ (theta):</strong> {cupedResults.theta.toFixed(4)} |
          <strong> 95% CI:</strong> [
          {cupedResults.confidence_interval.lower.toFixed(4)},{" "}
          {cupedResults.confidence_interval.upper.toFixed(4)}]
        </p>
      </div>
    </div>
  );
}

export default CUPEDSection;

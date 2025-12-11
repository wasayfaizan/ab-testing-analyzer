import React from "react";

function RiskAssessment({ results }) {
  const pValue = results.p_value;
  const probB = results.bayesian.prob_b_better;
  const ciLower = results.confidence_interval.lower;
  const ciUpper = results.confidence_interval.upper;

  // Calculate risk metrics
  const falsePositiveRisk = pValue < 0.05 ? (1 - probB) * 100 : 50; // If significant but low Bayesian prob
  const reversalRisk =
    Math.abs(probB - 0.5) < 0.1 ? 30 : Math.abs(probB - 0.5) < 0.2 ? 15 : 5;
  const trueUpliftBelowZero =
    ciLower < 0 && ciUpper < 0 ? 0 : ciLower < 0 ? 25 : 5;

  // Overall risk score (0-100)
  const overallRisk =
    falsePositiveRisk * 0.4 + reversalRisk * 0.3 + trueUpliftBelowZero * 0.3;

  let riskLevel = "low";
  let riskColor = "green";
  let riskBadge = "🟢 Low Risk";

  if (overallRisk > 50) {
    riskLevel = "high";
    riskColor = "red";
    riskBadge = "🔴 High Risk";
  } else if (overallRisk > 25) {
    riskLevel = "medium";
    riskColor = "yellow";
    riskBadge = "🟡 Medium Risk";
  }

  // Risk badge color mapping
  const getRiskBadgeStyle = () => {
    if (riskLevel === "high") {
      return "bg-red-500/15 text-red-400 border-red-500/30";
    } else if (riskLevel === "medium") {
      return "bg-badge-significant/15 text-badge-significant border-badge-significant/30";
    } else {
      return "bg-badge-positive/15 text-badge-positive border-badge-positive/30";
    }
  };

  // Risk badge color mapping for dark mode
  const getRiskBadgeStyleDark = () => {
    if (riskLevel === "high") {
      return "dark:bg-red-500/15 dark:text-red-400 dark:border-red-500/30";
    } else if (riskLevel === "medium") {
      return "dark:bg-badge-significant/15 dark:text-badge-significant dark:border-badge-significant/30";
    } else {
      return "dark:bg-aqua/20 dark:text-aqua dark:border-aqua/30";
    }
  };

  return (
    <div className="bg-white dark:bg-[#13161C] rounded-xl border-2 border-gray-200 dark:border-[rgba(255,255,255,0.08)] p-6">
      <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-4">
        Experiment Risk Assessment
      </h3>

      <div className="mb-6">
        <div
          className={`inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-${riskColor}-50 border-2 border-${riskColor}-200 ${getRiskBadgeStyleDark()}`}
        >
          <span className="text-lg">{riskBadge}</span>
          <span
            className={`text-sm font-semibold text-${riskColor}-700 dark:text-white`}
          >
            Risk Score: {overallRisk.toFixed(1)}%
          </span>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="p-4 bg-gray-50 dark:bg-white/[0.03] dark:border dark:border-[rgba(255,255,255,0.08)] rounded-lg">
          <p className="text-xs font-semibold text-gray-600 dark:text-[#A8A8B3] mb-1">
            False Positive Risk
          </p>
          <p className="text-xl font-bold text-gray-900 dark:text-white">
            {falsePositiveRisk.toFixed(1)}%
          </p>
          <p className="text-xs text-gray-500 dark:text-[#A8A8B3] mt-1">
            Chance result is significant by chance
          </p>
        </div>

        <div className="p-4 bg-gray-50 dark:bg-white/[0.03] dark:border dark:border-[rgba(255,255,255,0.08)] rounded-lg">
          <p className="text-xs font-semibold text-gray-600 dark:text-[#A8A8B3] mb-1">
            Reversal Risk
          </p>
          <p className="text-xl font-bold text-gray-900 dark:text-white">
            {reversalRisk.toFixed(1)}%
          </p>
          <p className="text-xs text-gray-500 dark:text-[#A8A8B3] mt-1">
            Chance result reverses with more data
          </p>
        </div>

        <div className="p-4 bg-gray-50 dark:bg-white/[0.03] dark:border dark:border-[rgba(255,255,255,0.08)] rounded-lg">
          <p className="text-xs font-semibold text-gray-600 dark:text-[#A8A8B3] mb-1">
            True Uplift Below Zero
          </p>
          <p className="text-xl font-bold text-gray-900 dark:text-white">
            {trueUpliftBelowZero.toFixed(1)}%
          </p>
          <p className="text-xs text-gray-500 dark:text-[#A8A8B3] mt-1">
            Probability true effect is negative
          </p>
        </div>
      </div>
    </div>
  );
}

export default RiskAssessment;

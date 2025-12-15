import React from "react";
import StatCard from "./StatCard";
import DayOfWeekInsights from "./DayOfWeekInsights";
import SegmentationForestPlot from "./SegmentationForestPlot";
import NumericCovariateEffects from "./NumericCovariateEffects";
import MarginalEffectsPlot from "./MarginalEffectsPlot";
import BinnedCovariateView from "./BinnedCovariateView";

function AdvancedAnalysisResults({ results, analysisMode }) {
  if (!results || !analysisMode) {
    return null;
  }

  // Multi-Variant Analysis Results
  if (analysisMode === "multi-variant") {
    return (
      <div className="space-y-6">
        <div className="bg-white dark:bg-[#13161C] rounded-xl border border-gray-200 dark:border-[rgba(255,255,255,0.08)] p-6">
          <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-4">
            Multi-Variant Test Results
          </h3>
          <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
            Control Group:{" "}
            <span className="font-semibold">{results.control_group}</span>
          </p>

          {results.overall_test && (
            <div className="mb-6 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
              <p className="text-sm font-semibold text-blue-900 dark:text-blue-300">
                Overall Test: {results.overall_test.test_type}
              </p>
              <p className="text-lg font-bold text-blue-600 dark:text-blue-400 mt-1">
                P-value: {results.overall_test.p_value.toFixed(4)}
              </p>
              <p
                className={`text-sm mt-2 ${
                  results.overall_test.significant
                    ? "text-green-600 dark:text-green-400"
                    : "text-gray-600 dark:text-gray-400"
                }`}
              >
                {results.overall_test.significant
                  ? "✓ Statistically Significant"
                  : "Not Significant"}
              </p>
            </div>
          )}

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {results.pairwise_comparisons?.map((comparison, idx) => (
              <div
                key={idx}
                className="bg-gray-50 dark:bg-white/[0.03] rounded-lg p-4 border border-gray-200 dark:border-[rgba(255,255,255,0.08)]"
              >
                <h4 className="font-semibold text-gray-900 dark:text-white mb-2">
                  {comparison.treatment_group} vs {comparison.control_group}
                </h4>
                <div className="space-y-2 text-sm">
                  <p className="text-gray-600 dark:text-gray-400">
                    Lift:{" "}
                    <span className="font-semibold text-gray-900 dark:text-white">
                      {comparison.lift > 0 ? "+" : ""}
                      {comparison.lift.toFixed(2)}%
                    </span>
                  </p>
                  <p className="text-gray-600 dark:text-gray-400">
                    P-value:{" "}
                    <span className="font-semibold text-gray-900 dark:text-white">
                      {comparison.p_value < 0.0001
                        ? comparison.p_value.toExponential(2)
                        : comparison.p_value.toFixed(4)}
                    </span>
                  </p>
                  <p
                    className={`text-xs ${
                      comparison.p_value < 0.05
                        ? "text-green-600 dark:text-green-400"
                        : "text-gray-500 dark:text-gray-500"
                    }`}
                  >
                    {comparison.p_value < 0.05
                      ? "✓ Significant"
                      : "Not Significant"}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  // Segmentation Analysis Results
  if (analysisMode === "segmentation") {
    // Calculate summary stats
    const segments = results.segments || [];
    const significantSegments = segments.filter((s) => s.p_value < 0.05);
    const numSignificant = significantSegments.length;

    // Best performing: highest lift among significant, or highest overall
    let bestSegment = null;
    if (significantSegments.length > 0) {
      bestSegment = significantSegments.reduce((best, current) =>
        current.lift > best.lift ? current : best
      );
    } else if (segments.length > 0) {
      bestSegment = segments.reduce((best, current) =>
        current.lift > best.lift ? current : best
      );
    }

    // Worst performing: lowest lift
    const worstSegment =
      segments.length > 0
        ? segments.reduce((worst, current) =>
            current.lift < worst.lift ? current : worst
          )
        : null;

    return (
      <div className="space-y-6">
        <div className="bg-white dark:bg-[#13161C] rounded-xl border border-gray-200 dark:border-[rgba(255,255,255,0.08)] p-6">
          <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-4">
            Segmentation Analysis
          </h3>
          <p className="text-sm text-gray-600 dark:text-gray-400 mb-6">
            Segmented by:{" "}
            <span className="font-semibold">{results.segment_column}</span>
          </p>

          {/* Summary Header */}
          {(bestSegment || worstSegment) && (
            <div className="mb-6 p-4 bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                {bestSegment && (
                  <div className="flex items-center gap-2">
                    <span className="text-2xl">⭐</span>
                    <div>
                      <p className="text-xs text-gray-600 dark:text-gray-400">
                        Best Performing Segment
                      </p>
                      <p className="font-bold text-gray-900 dark:text-white">
                        {bestSegment.segment}:{" "}
                        <span className="text-green-600 dark:text-green-400">
                          {bestSegment.lift > 0 ? "+" : ""}
                          {bestSegment.lift.toFixed(1)}% lift
                        </span>
                      </p>
                    </div>
                  </div>
                )}
                {worstSegment && (
                  <div className="flex items-center gap-2">
                    <span className="text-2xl">⚠️</span>
                    <div>
                      <p className="text-xs text-gray-600 dark:text-gray-400">
                        Weakest Segment
                      </p>
                      <p className="font-bold text-gray-900 dark:text-white">
                        {worstSegment.segment}:{" "}
                        <span
                          className={
                            worstSegment.lift < 0
                              ? "text-red-600 dark:text-red-400"
                              : "text-gray-600 dark:text-gray-400"
                          }
                        >
                          {worstSegment.lift > 0 ? "+" : ""}
                          {worstSegment.lift.toFixed(1)}%
                          {worstSegment.p_value >= 0.05 && ", not significant"}
                        </span>
                      </p>
                    </div>
                  </div>
                )}
                <div className="flex items-center gap-2">
                  <span className="text-2xl">📊</span>
                  <div>
                    <p className="text-xs text-gray-600 dark:text-gray-400">
                      Significant Segments
                    </p>
                    <p className="font-bold text-gray-900 dark:text-white">
                      {numSignificant} / {segments.length}
                    </p>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Segmentation Cards */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
            {segments.map((segment, idx) => {
              const isSignificant = segment.p_value < 0.05;
              return (
                <div
                  key={idx}
                  className={`bg-gray-50 dark:bg-white/[0.03] rounded-lg p-4 border-2 ${
                    isSignificant
                      ? "border-green-500 dark:border-green-400"
                      : "border-gray-200 dark:border-[rgba(255,255,255,0.08)]"
                  }`}
                >
                  <h4 className="font-semibold text-gray-900 dark:text-white mb-2">
                    {segment.segment} (n={segment.segment_size.toLocaleString()}
                    )
                  </h4>
                  <div className="space-y-2 text-sm">
                    <p className="text-gray-600 dark:text-gray-400">
                      Lift:{" "}
                      <span
                        className={`font-semibold ${
                          isSignificant
                            ? "text-green-600 dark:text-green-400"
                            : "text-gray-600 dark:text-gray-400"
                        }`}
                      >
                        {segment.lift > 0 ? "+" : ""}
                        {segment.lift.toFixed(2)}%
                      </span>{" "}
                      {isSignificant ? (
                        <span className="text-green-600 dark:text-green-400">
                          ✓ Significant (p &lt; 0.05)
                        </span>
                      ) : (
                        <span className="text-gray-500 dark:text-gray-500">
                          ✖ Not Significant
                        </span>
                      )}
                    </p>
                    <p className="text-gray-600 dark:text-gray-400">
                      P-value:{" "}
                      <span className="font-semibold text-gray-900 dark:text-white">
                        {segment.p_value < 0.0001
                          ? segment.p_value.toExponential(2)
                          : segment.p_value.toFixed(4)}
                      </span>
                    </p>
                    {segment.group_a && segment.group_b && (
                      <>
                        <p className="text-gray-600 dark:text-gray-400">
                          Control:{" "}
                          {segment.group_a.rate
                            ? (segment.group_a.rate * 100).toFixed(2) + "%"
                            : segment.group_a.mean.toFixed(2)}
                        </p>
                        <p className="text-gray-600 dark:text-gray-400">
                          Treatment:{" "}
                          {segment.group_b.rate
                            ? (segment.group_b.rate * 100).toFixed(2) + "%"
                            : segment.group_b.mean.toFixed(2)}
                        </p>
                      </>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Forest Plot */}
        {results.forest_plot_data && results.forest_plot_data.length > 0 && (
          <SegmentationForestPlot forestPlotData={results.forest_plot_data} />
        )}

        {/* Insights Summary */}
        {results.segmentation_summary && (
          <div className="bg-white dark:bg-[#13161C] rounded-xl border border-gray-200 dark:border-[rgba(255,255,255,0.08)] p-6">
            <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-4">
              Insights Summary
            </h3>
            <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
              <p className="text-sm text-blue-800 dark:text-blue-300 leading-relaxed">
                {results.segmentation_summary}
              </p>
            </div>
          </div>
        )}

        {/* Targeting Recommendations */}
        {results.targeting_recommendations && (
          <div className="bg-white dark:bg-[#13161C] rounded-xl border border-gray-200 dark:border-[rgba(255,255,255,0.08)] p-6">
            <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
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
                  d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"
                />
              </svg>
              Targeting Recommendations
            </h3>
            <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg border border-green-200 dark:border-green-800">
              <p className="text-sm text-green-800 dark:text-green-300 leading-relaxed">
                {results.targeting_recommendations}
              </p>
            </div>
          </div>
        )}

        {/* Interaction Effects (Treatment × Segment) */}
        {results.interaction_effects &&
          results.interaction_effects.length > 0 && (
            <div className="bg-white dark:bg-[#13161C] rounded-xl border border-gray-200 dark:border-[rgba(255,255,255,0.08)] p-6">
              <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-4">
                Interaction Effects (Treatment × Segment)
              </h3>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
                Regression analysis testing whether treatment effectiveness
                varies by segment. Significant interactions indicate that
                treatment has different effects across segments.
              </p>
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
                  <thead className="bg-gray-50 dark:bg-white/[0.03]">
                    <tr>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                        Segment
                      </th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                        Interaction Coef
                      </th>
                      {results.interaction_effects[0]?.odds_ratio !==
                        undefined && (
                        <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                          Odds Ratio
                        </th>
                      )}
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                        P-value
                      </th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                        95% CI
                      </th>
                    </tr>
                  </thead>
                  <tbody className="bg-white dark:bg-[#0D0F14] divide-y divide-gray-200 dark:divide-gray-700">
                    {results.interaction_effects.map((interaction, idx) => {
                      const isSignificant = interaction.significant;
                      return (
                        <tr
                          key={idx}
                          className={
                            isSignificant
                              ? "bg-green-50/50 dark:bg-green-900/10"
                              : ""
                          }
                        >
                          <td className="px-4 py-3 whitespace-nowrap text-sm font-medium text-gray-900 dark:text-white">
                            {interaction.segment}
                          </td>
                          <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-900 dark:text-white">
                            {interaction.interaction_coef > 0 ? "+" : ""}
                            {interaction.interaction_coef.toFixed(4)}
                          </td>
                          {interaction.odds_ratio !== undefined && (
                            <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-900 dark:text-white">
                              {interaction.odds_ratio.toFixed(4)}
                            </td>
                          )}
                          <td className="px-4 py-3 whitespace-nowrap">
                            <span
                              className={`text-sm font-semibold ${
                                isSignificant
                                  ? "text-green-600 dark:text-green-400"
                                  : "text-gray-600 dark:text-gray-400"
                              }`}
                            >
                              {interaction.p_value < 0.0001
                                ? interaction.p_value.toExponential(2)
                                : interaction.p_value.toFixed(4)}
                            </span>
                          </td>
                          <td className="px-4 py-3 whitespace-nowrap text-xs text-gray-600 dark:text-gray-400">
                            [{interaction.ci_lower.toFixed(4)},{" "}
                            {interaction.ci_upper.toFixed(4)}]
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
              <p className="text-xs text-gray-500 dark:text-gray-400 mt-4">
                * Positive interaction coefficients indicate treatment has a
                stronger effect in this segment compared to the baseline
                segment.
              </p>
            </div>
          )}
      </div>
    );
  }

  // Time-Based Analysis Results
  if (analysisMode === "time-based") {
    return (
      <div className="space-y-6">
        <div className="bg-white dark:bg-[#13161C] rounded-xl border border-gray-200 dark:border-[rgba(255,255,255,0.08)] p-6">
          <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-4">
            Time-Based Analysis
          </h3>
          <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
            Granularity:{" "}
            <span className="font-semibold">{results.period_name}</span>
          </p>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {results.time_periods?.map((period, idx) => (
              <div
                key={idx}
                className="bg-gray-50 dark:bg-white/[0.03] rounded-lg p-4 border border-gray-200 dark:border-[rgba(255,255,255,0.08)]"
              >
                <h4 className="font-semibold text-gray-900 dark:text-white mb-2">
                  {period.time_period_label}
                </h4>
                <div className="space-y-2 text-sm">
                  <p className="text-gray-600 dark:text-gray-400">
                    Lift:{" "}
                    <span className="font-semibold text-gray-900 dark:text-white">
                      {period.lift > 0 ? "+" : ""}
                      {period.lift.toFixed(2)}%
                    </span>
                  </p>
                  <p className="text-gray-600 dark:text-gray-400">
                    P-value:{" "}
                    <span className="font-semibold text-gray-900 dark:text-white">
                      {period.p_value < 0.0001
                        ? period.p_value.toExponential(2)
                        : period.p_value.toFixed(4)}
                    </span>
                  </p>
                  <p
                    className={`text-xs ${
                      period.p_value < 0.05
                        ? "text-green-600 dark:text-green-400"
                        : "text-gray-500 dark:text-gray-500"
                    }`}
                  >
                    {period.p_value < 0.05
                      ? "✓ Significant"
                      : "Not Significant"}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  // Regression Analysis Results
  if (analysisMode === "regression") {
    // Safety check - if results don't have expected structure, show error
    if (!results.model_type) {
      return (
        <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4">
          <p className="text-red-800 dark:text-red-300">
            Error: Regression results are missing required data. Please try
            running the analysis again.
          </p>
          <pre className="mt-2 text-xs overflow-auto">
            {JSON.stringify(results, null, 2)}
          </pre>
        </div>
      );
    }

    return (
      <div className="space-y-6">
        <div className="bg-white dark:bg-[#13161C] rounded-xl border border-gray-200 dark:border-[rgba(255,255,255,0.08)] p-6">
          <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-4">
            Regression Analysis
          </h3>
          <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
            Model Type:{" "}
            <span className="font-semibold">{results.model_type}</span>
            {results.n_observations && (
              <span className="ml-4">
                • N = {results.n_observations.toLocaleString()}
              </span>
            )}
          </p>

          {/* Warnings */}
          {results.warnings && results.warnings.length > 0 && (
            <div className="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg p-4 mb-4">
              <p className="text-sm font-semibold text-yellow-900 dark:text-yellow-300 mb-2">
                ⚠️ Warnings
              </p>
              <ul className="list-disc list-inside space-y-1">
                {results.warnings.map((warning, idx) => (
                  <li
                    key={idx}
                    className="text-xs text-yellow-700 dark:text-yellow-400"
                  >
                    {warning}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Treatment Effect Summary */}
          <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4 mb-6">
            <p className="text-sm font-semibold text-blue-900 dark:text-blue-300 mb-2">
              Treatment Effect
            </p>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div>
                <p className="text-xs text-blue-700 dark:text-blue-400 mb-1">
                  Coefficient
                </p>
                <p className="text-xl font-bold text-blue-600 dark:text-blue-400">
                  {results.treatment_effect !== undefined
                    ? `${
                        results.treatment_effect > 0 ? "+" : ""
                      }${results.treatment_effect.toFixed(4)}`
                    : "N/A"}
                </p>
              </div>
              {results.odds_ratio !== undefined && (
                <div>
                  <p className="text-xs text-blue-700 dark:text-blue-400 mb-1">
                    Odds Ratio
                  </p>
                  <p className="text-xl font-bold text-blue-600 dark:text-blue-400">
                    {results.odds_ratio.toFixed(4)}
                  </p>
                </div>
              )}
              <div>
                <p className="text-xs text-blue-700 dark:text-blue-400 mb-1">
                  P-value
                </p>
                <p
                  className={`text-xl font-bold ${
                    results.p_value !== undefined && results.p_value < 0.05
                      ? "text-green-600 dark:text-green-400"
                      : "text-blue-600 dark:text-blue-400"
                  }`}
                >
                  {results.p_value !== undefined
                    ? results.p_value < 0.0001
                      ? results.p_value.toExponential(2)
                      : results.p_value.toFixed(4)
                    : "N/A"}
                </p>
              </div>
              <div>
                <p className="text-xs text-blue-700 dark:text-blue-400 mb-1">
                  95% CI
                </p>
                <p className="text-sm font-semibold text-blue-600 dark:text-blue-400">
                  {results.ci_lower !== undefined &&
                  results.ci_upper !== undefined
                    ? `[${results.ci_lower.toFixed(
                        4
                      )}, ${results.ci_upper.toFixed(4)}]`
                    : "N/A"}
                </p>
              </div>
            </div>
            {results.p_value !== undefined && results.p_value < 0.05 && (
              <p className="text-xs text-green-600 dark:text-green-400 mt-2">
                ✓ Statistically significant (p &lt; 0.05)
              </p>
            )}
          </div>

          {/* Model Summary */}
          {results.model_summary && (
            <div className="bg-gray-50 dark:bg-white/[0.03] rounded-lg p-4 mb-6">
              <h4 className="font-semibold text-gray-900 dark:text-white mb-2">
                Model Summary
              </h4>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                {results.model_summary.r_squared !== undefined && (
                  <div>
                    <p className="text-gray-600 dark:text-gray-400">R²</p>
                    <p className="font-semibold text-gray-900 dark:text-white">
                      {results.model_summary.r_squared.toFixed(4)}
                    </p>
                  </div>
                )}
                {results.model_summary.adj_r_squared !== undefined && (
                  <div>
                    <p className="text-gray-600 dark:text-gray-400">Adj. R²</p>
                    <p className="font-semibold text-gray-900 dark:text-white">
                      {results.model_summary.adj_r_squared.toFixed(4)}
                    </p>
                  </div>
                )}
                {results.model_summary.pseudo_r_squared !== undefined && (
                  <div>
                    <p className="text-gray-600 dark:text-gray-400">
                      Pseudo R²
                    </p>
                    <p className="font-semibold text-gray-900 dark:text-white">
                      {results.model_summary.pseudo_r_squared.toFixed(4)}
                    </p>
                  </div>
                )}
                {results.model_summary.aic !== undefined && (
                  <div>
                    <p className="text-gray-600 dark:text-gray-400">AIC</p>
                    <p className="font-semibold text-gray-900 dark:text-white">
                      {results.model_summary.aic.toFixed(2)}
                    </p>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Detailed Coefficients Table */}
          {results.coefficients && results.coefficients.length > 0 && (
            <div>
              <h4 className="font-semibold text-gray-900 dark:text-white mb-3">
                Detailed Coefficients
              </h4>
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
                  <thead className="bg-gray-50 dark:bg-white/[0.03]">
                    <tr>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                        Variable
                      </th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                        Coef
                      </th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                        Std Err
                      </th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                        {results.model_type === "Logistic Regression"
                          ? "z"
                          : "t"}
                      </th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                        P-value
                      </th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                        95% CI
                      </th>
                      {results.model_type === "Logistic Regression" && (
                        <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                          Odds Ratio
                        </th>
                      )}
                    </tr>
                  </thead>
                  <tbody className="bg-white dark:bg-[#0D0F14] divide-y divide-gray-200 dark:divide-gray-700">
                    {results.coefficients.map((coef, idx) => (
                      <tr
                        key={idx}
                        className={
                          coef.name === "treatment"
                            ? "bg-blue-50/50 dark:bg-blue-900/10"
                            : ""
                        }
                      >
                        <td className="px-4 py-3 whitespace-nowrap">
                          <div className="text-sm font-medium text-gray-900 dark:text-white">
                            {coef.name}
                          </div>
                          {coef.original_name !== coef.name && (
                            <div className="text-xs text-gray-500 dark:text-gray-400">
                              ({coef.original_name}
                              {coef.category && `: ${coef.category}`})
                            </div>
                          )}
                        </td>
                        <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-900 dark:text-white">
                          {coef.coef !== undefined
                            ? `${coef.coef > 0 ? "+" : ""}${coef.coef.toFixed(
                                4
                              )}`
                            : "N/A"}
                        </td>
                        <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-600 dark:text-gray-400">
                          {coef.std_err !== undefined
                            ? coef.std_err.toFixed(4)
                            : "N/A"}
                        </td>
                        <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-600 dark:text-gray-400">
                          {coef.z !== undefined
                            ? coef.z.toFixed(3)
                            : coef.t !== undefined
                            ? coef.t.toFixed(3)
                            : "N/A"}
                        </td>
                        <td className="px-4 py-3 whitespace-nowrap">
                          {coef.p_value !== undefined ? (
                            <span
                              className={`text-sm font-semibold ${
                                coef.p_value < 0.05
                                  ? "text-green-600 dark:text-green-400"
                                  : "text-gray-600 dark:text-gray-400"
                              }`}
                            >
                              {coef.p_value < 0.0001
                                ? coef.p_value.toExponential(2)
                                : coef.p_value.toFixed(4)}
                            </span>
                          ) : (
                            "N/A"
                          )}
                        </td>
                        <td className="px-4 py-3 whitespace-nowrap text-xs text-gray-600 dark:text-gray-400">
                          {coef.ci_lower !== undefined &&
                          coef.ci_upper !== undefined
                            ? `[${coef.ci_lower.toFixed(
                                4
                              )}, ${coef.ci_upper.toFixed(4)}]`
                            : "N/A"}
                        </td>
                        {results.model_type === "Logistic Regression" && (
                          <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-900 dark:text-white">
                            {coef.odds_ratio !== undefined
                              ? coef.odds_ratio.toFixed(4)
                              : "N/A"}
                          </td>
                        )}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Day-of-Week Insights - Only for logistic regression with day effects */}
          {results.day_effects && results.day_effects.length > 0 && (
            <div className="mt-6">
              <DayOfWeekInsights
                dayEffects={results.day_effects}
                summary={results.day_effects_summary}
                recommendations={results.targeting_recommendations}
              />
            </div>
          )}
        </div>

        {/* Numeric Covariate Effects */}
        {results.numeric_effects &&
          Object.keys(results.numeric_effects).length > 0 && (
            <NumericCovariateEffects numericEffects={results.numeric_effects} />
          )}

        {/* Marginal Effects Plots */}
        {results.marginal_effects_data &&
          Object.keys(results.marginal_effects_data).length > 0 && (
            <MarginalEffectsPlot
              marginalEffectsData={results.marginal_effects_data}
              modelType={results.model_type}
            />
          )}

        {/* Binned View Toggle */}
        {results.binned_effects &&
          Object.keys(results.binned_effects).length > 0 && (
            <BinnedCovariateView binnedEffects={results.binned_effects} />
          )}

        {/* All Covariates Insights Summary */}
        {results.all_covariates_insights && (
          <div className="bg-white dark:bg-[#13161C] rounded-xl border border-gray-200 dark:border-[rgba(255,255,255,0.08)] p-6">
            <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-4">
              All Covariates Insights
            </h3>
            <div className="p-4 bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
              <p className="text-sm text-blue-800 dark:text-blue-300 leading-relaxed">
                {results.all_covariates_insights}
              </p>
            </div>
          </div>
        )}
      </div>
    );
  }

  // Non-Parametric Test Results
  if (analysisMode === "non-parametric") {
    return (
      <div className="space-y-6">
        <div className="bg-white dark:bg-[#13161C] rounded-xl border border-gray-200 dark:border-[rgba(255,255,255,0.08)] p-6">
          <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-4">
            Non-Parametric Test Results
          </h3>
          <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
            Test Type:{" "}
            <span className="font-semibold">{results.test_type}</span>
          </p>

          <div className="space-y-4">
            <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4">
              <p className="text-sm font-semibold text-blue-900 dark:text-blue-300 mb-1">
                Test Statistic: {results.statistic.toFixed(4)}
              </p>
              <p className="text-lg font-bold text-blue-600 dark:text-blue-400 mt-1">
                P-value:{" "}
                {results.p_value < 0.0001
                  ? results.p_value.toExponential(2)
                  : results.p_value.toFixed(4)}
              </p>
              <p
                className={`text-sm mt-2 ${
                  results.significant
                    ? "text-green-600 dark:text-green-400"
                    : "text-gray-600 dark:text-gray-400"
                }`}
              >
                {results.significant
                  ? "✓ Statistically Significant"
                  : "Not Significant"}
              </p>
              {results.effect_size && (
                <p className="text-sm text-blue-700 dark:text-blue-400 mt-2">
                  Effect Size: {results.effect_size.toFixed(4)} (
                  {results.effect_size_interpretation})
                </p>
              )}
            </div>

            <div>
              <h4 className="font-semibold text-gray-900 dark:text-white mb-2">
                Groups
              </h4>
              <div className="space-y-2">
                {results.groups?.map((group, idx) => (
                  <div
                    key={idx}
                    className="p-3 bg-gray-50 dark:bg-white/[0.03] rounded border border-gray-200 dark:border-[rgba(255,255,255,0.08)]"
                  >
                    <p className="font-semibold text-gray-900 dark:text-white">
                      {group.name}
                    </p>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      Size: {group.size} | Median: {group.median.toFixed(2)}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Fallback: if we have results but don't match any mode, show raw data
  return (
    <div className="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg p-4">
      <p className="text-yellow-800 dark:text-yellow-300 mb-2">
        Unknown analysis mode: {analysisMode}
      </p>
      <details className="mt-2">
        <summary className="cursor-pointer text-sm text-yellow-700 dark:text-yellow-400">
          View raw results
        </summary>
        <pre className="mt-2 text-xs overflow-auto bg-white dark:bg-[#0D0F14] p-2 rounded">
          {JSON.stringify(results, null, 2)}
        </pre>
      </details>
    </div>
  );
}

export default AdvancedAnalysisResults;

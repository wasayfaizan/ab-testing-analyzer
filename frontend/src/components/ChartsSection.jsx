import React, { useMemo } from "react";
import Plot from "react-plotly.js";

function ChartsSection({ results }) {
  const isBinary = results.metric_type === "binary";

  // Get group labels
  const groupALabel =
    results.group_a_name || results.sample_data?.group_a_label || "Group A";
  const groupBLabel =
    results.group_b_name || results.sample_data?.group_b_label || "Group B";

  // Bar chart data with error bars
  const barChartData = useMemo(() => {
    if (isBinary) {
      // Calculate error bars (95% CI for proportions)
      const se_a = Math.sqrt(
        (results.group_a.rate * (1 - results.group_a.rate)) /
          results.group_a.size
      );
      const se_b = Math.sqrt(
        (results.group_b.rate * (1 - results.group_b.rate)) /
          results.group_b.size
      );
      const z_critical = 1.96;

      return [
        {
          x: [groupALabel, groupBLabel],
          y: [results.group_a.rate, results.group_b.rate],
          type: "bar",
          marker: {
            color: ["#6366F1", "#14B8A6"],
          },
          text: [
            `${(results.group_a.rate * 100).toFixed(2)}%`,
            `${(results.group_b.rate * 100).toFixed(2)}%`,
          ],
          textposition: "outside",
          error_y: {
            type: "data",
            array: [z_critical * se_a, z_critical * se_b],
            visible: true,
            color: "#666",
            thickness: 2,
          },
        },
      ];
    } else {
      // Error bars for numeric (standard error)
      const se_a = results.group_a.std / Math.sqrt(results.group_a.size);
      const se_b = results.group_b.std / Math.sqrt(results.group_b.size);
      const t_critical = 1.96; // Approximate for large samples

      return [
        {
          x: [groupALabel, groupBLabel],
          y: [results.group_a.mean, results.group_b.mean],
          type: "bar",
          marker: {
            color: ["#6366F1", "#14B8A6"],
          },
          text: [
            results.group_a.mean.toFixed(2),
            results.group_b.mean.toFixed(2),
          ],
          textposition: "outside",
          error_y: {
            type: "data",
            array: [t_critical * se_a, t_critical * se_b],
            visible: true,
            color: "#666",
            thickness: 2,
          },
        },
      ];
    }
  }, [results, isBinary, groupALabel, groupBLabel]);

  // Distribution plot data
  const distributionData = useMemo(() => {
    if (isBinary) {
      // For binary, show conversion rates as bars
      return [
        {
          x: ["Converted", "Not Converted"],
          y: [
            results.group_a.conversions,
            results.group_a.size - results.group_a.conversions,
          ],
          name: groupALabel,
          type: "bar",
          marker: { color: "#6366F1" },
        },
        {
          x: ["Converted", "Not Converted"],
          y: [
            results.group_b.conversions,
            results.group_b.size - results.group_b.conversions,
          ],
          name: groupBLabel,
          type: "bar",
          marker: { color: "#14B8A6" },
        },
      ];
    } else {
      // For numeric, create histogram-like distribution
      const sampleA = results.sample_data?.group_a || [];
      const sampleB = results.sample_data?.group_b || [];

      return [
        {
          x: sampleA,
          type: "histogram",
          name: groupALabel,
          marker: { color: "#6366F1", opacity: 0.7 },
          nbinsx: 20,
        },
        {
          x: sampleB,
          type: "histogram",
          name: groupBLabel,
          marker: { color: "#14B8A6", opacity: 0.7 },
          nbinsx: 20,
        },
      ];
    }
  }, [results, isBinary]);

  // Confidence interval chart
  const confidenceIntervalData = useMemo(() => {
    const diff = results.difference;
    const ciLower = results.confidence_interval.lower;
    const ciUpper = results.confidence_interval.upper;

    return [
      {
        x: [diff],
        y: ["Difference"],
        type: "scatter",
        mode: "markers",
        marker: {
          size: 12,
          color:
            (ciLower > 0 && ciUpper > 0) || (ciLower < 0 && ciUpper < 0)
              ? "#14B8A6"
              : "#F87171",
        },
        name: "Point Estimate",
        error_x: {
          type: "data",
          symmetric: false,
          array: [[ciUpper - diff]],
          arrayminus: [[diff - ciLower]],
        },
      },
    ];
  }, [results]);

  // Bayesian posterior curves
  const bayesianData = useMemo(() => {
    if (isBinary) {
      // Beta distribution curves
      const alphaA = results.bayesian.posterior_a.alpha;
      const betaA = results.bayesian.posterior_a.beta;
      const alphaB = results.bayesian.posterior_b.alpha;
      const betaB = results.bayesian.posterior_b.beta;

      const x = Array.from({ length: 100 }, (_, i) => i / 100);

      // Approximate Beta PDF
      const betaPDF = (x, alpha, beta) => {
        // Simplified Beta PDF approximation
        return Math.pow(x, alpha - 1) * Math.pow(1 - x, beta - 1);
      };

      const yA = x.map((val) => betaPDF(val, alphaA, betaA));
      const yB = x.map((val) => betaPDF(val, alphaB, betaB));

      // Normalize
      const maxA = Math.max(...yA);
      const maxB = Math.max(...yB);
      const normalizedYA = yA.map((y) => y / maxA);
      const normalizedYB = yB.map((y) => y / maxB);

      return [
        {
          x: x,
          y: normalizedYA,
          type: "scatter",
          mode: "lines",
          name: `${groupALabel} Posterior`,
          line: { color: "#6366F1", width: 2 },
        },
        {
          x: x,
          y: normalizedYB,
          type: "scatter",
          mode: "lines",
          name: `${groupBLabel} Posterior`,
          line: { color: "#14B8A6", width: 2 },
        },
      ];
    } else {
      // Normal distribution curves
      const meanA = results.bayesian.posterior_a.mean;
      const stdA = results.bayesian.posterior_a.std;
      const meanB = results.bayesian.posterior_b.mean;
      const stdB = results.bayesian.posterior_b.std;

      const range = Math.max(
        Math.abs(meanA - meanB) + 4 * Math.max(stdA, stdB),
        10
      );
      const x = Array.from(
        { length: 200 },
        (_, i) => Math.min(meanA, meanB) - range / 2 + (range * i) / 200
      );

      const normalPDF = (x, mean, std) => {
        return (
          Math.exp(-0.5 * Math.pow((x - mean) / std, 2)) /
          (std * Math.sqrt(2 * Math.PI))
        );
      };

      return [
        {
          x: x,
          y: x.map((val) => normalPDF(val, meanA, stdA)),
          type: "scatter",
          mode: "lines",
          name: `${groupALabel} Posterior`,
          line: { color: "#6366F1", width: 2 },
          fill: "tozeroy",
          fillcolor: "rgba(99, 102, 241, 0.15)",
        },
        {
          x: x,
          y: x.map((val) => normalPDF(val, meanB, stdB)),
          type: "scatter",
          mode: "lines",
          name: `${groupBLabel} Posterior`,
          line: { color: "#14B8A6", width: 2 },
          fill: "tozeroy",
          fillcolor: "rgba(20, 184, 166, 0.15)",
        },
      ];
    }
  }, [results, isBinary]);

  const plotLayout = {
    paper_bgcolor: "white",
    plot_bgcolor: "white",
    font: { family: "Inter, sans-serif", size: 12 },
    margin: { l: 60, r: 20, t: 40, b: 60 },
  };

  return (
    <div className="space-y-6 mb-8">
      {/* Bar Chart */}
      <div className="bg-white dark:bg-[#13161C] rounded-2xl shadow-lg border border-purple-200/30 dark:border-[rgba(255,255,255,0.08)] p-6 hover:shadow-xl dark:hover:shadow-[0_8px_32px_rgba(138,63,252,0.15)] transition-all duration-300 animate-fade-in">
        <div className="flex items-center gap-3 mb-6">
          <div className="w-10 h-10 bg-gradient-to-br from-indigo-100 to-purple-100 dark:from-indigo-900/30 dark:to-purple-900/30 rounded-xl flex items-center justify-center">
            <svg
              className="w-5 h-5 text-indigo-600 dark:text-indigo-400"
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
          </div>
          <h3 className="text-lg font-bold text-gray-900 dark:text-white">
            {isBinary ? "Conversion Rates" : "Mean Values"}
          </h3>
        </div>
        <Plot
          data={barChartData}
          layout={{
            ...plotLayout,
            title: isBinary
              ? "Conversion Rate Comparison"
              : "Mean Value Comparison",
            xaxis: { title: "Group" },
            yaxis: { title: isBinary ? "Conversion Rate" : "Mean Value" },
            height: 400,
          }}
          config={{ responsive: true, displayModeBar: false }}
        />
      </div>

      {/* Distribution Plot */}
      <div className="bg-white dark:bg-[#13161C] rounded-2xl shadow-lg border border-purple-200/30 dark:border-[rgba(255,255,255,0.08)] p-6 hover:shadow-xl dark:hover:shadow-[0_8px_32px_rgba(138,63,252,0.15)] transition-all duration-300 animate-fade-in">
        <div className="flex items-center gap-3 mb-6">
          <div className="w-10 h-10 bg-gradient-to-br from-purple-100 to-pink-100 dark:from-purple-900/30 dark:to-pink-900/30 rounded-xl flex items-center justify-center">
            <svg
              className="w-5 h-5 text-purple-600 dark:text-purple-400"
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
          </div>
          <h3 className="text-lg font-bold text-gray-900 dark:text-white">
            {isBinary ? "Conversion Distribution" : "Value Distribution"}
          </h3>
        </div>
        <Plot
          data={distributionData}
          layout={{
            ...plotLayout,
            title: isBinary
              ? "Conversion Counts by Group"
              : "Value Distribution by Group",
            xaxis: { title: isBinary ? "Outcome" : "Value" },
            yaxis: { title: isBinary ? "Count" : "Frequency" },
            barmode: isBinary ? "group" : "overlay",
            height: 400,
          }}
          config={{ responsive: true, displayModeBar: false }}
        />
      </div>

      {/* Confidence Interval */}
      <div className="bg-white dark:bg-[#13161C] rounded-2xl shadow-lg border border-purple-200/30 dark:border-[rgba(255,255,255,0.08)] p-6 hover:shadow-xl dark:hover:shadow-[0_8px_32px_rgba(138,63,252,0.15)] transition-all duration-300 animate-fade-in">
        <div className="flex items-center gap-3 mb-6">
          <div className="w-10 h-10 bg-gradient-to-br from-teal-100 to-cyan-100 dark:from-teal-900/30 dark:to-cyan-900/30 rounded-xl flex items-center justify-center">
            <svg
              className="w-5 h-5 text-teal-600 dark:text-teal-400"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
              />
            </svg>
          </div>
          <h3 className="text-lg font-bold text-gray-900 dark:text-white">
            95% Confidence Interval for Difference
          </h3>
        </div>
        <Plot
          data={confidenceIntervalData}
          layout={{
            ...plotLayout,
            title: "Difference with 95% Confidence Interval",
            xaxis: { title: `Difference (${groupBLabel} - ${groupALabel})` },
            yaxis: { title: "" },
            height: 300,
            shapes: [
              {
                type: "line",
                x0: 0,
                x1: 0,
                y0: -0.5,
                y1: 0.5,
                line: { color: "gray", width: 1, dash: "dash" },
              },
            ],
          }}
          config={{ responsive: true, displayModeBar: false }}
        />
      </div>

      {/* Bayesian Posterior Curves */}
      <div className="bg-white dark:bg-[#13161C] rounded-2xl shadow-lg border border-purple-200/30 dark:border-[rgba(255,255,255,0.08)] p-6 hover:shadow-xl dark:hover:shadow-[0_8px_32px_rgba(138,63,252,0.15)] transition-all duration-300 animate-fade-in">
        <div className="flex items-center gap-3 mb-6">
          <div className="w-10 h-10 bg-gradient-to-br from-indigo-100 to-purple-100 dark:from-indigo-900/30 dark:to-purple-900/30 rounded-xl flex items-center justify-center">
            <svg
              className="w-5 h-5 text-indigo-600 dark:text-indigo-400"
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
          </div>
          <h3 className="text-lg font-bold text-gray-900 dark:text-white">
            Bayesian Posterior Distributions
          </h3>
        </div>
        <Plot
          data={bayesianData}
          layout={{
            ...plotLayout,
            title: "Posterior Probability Distributions",
            xaxis: { title: isBinary ? "Conversion Rate" : "Value" },
            yaxis: { title: "Density" },
            height: 400,
            legend: { x: 0.7, y: 0.9 },
          }}
          config={{ responsive: true, displayModeBar: false }}
        />
      </div>
    </div>
  );
}

export default ChartsSection;

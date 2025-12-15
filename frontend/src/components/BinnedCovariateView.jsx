import React from "react";
import Plot from "react-plotly.js";

function BinnedCovariateView({ binnedEffects }) {
  const [expandedVars, setExpandedVars] = React.useState(new Set());

  if (!binnedEffects || Object.keys(binnedEffects).length === 0) {
    return null;
  }

  const formatVariableName = (name) => {
    return name.replace(/_/g, " ").replace(/\b\w/g, (l) => l.toUpperCase());
  };

  const toggleVar = (varName) => {
    const newExpanded = new Set(expandedVars);
    if (newExpanded.has(varName)) {
      newExpanded.delete(varName);
    } else {
      newExpanded.add(varName);
    }
    setExpandedVars(newExpanded);
  };

  return (
    <div className="bg-white dark:bg-[#13161C] rounded-xl border border-gray-200 dark:border-[rgba(255,255,255,0.08)] p-6">
      <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-4">
        Binned View (Categorical Comparison)
      </h3>
      <p className="text-sm text-gray-600 dark:text-gray-400 mb-6">
        Numeric variables grouped into categories for easier comparison. Toggle
        to view binned effects.
      </p>

      <div className="space-y-6">
        {Object.entries(binnedEffects).map(([varName, data]) => {
          const readableName = formatVariableName(
            data.original_name || varName
          );
          const isExpanded = expandedVars.has(varName);
          const bins = data.bins || [];

          // Prepare chart data
          const binNames = bins.map((b) => b.bin);
          const oddsRatios = bins.map((b) => b.odds_ratio);
          const colors = bins.map((b) =>
            b.is_baseline ? "#94A3B8" : b.odds_ratio > 1 ? "#25E6D1" : "#94A3B8"
          );

          const chartData = [
            {
              type: "bar",
              orientation: "h",
              x: oddsRatios,
              y: binNames,
              marker: {
                color: colors,
                line: {
                  color: "rgba(0,0,0,0.1)",
                  width: 1,
                },
              },
              text: bins.map(
                (b) =>
                  `Odds Ratio: ${b.odds_ratio.toFixed(3)}<br>` +
                  `Conversion Rate: ${(b.conversion_rate * 100).toFixed(
                    2
                  )}%<br>` +
                  `Lift: ${b.lift > 0 ? "+" : ""}${b.lift.toFixed(1)}%<br>` +
                  `n: ${b.n.toLocaleString()}`
              ),
              textposition: "none",
              hovertemplate:
                "<b>%{y}</b><br>" + "%{text}<br>" + "<extra></extra>",
            },
          ];

          const chartLayout = {
            title: {
              text: readableName,
              font: { size: 14, color: "#1F2937" },
            },
            xaxis: {
              title: "Odds Ratio (vs Baseline)",
              showgrid: true,
              gridcolor: "rgba(0,0,0,0.1)",
              zeroline: true,
              zerolinecolor: "rgba(0,0,0,0.3)",
            },
            yaxis: {
              title: "",
              showgrid: false,
            },
            margin: { l: 120, r: 20, t: 50, b: 50 },
            height: Math.max(250, bins.length * 60),
            plot_bgcolor: "rgba(0,0,0,0)",
            paper_bgcolor: "rgba(0,0,0,0)",
            font: { color: "#1F2937" },
          };

          const chartConfig = {
            displayModeBar: false,
            responsive: true,
          };

          return (
            <div
              key={varName}
              className="bg-gray-50 dark:bg-white/[0.03] rounded-lg p-4 border border-gray-200 dark:border-[rgba(255,255,255,0.08)]"
            >
              <button
                onClick={() => toggleVar(varName)}
                className="w-full flex items-center justify-between text-left"
              >
                <h4 className="font-semibold text-gray-900 dark:text-white">
                  {readableName}
                </h4>
                <span className="text-gray-500 dark:text-gray-400">
                  {isExpanded ? "▼" : "▶"}
                </span>
              </button>

              {isExpanded && (
                <div className="mt-4">
                  <Plot
                    data={chartData}
                    layout={chartLayout}
                    config={chartConfig}
                    style={{ width: "100%", height: "100%" }}
                  />
                  <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-2 text-xs">
                    {bins.map((bin, idx) => (
                      <div
                        key={idx}
                        className={`p-2 rounded ${
                          bin.is_baseline
                            ? "bg-blue-50 dark:bg-blue-900/20"
                            : bin.odds_ratio > 1
                            ? "bg-green-50 dark:bg-green-900/20"
                            : "bg-gray-50 dark:bg-white/[0.03]"
                        }`}
                      >
                        <p className="font-semibold text-gray-900 dark:text-white">
                          {bin.bin}
                        </p>
                        <p className="text-gray-600 dark:text-gray-400">
                          {bin.is_baseline
                            ? "Baseline"
                            : `+${bin.lift.toFixed(1)}%`}
                        </p>
                        <p className="text-gray-500 dark:text-gray-500 text-xs">
                          n={bin.n.toLocaleString()}
                        </p>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

export default BinnedCovariateView;

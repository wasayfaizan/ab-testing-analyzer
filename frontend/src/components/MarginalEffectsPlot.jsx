import React from "react";
import Plot from "react-plotly.js";

function MarginalEffectsPlot({
  marginalEffectsData,
  modelType = "Logistic Regression",
}) {
  if (!marginalEffectsData || Object.keys(marginalEffectsData).length === 0) {
    return null;
  }

  const formatVariableName = (name) => {
    return name.replace(/_/g, " ").replace(/\b\w/g, (l) => l.toUpperCase());
  };

  const isLogistic = modelType === "Logistic Regression";
  const yAxisTitle = isLogistic
    ? "Predicted Conversion Probability"
    : "Predicted Value";

  return (
    <div className="bg-white dark:bg-[#13161C] rounded-xl border border-gray-200 dark:border-[rgba(255,255,255,0.08)] p-6">
      <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-4">
        Marginal Effects Plots
      </h3>
      <p className="text-sm text-gray-600 dark:text-gray-400 mb-6">
        {isLogistic
          ? "Predicted conversion probability as a function of each numeric covariate, holding other variables at their mean values."
          : "Predicted outcome value as a function of each numeric covariate, holding other variables at their mean values."}
      </p>

      <div className="space-y-8">
        {Object.entries(marginalEffectsData).map(([varName, data]) => {
          const readableName = formatVariableName(
            data.original_name || varName
          );
          const isSignificant = data.significant;

          const chartData = [
            {
              type: "scatter",
              mode: "lines",
              name: "Predicted Probability",
              x: data.x_values,
              y: data.predicted_probs,
              line: {
                color: isSignificant ? "#25E6D1" : "#94A3B8",
                width: 2,
              },
              hovertemplate:
                `<b>${readableName}</b><br>` +
                `Value: %{x:.2f}<br>` +
                `${
                  isLogistic ? "Predicted Probability" : "Predicted Value"
                }: %{y:.4f}<br>` +
                `<extra></extra>`,
            },
            {
              type: "scatter",
              mode: "lines",
              name: "95% CI Upper",
              x: data.x_values,
              y: data.ci_upper,
              line: { width: 0 },
              showlegend: false,
              hoverinfo: "skip",
            },
            {
              type: "scatter",
              mode: "lines",
              name: "95% CI",
              x: data.x_values,
              y: data.ci_lower,
              line: { width: 0 },
              fill: "tonexty",
              fillcolor: isSignificant
                ? "rgba(37, 230, 209, 0.1)"
                : "rgba(148, 163, 184, 0.1)",
              showlegend: false,
              hoverinfo: "skip",
            },
          ];

          const chartLayout = {
            title: {
              text: readableName,
              font: { size: 14, color: "#1F2937" },
            },
            xaxis: {
              title: readableName,
              showgrid: true,
              gridcolor: "rgba(0,0,0,0.1)",
            },
            yaxis: {
              title: yAxisTitle,
              showgrid: true,
              gridcolor: "rgba(0,0,0,0.1)",
              range: isLogistic
                ? [0, Math.min(1, Math.max(...data.predicted_probs) * 1.1)]
                : undefined,
            },
            margin: { l: 60, r: 20, t: 50, b: 50 },
            height: 350,
            plot_bgcolor: "rgba(0,0,0,0)",
            paper_bgcolor: "rgba(0,0,0,0)",
            font: { color: "#1F2937" },
            legend: {
              x: 0.02,
              y: 0.98,
              bgcolor: "rgba(255,255,255,0.8)",
            },
          };

          const chartConfig = {
            displayModeBar: false,
            responsive: true,
          };

          return (
            <div
              key={varName}
              className="bg-gray-50 dark:bg-white/[0.03] rounded-lg p-4"
            >
              <Plot
                data={chartData}
                layout={chartLayout}
                config={chartConfig}
                style={{ width: "100%", height: "100%" }}
              />
              <div className="mt-2 text-xs text-gray-500 dark:text-gray-400">
                Coefficient: {data.coef > 0 ? "+" : ""}
                {data.coef.toFixed(4)} | Odds Ratio:{" "}
                {data.odds_ratio.toFixed(4)} | P-value:{" "}
                {data.p_value < 0.0001
                  ? data.p_value.toExponential(2)
                  : data.p_value.toFixed(4)}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export default MarginalEffectsPlot;

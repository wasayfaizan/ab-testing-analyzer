import React from "react";
import Plot from "react-plotly.js";

function DayOfWeekInsights({ dayEffects, summary, recommendations }) {
  if (!dayEffects || dayEffects.length === 0) {
    return null;
  }

  // Prepare data for chart
  const days = dayEffects.map((d) => d.day);
  const oddsRatios = dayEffects.map((d) => {
    // Cap extreme values for readability (between 0.1 and 10)
    const or = d.odds_ratio;
    return Math.max(0.1, Math.min(10, or));
  });
  const colors = dayEffects.map((d) => (d.significant ? "#25E6D1" : "#94A3B8"));
  const pValues = dayEffects.map((d) => d.p_value);

  // Calculate summary stats
  const topDay = dayEffects[0];
  const bottomDay = dayEffects[dayEffects.length - 1];
  const numSignificant = dayEffects.filter((d) => d.significant).length;

  // Correctly detect baseline day: the missing day from dummy encoding
  const expectedDays = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
  ];
  const returnedDays = dayEffects.map((d) => d.day);
  const baselineDay =
    expectedDays.find((day) => !returnedDays.includes(day)) || "Reference";

  // Chart data
  const chartData = [
    {
      type: "bar",
      orientation: "h",
      x: oddsRatios,
      y: days,
      marker: {
        color: colors,
        line: {
          color: "rgba(0,0,0,0.1)",
          width: 1,
        },
      },
      text: dayEffects.map(
        (d, i) =>
          `Odds Ratio: ${d.odds_ratio.toFixed(3)}<br>` +
          `Coefficient: ${d.coef.toFixed(4)}<br>` +
          `P-value: ${
            d.p_value < 0.0001
              ? d.p_value.toExponential(2)
              : d.p_value.toFixed(4)
          }<br>` +
          `95% CI: [${d.ci_lower.toFixed(4)}, ${d.ci_upper.toFixed(4)}]`
      ),
      textposition: "none",
      hovertemplate: "<b>%{y}</b><br>" + "%{text}<br>" + "<extra></extra>",
    },
  ];

  const chartLayout = {
    title: {
      text: "Day-of-Week Conversion Effectiveness",
      font: { size: 16, color: "#1F2937" },
    },
    xaxis: {
      title: "Odds Ratio",
      showgrid: true,
      gridcolor: "rgba(0,0,0,0.1)",
      zeroline: true,
      zerolinecolor: "rgba(0,0,0,0.3)",
    },
    yaxis: {
      title: "",
      showgrid: false,
    },
    margin: { l: 100, r: 20, t: 50, b: 50 },
    height: 400,
    plot_bgcolor: "rgba(0,0,0,0)",
    paper_bgcolor: "rgba(0,0,0,0)",
    font: { color: "#1F2937" },
  };

  const chartConfig = {
    displayModeBar: false,
    responsive: true,
  };

  return (
    <div className="bg-white dark:bg-[#13161C] rounded-xl border border-gray-200 dark:border-[rgba(255,255,255,0.08)] p-6">
      <div className="mb-6">
        <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-2">
          Day-of-Week Conversion Insights
        </h3>
        <p className="text-sm text-gray-600 dark:text-gray-400">
          How user behavior varies by weekday
        </p>
      </div>

      {/* Summary Panel */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4 border border-blue-200 dark:border-blue-800">
          <p className="text-xs font-semibold text-blue-700 dark:text-blue-300 mb-1">
            Top Performing Day
          </p>
          <p className="text-lg font-bold text-blue-900 dark:text-blue-200">
            {topDay.day}
          </p>
          <p className="text-xs text-blue-600 dark:text-blue-400 mt-1">
            {((topDay.odds_ratio - 1) * 100).toFixed(1)}% higher odds
          </p>
        </div>

        <div className="bg-gray-50 dark:bg-white/[0.03] rounded-lg p-4 border border-gray-200 dark:border-[rgba(255,255,255,0.08)]">
          <p className="text-xs font-semibold text-gray-700 dark:text-gray-300 mb-1">
            Lowest Performing Day
          </p>
          <p className="text-lg font-bold text-gray-900 dark:text-white">
            {bottomDay.day}
          </p>
          <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">
            {((bottomDay.odds_ratio - 1) * 100).toFixed(1)}% vs baseline
          </p>
        </div>

        <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-4 border border-green-200 dark:border-green-800">
          <p className="text-xs font-semibold text-green-700 dark:text-green-300 mb-1">
            Significant Days
          </p>
          <p className="text-lg font-bold text-green-900 dark:text-green-200">
            {numSignificant}
          </p>
          <p className="text-xs text-green-600 dark:text-green-400 mt-1">
            out of {dayEffects.length} days
          </p>
        </div>

        <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-4 border border-purple-200 dark:border-purple-800">
          <p className="text-xs font-semibold text-purple-700 dark:text-purple-300 mb-1">
            Baseline Day
          </p>
          <p className="text-lg font-bold text-purple-900 dark:text-purple-200">
            {baselineDay}
          </p>
          <p className="text-xs text-purple-600 dark:text-purple-400 mt-1">
            Reference category
          </p>
        </div>
      </div>

      {/* Chart */}
      <div className="mb-6">
        <div className="bg-gray-50 dark:bg-white/[0.03] rounded-lg p-4">
          <Plot
            data={chartData}
            layout={chartLayout}
            config={chartConfig}
            style={{ width: "100%", height: "100%" }}
          />
        </div>
        <div className="flex items-center justify-center gap-4 mt-4 text-xs">
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded bg-[#25E6D1]"></div>
            <span className="text-gray-600 dark:text-gray-400">
              Significant (p &lt; 0.05)
            </span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded bg-[#94A3B8]"></div>
            <span className="text-gray-600 dark:text-gray-400">
              Not Significant
            </span>
          </div>
        </div>
      </div>

      {/* Interpretation */}
      {summary && (
        <div className="mb-6 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
          <h4 className="text-sm font-semibold text-blue-900 dark:text-blue-300 mb-2">
            Interpretation
          </h4>
          <p className="text-sm text-blue-800 dark:text-blue-300 leading-relaxed">
            {summary}
          </p>
        </div>
      )}

      {/* Recommendations */}
      {recommendations && (
        <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg border border-green-200 dark:border-green-800">
          <h4 className="text-sm font-semibold text-green-900 dark:text-green-300 mb-2 flex items-center gap-2">
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
          </h4>
          <p className="text-sm text-green-800 dark:text-green-300 leading-relaxed">
            {recommendations}
          </p>
        </div>
      )}
    </div>
  );
}

export default DayOfWeekInsights;

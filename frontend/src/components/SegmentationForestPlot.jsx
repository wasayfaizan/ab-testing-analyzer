import React from "react";
import Plot from "react-plotly.js";

function SegmentationForestPlot({ forestPlotData }) {
  if (!forestPlotData || forestPlotData.length === 0) {
    return null;
  }

  // Sort by effect size (descending)
  const sortedData = [...forestPlotData].sort((a, b) => b.effect_size - a.effect_size);

  const segments = sortedData.map((d) => d.segment);
  const effectSizes = sortedData.map((d) => d.effect_size);
  
  // CI values from backend are absolute differences (treatment_rate - control_rate)
  // Convert to percentage lift terms: (diff / control_rate) * 100
  const ciLower = sortedData.map((d) => {
    if (d.control_rate > 0) {
      // Convert absolute difference to percentage lift
      return (d.ci_lower / d.control_rate) * 100;
    }
    return d.ci_lower;
  });
  const ciUpper = sortedData.map((d) => {
    if (d.control_rate > 0) {
      return (d.ci_upper / d.control_rate) * 100;
    }
    return d.ci_upper;
  });
  
  const colors = sortedData.map((d) => (d.significant ? "#25E6D1" : "#94A3B8"));

  // Create error bars for confidence intervals
  const errorBars = {
    type: "data",
    symmetric: false,
    array: sortedData.map((d, i) => Math.max(0, ciUpper[i] - effectSizes[i])),
    arrayminus: sortedData.map((d, i) => Math.max(0, effectSizes[i] - ciLower[i])),
    color: "rgba(0,0,0,0.3)",
    thickness: 1.5,
    width: 3,
  };

  // Create hover text
  const hoverText = sortedData.map(
    (d, i) =>
      `<b>${d.segment}</b><br>` +
      `Control: ${(d.control_rate * 100).toFixed(2)}%<br>` +
      `Treatment: ${(d.treatment_rate * 100).toFixed(2)}%<br>` +
      `Lift: ${d.lift > 0 ? "+" : ""}${d.lift.toFixed(2)}%<br>` +
      `P-value: ${d.p_value < 0.0001 ? d.p_value.toExponential(2) : d.p_value.toFixed(4)}<br>` +
      `n: ${d.n.toLocaleString()}`
  );

  const chartData = [
    {
      type: "bar",
      orientation: "h",
      x: effectSizes,
      y: segments,
      marker: {
        color: colors,
        line: {
          color: "rgba(0,0,0,0.1)",
          width: 1,
        },
      },
      error_x: errorBars,
      text: hoverText,
      textposition: "none",
      hovertemplate: "%{text}<extra></extra>",
    },
  ];

  // Find max absolute value for symmetric x-axis
  const maxAbs = Math.max(
    ...effectSizes.map(Math.abs),
    ...ciUpper.map(Math.abs),
    ...ciLower.map(Math.abs)
  );
  const xAxisRange = [-maxAbs * 1.1, maxAbs * 1.1];

  const chartLayout = {
    title: {
      text: "Segment Treatment Effect Forest Plot",
      font: { size: 16, color: "#1F2937" },
    },
    xaxis: {
      title: "Treatment Lift (%)",
      showgrid: true,
      gridcolor: "rgba(0,0,0,0.1)",
      zeroline: true,
      zerolinecolor: "rgba(0,0,0,0.3)",
      zerolinewidth: 2,
      range: xAxisRange,
    },
    yaxis: {
      title: "",
      showgrid: false,
      autorange: "reversed", // Top to bottom
    },
    margin: { l: 120, r: 20, t: 50, b: 50 },
    height: Math.max(400, segments.length * 50),
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
      <div className="mb-4">
        <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-2">
          Segment Treatment Effect Forest Plot
        </h3>
        <p className="text-sm text-gray-600 dark:text-gray-400">
          Treatment lift by segment with 95% confidence intervals
        </p>
      </div>

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
  );
}

export default SegmentationForestPlot;

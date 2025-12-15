import React, { useState, useEffect } from "react";
import { apiClient } from "../config/api";

function PowerCalculator() {
  const [baselineRate, setBaselineRate] = useState(0.1);
  const [expectedLift, setExpectedLift] = useState(0.1);
  const [alpha, setAlpha] = useState(0.05);
  const [power, setPower] = useState(0.8);
  const [requiredSampleSize, setRequiredSampleSize] = useState(null);
  const [effectSize, setEffectSize] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [calculationMode, setCalculationMode] = useState("sample_size"); // 'sample_size' or 'mde'

  // For MDE calculation
  const [sampleSizePerGroup, setSampleSizePerGroup] = useState(1000);
  const [mdeResult, setMdeResult] = useState(null);

  const calculateSampleSize = async () => {
    setLoading(true);
    setError("");
    try {
      const response = await axios.post("/api/power-analysis", {
        baseline_rate: baselineRate,
        expected_lift: expectedLift > 1 ? expectedLift / 100 : expectedLift,
        alpha: alpha,
        power: power,
      });
      setRequiredSampleSize(response.data.sample_size_per_group);
      setEffectSize(response.data.effect_size);
    } catch (err) {
      setError(err.response?.data?.detail || "Error calculating sample size");
      setRequiredSampleSize(null);
    } finally {
      setLoading(false);
    }
  };

  const calculateMDE = async () => {
    setLoading(true);
    setError("");
    try {
      const response = await apiClient.post("/api/mde-analysis", {
        baseline_rate: baselineRate,
        sample_size_per_group: sampleSizePerGroup,
        alpha: alpha,
        power: power,
      });
      setMdeResult(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || "Error calculating MDE");
      setMdeResult(null);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (calculationMode === "sample_size") {
      calculateSampleSize();
    } else {
      calculateMDE();
    }
  }, [
    baselineRate,
    expectedLift,
    alpha,
    power,
    calculationMode,
    sampleSizePerGroup,
  ]);

  return (
    <div className="bg-white dark:bg-[#13161C] rounded-xl border border-gray-200 dark:border-[rgba(255,255,255,0.08)] p-6">
      <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-4">
        Sample Size Calculator
      </h3>

      {/* Mode Toggle */}
      <div className="flex gap-2 mb-6">
        <button
          onClick={() => setCalculationMode("sample_size")}
          className={`px-4 py-2 rounded-lg border-2 transition-colors ${
            calculationMode === "sample_size"
              ? "border-blue-500 bg-blue-50 dark:bg-blue-900/20 text-blue-700 dark:text-blue-300"
              : "border-gray-300 dark:border-gray-600 hover:border-gray-400"
          }`}
        >
          Calculate Sample Size
        </button>
        <button
          onClick={() => setCalculationMode("mde")}
          className={`px-4 py-2 rounded-lg border-2 transition-colors ${
            calculationMode === "mde"
              ? "border-blue-500 bg-blue-50 dark:bg-blue-900/20 text-blue-700 dark:text-blue-300"
              : "border-gray-300 dark:border-gray-600 hover:border-gray-400"
          }`}
        >
          Calculate MDE
        </button>
      </div>

      {error && (
        <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-3 mb-4">
          <p className="text-sm text-red-800 dark:text-red-300">{error}</p>
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Baseline Conversion Rate
          </label>
          <input
            type="number"
            step="0.01"
            min="0"
            max="1"
            value={baselineRate}
            onChange={(e) => setBaselineRate(parseFloat(e.target.value) || 0)}
            className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-[#0D0F14] text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          />
          <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
            Enter as decimal (e.g., 0.1 for 10%)
          </p>
        </div>

        {calculationMode === "sample_size" ? (
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Expected Lift
            </label>
            <input
              type="number"
              step="0.01"
              min="0"
              value={expectedLift}
              onChange={(e) => setExpectedLift(parseFloat(e.target.value) || 0)}
              className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-[#0D0F14] text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            />
            <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
              Enter as decimal (e.g., 0.1 for 10%) or percentage (e.g., 10)
            </p>
          </div>
        ) : (
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Sample Size per Group
            </label>
            <input
              type="number"
              step="100"
              min="10"
              value={sampleSizePerGroup}
              onChange={(e) =>
                setSampleSizePerGroup(parseInt(e.target.value) || 1000)
              }
              className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-[#0D0F14] text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            />
          </div>
        )}

        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Alpha (Significance Level)
          </label>
          <input
            type="number"
            step="0.01"
            min="0"
            max="1"
            value={alpha}
            onChange={(e) => setAlpha(parseFloat(e.target.value) || 0.05)}
            className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-[#0D0F14] text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          />
          <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
            Typically 0.05 (5%)
          </p>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Power (1 - Beta)
          </label>
          <input
            type="number"
            step="0.01"
            min="0"
            max="1"
            value={power}
            onChange={(e) => setPower(parseFloat(e.target.value) || 0.8)}
            className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-[#0D0F14] text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          />
          <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
            Typically 0.8 (80%) or 0.9 (90%)
          </p>
        </div>
      </div>

      {loading && (
        <div className="text-center py-4">
          <div className="inline-block animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600"></div>
        </div>
      )}

      {calculationMode === "sample_size" && requiredSampleSize && (
        <div className="bg-blue-50 dark:bg-blue-900/20 border-2 border-blue-200 dark:border-blue-800 rounded-lg p-4">
          <p className="text-sm font-semibold text-blue-900 dark:text-blue-300 mb-1">
            Required Sample Size per Group
          </p>
          <p className="text-2xl font-bold text-blue-600 dark:text-blue-400">
            {requiredSampleSize.toLocaleString()}
          </p>
          <p className="text-xs text-blue-700 dark:text-blue-400 mt-2">
            Total sample size needed:{" "}
            {(requiredSampleSize * 2).toLocaleString()}
          </p>
          {effectSize && (
            <p className="text-xs text-blue-700 dark:text-blue-400 mt-1">
              Effect size (Cohen's h): {effectSize.toFixed(3)} (
              {effectSize < 0.2
                ? "small"
                : effectSize < 0.5
                ? "medium"
                : "large"}
              )
            </p>
          )}
        </div>
      )}

      {calculationMode === "mde" && mdeResult && (
        <div className="bg-green-50 dark:bg-green-900/20 border-2 border-green-200 dark:border-green-800 rounded-lg p-4">
          <p className="text-sm font-semibold text-green-900 dark:text-green-300 mb-1">
            Minimum Detectable Effect
          </p>
          <p className="text-2xl font-bold text-green-600 dark:text-green-400">
            {mdeResult.mde_lift?.toFixed(2) || "N/A"}%
          </p>
          <p className="text-xs text-green-700 dark:text-green-400 mt-2">
            With {sampleSizePerGroup.toLocaleString()} per group, you can detect
            a minimum lift of {mdeResult.mde_lift?.toFixed(2)}%
          </p>
        </div>
      )}
    </div>
  );
}

export default PowerCalculator;

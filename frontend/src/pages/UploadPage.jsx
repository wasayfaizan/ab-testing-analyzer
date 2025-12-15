import React, { useState, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import { useDropzone } from "react-dropzone";
import axios from "axios";

function UploadPage({ setAnalysisResults }) {
  const [file, setFile] = useState(null);
  const [csvData, setCsvData] = useState(null);
  const [columns, setColumns] = useState([]);
  const [groupColumn, setGroupColumn] = useState("");
  const [metricColumn, setMetricColumn] = useState("");
  const [metricType, setMetricType] = useState("binary");
  const [availableGroups, setAvailableGroups] = useState([]);
  const [groupA, setGroupA] = useState(""); // Control group
  const [groupB, setGroupB] = useState(""); // Treatment group
  const [enableCuped, setEnableCuped] = useState(false);
  const [covariateColumn, setCovariateColumn] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [analysisMode, setAnalysisMode] = useState("ab"); // 'ab', 'multi-variant', 'segmentation', 'time-based', 'regression', 'non-parametric'
  const [selectedTreatmentGroups, setSelectedTreatmentGroups] = useState([]); // For multi-variant
  const [segmentColumn, setSegmentColumn] = useState(""); // For segmentation
  const [timestampColumn, setTimestampColumn] = useState(""); // For time-based
  const [timeGranularity, setTimeGranularity] = useState("day"); // For time-based
  const [regressionCovariates, setRegressionCovariates] = useState([]); // For regression
  const [nonParametricTestType, setNonParametricTestType] =
    useState("mann_whitney"); // For non-parametric
  const navigate = useNavigate();

  // Extract groups from CSV when group column is selected
  const extractGroups = useCallback((file, groupColName) => {
    if (!file || !groupColName) {
      setAvailableGroups([]);
      setGroupA("");
      setGroupB("");
      return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
      const text = e.target.result;
      const lines = text.split("\n");
      if (lines.length < 2) return;

      const headers = lines[0].split(",").map((h) => h.trim());
      const groupIndex = headers.indexOf(groupColName);

      if (groupIndex === -1) return;

      const groups = new Set();
      for (let i = 1; i < lines.length; i++) {
        const values = lines[i].split(",");
        if (values[groupIndex]) {
          const group = values[groupIndex].trim();
          if (group && group.toLowerCase() !== "nan" && group !== "") {
            groups.add(group);
          }
        }
      }

      const uniqueGroups = Array.from(groups);
      setAvailableGroups(uniqueGroups);

      // Auto-select first two groups if available
      if (uniqueGroups.length >= 2) {
        setGroupA(uniqueGroups[0]);
        setGroupB(uniqueGroups[1]);
        // For multi-variant, select remaining groups
        if (uniqueGroups.length > 2) {
          setSelectedTreatmentGroups(uniqueGroups.slice(2));
        }
      } else if (uniqueGroups.length === 1) {
        setGroupA(uniqueGroups[0]);
        setGroupB("");
      }
    };
    reader.readAsText(file);
  }, []);

  const onDrop = useCallback(
    (acceptedFiles) => {
      const file = acceptedFiles[0];
      if (file) {
        setFile(file);
        setError("");

        // Read CSV to preview
        const reader = new FileReader();
        reader.onload = (e) => {
          const text = e.target.result;
          const lines = text.split("\n").slice(0, 6); // First 5 rows
          const headers = lines[0].split(",");
          setColumns(headers.map((h) => h.trim()));
          setCsvData(lines);

          // Auto-select common column names
          const groupCol = headers.find(
            (h) =>
              h.toLowerCase().includes("group") ||
              h.toLowerCase().includes("variant") ||
              h.toLowerCase() === "a" ||
              h.toLowerCase() === "b"
          );
          const metricCol = headers.find(
            (h) =>
              h.toLowerCase().includes("conversion") ||
              h.toLowerCase().includes("metric") ||
              h.toLowerCase().includes("revenue") ||
              h.toLowerCase().includes("value")
          );

          if (groupCol) {
            const colName = groupCol.trim();
            setGroupColumn(colName);
            // Extract groups after setting column
            setTimeout(() => extractGroups(file, colName), 100);
          }
          if (metricCol) setMetricColumn(metricCol.trim());
        };
        reader.readAsText(file);
      }
    },
    [extractGroups]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "text/csv": [".csv"],
      "text/plain": [".csv"],
    },
    multiple: false,
  });

  const handleAnalyze = async () => {
    console.log("handleAnalyze called", {
      file: file?.name,
      groupColumn,
      metricColumn,
      groupA,
      groupB,
      analysisMode,
    });

    if (!file || !groupColumn || !metricColumn) {
      setError("Please select a file and specify group and metric columns");
      return;
    }

    // Validation based on analysis mode
    if (analysisMode === "multi-variant") {
      if (!groupA || selectedTreatmentGroups.length === 0) {
        setError(
          "Please select a control group and at least one treatment group"
        );
        return;
      }
    } else if (analysisMode === "segmentation") {
      if (!groupA || !groupB || !segmentColumn) {
        setError(
          "Please select control group, treatment group, and segment column"
        );
        return;
      }
    } else if (analysisMode === "time-based") {
      if (!groupA || !groupB || !timestampColumn) {
        setError(
          "Please select control group, treatment group, and timestamp column"
        );
        return;
      }
    } else if (analysisMode === "regression") {
      if (!groupA || !groupB || regressionCovariates.length === 0) {
        setError(
          "Please select control group, treatment group, and at least one covariate"
        );
        return;
      }
    } else if (analysisMode === "non-parametric") {
      if (!groupA) {
        setError("Please select at least one group");
        return;
      }
      if (nonParametricTestType === "mann_whitney" && !groupB) {
        setError("Please select both groups for Mann-Whitney test");
        return;
      }
    } else {
      // Standard A/B test
      if (!groupA || !groupB) {
        setError(
          "Please select which group is Control (A) and which is Treatment (B)"
        );
        return;
      }
      if (groupA === groupB) {
        setError("Control and Treatment groups must be different");
        return;
      }
    }

    setLoading(true);
    setError("");
    console.log("Starting analysis...");

    try {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("group_column", groupColumn);
      formData.append("metric_column", metricColumn);
      formData.append("metric_type", metricType);

      let endpoint = "/api/analyze";

      // Route to appropriate endpoint based on analysis mode
      if (analysisMode === "multi-variant") {
        endpoint = "/api/multi-variant-analysis";
        formData.append("control_group", groupA);
        formData.append("treatment_groups", selectedTreatmentGroups.join(","));
      } else if (analysisMode === "segmentation") {
        endpoint = "/api/segmentation-analysis";
        formData.append("group_a_name", groupA);
        formData.append("group_b_name", groupB);
        formData.append("segment_column", segmentColumn);
      } else if (analysisMode === "time-based") {
        endpoint = "/api/time-based-analysis";
        formData.append("group_a_name", groupA);
        formData.append("group_b_name", groupB);
        formData.append("timestamp_column", timestampColumn);
        formData.append("time_granularity", timeGranularity);
      } else if (analysisMode === "regression") {
        endpoint = "/api/regression-analysis";
        formData.append("group_a_name", groupA);
        formData.append("group_b_name", groupB);
        formData.append("covariate_columns", regressionCovariates.join(","));
      } else if (analysisMode === "non-parametric") {
        endpoint = "/api/non-parametric-test";
        formData.append("test_type", nonParametricTestType);
        formData.append("group_a_name", groupA);
        if (groupB) formData.append("group_b_name", groupB);
        if (
          nonParametricTestType === "kruskal_wallis" &&
          selectedTreatmentGroups.length > 0
        ) {
          formData.append(
            "additional_groups",
            selectedTreatmentGroups.join(",")
          );
        }
      } else {
        // Standard A/B test
        formData.append("group_a_name", groupA);
        formData.append("group_b_name", groupB);
        formData.append("enable_cuped", enableCuped.toString());
        if (enableCuped && covariateColumn) {
          formData.append("covariate_column", covariateColumn);
        }
      }

      console.log("Sending request to backend...", endpoint);
      const response = await axios.post(endpoint, formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });
      console.log("Analysis response received:", response.data);

      // For standard A/B test, get AI interpretation
      if (analysisMode === "ab") {
        console.log("Requesting AI interpretation...");
        const interpretationResponse = await axios.post(
          "/api/generate-interpretation",
          {
            lift: response.data.lift,
            p_value: response.data.p_value,
            effect_size: response.data.effect_size,
            ci_lower: response.data.confidence_interval.lower,
            ci_upper: response.data.confidence_interval.upper,
            prob_b_better: response.data.bayesian.prob_b_better,
            metric_type: response.data.metric_type,
            test_type: response.data.test_type,
          }
        );
        console.log("Interpretation received");

        const results = {
          ...response.data,
          interpretation: interpretationResponse.data.interpretation,
          analysis_mode: analysisMode,
        };

        // Save to localStorage
        localStorage.setItem("abTestResults", JSON.stringify(results));

        setAnalysisResults(results);
        navigate("/results");
      } else {
        // For advanced analyses, store results with mode
        const results = {
          ...response.data,
          analysis_mode: analysisMode,
        };
        console.log("Storing advanced analysis results:", results);
        console.log("Analysis mode:", analysisMode);
        localStorage.setItem("abTestResults", JSON.stringify(results));
        setAnalysisResults(results);
        navigate("/results");
      }
    } catch (err) {
      console.error("Analysis error:", err);
      if (
        err.code === "ERR_NETWORK" ||
        err.message?.includes("Network Error")
      ) {
        setError(
          "Network error: Could not connect to the backend server. Please make sure the backend is running on http://localhost:8000"
        );
      } else if (err.response?.data?.detail) {
        setError(err.response.data.detail);
      } else if (err.response?.data?.message) {
        setError(err.response.data.message);
      } else if (err.message) {
        setError(err.message);
      } else {
        setError(
          "Error analyzing data. Please check your CSV file and try again."
        );
      }
    } finally {
      setLoading(false);
    }
  };

  // Format file size
  const formatFileSize = (bytes) => {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + " " + sizes[i];
  };

  // Copy sample CSV to clipboard
  const copySampleCSV = () => {
    const sampleCSV = `group,conversion,revenue
A,1,25.50
A,0,0.00
A,1,30.00
B,1,28.00
B,1,32.50
B,0,0.00`;
    navigator.clipboard.writeText(sampleCSV);
    // You could add a toast notification here
  };

  // Determine current step
  const currentStep = file
    ? columns.length > 0 && groupColumn && metricColumn
      ? 3
      : 2
    : 1;

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 via-white to-gray-50 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-4xl mx-auto">
        {/* Hero Card */}
        <div className="bg-white rounded-2xl shadow-lg border border-gray-200 p-8 mb-8">
          <div className="text-center">
            <div className="flex items-center justify-center gap-3 mb-4">
              <span className="text-4xl">📊</span>
              <span className="text-4xl">📈</span>
              <span className="text-4xl">🧪</span>
            </div>
            <h1 className="text-4xl font-bold text-gray-900 mb-3">
              A/B Testing Experiment Analyzer
            </h1>
            <p className="text-lg text-gray-600 max-w-2xl mx-auto">
              Upload your experiment data and run statistically rigorous
              analysis
            </p>
          </div>
        </div>

        {/* Step Progress Indicator */}
        <div className="bg-white rounded-xl shadow-md border border-gray-200 p-6 mb-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center flex-1">
              <div
                className={`flex items-center ${
                  currentStep >= 1 ? "text-blue-600" : "text-gray-400"
                }`}
              >
                <div
                  className={`w-10 h-10 rounded-full flex items-center justify-center font-semibold ${
                    currentStep >= 1
                      ? "bg-blue-100 text-blue-600"
                      : "bg-gray-100 text-gray-400"
                  }`}
                >
                  {currentStep > 1 ? "✓" : "1"}
                </div>
                <span className="ml-3 font-medium">Upload CSV</span>
              </div>
              <div
                className={`flex-1 h-1 mx-4 ${
                  currentStep >= 2 ? "bg-blue-600" : "bg-gray-200"
                }`}
              ></div>
            </div>
            <div className="flex items-center flex-1">
              <div
                className={`flex items-center ${
                  currentStep >= 2 ? "text-blue-600" : "text-gray-400"
                }`}
              >
                <div
                  className={`w-10 h-10 rounded-full flex items-center justify-center font-semibold ${
                    currentStep >= 2
                      ? "bg-blue-100 text-blue-600"
                      : "bg-gray-100 text-gray-400"
                  }`}
                >
                  {currentStep > 2 ? "✓" : "2"}
                </div>
                <span className="ml-3 font-medium">Configure Analysis</span>
              </div>
              <div
                className={`flex-1 h-1 mx-4 ${
                  currentStep >= 3 ? "bg-blue-600" : "bg-gray-200"
                }`}
              ></div>
            </div>
            <div className="flex items-center">
              <div
                className={`flex items-center ${
                  currentStep >= 3 ? "text-blue-600" : "text-gray-400"
                }`}
              >
                <div
                  className={`w-10 h-10 rounded-full flex items-center justify-center font-semibold ${
                    currentStep >= 3
                      ? "bg-blue-100 text-blue-600"
                      : "bg-gray-100 text-gray-400"
                  }`}
                >
                  3
                </div>
                <span className="ml-3 font-medium">View Results</span>
              </div>
            </div>
          </div>
        </div>

        {/* Analysis Mode Selection */}
        {columns.length > 0 && (
          <div className="bg-white rounded-xl shadow-md border border-gray-200 p-6 mb-6">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">
              Analysis Mode
            </h2>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
              <button
                onClick={() => setAnalysisMode("ab")}
                className={`px-4 py-2 rounded-lg border-2 transition-colors ${
                  analysisMode === "ab"
                    ? "border-blue-500 bg-blue-50 text-blue-700"
                    : "border-gray-300 hover:border-gray-400"
                }`}
              >
                A/B Test
              </button>
              <button
                onClick={() => setAnalysisMode("multi-variant")}
                className={`px-4 py-2 rounded-lg border-2 transition-colors ${
                  analysisMode === "multi-variant"
                    ? "border-blue-500 bg-blue-50 text-blue-700"
                    : "border-gray-300 hover:border-gray-400"
                }`}
              >
                Multi-Variant
              </button>
              <button
                onClick={() => setAnalysisMode("segmentation")}
                className={`px-4 py-2 rounded-lg border-2 transition-colors ${
                  analysisMode === "segmentation"
                    ? "border-blue-500 bg-blue-50 text-blue-700"
                    : "border-gray-300 hover:border-gray-400"
                }`}
              >
                Segmentation
              </button>
              <button
                onClick={() => setAnalysisMode("time-based")}
                className={`px-4 py-2 rounded-lg border-2 transition-colors ${
                  analysisMode === "time-based"
                    ? "border-blue-500 bg-blue-50 text-blue-700"
                    : "border-gray-300 hover:border-gray-400"
                }`}
              >
                Time-Based
              </button>
              <button
                onClick={() => setAnalysisMode("regression")}
                className={`px-4 py-2 rounded-lg border-2 transition-colors ${
                  analysisMode === "regression"
                    ? "border-blue-500 bg-blue-50 text-blue-700"
                    : "border-gray-300 hover:border-gray-400"
                }`}
              >
                Regression
              </button>
              <button
                onClick={() => setAnalysisMode("non-parametric")}
                className={`px-4 py-2 rounded-lg border-2 transition-colors ${
                  analysisMode === "non-parametric"
                    ? "border-blue-500 bg-blue-50 text-blue-700"
                    : "border-gray-300 hover:border-gray-400"
                }`}
              >
                Non-Parametric
              </button>
            </div>
          </div>
        )}

        {/* Upload Card */}
        <div className="bg-white rounded-xl shadow-md border border-gray-200 p-8 mb-6">
          <h2 className="text-2xl font-bold text-gray-900 mb-6">
            Upload CSV File
          </h2>

          <div
            {...getRootProps()}
            className={`border-2 border-dashed rounded-xl p-12 text-center cursor-pointer transition-all ${
              isDragActive
                ? "border-blue-500 bg-blue-50 shadow-inner"
                : file
                ? "border-green-300 bg-green-50/30"
                : "border-gray-300 hover:border-blue-400 hover:bg-gray-50"
            }`}
          >
            <input {...getInputProps()} />
            <div className="space-y-4">
              {file ? (
                <>
                  <div className="flex items-center justify-center">
                    <div className="w-16 h-16 rounded-full bg-green-100 flex items-center justify-center">
                      <svg
                        className="w-8 h-8 text-green-600"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M5 13l4 4L19 7"
                        />
                      </svg>
                    </div>
                  </div>
                  <div>
                    <p className="text-lg font-semibold text-gray-900">
                      {file.name}
                    </p>
                    <p className="text-sm text-gray-600 mt-1">
                      {formatFileSize(file.size)}
                    </p>
                  </div>
                  <p className="text-xs text-gray-500">
                    Click or drag to replace file
                  </p>
                </>
              ) : (
                <>
                  <svg
                    className="mx-auto h-16 w-16 text-gray-400"
                    stroke="currentColor"
                    fill="none"
                    viewBox="0 0 48 48"
                  >
                    <path
                      d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02"
                      strokeWidth={2}
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                  </svg>
                  <div>
                    <p className="text-lg font-medium text-gray-900">
                      Drag & drop your CSV file or click to browse
                    </p>
                    <p className="text-sm text-gray-500 mt-2">
                      CSV only • Auto-detects columns
                    </p>
                  </div>
                </>
              )}
            </div>
          </div>

          {/* Detected Columns */}
          {columns.length > 0 && (
            <div className="mt-6 p-4 bg-blue-50 rounded-lg border border-blue-200">
              <p className="text-sm font-medium text-blue-900 mb-2">
                ✓ Detected {columns.length} column
                {columns.length !== 1 ? "s" : ""}:
              </p>
              <div className="flex flex-wrap gap-2">
                {columns.map((col, idx) => (
                  <span
                    key={idx}
                    className="px-3 py-1 bg-white rounded-md text-xs font-mono text-gray-700 border border-blue-200"
                  >
                    {col}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Privacy Reassurance */}
          <div className="mt-4 flex items-start gap-2 text-xs text-gray-500">
            <svg
              className="w-4 h-4 text-gray-400 mt-0.5 flex-shrink-0"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z"
              />
            </svg>
            <span>No data is stored — analysis runs locally</span>
          </div>

          {/* CSV Preview */}
          {csvData && (
            <div className="mt-6">
              <h3 className="text-sm font-medium text-gray-700 mb-3">
                CSV Preview
              </h3>
              <div className="overflow-x-auto border border-gray-200 rounded-lg shadow-sm">
                <table className="min-w-full divide-y divide-gray-200 text-xs">
                  <thead className="bg-gray-50">
                    <tr>
                      {csvData[0].split(",").map((col, idx) => (
                        <th
                          key={idx}
                          className="px-3 py-2 text-left font-medium text-gray-700"
                        >
                          {col.trim()}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    {csvData.slice(1, 6).map((row, rowIdx) => (
                      <tr key={rowIdx}>
                        {row.split(",").map((cell, cellIdx) => (
                          <td key={cellIdx} className="px-3 py-2 text-gray-600">
                            {cell.trim()}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>

        {/* Configuration Card */}
        {columns.length > 0 && (
          <div className="bg-white rounded-xl shadow-md border border-gray-200 p-8 mb-6">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">
              Configuration
            </h2>

            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Group Column
                </label>
                <select
                  value={groupColumn}
                  onChange={(e) => {
                    const newCol = e.target.value;
                    setGroupColumn(newCol);
                    setGroupA("");
                    setGroupB("");
                    setAvailableGroups([]);
                    if (file && newCol) {
                      extractGroups(file, newCol);
                    }
                  }}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                >
                  <option value="">Select column...</option>
                  {columns.map((col) => (
                    <option key={col} value={col}>
                      {col}
                    </option>
                  ))}
                </select>
              </div>

              {/* Multi-Variant Mode */}
              {analysisMode === "multi-variant" &&
                groupColumn &&
                availableGroups.length > 0 && (
                  <>
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Control Group <span className="text-red-500">*</span>
                      </label>
                      <select
                        value={groupA}
                        onChange={(e) => setGroupA(e.target.value)}
                        className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                      >
                        <option value="">Select control group...</option>
                        {availableGroups.map((group) => (
                          <option key={group} value={group}>
                            {group}
                          </option>
                        ))}
                      </select>
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Treatment Groups <span className="text-red-500">*</span>
                      </label>
                      <div className="space-y-2">
                        {availableGroups
                          .filter((g) => g !== groupA)
                          .map((group) => (
                            <label key={group} className="flex items-center">
                              <input
                                type="checkbox"
                                checked={selectedTreatmentGroups.includes(
                                  group
                                )}
                                onChange={(e) => {
                                  if (e.target.checked) {
                                    setSelectedTreatmentGroups([
                                      ...selectedTreatmentGroups,
                                      group,
                                    ]);
                                  } else {
                                    setSelectedTreatmentGroups(
                                      selectedTreatmentGroups.filter(
                                        (g) => g !== group
                                      )
                                    );
                                  }
                                }}
                                className="mr-2"
                              />
                              <span>{group}</span>
                            </label>
                          ))}
                      </div>
                    </div>
                  </>
                )}

              {/* Standard A/B, Segmentation, Time-Based, Regression, Non-Parametric Modes */}
              {analysisMode !== "multi-variant" &&
                groupColumn &&
                availableGroups.length > 0 && (
                  <>
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Control Group (A){" "}
                        <span className="text-red-500">*</span>
                      </label>
                      <select
                        value={groupA}
                        onChange={(e) => setGroupA(e.target.value)}
                        className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                      >
                        <option value="">Select control group...</option>
                        {availableGroups.map((group) => (
                          <option key={group} value={group}>
                            {group}
                          </option>
                        ))}
                      </select>
                      <p className="text-xs text-gray-500 mt-1">
                        This is your baseline/control group
                      </p>
                    </div>

                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Treatment Group (B){" "}
                        <span className="text-red-500">*</span>
                      </label>
                      <select
                        value={groupB}
                        onChange={(e) => setGroupB(e.target.value)}
                        className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                      >
                        <option value="">Select treatment group...</option>
                        {availableGroups.map((group) => (
                          <option key={group} value={group}>
                            {group}
                          </option>
                        ))}
                      </select>
                      <p className="text-xs text-gray-500 mt-1">
                        This is your test/treatment group
                      </p>
                    </div>
                  </>
                )}

              {/* Segmentation Mode */}
              {analysisMode === "segmentation" && (
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Segment Column <span className="text-red-500">*</span>
                  </label>
                  <select
                    value={segmentColumn}
                    onChange={(e) => setSegmentColumn(e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                  >
                    <option value="">Select segment column...</option>
                    {columns
                      .filter(
                        (col) => col !== groupColumn && col !== metricColumn
                      )
                      .map((col) => (
                        <option key={col} value={col}>
                          {col}
                        </option>
                      ))}
                  </select>
                  <p className="text-xs text-gray-500 mt-1">
                    Column to segment by (e.g., demographics, cohorts)
                  </p>
                </div>
              )}

              {/* Time-Based Mode */}
              {analysisMode === "time-based" && (
                <>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Timestamp Column <span className="text-red-500">*</span>
                    </label>
                    <select
                      value={timestampColumn}
                      onChange={(e) => setTimestampColumn(e.target.value)}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                    >
                      <option value="">Select timestamp column...</option>
                      {columns
                        .filter(
                          (col) => col !== groupColumn && col !== metricColumn
                        )
                        .map((col) => (
                          <option key={col} value={col}>
                            {col}
                          </option>
                        ))}
                    </select>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Time Granularity
                    </label>
                    <select
                      value={timeGranularity}
                      onChange={(e) => setTimeGranularity(e.target.value)}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                    >
                      <option value="hour">Hour of Day</option>
                      <option value="day">Day of Week</option>
                      <option value="week">Week of Year</option>
                    </select>
                  </div>
                </>
              )}

              {/* Regression Mode */}
              {analysisMode === "regression" && (
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Covariate Columns <span className="text-red-500">*</span>
                  </label>
                  <div className="space-y-2">
                    {columns
                      .filter(
                        (col) => col !== groupColumn && col !== metricColumn
                      )
                      .map((col) => (
                        <label key={col} className="flex items-center">
                          <input
                            type="checkbox"
                            checked={regressionCovariates.includes(col)}
                            onChange={(e) => {
                              if (e.target.checked) {
                                setRegressionCovariates([
                                  ...regressionCovariates,
                                  col,
                                ]);
                              } else {
                                setRegressionCovariates(
                                  regressionCovariates.filter((c) => c !== col)
                                );
                              }
                            }}
                            className="mr-2"
                          />
                          <span>{col}</span>
                        </label>
                      ))}
                  </div>
                  <p className="text-xs text-gray-500 mt-1">
                    Select covariates to control for in regression
                  </p>
                </div>
              )}

              {/* Non-Parametric Mode */}
              {analysisMode === "non-parametric" && (
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Test Type
                  </label>
                  <select
                    value={nonParametricTestType}
                    onChange={(e) => setNonParametricTestType(e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                  >
                    <option value="mann_whitney">
                      Mann-Whitney U (2 groups)
                    </option>
                    <option value="kruskal_wallis">
                      Kruskal-Wallis (3+ groups)
                    </option>
                  </select>
                  {nonParametricTestType === "kruskal_wallis" && (
                    <p className="text-xs text-gray-500 mt-1">
                      Select all groups you want to compare
                    </p>
                  )}
                </div>
              )}

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Metric Column
                </label>
                <select
                  value={metricColumn}
                  onChange={(e) => setMetricColumn(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                >
                  <option value="">Select column...</option>
                  {columns.map((col) => (
                    <option key={col} value={col}>
                      {col}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Metric Type
                </label>
                <select
                  value={metricType}
                  onChange={(e) => setMetricType(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                >
                  <option value="binary">Binary (Conversion/Click)</option>
                  <option value="numeric">Numeric (Revenue/Time/etc.)</option>
                </select>
              </div>

              {/* CUPED Option */}
              <div className="pt-4 border-t border-gray-200">
                <div className="flex items-center gap-3 mb-3">
                  <input
                    type="checkbox"
                    id="enableCuped"
                    checked={enableCuped}
                    onChange={(e) => {
                      setEnableCuped(e.target.checked);
                      if (!e.target.checked) setCovariateColumn("");
                    }}
                    className="w-4 h-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
                  />
                  <label
                    htmlFor="enableCuped"
                    className="text-sm font-medium text-gray-700"
                  >
                    Enable CUPED Variance Reduction
                  </label>
                </div>
                {enableCuped && (
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Covariate Column (for CUPED)
                    </label>
                    <select
                      value={covariateColumn}
                      onChange={(e) => setCovariateColumn(e.target.value)}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                    >
                      <option value="">Select covariate column...</option>
                      {columns
                        .filter(
                          (col) => col !== groupColumn && col !== metricColumn
                        )
                        .map((col) => (
                          <option key={col} value={col}>
                            {col}
                          </option>
                        ))}
                    </select>
                    <p className="text-xs text-gray-500 mt-1">
                      CUPED uses a pre-experiment covariate to reduce variance
                      and increase power
                    </p>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Sample CSV Accordion */}
        <div className="bg-white rounded-xl shadow-md border border-gray-200 mb-6 overflow-hidden">
          <details className="group">
            <summary className="cursor-pointer p-6 flex items-center justify-between hover:bg-gray-50 transition-colors">
              <div>
                <h3 className="text-lg font-semibold text-gray-900 mb-1">
                  Sample CSV Format
                </h3>
                <p className="text-sm text-gray-600">
                  View example format and column explanations
                </p>
              </div>
              <svg
                className="w-5 h-5 text-gray-400 transform transition-transform group-open:rotate-180"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M19 9l-7 7-7-7"
                />
              </svg>
            </summary>
            <div className="px-6 pb-6 border-t border-gray-200 pt-6">
              <div className="bg-gray-50 rounded-lg border border-gray-200 p-4 mb-4 relative">
                <button
                  onClick={copySampleCSV}
                  className="absolute top-3 right-3 px-3 py-1.5 text-xs font-medium text-blue-600 bg-white border border-blue-200 rounded-md hover:bg-blue-50 transition-colors"
                >
                  📋 Copy to Clipboard
                </button>
                <pre className="font-mono text-xs text-gray-800 overflow-x-auto">{`group,conversion,revenue
A,1,25.50
A,0,0.00
A,1,30.00
B,1,28.00
B,1,32.50
B,0,0.00`}</pre>
              </div>
              <div className="space-y-3 text-sm">
                <div className="flex items-start gap-3">
                  <span className="font-semibold text-gray-700 min-w-[80px]">
                    group:
                  </span>
                  <span className="text-gray-600">
                    Experiment group identifier (A = control, B = treatment)
                  </span>
                </div>
                <div className="flex items-start gap-3">
                  <span className="font-semibold text-gray-700 min-w-[80px]">
                    conversion:
                  </span>
                  <span className="text-gray-600">
                    Binary metric (0 = no conversion, 1 = converted)
                  </span>
                </div>
                <div className="flex items-start gap-3">
                  <span className="font-semibold text-gray-700 min-w-[80px]">
                    revenue:
                  </span>
                  <span className="text-gray-600">
                    Numeric metric (optional, for revenue-based analysis)
                  </span>
                </div>
              </div>
            </div>
          </details>
        </div>

        {/* Error Message */}
        {error && (
          <div className="bg-red-50 border-2 border-red-200 rounded-xl p-4 mb-6 shadow-sm">
            <div className="flex items-start gap-3">
              <svg
                className="w-5 h-5 text-red-600 mt-0.5 flex-shrink-0"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                />
              </svg>
              <p className="text-sm text-red-800 font-medium">{error}</p>
            </div>
          </div>
        )}

        {/* Primary CTA Button */}
        <div className="text-center mb-6">
          <button
            onClick={handleAnalyze}
            disabled={
              loading ||
              !file ||
              !groupColumn ||
              !metricColumn ||
              (analysisMode === "multi-variant" &&
                (!groupA || selectedTreatmentGroups.length === 0)) ||
              (analysisMode === "segmentation" &&
                (!groupA || !groupB || !segmentColumn)) ||
              (analysisMode === "time-based" &&
                (!groupA || !groupB || !timestampColumn)) ||
              (analysisMode === "regression" &&
                (!groupA || !groupB || regressionCovariates.length === 0)) ||
              (analysisMode === "non-parametric" &&
                (!groupA ||
                  (nonParametricTestType === "mann_whitney" && !groupB))) ||
              (analysisMode === "ab" && (!groupA || !groupB))
            }
            className={`relative px-10 py-4 rounded-xl font-bold text-white text-lg transition-all transform ${
              loading || !file || !groupColumn || !metricColumn
                ? "bg-gray-400 cursor-not-allowed"
                : "bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 hover:shadow-xl hover:scale-105 active:scale-100"
            } shadow-lg`}
          >
            {loading ? (
              <span className="flex items-center justify-center">
                <svg
                  className="animate-spin -ml-1 mr-3 h-5 w-5 text-white"
                  xmlns="http://www.w3.org/2000/svg"
                  fill="none"
                  viewBox="0 0 24 24"
                >
                  <circle
                    className="opacity-25"
                    cx="12"
                    cy="12"
                    r="10"
                    stroke="currentColor"
                    strokeWidth="4"
                  ></circle>
                  <path
                    className="opacity-75"
                    fill="currentColor"
                    d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                  ></path>
                </svg>
                Analyzing...
              </span>
            ) : (
              <>🚀 Run Statistical Analysis</>
            )}
          </button>
          <p className="text-sm text-gray-500 mt-3">
            A/B tests, regression, segmentation & time-based insights
          </p>
        </div>
      </div>
    </div>
  );
}

export default UploadPage;

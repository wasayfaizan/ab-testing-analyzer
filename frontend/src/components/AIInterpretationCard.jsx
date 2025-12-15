import React, { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { apiClient } from "../config/api";

function AIInterpretationCard({ interpretation, results }) {
  const [selectedMode, setSelectedMode] = useState("executive");
  const [aiOutput, setAiOutput] = useState(interpretation || "");
  const [loading, setLoading] = useState(false);
  const [expandedSections, setExpandedSections] = useState({
    summary: true,
    interpretation: true,
    recommendation: true,
  });
  const [copied, setCopied] = useState(false);
  const contentRef = useRef(null);

  const modes = [
    { id: "executive", label: "Executive", icon: "👔" },
    { id: "non-technical", label: "Non-Technical", icon: "💬" },
    { id: "slack-email", label: "Slack/Email", icon: "📧" },
    { id: "recommendations", label: "Detailed", icon: "💡" },
  ];

  useEffect(() => {
    if (interpretation) {
      setAiOutput(interpretation);
    }
  }, [interpretation]);

  const generateInterpretation = async (mode) => {
    if (!results) return;

    setLoading(true);
    try {
      const response = await apiClient.post(
        "/api/generate-interpretation-mode",
        {
          lift: results.lift,
          p_value: results.p_value,
          effect_size: results.effect_size,
          ci_lower: results.confidence_interval.lower,
          ci_upper: results.confidence_interval.upper,
          prob_b_better: results.bayesian.prob_b_better,
          metric_type: results.metric_type,
          test_type: results.test_type,
          mode: mode,
          group_a_name: results.group_a_name,
          group_b_name: results.group_b_name,
        }
      );
      setAiOutput(response.data.interpretation);
      setSelectedMode(mode);
    } catch (error) {
      console.error("Error generating interpretation:", error);
      if (
        error.code === "ECONNREFUSED" ||
        error.message?.includes("Network Error")
      ) {
        setAiOutput(
          "Unable to connect to the backend server. Please ensure the backend is running on http://localhost:8000"
        );
      } else {
        setAiOutput("Error generating interpretation. Please try again.");
      }
    } finally {
      setLoading(false);
    }
  };

  const toggleSection = (section) => {
    setExpandedSections((prev) => ({
      ...prev,
      [section]: !prev[section],
    }));
  };

  const copyToClipboard = async () => {
    if (contentRef.current) {
      const text = contentRef.current.innerText || aiOutput;
      try {
        await navigator.clipboard.writeText(text);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
      } catch (err) {
        console.error("Failed to copy:", err);
      }
    }
  };

  const downloadPDF = () => {
    // This would integrate with your PDF generator
    // For now, we'll create a simple text download
    const blob = new Blob([aiOutput], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "ai-interpretation-summary.txt";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  // Parse AI output to extract sections
  const parseSections = (text) => {
    const sections = {
      summary: "",
      interpretation: "",
      recommendation: "",
    };

    const lines = text.split("\n");
    let currentSection = null;
    let currentContent = [];

    lines.forEach((line) => {
      const trimmed = line.trim();

      if (trimmed.startsWith("## ")) {
        // Save previous section
        if (currentSection) {
          sections[currentSection] = currentContent.join("\n");
        }

        // Start new section
        const title = trimmed.replace("## ", "").toLowerCase();
        if (title.includes("summary") || title.includes("results")) {
          currentSection = "summary";
        } else if (
          title.includes("interpretation") ||
          title.includes("analysis")
        ) {
          currentSection = "interpretation";
        } else if (title.includes("recommendation")) {
          currentSection = "recommendation";
        } else {
          currentSection = "interpretation";
        }
        currentContent = [line];
      } else {
        currentContent.push(line);
      }
    });

    // Save last section
    if (currentSection) {
      sections[currentSection] = currentContent.join("\n");
    } else {
      // If no sections found, put everything in interpretation
      sections.interpretation = text;
    }

    return sections;
  };

  const sections = parseSections(aiOutput);

  if (!aiOutput && !interpretation) {
    return null;
  }

  return (
    <div className="bg-white dark:bg-[#13161C] rounded-2xl shadow-xl border border-gray-200/50 dark:border-[rgba(255,255,255,0.08)] overflow-hidden mb-8 animate-fade-in">
      {/* Header Bar */}
      <div className="bg-gradient-to-r from-indigo-500 via-purple-500 to-indigo-600 dark:bg-[#13161C] border-b border-indigo-200/50 dark:border-[rgba(255,255,255,0.08)] px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="w-10 h-10 bg-white/20 backdrop-blur-sm rounded-xl flex items-center justify-center shadow-lg">
              <svg
                className="w-5 h-5 text-white"
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
            </div>
            <div>
              <h2 className="text-xl font-bold text-white">
                AI-Powered Interpretation
              </h2>
              <p className="text-sm text-indigo-100/90">
                Intelligent analysis powered by GPT-4
              </p>
            </div>
          </div>

          {/* Action Buttons */}
          <div className="flex items-center gap-2">
            <button
              onClick={copyToClipboard}
              className="px-3 py-1.5 text-sm font-medium text-white bg-white/20 hover:bg-white/30 backdrop-blur-sm border border-white/30 rounded-xl transition-all flex items-center gap-2 hover:shadow-lg hover:-translate-y-0.5"
            >
              {copied ? (
                <>
                  <svg
                    className="w-4 h-4 text-green-600"
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
                  Copied!
                </>
              ) : (
                <>
                  <svg
                    className="w-4 h-4"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z"
                    />
                  </svg>
                  Copy
                </>
              )}
            </button>
            <button
              onClick={downloadPDF}
              className="px-3 py-1.5 text-sm font-medium text-white bg-white/20 hover:bg-white/30 backdrop-blur-sm border border-white/30 rounded-xl transition-all flex items-center gap-2 hover:shadow-lg hover:-translate-y-0.5"
            >
              <svg
                className="w-4 h-4"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                />
              </svg>
              PDF
            </button>
          </div>
        </div>
      </div>

      {/* Mode Selector */}
      {results && (
        <div className="px-4 sm:px-6 py-4 bg-purple-50/50 dark:bg-white/[0.03] border-b border-purple-200/30 dark:border-[rgba(255,255,255,0.08)]">
          <div className="flex flex-col sm:flex-row items-start sm:items-center gap-3">
            <span className="text-xs font-semibold text-purple-600 dark:text-[#A8A8B3] uppercase tracking-wider whitespace-nowrap">
              View as:
            </span>
            <div className="flex flex-wrap items-center gap-1 bg-white dark:bg-white/[0.05] rounded-xl p-1 border border-purple-200/50 dark:border-[rgba(255,255,255,0.08)] w-full sm:w-auto shadow-sm">
              {modes.map((mode) => (
                <button
                  key={mode.id}
                  onClick={() => generateInterpretation(mode.id)}
                  disabled={loading}
                  className={`px-3 sm:px-4 py-2 text-xs sm:text-sm font-medium rounded-lg transition-all duration-200 flex-1 sm:flex-initial ${
                    selectedMode === mode.id
                      ? "bg-gradient-to-r from-indigo-500 to-purple-600 text-white shadow-md"
                      : "text-gray-700 dark:text-[#A8A8B3] hover:bg-purple-50 dark:hover:bg-white/[0.05]"
                  } ${loading ? "opacity-50 cursor-not-allowed" : ""}`}
                >
                  <span className="mr-1.5">{mode.icon}</span>
                  <span className="hidden sm:inline">{mode.label}</span>
                  <span className="sm:hidden">{mode.label.split(" ")[0]}</span>
                </button>
              ))}
            </div>
            {loading && (
              <div className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
                <div className="w-4 h-4 border-2 border-blue-600 border-t-transparent rounded-full animate-spin"></div>
                <span className="hidden sm:inline">Generating...</span>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Content */}
      <div className="p-6" ref={contentRef}>
        {/* Results Summary Section */}
        {sections.summary && (
          <div className="mb-6">
            <button
              onClick={() => toggleSection("summary")}
              className="w-full flex items-center justify-between py-3 px-4 bg-indigo-50/50 dark:bg-white/[0.03] rounded-xl hover:bg-indigo-50 dark:hover:bg-white/[0.05] transition-all group sticky top-0 z-10 backdrop-blur-sm"
            >
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 bg-gradient-primary/20 dark:bg-gradient-primary/20 rounded-lg flex items-center justify-center">
                  <svg
                    className="w-4 h-4 text-aqua dark:text-aqua"
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
                  Results Summary
                </h3>
              </div>
              <svg
                className={`w-5 h-5 text-gray-500 dark:text-gray-400 transition-transform duration-300 ${
                  expandedSections.summary ? "rotate-180" : ""
                }`}
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
            </button>
            <div
              className={`overflow-hidden transition-all duration-500 ease-in-out ${
                expandedSections.summary
                  ? "max-h-[2000px] opacity-100"
                  : "max-h-0 opacity-0"
              }`}
            >
              <div className="pt-4 px-4">
                {results && (
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 sm:gap-4 mb-4">
                    <div className="bg-gradient-to-br from-blue-50 to-indigo-50 dark:bg-[#13161C] rounded-xl p-4 border border-blue-200/50 dark:border-[rgba(255,255,255,0.08)]">
                      <div className="text-xs font-semibold text-blue-600 dark:text-aqua uppercase tracking-wide mb-1">
                        Lift
                      </div>
                      <div className="text-xl sm:text-2xl font-bold text-gray-900 dark:text-white">
                        {results.lift?.toFixed(2) || "0.00"}%
                      </div>
                    </div>
                    <div className="bg-gradient-to-br from-purple-50 to-pink-50 dark:bg-[#13161C] rounded-xl p-4 border border-purple-200/50 dark:border-[rgba(255,255,255,0.08)]">
                      <div className="text-xs font-semibold text-purple-600 dark:text-aqua uppercase tracking-wide mb-1">
                        P-Value
                      </div>
                      <div className="text-xl sm:text-2xl font-bold text-gray-900 dark:text-white">
                        {results.p_value && results.p_value < 0.0001
                          ? results.p_value.toExponential(2)
                          : results.p_value?.toFixed(4) || "N/A"}
                      </div>
                    </div>
                    <div className="bg-gradient-to-br from-green-50 to-emerald-50 dark:bg-[#13161C] rounded-xl p-4 border border-green-200/50 dark:border-[rgba(255,255,255,0.08)]">
                      <div className="text-xs font-semibold text-green-600 dark:text-aqua uppercase tracking-wide mb-1">
                        Effect Size
                      </div>
                      <div className="text-xl sm:text-2xl font-bold text-gray-900 dark:text-white">
                        {results.effect_size?.toFixed(3) || "0.000"}
                      </div>
                    </div>
                    <div className="bg-gradient-to-br from-amber-50 to-orange-50 dark:bg-[#13161C] rounded-xl p-4 border border-amber-200/50 dark:border-[rgba(255,255,255,0.08)]">
                      <div className="text-xs font-semibold text-amber-600 dark:text-aqua uppercase tracking-wide mb-1">
                        Bayesian Prob
                      </div>
                      <div className="text-xl sm:text-2xl font-bold text-gray-900 dark:text-white">
                        {(results.bayesian?.prob_b_better * 100)?.toFixed(1) ||
                          "50.0"}
                        %
                      </div>
                    </div>
                  </div>
                )}
                <div className="prose prose-sm max-w-none dark:prose-invert prose-headings:font-bold prose-p:text-gray-700 dark:prose-p:text-[#A8A8B3] prose-strong:text-gray-900 dark:prose-strong:text-white">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>
                    {sections.summary}
                  </ReactMarkdown>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Interpretation Section */}
        {sections.interpretation && (
          <div className="mb-6">
            <button
              onClick={() => toggleSection("interpretation")}
              className="w-full flex items-center justify-between py-3 px-4 bg-indigo-50/50 dark:bg-white/[0.03] rounded-xl hover:bg-indigo-50 dark:hover:bg-white/[0.05] transition-all group sticky top-0 z-10 backdrop-blur-sm"
            >
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 bg-gradient-primary/20 dark:bg-gradient-primary/20 rounded-lg flex items-center justify-center">
                  <svg
                    className="w-4 h-4 text-aqua dark:text-aqua"
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
                </div>
                <h3 className="text-lg font-bold text-gray-900 dark:text-white">
                  Interpretation
                </h3>
              </div>
              <svg
                className={`w-5 h-5 text-gray-500 dark:text-gray-400 transition-transform duration-300 ${
                  expandedSections.interpretation ? "rotate-180" : ""
                }`}
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
            </button>
            <div
              className={`overflow-hidden transition-all duration-500 ease-in-out ${
                expandedSections.interpretation
                  ? "max-h-[2000px] opacity-100"
                  : "max-h-0 opacity-0"
              }`}
            >
              <div className="pt-4 px-4">
                <div className="prose prose-sm max-w-none dark:prose-invert prose-headings:font-bold prose-p:text-gray-700 dark:prose-p:text-gray-300 prose-strong:text-gray-900 dark:prose-strong:text-white prose-ul:list-disc prose-ol:list-decimal">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>
                    {sections.interpretation}
                  </ReactMarkdown>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Recommendation Section */}
        {sections.recommendation && (
          <div>
            <button
              onClick={() => toggleSection("recommendation")}
              className="w-full flex items-center justify-between py-3 px-4 bg-indigo-50/50 dark:bg-white/[0.03] rounded-xl hover:bg-indigo-50 dark:hover:bg-white/[0.05] transition-all group sticky top-0 z-10 backdrop-blur-sm"
            >
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 bg-aqua/20 rounded-lg flex items-center justify-center">
                  <svg
                    className="w-4 h-4 text-aqua"
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
                  Recommendation
                </h3>
              </div>
              <svg
                className={`w-5 h-5 text-gray-500 dark:text-gray-400 transition-transform duration-300 ${
                  expandedSections.recommendation ? "rotate-180" : ""
                }`}
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
            </button>
            <div
              className={`overflow-hidden transition-all duration-500 ease-in-out ${
                expandedSections.recommendation
                  ? "max-h-[2000px] opacity-100"
                  : "max-h-0 opacity-0"
              }`}
            >
              <div className="pt-4 px-4">
                <div className="bg-gradient-to-r from-[#ECFDF5] to-[#D1FAE5] dark:bg-[#13161C] rounded-xl p-6 border-l-4 border-teal-500 dark:border-aqua/30 shadow-sm dark:border-[rgba(255,255,255,0.08)]">
                  <div className="prose prose-sm max-w-none dark:prose-invert prose-headings:font-bold prose-p:text-gray-800 dark:prose-p:text-[#A8A8B3] prose-strong:text-gray-900 dark:prose-strong:text-white">
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>
                      {sections.recommendation}
                    </ReactMarkdown>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Fallback: If no sections detected, show full content */}
        {!sections.summary &&
          !sections.interpretation &&
          !sections.recommendation && (
            <div className="prose prose-sm max-w-none prose-invert prose-headings:font-bold prose-p:text-text-secondary prose-strong:text-text-primary">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {aiOutput}
              </ReactMarkdown>
            </div>
          )}
      </div>
    </div>
  );
}

export default AIInterpretationCard;

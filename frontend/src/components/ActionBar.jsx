import React, { useState, useEffect, useRef } from "react";
import { createPortal } from "react-dom";
import { useNavigate } from "react-router-dom";
import { generatePDFReport } from "../utils/pdfGenerator";
import {
  exportToCSV,
  exportToJSON,
  saveExperiment,
} from "../utils/exportUtils";
import ThemeToggle from "./ThemeToggle";

function ActionBar({ results }) {
  const navigate = useNavigate();
  const [showSaveDialog, setShowSaveDialog] = useState(false);
  const [showExportMenu, setShowExportMenu] = useState(false);
  const [experimentName, setExperimentName] = useState("");
  const exportButtonRef = useRef(null);
  const [dropdownPosition, setDropdownPosition] = useState({
    top: 0,
    right: 0,
  });

  const handleDownloadPDF = () => {
    generatePDFReport(results);
    setShowExportMenu(false);
  };

  const handleExportCSV = () => {
    exportToCSV(results);
    setShowExportMenu(false);
  };

  const handleExportJSON = () => {
    exportToJSON(results);
    setShowExportMenu(false);
  };

  const handleSave = () => {
    if (experimentName.trim()) {
      saveExperiment(results, experimentName.trim());
      setShowSaveDialog(false);
      setExperimentName("");
      alert("Experiment saved successfully!");
    }
  };

  // Update dropdown position when menu opens
  useEffect(() => {
    if (showExportMenu && exportButtonRef.current) {
      const rect = exportButtonRef.current.getBoundingClientRect();
      setDropdownPosition({
        top: rect.bottom + window.scrollY + 8,
        right: window.innerWidth - rect.right,
      });
    }
  }, [showExportMenu]);

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (
        showExportMenu &&
        exportButtonRef.current &&
        !exportButtonRef.current.contains(event.target) &&
        !event.target.closest("[data-export-menu]")
      ) {
        setShowExportMenu(false);
      }
    };

    if (showExportMenu) {
      document.addEventListener("mousedown", handleClickOutside);
      return () => {
        document.removeEventListener("mousedown", handleClickOutside);
      };
    }
  }, [showExportMenu]);

  return (
    <div className="mb-8">
      {/* Gradient Header */}
      <div className="bg-gradient-to-r from-indigo-500 via-purple-500 to-indigo-600 rounded-2xl shadow-xl border border-indigo-200/50 overflow-hidden">
        <div className="backdrop-blur-sm bg-white/10 px-6 py-5">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-white mb-1">
                Analysis Results
              </h1>
              <p className="text-sm text-indigo-100/90">
                Comprehensive statistical analysis of your A/B test
              </p>
            </div>
            <div className="flex items-center gap-3">
              <ThemeToggle />

              {/* Run New Analysis */}
              <button
                onClick={() => navigate("/")}
                className="px-5 py-2.5 text-sm font-medium text-white bg-white/20 hover:bg-white/30 backdrop-blur-sm border border-white/30 rounded-xl transition-all duration-200 hover:shadow-lg hover:-translate-y-0.5 flex items-center gap-2"
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
                    d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
                  />
                </svg>
                Run Analysis
              </button>

              {/* Saved Experiments */}
              <button
                onClick={() => navigate("/saved")}
                className="px-5 py-2.5 text-sm font-medium text-white bg-white/20 hover:bg-white/30 backdrop-blur-sm border border-white/30 rounded-xl transition-all duration-200 hover:shadow-lg hover:-translate-y-0.5 flex items-center gap-2"
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
                    d="M5 5a2 2 0 012-2h10a2 2 0 012 2v16l-7-3.5L5 21V5z"
                  />
                </svg>
                Saved
              </button>

              {/* Save Experiment */}
              <div className="relative">
                <button
                  onClick={() => setShowSaveDialog(!showSaveDialog)}
                  className="px-5 py-2.5 text-sm font-medium text-white bg-white/20 hover:bg-white/30 backdrop-blur-sm border border-white/30 rounded-xl transition-all duration-200 hover:shadow-lg hover:-translate-y-0.5 flex items-center gap-2"
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
                      d="M8 7H5a2 2 0 00-2 2v9a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-3m-1 4l-3 3m0 0l-3-3m3 3V4"
                    />
                  </svg>
                  Save
                </button>
              </div>

              {/* Export Menu */}
              <div className="relative">
                <button
                  ref={exportButtonRef}
                  onClick={() => setShowExportMenu(!showExportMenu)}
                  className="px-5 py-2.5 text-sm font-medium text-white bg-white/20 hover:bg-white/30 backdrop-blur-sm border border-white/30 rounded-xl transition-all duration-200 hover:shadow-lg hover:-translate-y-0.5 flex items-center gap-2"
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
                  Export
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
                      d="M19 9l-7 7-7-7"
                    />
                  </svg>
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
      {/* Save Dialog - Centered Modal */}
      {showSaveDialog &&
        createPortal(
          <div
            className="fixed inset-0 z-50 flex items-center justify-center bg-black/20 dark:bg-black/40 backdrop-blur-sm animate-fade-in"
            onClick={() => setShowSaveDialog(false)}
          >
            <div
              className="w-72 bg-white dark:bg-[#13161C] rounded-xl shadow-2xl border border-purple-200/50 dark:border-[rgba(255,255,255,0.08)] p-4 animate-fade-in"
              onClick={(e) => e.stopPropagation()}
            >
              <input
                type="text"
                value={experimentName}
                onChange={(e) => setExperimentName(e.target.value)}
                placeholder="Experiment name..."
                className="w-full px-4 py-2.5 border border-purple-200 dark:border-[rgba(255,255,255,0.08)] rounded-lg mb-3 text-sm focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent bg-white dark:bg-white/[0.05] text-gray-900 dark:text-white placeholder-gray-400 dark:placeholder-[#A8A8B3]"
                onKeyPress={(e) => {
                  if (e.key === "Enter") handleSave();
                }}
                autoFocus
              />
              <div className="flex gap-2">
                <button
                  onClick={handleSave}
                  className="flex-1 px-4 py-2 bg-gradient-to-r from-indigo-500 to-purple-600 text-white rounded-lg text-sm font-medium hover:shadow-lg transition-all duration-200 hover:-translate-y-0.5"
                >
                  Save
                </button>
                <button
                  onClick={() => {
                    setShowSaveDialog(false);
                    setExperimentName("");
                  }}
                  className="px-4 py-2 bg-gray-100 dark:bg-white/[0.05] dark:border dark:border-[rgba(255,255,255,0.08)] text-gray-700 dark:text-[#A8A8B3] rounded-lg text-sm font-medium hover:bg-gray-200 dark:hover:bg-white/[0.10] transition-all"
                >
                  Cancel
                </button>
              </div>
            </div>
          </div>,
          document.body
        )}
      {/* Export Menu Dropdown - Rendered via Portal to escape overflow-hidden */}
      {showExportMenu &&
        createPortal(
          <div
            data-export-menu
            className="fixed w-48 bg-white dark:bg-gray-800 rounded-xl shadow-2xl border border-purple-200/50 dark:border-gray-700 overflow-hidden z-[60] animate-fade-in"
            style={{
              top: `${dropdownPosition.top}px`,
              right: `${dropdownPosition.right}px`,
            }}
          >
            <button
              onClick={handleDownloadPDF}
              className="w-full text-left px-4 py-3 text-sm text-gray-700 dark:text-gray-300 hover:bg-purple-50 dark:hover:bg-gray-700 transition-colors flex items-center gap-2"
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
                  d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z"
                />
              </svg>
              PDF Report
            </button>
            <button
              onClick={handleExportCSV}
              className="w-full text-left px-4 py-3 text-sm text-gray-700 dark:text-gray-300 hover:bg-purple-50 dark:hover:bg-gray-700 transition-colors flex items-center gap-2 border-t border-gray-100 dark:border-gray-700"
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
                  d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                />
              </svg>
              CSV Summary
            </button>
            <button
              onClick={handleExportJSON}
              className="w-full text-left px-4 py-3 text-sm text-gray-700 dark:text-gray-300 hover:bg-purple-50 dark:hover:bg-gray-700 transition-colors flex items-center gap-2 border-t border-gray-100 dark:border-gray-700"
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
                  d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4"
                />
              </svg>
              JSON Data
            </button>
          </div>,
          document.body
        )}
    </div>
  );
}

export default ActionBar;

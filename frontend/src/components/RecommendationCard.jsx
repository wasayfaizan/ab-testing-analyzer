import React from "react";

function RecommendationCard({ recommendation, color = "green" }) {
  return (
    <div className="bg-gradient-to-br from-[#ECFDF5] to-[#D1FAE5] dark:bg-[#13161C] rounded-2xl border-l-4 border-teal-500 dark:border-aqua/30 p-6 shadow-lg dark:border-[rgba(255,255,255,0.08)] animate-fade-in">
      <div className="flex items-center gap-4">
        <div className="w-14 h-14 rounded-xl bg-white/80 dark:bg-aqua/20 backdrop-blur-sm flex items-center justify-center shadow-md flex-shrink-0">
          <svg
            className="w-7 h-7 text-teal-600 dark:text-aqua"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2.5}
              d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
            />
          </svg>
        </div>
        <div className="flex-1">
          <p className="text-xs font-semibold uppercase tracking-wider text-teal-700 dark:text-aqua mb-2">
            Recommendation
          </p>
          <p className="text-2xl font-bold text-gray-900 dark:text-white leading-tight">
            {recommendation}
          </p>
        </div>
      </div>
    </div>
  );
}

export default RecommendationCard;

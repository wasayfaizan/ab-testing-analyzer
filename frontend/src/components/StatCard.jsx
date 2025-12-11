import React from "react";
import Tooltip from "./Tooltip";

const tooltipContent = {
  Lift: "The percentage improvement of the treatment group over the control group. Calculated as (conversion_B - conversion_A) / conversion_A × 100%",
  "P-Value":
    "The probability of observing this result (or more extreme) if there is no true difference between groups. Lower values indicate stronger evidence against the null hypothesis.",
  "Effect Size":
    "A standardized measure of the magnitude of the difference between groups. Cohen's h (for proportions) or Cohen's d (for continuous metrics) helps assess practical significance.",
  "95% Confidence Interval":
    "A range of values that, with 95% confidence, contains the true difference between groups. If the interval doesn't include zero, the result is statistically significant.",
  "Bayesian Probability":
    "The probability that treatment (B) performs better than control (A) based on Bayesian analysis. Uses Beta-Binomial (binary) or Normal (continuous) posterior distributions.",
};

function StatCard({
  icon,
  label,
  value,
  description,
  color = "indigo",
  badge,
  tooltip,
}) {
  // Badge color mapping
  const getBadgeColor = (badgeText) => {
    if (badgeText === "Positive" || badgeText === "Significant") {
      return "bg-badge-positive/15 text-badge-positive border-badge-positive/30";
    } else if (badgeText === "Not Significant" || badgeText === "Negative") {
      return "bg-badge-neutral/15 text-badge-neutral border-badge-neutral/30";
    } else {
      return "bg-badge-significant/15 text-badge-significant border-badge-significant/30";
    }
  };

  const tooltipText = tooltip || tooltipContent[label] || description;

  const colorClasses = {
    indigo:
      "border-indigo-200/60 bg-gradient-to-br from-indigo-50/50 to-purple-50/50",
    purple:
      "border-purple-200/60 bg-gradient-to-br from-purple-50/50 to-pink-50/50",
    teal: "border-teal-200/60 bg-gradient-to-br from-teal-50/50 to-cyan-50/50",
    gray: "border-gray-200/60 bg-gradient-to-br from-gray-50/50 to-slate-50/50",
    green:
      "border-teal-200/60 bg-gradient-to-br from-emerald-50/50 to-teal-50/50",
  };

  const iconBgClasses = {
    indigo: "bg-gradient-to-br from-indigo-100 to-purple-100 text-indigo-600",
    purple: "bg-gradient-to-br from-purple-100 to-pink-100 text-purple-600",
    teal: "bg-gradient-to-br from-teal-100 to-cyan-100 text-teal-600",
    gray: "bg-gradient-to-br from-gray-100 to-slate-100 text-gray-600",
    green: "bg-gradient-to-br from-emerald-100 to-teal-100 text-teal-600",
  };

  // Badge color mapping for light theme
  const getBadgeColorLight = (badgeText) => {
    if (badgeText === "Positive" || badgeText === "Significant") {
      return "bg-teal-100 text-teal-700 border-teal-200/50";
    } else if (badgeText === "Not Significant" || badgeText === "Negative") {
      return "bg-gray-100 text-gray-700 border-gray-200/50";
    } else {
      return "bg-amber-100 text-amber-700 border-amber-200/50";
    }
  };

  // Dark mode badge colors
  const getBadgeColorDark = (badgeText) => {
    if (badgeText === "Positive" || badgeText === "Significant") {
      return "dark:bg-aqua/20 dark:text-aqua dark:border-aqua/30";
    } else if (badgeText === "Not Significant" || badgeText === "Negative") {
      return "dark:bg-gray-700/50 dark:text-gray-400 dark:border-gray-600/50";
    } else {
      return "dark:bg-badge-significant/20 dark:text-badge-significant dark:border-badge-significant/30";
    }
  };

  return (
    <div
      className={`group relative bg-white dark:bg-[#13161C] rounded-2xl border-2 dark:border-[rgba(255,255,255,0.08)] p-6 shadow-sm hover:shadow-xl dark:hover:shadow-[0_8px_32px_rgba(138,63,252,0.15)] transition-all duration-300 animate-fade-in ${colorClasses[color]} dark:border-[rgba(255,255,255,0.08)]`}
    >
      {/* Icon and Badge Container - Perfectly Aligned */}
      <div className="flex items-start justify-between mb-4">
        {/* Icon - Top Left - Larger and Clearer */}
        <div
          className={`w-12 h-12 rounded-xl flex items-center justify-center ${iconBgClasses[color]} dark:bg-gradient-primary/20 dark:text-aqua transition-transform duration-300 group-hover:scale-110 group-hover:rotate-3`}
        >
          {icon}
        </div>
        {/* Badge - Top Right - Consistent Pill Style */}
        {badge && (
          <span
            className={`px-3 py-1 text-xs font-semibold rounded-full border ${getBadgeColorLight(
              badge
            )} ${getBadgeColorDark(badge)}`}
          >
            {badge}
          </span>
        )}
      </div>

      {/* Content - Compact and Balanced */}
      <div className="space-y-2">
        {/* Label with Tooltip */}
        <div className="flex items-center gap-2">
          <p className="text-xs font-semibold uppercase tracking-wider text-gray-500 dark:text-[#A8A8B3]">
            {label}
          </p>
          {tooltipText && (
            <Tooltip content={tooltipText}>
              <svg
                className="w-4 h-4 text-gray-400 hover:text-indigo-500 dark:text-[#A8A8B3] dark:hover:text-aqua cursor-help transition-colors"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                />
              </svg>
            </Tooltip>
          )}
        </div>

        {/* Value - Big, Bold, Readable */}
        <p className="text-3xl font-semibold text-gray-900 dark:text-white">
          {value}
        </p>

        {/* Description - Smaller muted */}
        {description && (
          <p className="text-xs text-gray-600 dark:text-[#A8A8B3] mt-2 font-medium">
            {description}
          </p>
        )}
      </div>
    </div>
  );
}

export default StatCard;

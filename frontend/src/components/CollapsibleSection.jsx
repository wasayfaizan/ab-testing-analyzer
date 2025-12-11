import React, { useState } from "react";

function CollapsibleSection({
  title,
  icon,
  children,
  defaultOpen = true,
  gradient = "indigo",
}) {
  const [isOpen, setIsOpen] = useState(defaultOpen);

  const gradientClasses = {
    indigo: "from-indigo-500 to-purple-600",
    purple: "from-purple-500 to-pink-500",
    teal: "from-teal-500 to-cyan-500",
    gray: "from-gray-500 to-slate-500",
  };

  return (
    <div className="bg-white dark:bg-[#13161C] rounded-2xl shadow-lg border border-purple-200/30 dark:border-[rgba(255,255,255,0.08)] overflow-hidden mb-6 hover:shadow-xl dark:hover:shadow-[0_8px_32px_rgba(138,63,252,0.15)] transition-all duration-300">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className={`w-full px-6 py-4 flex items-center justify-between bg-gradient-to-r ${gradientClasses[gradient]} text-white hover:opacity-95 transition-all duration-200 backdrop-blur-sm`}
      >
        <div className="flex items-center gap-3">
          {icon && <div className="w-6 h-6">{icon}</div>}
          <h3 className="text-lg font-bold">{title}</h3>
        </div>
        <svg
          className={`w-5 h-5 transition-transform duration-300 ${
            isOpen ? "rotate-180" : ""
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
        className={`transition-all duration-500 ease-in-out ${
          isOpen
            ? "max-h-[10000px] opacity-100"
            : "max-h-0 opacity-0 overflow-hidden"
        }`}
      >
        <div className="p-6">{children}</div>
      </div>
    </div>
  );
}

export default CollapsibleSection;

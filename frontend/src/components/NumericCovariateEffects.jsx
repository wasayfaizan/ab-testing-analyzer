import React from "react";

function NumericCovariateEffects({ numericEffects }) {
  if (!numericEffects || Object.keys(numericEffects).length === 0) {
    return null;
  }

  const formatVariableName = (name) => {
    return name.replace(/_/g, " ").replace(/\b\w/g, (l) => l.toUpperCase());
  };

  const calculateEffectText = (coef, oddsRatio) => {
    // For odds ratio, calculate percentage change per unit
    const percentChange = ((oddsRatio - 1) * 100).toFixed(2);
    return `Each additional unit increases odds by ${
      percentChange > 0 ? "+" : ""
    }${percentChange}%`;
  };

  return (
    <div className="bg-white dark:bg-[#13161C] rounded-xl border border-gray-200 dark:border-[rgba(255,255,255,0.08)] p-6">
      <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-4">
        Numeric Covariate Effects (Marginal Impact)
      </h3>
      <p className="text-sm text-gray-600 dark:text-gray-400 mb-6">
        Effect of each numeric covariate on conversion probability, holding
        other variables constant.
      </p>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {Object.entries(numericEffects).map(([varName, effect]) => {
          const isSignificant = effect.significant;
          const readableName = formatVariableName(
            effect.original_name || varName
          );
          const effectText = calculateEffectText(
            effect.coef,
            effect.odds_ratio
          );

          return (
            <div
              key={varName}
              className={`bg-gray-50 dark:bg-white/[0.03] rounded-lg p-4 border-2 ${
                isSignificant
                  ? "border-green-500 dark:border-green-400"
                  : "border-gray-200 dark:border-[rgba(255,255,255,0.08)]"
              }`}
            >
              <h4 className="font-semibold text-gray-900 dark:text-white mb-3">
                {readableName}
              </h4>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-600 dark:text-gray-400">
                    Coefficient (β):
                  </span>
                  <span className="font-semibold text-gray-900 dark:text-white">
                    {effect.coef > 0 ? "+" : ""}
                    {effect.coef.toFixed(4)}
                  </span>
                </div>
                {effect.odds_ratio !== undefined &&
                  effect.odds_ratio !== null && (
                    <div className="flex justify-between">
                      <span className="text-gray-600 dark:text-gray-400">
                        Odds Ratio:
                      </span>
                      <span className="font-semibold text-gray-900 dark:text-white">
                        {effect.odds_ratio.toFixed(4)}
                      </span>
                    </div>
                  )}
                <div className="pt-2 border-t border-gray-200 dark:border-gray-700">
                  <p className="text-gray-700 dark:text-gray-300 italic">
                    {effect.odds_ratio !== undefined &&
                    effect.odds_ratio !== null
                      ? effectText
                      : `Each additional unit increases outcome by ${
                          effect.coef > 0 ? "+" : ""
                        }${effect.coef.toFixed(4)}`}
                  </p>
                </div>
                <div className="flex items-center justify-between pt-2">
                  <span className="text-gray-600 dark:text-gray-400">
                    P-value:
                  </span>
                  <span
                    className={`font-semibold ${
                      isSignificant
                        ? "text-green-600 dark:text-green-400"
                        : "text-gray-600 dark:text-gray-400"
                    }`}
                  >
                    {effect.p_value < 0.0001
                      ? effect.p_value.toExponential(2)
                      : effect.p_value.toFixed(4)}
                  </span>
                </div>
                {isSignificant && (
                  <div className="flex items-center gap-2 pt-1">
                    <span className="text-green-600 dark:text-green-400">
                      ✓
                    </span>
                    <span className="text-xs text-green-600 dark:text-green-400">
                      Significant (p &lt; 0.05)
                    </span>
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export default NumericCovariateEffects;

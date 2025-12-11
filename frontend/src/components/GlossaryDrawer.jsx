import React, { useState } from 'react'

const glossaryTerms = {
  'Lift': 'The percentage improvement of the treatment group over the control group. Calculated as (conversion_B - conversion_A) / conversion_A × 100%.',
  'P-Value': 'The probability of observing the current result (or more extreme) if there is no true difference between groups. Lower values indicate stronger evidence against the null hypothesis. Typically, p < 0.05 is considered statistically significant.',
  'Effect Size': 'A standardized measure of the magnitude of the difference between groups. Cohen\'s h (for proportions) or Cohen\'s d (for continuous metrics) helps assess practical significance beyond statistical significance.',
  'Confidence Interval (CI)': 'A range of values that, with a specified confidence level (typically 95%), contains the true difference between groups. If the interval doesn\'t include zero, the result is statistically significant.',
  'Two-Proportion Z-Test': 'A statistical test used to compare two proportions (conversion rates). It tests whether the difference between two proportions is statistically significant.',
  'Bayesian Inference': 'A statistical approach that incorporates prior beliefs and updates them with observed data to produce posterior probabilities. In A/B testing, it provides the probability that one variant is better than another.',
  'CUPED': 'Controlled-experiment Using Pre-Experiment Data. A variance reduction technique that uses a covariate (pre-experiment metric) to reduce variance and increase statistical power.',
  'MDE (Minimum Detectable Effect)': 'The smallest effect size that can be detected with a given sample size, power, and significance level. Helps in experiment planning.',
  'Statistical Power': 'The probability of correctly rejecting the null hypothesis when it is false (i.e., detecting a true effect). Typically set to 80% (0.8).',
  'Cohen\'s h': 'An effect size measure for comparing two proportions. h = 2 × (arcsin(√pB) - arcsin(√pA)). Values: < 0.2 (small), 0.2-0.5 (medium), > 0.5 (large).',
  'Cohen\'s d': 'An effect size measure for comparing two means. d = (mean_B - mean_A) / pooled_std. Values: < 0.2 (small), 0.2-0.5 (medium), 0.5-0.8 (large), > 0.8 (very large).'
}

function GlossaryDrawer() {
  const [isOpen, setIsOpen] = useState(false)

  return (
    <>
      <button
        onClick={() => setIsOpen(true)}
        className="fixed bottom-6 right-6 w-12 h-12 bg-blue-600 text-white rounded-full shadow-lg hover:bg-blue-700 transition-all duration-200 flex items-center justify-center z-50"
        aria-label="Open glossary"
      >
        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
      </button>

      {isOpen && (
        <>
          <div
            className="fixed inset-0 bg-black/50 z-50"
            onClick={() => setIsOpen(false)}
          />
          <div className="fixed right-0 top-0 h-full w-96 bg-white dark:bg-gray-900 shadow-2xl z-50 overflow-y-auto animate-slide-down">
            <div className="p-6 border-b border-gray-200 dark:border-gray-700 sticky top-0 bg-white dark:bg-gray-900">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-xl font-bold text-gray-900 dark:text-white">Glossary</h2>
                <button
                  onClick={() => setIsOpen(false)}
                  className="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
                >
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
              <p className="text-sm text-gray-600 dark:text-gray-400">Definitions of statistical terms used in A/B testing</p>
            </div>
            
            <div className="p-6 space-y-6">
              {Object.entries(glossaryTerms).map(([term, definition]) => (
                <div key={term} className="pb-6 border-b border-gray-200 dark:border-gray-700 last:border-0">
                  <h3 className="font-bold text-gray-900 dark:text-white mb-2">{term}</h3>
                  <p className="text-sm text-gray-700 dark:text-gray-300 leading-relaxed">{definition}</p>
                </div>
              ))}
            </div>
          </div>
        </>
      )}
    </>
  )
}

export default GlossaryDrawer


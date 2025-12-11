import React from 'react'
import StatCard from './StatCard'

// Format p-value for display (handles very small values with scientific notation)
function formatPValue(pValue) {
  if (pValue < 0.0001) {
    return pValue.toExponential(3)
  } else if (pValue < 0.01) {
    return pValue.toFixed(4)
  } else {
    return pValue.toFixed(4)
  }
}

function SummaryCards({ results }) {
  const isSignificant = results.p_value < 0.05
  const lift = results.lift
  const probB = results.bayesian.prob_b_better

  const cards = [
    {
      icon: (
        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
        </svg>
      ),
      label: 'Lift',
      value: `${lift > 0 ? '+' : ''}${lift.toFixed(2)}%`,
      description: 'Percentage improvement',
      color: lift > 0 ? 'teal' : 'gray',
      badge: lift > 0 ? 'Positive' : lift < 0 ? 'Negative' : 'Neutral'
    },
    {
      icon: (
        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
        </svg>
      ),
      label: 'P-Value',
      value: formatPValue(results.p_value),
      description: results.test_type,
      color: isSignificant ? 'teal' : 'gray',
      badge: isSignificant ? 'Significant' : 'Not Significant'
    },
    {
      icon: (
        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 12l3-3 3 3 4-4M8 21l4-4 4 4M3 4h18M4 4h16v12a1 1 0 01-1 1H5a1 1 0 01-1-1V4z" />
        </svg>
      ),
      label: 'Effect Size',
      value: Math.abs(results.effect_size).toFixed(3),
      description: results.effect_size_type,
      color: Math.abs(results.effect_size) > 0.5 ? 'purple' : 'indigo'
    },
    {
      icon: (
        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
      ),
      label: '95% Confidence Interval',
      value: `[${results.confidence_interval.lower.toFixed(4)}, ${results.confidence_interval.upper.toFixed(4)}]`,
      description: 'Difference between groups',
      color: (results.confidence_interval.lower > 0 && results.confidence_interval.upper > 0) ||
             (results.confidence_interval.lower < 0 && results.confidence_interval.upper < 0) ? 'teal' : 'indigo'
    },
    {
      icon: (
        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
        </svg>
      ),
      label: 'Bayesian Probability',
      value: `${(probB * 100).toFixed(1)}%`,
      description: 'P(B > A)',
      color: probB > 0.95 ? 'teal' : probB < 0.05 ? 'gray' : 'purple'
    }
  ]

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
      {cards.map((card, idx) => (
        <StatCard key={idx} {...card} />
      ))}
    </div>
  )
}

export default SummaryCards

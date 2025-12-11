import React, { useState } from 'react'

function PowerCalculator() {
  const [baselineRate, setBaselineRate] = useState(0.1)
  const [expectedLift, setExpectedLift] = useState(0.1)
  const [alpha, setAlpha] = useState(0.05)
  const [power, setPower] = useState(0.8)
  const [requiredSampleSize, setRequiredSampleSize] = useState(null)

  const calculateSampleSize = () => {
    // Two-proportion z-test sample size calculation
    // n = (Z_alpha/2 + Z_beta)^2 * (p1(1-p1) + p2(1-p2)) / (p1 - p2)^2
    
    const p1 = baselineRate
    const p2 = baselineRate * (1 + expectedLift)
    
    // Z-scores
    const zAlpha = 1.96 // For alpha = 0.05 (two-tailed)
    const zBeta = 0.84 // For power = 0.8
    
    if (p1 === p2) {
      setRequiredSampleSize(null)
      return
    }
    
    const numerator = Math.pow(zAlpha + zBeta, 2) * (p1 * (1 - p1) + p2 * (1 - p2))
    const denominator = Math.pow(p1 - p2, 2)
    const n = Math.ceil(numerator / denominator)
    
    setRequiredSampleSize(n)
  }

  React.useEffect(() => {
    calculateSampleSize()
  }, [baselineRate, expectedLift, alpha, power])

  return (
    <div className="bg-white rounded-xl border border-gray-200 p-6">
      <h3 className="text-lg font-bold text-gray-900 mb-4">Statistical Power Calculator</h3>
      <p className="text-sm text-gray-600 mb-6">Calculate the required sample size for your experiment</p>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Baseline Conversion Rate
          </label>
          <input
            type="number"
            step="0.01"
            min="0"
            max="1"
            value={baselineRate}
            onChange={(e) => setBaselineRate(parseFloat(e.target.value) || 0)}
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          />
        </div>
        
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Expected Lift (decimal, e.g., 0.1 for 10%)
          </label>
          <input
            type="number"
            step="0.01"
            min="0"
            value={expectedLift}
            onChange={(e) => setExpectedLift(parseFloat(e.target.value) || 0)}
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          />
        </div>
        
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Alpha (Significance Level)
          </label>
          <input
            type="number"
            step="0.01"
            min="0"
            max="1"
            value={alpha}
            onChange={(e) => setAlpha(parseFloat(e.target.value) || 0.05)}
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          />
        </div>
        
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Power (1 - Beta)
          </label>
          <input
            type="number"
            step="0.01"
            min="0"
            max="1"
            value={power}
            onChange={(e) => setPower(parseFloat(e.target.value) || 0.8)}
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          />
        </div>
      </div>
      
      {requiredSampleSize && (
        <div className="bg-blue-50 border-2 border-blue-200 rounded-lg p-4">
          <p className="text-sm font-semibold text-blue-900 mb-1">Required Sample Size per Group</p>
          <p className="text-2xl font-bold text-blue-600">{requiredSampleSize.toLocaleString()}</p>
          <p className="text-xs text-blue-700 mt-2">Total sample size needed: {(requiredSampleSize * 2).toLocaleString()}</p>
        </div>
      )}
    </div>
  )
}

export default PowerCalculator


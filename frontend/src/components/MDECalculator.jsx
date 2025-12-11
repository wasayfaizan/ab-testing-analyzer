import React, { useState } from 'react'

function MDECalculator() {
  const [power, setPower] = useState(0.8)
  const [sampleSize, setSampleSize] = useState(1000)
  const [baselineRate, setBaselineRate] = useState(0.1)
  const [alpha, setAlpha] = useState(0.05)
  const [mde, setMDE] = useState(null)

  const calculateMDE = () => {
    // Reverse power calculation to find MDE
    // Starting from sample size, find minimum detectable effect
    
    const zAlpha = 1.96 // For alpha = 0.05
    const zBeta = 0.84 // For power = 0.8
    
    // Solve for p2 given n, p1, zAlpha, zBeta
    // n = (Z_alpha/2 + Z_beta)^2 * (p1(1-p1) + p2(1-p2)) / (p1 - p2)^2
    
    const p1 = baselineRate
    const n = sampleSize
    
    // Approximate MDE using iterative approach or formula
    // For simplicity, we'll use an approximation
    const se = Math.sqrt((p1 * (1 - p1)) / n)
    const criticalValue = zAlpha * se * Math.sqrt(2) // Two-sample
    
    // Minimum detectable difference
    const minDiff = criticalValue * Math.sqrt(2) / (zAlpha + zBeta)
    const mdeValue = minDiff / p1
    
    setMDE(mdeValue)
  }

  React.useEffect(() => {
    calculateMDE()
  }, [power, sampleSize, baselineRate, alpha])

  return (
    <div className="bg-white rounded-xl border border-gray-200 p-6">
      <h3 className="text-lg font-bold text-gray-900 mb-4">Minimum Detectable Effect (MDE) Calculator</h3>
      <p className="text-sm text-gray-600 mb-6">Calculate the smallest lift you can detect with your sample size</p>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
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
        
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Sample Size per Group
          </label>
          <input
            type="number"
            step="100"
            min="10"
            value={sampleSize}
            onChange={(e) => setSampleSize(parseInt(e.target.value) || 1000)}
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          />
        </div>
        
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
            onChange={(e) => setBaselineRate(parseFloat(e.target.value) || 0.1)}
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
      </div>
      
      {mde !== null && (
        <div className="bg-purple-50 border-2 border-purple-200 rounded-lg p-4">
          <p className="text-sm font-semibold text-purple-900 mb-1">Minimum Detectable Effect</p>
          <p className="text-2xl font-bold text-purple-600">{(mde * 100).toFixed(2)}%</p>
          <p className="text-xs text-purple-700 mt-2">
            You can detect lifts of {(mde * 100).toFixed(2)}% or larger with {power * 100}% power
          </p>
        </div>
      )}
    </div>
  )
}

export default MDECalculator


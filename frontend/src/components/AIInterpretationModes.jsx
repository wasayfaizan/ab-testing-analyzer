import React, { useState } from 'react'
import axios from 'axios'

function AIInterpretationModes({ results }) {
  const [selectedMode, setSelectedMode] = useState('executive')
  const [interpretation, setInterpretation] = useState('')
  const [loading, setLoading] = useState(false)

  const modes = [
    { id: 'executive', label: 'Executive Summary', icon: '👔' },
    { id: 'non-technical', label: 'Non-Technical', icon: '💬' },
    { id: 'slack-email', label: 'Slack/Email', icon: '📧' },
    { id: 'recommendations', label: 'Recommendations', icon: '💡' }
  ]

  const generateInterpretation = async (mode) => {
    setLoading(true)
    try {
      const response = await axios.post('/api/generate-interpretation-mode', {
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
        group_b_name: results.group_b_name
      })
      setInterpretation(response.data.interpretation)
      setSelectedMode(mode)
    } catch (error) {
      console.error('Error generating interpretation:', error)
      setInterpretation('Error generating interpretation. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  const formatText = (text) => {
    const lines = text.split('\n')
    return lines.map((line, idx) => {
      if (line.startsWith('## ')) {
        return <h3 key={idx} className="text-lg font-bold text-gray-900 dark:text-white mt-4 mb-2">{line.replace('## ', '')}</h3>
      } else if (line.startsWith('• ') || line.startsWith('✅ ') || line.startsWith('❌ ') || line.startsWith('⚠️ ') || line.startsWith('📊 ') || line.startsWith('🎯 ') || line.startsWith('🔬 ') || line.startsWith('👥 ') || line.startsWith('📈 ') || line.startsWith('🎲 ') || line.startsWith('🔄 ')) {
        return <p key={idx} className="text-gray-700 dark:text-gray-300 mb-2">{line}</p>
      } else if (line.trim() === '') {
        return <br key={idx} />
      } else {
        return <p key={idx} className="text-gray-700 dark:text-gray-300 mb-2">{line}</p>
      }
    })
  }

  return (
    <div className="bg-white dark:bg-gray-800 rounded-xl border-2 border-gray-200 dark:border-gray-700 p-6">
      <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-4">AI Interpretation Modes</h3>
      
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-6">
        {modes.map((mode) => (
          <button
            key={mode.id}
            onClick={() => generateInterpretation(mode.id)}
            disabled={loading}
            className={`px-4 py-3 rounded-lg border-2 transition-all ${
              selectedMode === mode.id
                ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20 text-blue-700 dark:text-blue-300'
                : 'border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600 text-gray-700 dark:text-gray-300'
            } ${loading ? 'opacity-50 cursor-not-allowed' : ''}`}
          >
            <div className="text-2xl mb-1">{mode.icon}</div>
            <div className="text-xs font-medium">{mode.label}</div>
          </button>
        ))}
      </div>

      {loading && (
        <div className="text-center py-8">
          <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
          <p className="text-sm text-gray-600 dark:text-gray-400 mt-2">Generating interpretation...</p>
        </div>
      )}

      {interpretation && !loading && (
        <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
          <div className="prose max-w-none dark:prose-invert">
            {formatText(interpretation)}
          </div>
        </div>
      )}
    </div>
  )
}

export default AIInterpretationModes


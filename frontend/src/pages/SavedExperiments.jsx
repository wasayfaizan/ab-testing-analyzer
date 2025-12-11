import React, { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { getSavedExperiments, deleteExperiment, renameExperiment } from '../utils/exportUtils'

function SavedExperiments({ setAnalysisResults }) {
  const navigate = useNavigate()
  const [experiments, setExperiments] = useState([])
  const [editingId, setEditingId] = useState(null)
  const [editName, setEditName] = useState('')

  useEffect(() => {
    setExperiments(getSavedExperiments())
  }, [])

  const handleLoad = (experiment) => {
    setAnalysisResults(experiment.results)
    localStorage.setItem('abTestResults', JSON.stringify(experiment.results))
    navigate('/results')
  }

  const handleDelete = (id) => {
    if (window.confirm('Are you sure you want to delete this experiment?')) {
      deleteExperiment(id)
      setExperiments(getSavedExperiments())
    }
  }

  const handleRename = (id) => {
    if (editName.trim()) {
      renameExperiment(id, editName.trim())
      setExperiments(getSavedExperiments())
      setEditingId(null)
      setEditName('')
    }
  }

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-6xl mx-auto">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">Saved Experiments</h1>
          <p className="text-gray-600 dark:text-gray-400">View and manage your past A/B test analyses</p>
        </div>

        {experiments.length === 0 ? (
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-12 text-center">
            <svg className="mx-auto h-12 w-12 text-gray-400 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
            <p className="text-gray-600 dark:text-gray-400 mb-4">No saved experiments yet</p>
            <button
              onClick={() => navigate('/')}
              className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
            >
              Run New Analysis
            </button>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {experiments.map((exp) => (
              <div
                key={exp.id}
                className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6 hover:shadow-md transition-shadow"
              >
                <div className="flex items-start justify-between mb-4">
                  {editingId === exp.id ? (
                    <div className="flex-1 mr-2">
                      <input
                        type="text"
                        value={editName}
                        onChange={(e) => setEditName(e.target.value)}
                        className="w-full px-2 py-1 border border-gray-300 dark:border-gray-600 rounded text-sm"
                        autoFocus
                        onBlur={() => handleRename(exp.id)}
                        onKeyPress={(e) => {
                          if (e.key === 'Enter') handleRename(exp.id)
                        }}
                      />
                    </div>
                  ) : (
                    <h3 className="text-lg font-semibold text-gray-900 dark:text-white flex-1">
                      {exp.name}
                    </h3>
                  )}
                  <div className="flex gap-2">
                    {editingId !== exp.id && (
                      <button
                        onClick={() => {
                          setEditingId(exp.id)
                          setEditName(exp.name)
                        }}
                        className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
                      >
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
                        </svg>
                      </button>
                    )}
                    <button
                      onClick={() => handleDelete(exp.id)}
                      className="text-red-400 hover:text-red-600"
                    >
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                      </svg>
                    </button>
                  </div>
                </div>
                
                <div className="space-y-2 mb-4">
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    {new Date(exp.timestamp).toLocaleString()}
                  </p>
                  <div className="flex gap-4 text-sm">
                    <span className="text-gray-600 dark:text-gray-400">
                      Lift: <span className="font-semibold">{exp.results.lift > 0 ? '+' : ''}{exp.results.lift.toFixed(2)}%</span>
                    </span>
                    <span className="text-gray-600 dark:text-gray-400">
                      P: <span className="font-semibold">{exp.results.p_value < 0.0001 ? exp.results.p_value.toExponential(2) : exp.results.p_value.toFixed(4)}</span>
                    </span>
                  </div>
                </div>
                
                <button
                  onClick={() => handleLoad(exp)}
                  className="w-full px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                >
                  View Results
                </button>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

export default SavedExperiments


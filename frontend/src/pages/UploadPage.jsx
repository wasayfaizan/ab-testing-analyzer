import React, { useState, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { useDropzone } from 'react-dropzone'
import axios from 'axios'

function UploadPage({ setAnalysisResults }) {
  const [file, setFile] = useState(null)
  const [csvData, setCsvData] = useState(null)
  const [columns, setColumns] = useState([])
  const [groupColumn, setGroupColumn] = useState('')
  const [metricColumn, setMetricColumn] = useState('')
  const [metricType, setMetricType] = useState('binary')
  const [availableGroups, setAvailableGroups] = useState([])
  const [groupA, setGroupA] = useState('')  // Control group
  const [groupB, setGroupB] = useState('')  // Treatment group
  const [enableCuped, setEnableCuped] = useState(false)
  const [covariateColumn, setCovariateColumn] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const navigate = useNavigate()

  // Extract groups from CSV when group column is selected
  const extractGroups = useCallback((file, groupColName) => {
    if (!file || !groupColName) {
      setAvailableGroups([])
      setGroupA('')
      setGroupB('')
      return
    }

    const reader = new FileReader()
    reader.onload = (e) => {
      const text = e.target.result
      const lines = text.split('\n')
      if (lines.length < 2) return
      
      const headers = lines[0].split(',').map(h => h.trim())
      const groupIndex = headers.indexOf(groupColName)
      
      if (groupIndex === -1) return
      
      const groups = new Set()
      for (let i = 1; i < lines.length; i++) {
        const values = lines[i].split(',')
        if (values[groupIndex]) {
          const group = values[groupIndex].trim()
          if (group && group.toLowerCase() !== 'nan' && group !== '') {
            groups.add(group)
          }
        }
      }
      
      const uniqueGroups = Array.from(groups)
      setAvailableGroups(uniqueGroups)
      
      // Auto-select first two groups if available
      if (uniqueGroups.length >= 2) {
        setGroupA(uniqueGroups[0])
        setGroupB(uniqueGroups[1])
      } else if (uniqueGroups.length === 1) {
        setGroupA(uniqueGroups[0])
        setGroupB('')
      }
    }
    reader.readAsText(file)
  }, [])

  const onDrop = useCallback((acceptedFiles) => {
    const file = acceptedFiles[0]
    if (file) {
      setFile(file)
      setError('')
      
      // Read CSV to preview
      const reader = new FileReader()
      reader.onload = (e) => {
        const text = e.target.result
        const lines = text.split('\n').slice(0, 6) // First 5 rows
        const headers = lines[0].split(',')
        setColumns(headers.map(h => h.trim()))
        setCsvData(lines)
        
        // Auto-select common column names
        const groupCol = headers.find(h => 
          h.toLowerCase().includes('group') || 
          h.toLowerCase().includes('variant') ||
          h.toLowerCase() === 'a' ||
          h.toLowerCase() === 'b'
        )
        const metricCol = headers.find(h => 
          h.toLowerCase().includes('conversion') ||
          h.toLowerCase().includes('metric') ||
          h.toLowerCase().includes('revenue') ||
          h.toLowerCase().includes('value')
        )
        
        if (groupCol) {
          const colName = groupCol.trim()
          setGroupColumn(colName)
          // Extract groups after setting column
          setTimeout(() => extractGroups(file, colName), 100)
        }
        if (metricCol) setMetricColumn(metricCol.trim())
      }
      reader.readAsText(file)
    }
  }, [extractGroups])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/csv': ['.csv'],
      'text/plain': ['.csv']
    },
    multiple: false
  })

  const handleAnalyze = async () => {
    console.log('handleAnalyze called', { file: file?.name, groupColumn, metricColumn, groupA, groupB })
    
    if (!file || !groupColumn || !metricColumn) {
      setError('Please select a file and specify group and metric columns')
      return
    }

    if (!groupA || !groupB) {
      setError('Please select which group is Control (A) and which is Treatment (B)')
      return
    }

    if (groupA === groupB) {
      setError('Control and Treatment groups must be different')
      return
    }

    setLoading(true)
    setError('')
    console.log('Starting analysis...')

    try {
      const formData = new FormData()
      formData.append('file', file)
      formData.append('group_column', groupColumn)
      formData.append('metric_column', metricColumn)
      formData.append('metric_type', metricType)
      formData.append('group_a_name', groupA)  // Control group
      formData.append('group_b_name', groupB)  // Treatment group
      formData.append('enable_cuped', enableCuped.toString())
      if (enableCuped && covariateColumn) {
        formData.append('covariate_column', covariateColumn)
      }

      console.log('Sending request to backend...')
      const response = await axios.post('/api/analyze', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      })
      console.log('Analysis response received:', response.data)

      // Get AI interpretation
      console.log('Requesting AI interpretation...')
      const interpretationResponse = await axios.post(
        '/api/generate-interpretation',
        {
          lift: response.data.lift,
          p_value: response.data.p_value,
          effect_size: response.data.effect_size,
          ci_lower: response.data.confidence_interval.lower,
          ci_upper: response.data.confidence_interval.upper,
          prob_b_better: response.data.bayesian.prob_b_better,
          metric_type: response.data.metric_type,
          test_type: response.data.test_type
        }
      )
      console.log('Interpretation received')

      const results = {
        ...response.data,
        interpretation: interpretationResponse.data.interpretation
      }

      // Save to localStorage
      localStorage.setItem('abTestResults', JSON.stringify(results))
      
      setAnalysisResults(results)
      navigate('/results')
    } catch (err) {
      console.error('Analysis error:', err)
      if (err.code === 'ERR_NETWORK' || err.message?.includes('Network Error')) {
        setError('Network error: Could not connect to the backend server. Please make sure the backend is running on http://localhost:8000')
      } else if (err.response?.data?.detail) {
        setError(err.response.data.detail)
      } else if (err.response?.data?.message) {
        setError(err.response.data.message)
      } else if (err.message) {
        setError(err.message)
      } else {
        setError('Error analyzing data. Please check your CSV file and try again.')
      }
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gray-50 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">
            A/B Testing Experiment Analyzer
          </h1>
          <p className="text-lg text-gray-600">
            Upload your A/B test data and get comprehensive statistical analysis
          </p>
        </div>

        {/* Upload Card */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 mb-6">
          <h2 className="text-xl font-semibold text-gray-900 mb-4">Upload CSV File</h2>
          
          <div
            {...getRootProps()}
            className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${
              isDragActive
                ? 'border-blue-500 bg-blue-50'
                : 'border-gray-300 hover:border-gray-400'
            }`}
          >
            <input {...getInputProps()} />
            <div className="space-y-2">
              <svg
                className="mx-auto h-12 w-12 text-gray-400"
                stroke="currentColor"
                fill="none"
                viewBox="0 0 48 48"
              >
                <path
                  d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02"
                  strokeWidth={2}
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
              {file ? (
                <div>
                  <p className="text-sm font-medium text-gray-900">{file.name}</p>
                  <p className="text-xs text-gray-500 mt-1">
                    Click or drag to replace
                  </p>
                </div>
              ) : (
                <div>
                  <p className="text-sm text-gray-600">
                    Drag and drop your CSV file here, or click to browse
                  </p>
                  <p className="text-xs text-gray-500 mt-1">
                    CSV files only
                  </p>
                </div>
              )}
            </div>
          </div>

          {/* CSV Preview */}
          {csvData && (
            <div className="mt-6">
              <h3 className="text-sm font-medium text-gray-700 mb-2">CSV Preview</h3>
              <div className="overflow-x-auto border border-gray-200 rounded-lg">
                <table className="min-w-full divide-y divide-gray-200 text-xs">
                  <thead className="bg-gray-50">
                    <tr>
                      {csvData[0].split(',').map((col, idx) => (
                        <th key={idx} className="px-3 py-2 text-left font-medium text-gray-700">
                          {col.trim()}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    {csvData.slice(1, 6).map((row, rowIdx) => (
                      <tr key={rowIdx}>
                        {row.split(',').map((cell, cellIdx) => (
                          <td key={cellIdx} className="px-3 py-2 text-gray-600">
                            {cell.trim()}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>

        {/* Configuration Card */}
        {columns.length > 0 && (
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 mb-6">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">Configuration</h2>
            
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Group Column
                </label>
                <select
                  value={groupColumn}
                  onChange={(e) => {
                    const newCol = e.target.value
                    setGroupColumn(newCol)
                    setGroupA('')
                    setGroupB('')
                    setAvailableGroups([])
                    if (file && newCol) {
                      extractGroups(file, newCol)
                    }
                  }}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                >
                  <option value="">Select column...</option>
                  {columns.map((col) => (
                    <option key={col} value={col}>
                      {col}
                    </option>
                  ))}
                </select>
              </div>

              {groupColumn && availableGroups.length > 0 && (
                <>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Control Group (A) <span className="text-red-500">*</span>
                    </label>
                    <select
                      value={groupA}
                      onChange={(e) => setGroupA(e.target.value)}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                    >
                      <option value="">Select control group...</option>
                      {availableGroups.map((group) => (
                        <option key={group} value={group}>
                          {group}
                        </option>
                      ))}
                    </select>
                    <p className="text-xs text-gray-500 mt-1">This is your baseline/control group</p>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Treatment Group (B) <span className="text-red-500">*</span>
                    </label>
                    <select
                      value={groupB}
                      onChange={(e) => setGroupB(e.target.value)}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                    >
                      <option value="">Select treatment group...</option>
                      {availableGroups.map((group) => (
                        <option key={group} value={group}>
                          {group}
                        </option>
                      ))}
                    </select>
                    <p className="text-xs text-gray-500 mt-1">This is your test/treatment group</p>
                  </div>
                </>
              )}

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Metric Column
                </label>
                <select
                  value={metricColumn}
                  onChange={(e) => setMetricColumn(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                >
                  <option value="">Select column...</option>
                  {columns.map((col) => (
                    <option key={col} value={col}>
                      {col}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Metric Type
                </label>
                <select
                  value={metricType}
                  onChange={(e) => setMetricType(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                >
                  <option value="binary">Binary (Conversion/Click)</option>
                  <option value="numeric">Numeric (Revenue/Time/etc.)</option>
                </select>
              </div>

              {/* CUPED Option */}
              <div className="pt-4 border-t border-gray-200">
                <div className="flex items-center gap-3 mb-3">
                  <input
                    type="checkbox"
                    id="enableCuped"
                    checked={enableCuped}
                    onChange={(e) => {
                      setEnableCuped(e.target.checked)
                      if (!e.target.checked) setCovariateColumn('')
                    }}
                    className="w-4 h-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
                  />
                  <label htmlFor="enableCuped" className="text-sm font-medium text-gray-700">
                    Enable CUPED Variance Reduction
                  </label>
                </div>
                {enableCuped && (
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Covariate Column (for CUPED)
                    </label>
                    <select
                      value={covariateColumn}
                      onChange={(e) => setCovariateColumn(e.target.value)}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                    >
                      <option value="">Select covariate column...</option>
                      {columns.filter(col => col !== groupColumn && col !== metricColumn).map((col) => (
                        <option key={col} value={col}>
                          {col}
                        </option>
                      ))}
                    </select>
                    <p className="text-xs text-gray-500 mt-1">
                      CUPED uses a pre-experiment covariate to reduce variance and increase power
                    </p>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Sample Data Card */}
        <div className="bg-blue-50 rounded-lg border border-blue-200 p-6 mb-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-2">Sample CSV Format</h3>
          <p className="text-sm text-gray-600 mb-4">
            Your CSV should have a column for groups (A/B) and a column for the metric:
          </p>
          <div className="bg-white rounded border border-gray-200 p-4 font-mono text-xs overflow-x-auto">
            <pre>{`group,conversion,revenue
A,1,25.50
A,0,0.00
A,1,30.00
B,1,28.00
B,1,32.50
B,0,0.00`}</pre>
          </div>
        </div>

        {/* Error Message */}
        {error && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6">
            <p className="text-sm text-red-800">{error}</p>
          </div>
        )}

        {/* Analyze Button */}
        <div className="text-center">
          <button
            onClick={handleAnalyze}
            disabled={loading || !file || !groupColumn || !metricColumn || !groupA || !groupB}
            className={`px-8 py-3 rounded-lg font-semibold text-white transition-colors ${
              loading || !file || !groupColumn || !metricColumn || !groupA || !groupB
                ? 'bg-gray-400 cursor-not-allowed'
                : 'bg-blue-600 hover:bg-blue-700'
            }`}
          >
            {loading ? (
              <span className="flex items-center justify-center">
                <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                Analyzing...
              </span>
            ) : (
              'Run Analysis'
            )}
          </button>
        </div>
      </div>
    </div>
  )
}

export default UploadPage


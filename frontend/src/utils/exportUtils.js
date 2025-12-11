import html2canvas from 'html2canvas'
import jsPDF from 'jspdf'

export const exportToCSV = (results) => {
  const csvRows = []
  
  // Header
  csvRows.push(['Metric', 'Value'])
  
  // Add all metrics
  csvRows.push(['Lift (%)', results.lift.toFixed(2)])
  csvRows.push(['P-Value', results.p_value])
  csvRows.push(['Effect Size', results.effect_size.toFixed(4)])
  csvRows.push(['Effect Size Type', results.effect_size_type])
  csvRows.push(['95% CI Lower', results.confidence_interval.lower.toFixed(4)])
  csvRows.push(['95% CI Upper', results.confidence_interval.upper.toFixed(4)])
  csvRows.push(['Bayesian P(B > A)', (results.bayesian.prob_b_better * 100).toFixed(2) + '%'])
  csvRows.push(['Group A Size', results.group_a.size])
  csvRows.push(['Group B Size', results.group_b.size])
  
  if (results.metric_type === 'binary') {
    csvRows.push(['Group A Conversion Rate', results.group_a.rate.toFixed(4)])
    csvRows.push(['Group B Conversion Rate', results.group_b.rate.toFixed(4)])
  } else {
    csvRows.push(['Group A Mean', results.group_a.mean.toFixed(4)])
    csvRows.push(['Group B Mean', results.group_b.mean.toFixed(4)])
  }
  
  const csvContent = csvRows.map(row => row.join(',')).join('\n')
  const blob = new Blob([csvContent], { type: 'text/csv' })
  const url = window.URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `ab-test-results-${Date.now()}.csv`
  a.click()
  window.URL.revokeObjectURL(url)
}

export const exportToPNG = async (elementId, filename) => {
  const element = document.getElementById(elementId)
  if (!element) return
  
  const canvas = await html2canvas(element, {
    backgroundColor: '#ffffff',
    scale: 2
  })
  
  const url = canvas.toDataURL('image/png')
  const a = document.createElement('a')
  a.href = url
  a.download = filename || `chart-${Date.now()}.png`
  a.click()
}

export const exportToJSON = (results) => {
  const jsonStr = JSON.stringify(results, null, 2)
  const blob = new Blob([jsonStr], { type: 'application/json' })
  const url = window.URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `ab-test-results-${Date.now()}.json`
  a.click()
  window.URL.revokeObjectURL(url)
}

export const saveExperiment = (results, name) => {
  const saved = JSON.parse(localStorage.getItem('savedExperiments') || '[]')
  const experiment = {
    id: Date.now().toString(),
    name: name || `Experiment ${new Date().toLocaleString()}`,
    timestamp: new Date().toISOString(),
    results: results
  }
  saved.push(experiment)
  localStorage.setItem('savedExperiments', JSON.stringify(saved))
  return experiment.id
}

export const getSavedExperiments = () => {
  return JSON.parse(localStorage.getItem('savedExperiments') || '[]')
}

export const deleteExperiment = (id) => {
  const saved = getSavedExperiments()
  const filtered = saved.filter(exp => exp.id !== id)
  localStorage.setItem('savedExperiments', JSON.stringify(filtered))
}

export const renameExperiment = (id, newName) => {
  const saved = getSavedExperiments()
  const experiment = saved.find(exp => exp.id === id)
  if (experiment) {
    experiment.name = newName
    localStorage.setItem('savedExperiments', JSON.stringify(saved))
  }
}


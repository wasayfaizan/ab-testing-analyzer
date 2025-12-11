import jsPDF from 'jspdf'
import html2canvas from 'html2canvas'

export async function generatePDFReport(results) {
  // Create a temporary container for PDF content
  const pdfContainer = document.createElement('div')
  pdfContainer.style.width = '210mm'
  pdfContainer.style.padding = '20mm'
  pdfContainer.style.backgroundColor = 'white'
  pdfContainer.style.fontFamily = 'Arial, sans-serif'
  pdfContainer.style.position = 'absolute'
  pdfContainer.style.left = '-9999px'
  document.body.appendChild(pdfContainer)

  // Build HTML content
  const isSignificant = results.p_value < 0.05
  const lift = results.lift
  const probB = results.bayesian.prob_b_better

  let recommendation = "Continue Experiment"
  if (isSignificant && lift > 0) {
    recommendation = "Choose B"
  } else if (isSignificant && lift < 0) {
    recommendation = "Choose A"
  } else if (probB > 0.95) {
    recommendation = "Choose B"
  } else if (probB < 0.05) {
    recommendation = "Choose A"
  } else if (!isSignificant && Math.abs(lift) < 1) {
    recommendation = "No statistically significant difference"
  }

  pdfContainer.innerHTML = `
    <div style="color: #1f2937;">
      <h1 style="font-size: 28px; font-weight: bold; margin-bottom: 10px; color: #111827;">
        A/B Testing Experiment Analyzer
      </h1>
      <p style="color: #6b7280; margin-bottom: 30px;">Statistical Analysis Report</p>
      
      <h2 style="font-size: 20px; font-weight: bold; margin-top: 30px; margin-bottom: 15px; color: #111827;">
        Summary Metrics
      </h2>
      <table style="width: 100%; border-collapse: collapse; margin-bottom: 30px;">
        <tr style="background-color: #f3f4f6;">
          <th style="padding: 10px; text-align: left; border: 1px solid #d1d5db;">Metric</th>
          <th style="padding: 10px; text-align: left; border: 1px solid #d1d5db;">Value</th>
        </tr>
        <tr>
          <td style="padding: 10px; border: 1px solid #d1d5db;"><strong>Lift</strong></td>
          <td style="padding: 10px; border: 1px solid #d1d5db;">${lift > 0 ? '+' : ''}${lift.toFixed(2)}%</td>
        </tr>
        <tr style="background-color: #f9fafb;">
          <td style="padding: 10px; border: 1px solid #d1d5db;"><strong>P-Value</strong></td>
          <td style="padding: 10px; border: 1px solid #d1d5db;">${results.p_value < 0.0001 ? results.p_value.toExponential(3) : results.p_value.toFixed(4)}</td>
        </tr>
        <tr>
          <td style="padding: 10px; border: 1px solid #d1d5db;"><strong>Test Type</strong></td>
          <td style="padding: 10px; border: 1px solid #d1d5db;">${results.test_type}</td>
        </tr>
        <tr style="background-color: #f9fafb;">
          <td style="padding: 10px; border: 1px solid #d1d5db;"><strong>Effect Size</strong></td>
          <td style="padding: 10px; border: 1px solid #d1d5db;">${Math.abs(results.effect_size).toFixed(3)} (${results.effect_size_type})</td>
        </tr>
        <tr>
          <td style="padding: 10px; border: 1px solid #d1d5db;"><strong>95% Confidence Interval</strong></td>
          <td style="padding: 10px; border: 1px solid #d1d5db;">[${results.confidence_interval.lower.toFixed(4)}, ${results.confidence_interval.upper.toFixed(4)}]</td>
        </tr>
        <tr style="background-color: #f9fafb;">
          <td style="padding: 10px; border: 1px solid #d1d5db;"><strong>Bayesian Probability (B > A)</strong></td>
          <td style="padding: 10px; border: 1px solid #d1d5db;">${(probB * 100).toFixed(1)}%</td>
        </tr>
        <tr>
          <td style="padding: 10px; border: 1px solid #d1d5db;"><strong>Recommendation</strong></td>
          <td style="padding: 10px; border: 1px solid #d1d5db; font-weight: bold;">${recommendation}</td>
        </tr>
      </table>

      <h2 style="font-size: 20px; font-weight: bold; margin-top: 30px; margin-bottom: 15px; color: #111827;">
        Group Statistics
      </h2>
      <table style="width: 100%; border-collapse: collapse; margin-bottom: 30px;">
        <tr style="background-color: #f3f4f6;">
          <th style="padding: 10px; text-align: left; border: 1px solid #d1d5db;">Group</th>
          <th style="padding: 10px; text-align: left; border: 1px solid #d1d5db;">Sample Size</th>
          <th style="padding: 10px; text-align: left; border: 1px solid #d1d5db;">${results.metric_type === 'binary' ? 'Conversion Rate' : 'Mean'}</th>
          ${results.metric_type === 'numeric' ? '<th style="padding: 10px; text-align: left; border: 1px solid #d1d5db;">Std Dev</th>' : ''}
        </tr>
        <tr>
          <td style="padding: 10px; border: 1px solid #d1d5db;"><strong>Group A</strong></td>
          <td style="padding: 10px; border: 1px solid #d1d5db;">${results.group_a.size}</td>
          <td style="padding: 10px; border: 1px solid #d1d5db;">${results.metric_type === 'binary' ? (results.group_a.rate * 100).toFixed(2) + '%' : results.group_a.mean.toFixed(2)}</td>
          ${results.metric_type === 'numeric' ? `<td style="padding: 10px; border: 1px solid #d1d5db;">${results.group_a.std.toFixed(2)}</td>` : ''}
        </tr>
        <tr style="background-color: #f9fafb;">
          <td style="padding: 10px; border: 1px solid #d1d5db;"><strong>Group B</strong></td>
          <td style="padding: 10px; border: 1px solid #d1d5db;">${results.group_b.size}</td>
          <td style="padding: 10px; border: 1px solid #d1d5db;">${results.metric_type === 'binary' ? (results.group_b.rate * 100).toFixed(2) + '%' : results.group_b.mean.toFixed(2)}</td>
          ${results.metric_type === 'numeric' ? `<td style="padding: 10px; border: 1px solid #d1d5db;">${results.group_b.std.toFixed(2)}</td>` : ''}
        </tr>
      </table>

      <h2 style="font-size: 20px; font-weight: bold; margin-top: 30px; margin-bottom: 15px; color: #111827;">
        AI Interpretation
      </h2>
      <div style="background-color: #f9fafb; padding: 20px; border-radius: 8px; border: 1px solid #e5e7eb; margin-bottom: 30px;">
        ${results.interpretation.replace(/\n/g, '<br>').replace(/## /g, '<h3 style="font-size: 16px; font-weight: bold; margin-top: 15px; margin-bottom: 10px;">').replace(/\*\*/g, '<strong>').replace(/\*\*/g, '</strong>')}
      </div>

      <div style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #e5e7eb; color: #6b7280; font-size: 12px;">
        <p>Generated by A/B Testing Experiment Analyzer</p>
        <p>Report generated on ${new Date().toLocaleString()}</p>
      </div>
    </div>
  `

  // Wait for rendering
  await new Promise(resolve => setTimeout(resolve, 500))

  // Convert to canvas then PDF
  const canvas = await html2canvas(pdfContainer, {
    scale: 2,
    useCORS: true,
    logging: false
  })

  const imgData = canvas.toDataURL('image/png')
  const pdf = new jsPDF('p', 'mm', 'a4')
  const imgWidth = 210
  const pageHeight = 297
  const imgHeight = (canvas.height * imgWidth) / canvas.width
  let heightLeft = imgHeight

  let position = 0

  pdf.addImage(imgData, 'PNG', 0, position, imgWidth, imgHeight)
  heightLeft -= pageHeight

  while (heightLeft >= 0) {
    position = heightLeft - imgHeight
    pdf.addPage()
    pdf.addImage(imgData, 'PNG', 0, position, imgWidth, imgHeight)
    heightLeft -= pageHeight
  }

  // Clean up
  document.body.removeChild(pdfContainer)

  // Save PDF
  pdf.save(`ab-test-analysis-${Date.now()}.pdf`)
}


<body>
<div class="container">

<!-- HEADER -->
<div class="header">
<div class="icons">📊 📈 🧪</div>
<h1>A/B Testing Experiment Analyzer</h1>
<p class="subtitle">
A professional, dataset-agnostic platform for rigorous experimentation analysis
</p>
</div>
<p align="center">
  <img src="https://github.com/user-attachments/assets/04046335-f74a-43d4-86b1-f1c085b72d9d" width="85%" />
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/8cbbeeed-42a3-452f-b407-8c4b07c95057" width="85%" />
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/81cbdc15-f467-4d57-97e6-b1c84da94093" width="85%" />
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/81b4d5ca-0c71-4a44-8a1c-3fa1e4ca0993" width="85%" />
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/6bd0596f-6481-4fb4-b0b8-b1b6d95e8ace" width="85%" />
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/61dfbfc2-6af6-4618-b4ba-3d1c019f241f" width="85%" />
</p>






<div class="content">

<!-- OVERVIEW -->
<div class="section">
<h2>📋 Overview</h2>
<p>
<strong>A/B Testing Experiment Analyzer</strong> is a full-stack analytics platform
designed to evaluate controlled experiments using statistically rigorous methods.
The tool is <strong>dataset-agnostic</strong> and adapts dynamically to user-selected
columns, making it suitable for a wide range of experimentation use cases.
</p>
<p>
It supports simple A/B tests as well as complex multi-variant experiments involving
segmentation, regression with covariates, non-parametric testing, and time-based analysis.
The goal is to provide trustworthy statistical inference and clear, actionable insights.
</p>
</div>

<!-- KEY FEATURES -->
<div class="section">
<h2>✨ Key Features</h2>

<div class="feature-grid">

<div class="feature-card">
<h4>Standard A/B Testing</h4>
<ul>
<li>Binary and numeric outcome metrics</li>
<li>Two-proportion z-test</li>
<li>Two-sample t-test</li>
<li>Bayesian A/B analysis</li>
<li>CUPED variance reduction</li>
</ul>
</div>

<div class="feature-card">
<h4>Advanced Experimentation</h4>
<ul>
<li>Multi-variant testing (A/B/C/D)</li>
<li>Segmentation analysis</li>
<li>Time-based analysis</li>
<li>Regression with covariates</li>
<li>Non-parametric tests</li>
</ul>
</div>

<div class="feature-card">
<h4>Visualization & Insights</h4>
<ul>
<li>Forest plots</li>
<li>Ranking and comparison charts</li>
<li>Marginal effects plots</li>
<li>Interactive visualizations</li>
<li>Automated insights summaries</li>
</ul>
</div>

<div class="feature-card">
<h4>Statistical Tooling</h4>
<ul>
<li>Power analysis</li>
<li>Minimum Detectable Effect (MDE)</li>
<li>Confidence intervals</li>
<li>Effect sizes (Cohen’s d / h)</li>
<li>Risk and uplift assessment</li>
</ul>
</div>

</div>
</div>

<!-- QUICK START -->
<div class="section">
<h2>🚀 Quick Start</h2>

<h3>Prerequisites</h3>
<ul style="margin-left: 20px;">
<li>Python 3.8+</li>
<li>Node.js 16+</li>
<li>npm or yarn</li>
</ul>

<h3>Backend Setup</h3>
<div class="code-block">
<code>cd backend
pip install -r requirements.txt
uvicorn main:app --reload</code>
</div>
<p>Backend runs at <code>http://localhost:8000</code></p>

<h3>Frontend Setup</h3>
<div class="code-block">
<code>cd frontend
npm install
npm run dev</code>
</div>
<p>Frontend runs at <code>http://localhost:5173</code></p>
</div>

<!-- ANALYSIS MODES -->
<div class="section">
<h2>📊 Analysis Modes</h2>

<div class="feature-grid">

<div class="feature-card">
<h4>Standard A/B Test</h4>
<p>
Compare two experimental groups using binary or numeric outcomes.
Includes classical hypothesis testing, Bayesian inference, and confidence intervals.
</p>
</div>

<div class="feature-card">
<h4>Multi-Variant Testing</h4>
<p>
Analyze experiments with more than two variants using global and pairwise
significance testing.
</p>
</div>

<div class="feature-card">
<h4>Segmentation Analysis</h4>
<p>
Evaluate how treatment effects vary across user-defined segments.
Includes forest plots, summaries, and targeting recommendations.
</p>
</div>

<div class="feature-card">
<h4>Time-Based Analysis</h4>
<p>
Identify temporal patterns using timestamp data, such as daily, weekly,
or hourly performance trends.
</p>
</div>

<div class="feature-card">
<h4>Regression Analysis</h4>
<p>
Control for multiple covariates simultaneously using regression models
with full statistical inference.
</p>
</div>

<div class="feature-card">
<h4>Non-Parametric Tests</h4>
<p>
Distribution-free statistical tests (Mann-Whitney U, Kruskal-Wallis)
for skewed or non-normal metrics.
</p>
</div>

</div>
</div>

<!-- SPECIAL FEATURES -->
<div class="section">
<h2>🎯 Special Features</h2>

<h3>Automated Insights</h3>
<p>
The analyzer generates plain-language interpretations of statistical results,
helping non-technical stakeholders understand experimental outcomes.
</p>

<h3>Forest Plots & Rankings</h3>
<p>
Segment-level treatment effects are visualized using forest plots with
confidence intervals and significance highlighting.
</p>

<h3>Privacy-First Design</h3>
<p>
All data processing occurs locally. No experiment data is stored or transmitted
externally, except for optional AI-powered interpretation.
</p>
</div>

<!-- TECH STACK -->
<div class="section">
<h2>🛠️ Technology Stack</h2>
<div class="tech-stack">
<span class="tech-item"><strong>Backend:</strong> Python, FastAPI</span>
<span class="tech-item"><strong>Statistics:</strong> SciPy, Statsmodels, NumPy, Pandas</span>
<span class="tech-item"><strong>Frontend:</strong> React, Vite</span>
<span class="tech-item"><strong>Styling:</strong> Tailwind CSS</span>
<span class="tech-item"><strong>Visualization:</strong> Plotly.js</span>
<span class="tech-item"><strong>AI:</strong> OpenAI API</span>
</div>
</div>

<!-- DATA FORMAT -->
<div class="section">
<h2>📝 Data Format</h2>

<p>
The analyzer is <strong>fully flexible</strong> and does not require a fixed schema.
Users select column roles directly in the interface.
</p>

<h3>Minimum Requirements</h3>
<ul style="margin-left: 20px;">
<li><strong>Group column</strong> — identifies experimental variants</li>
<li><strong>Metric column</strong> — outcome being measured (binary or numeric)</li>
</ul>

<h3>Optional Columns</h3>
<ul style="margin-left: 20px;">
<li>Segmentation variables (categorical)</li>
<li>Timestamp column (for time-based analysis)</li>
<li>Covariates (numeric or categorical)</li>
</ul>

<div class="code-block">
<code>variant,metric,segment,timestamp,covariate_1,covariate_2
A,1,segment_1,2024-01-01 10:15:00,12,0.45
A,0,segment_2,2024-01-01 10:20:00,8,0.31
B,1,segment_1,2024-01-01 10:17:00,15,0.52
B,1,segment_2,2024-01-01 10:25:00,20,0.61</code>
</div>
</div>




</div>
</body>
</html>


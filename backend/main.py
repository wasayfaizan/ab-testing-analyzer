"""
A/B Testing Experiment Analyzer - Backend API
FastAPI backend for statistical analysis of A/B test data
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, Literal, List
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest
import io
import json
import os
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client (will be None if API key is not set)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

app = FastAPI(title="A/B Testing Experiment Analyzer API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalysisRequest(BaseModel):
    group_column: str
    metric_column: str
    metric_type: Literal["binary", "numeric"]


# Pydantic models for new endpoints
class PowerAnalysisRequest(BaseModel):
    baseline_rate: float = Field(..., ge=0, le=1, description="Baseline conversion rate (0-1)")
    expected_lift: float = Field(..., description="Expected lift as percentage or decimal (e.g., 0.1 for 10%)")
    alpha: float = Field(0.05, ge=0, le=1, description="Significance level (default 0.05)")
    power: float = Field(0.8, ge=0, le=1, description="Statistical power (default 0.8)")


class MDEAnalysisRequest(BaseModel):
    baseline_rate: float = Field(..., ge=0, le=1, description="Baseline conversion rate (0-1)")
    sample_size_per_group: int = Field(..., gt=0, description="Sample size per group")
    alpha: float = Field(0.05, ge=0, le=1, description="Significance level (default 0.05)")
    power: float = Field(0.8, ge=0, le=1, description="Statistical power (default 0.8)")


class SequentialTestingRequest(BaseModel):
    timestamp_column: str = Field(..., description="Column name containing timestamps")
    metric_column: str = Field(..., description="Column name containing metric values")
    group_column: str = Field(..., description="Column name containing group labels")
    group_a_name: str = Field(..., description="Control group name")
    group_b_name: str = Field(..., description="Treatment group name")
    metric_type: Literal["binary", "numeric"] = Field("binary", description="Type of metric")


class UpliftBootstrapRequest(BaseModel):
    metric_type: Literal["binary", "numeric"] = Field(..., description="Type of metric")
    metric_column: str = Field(..., description="Column name for metric")
    group_column: str = Field(..., description="Column name for groups")
    group_a_name: str = Field(..., description="Control group name")
    group_b_name: str = Field(..., description="Treatment group name")
    resamples: int = Field(5000, ge=100, le=10000, description="Number of bootstrap resamples")


class BayesianGridRequest(BaseModel):
    alpha_range: List[float] = Field(..., description="Range of alpha values for Beta prior")
    beta_range: List[float] = Field(..., description="Range of beta values for Beta prior")
    metric_type: Literal["binary", "numeric"] = Field(..., description="Type of metric")
    conversions_a: Optional[int] = Field(None, description="Conversions in group A (binary only)")
    conversions_b: Optional[int] = Field(None, description="Conversions in group B (binary only)")
    n_a: Optional[int] = Field(None, description="Sample size group A")
    n_b: Optional[int] = Field(None, description="Sample size group B")
    mean_a: Optional[float] = Field(None, description="Mean of group A (numeric only)")
    mean_b: Optional[float] = Field(None, description="Mean of group B (numeric only)")
    std_a: Optional[float] = Field(None, description="Std dev of group A (numeric only)")
    std_b: Optional[float] = Field(None, description="Std dev of group B (numeric only)")


class RiskAssessmentRequest(BaseModel):
    lift: float = Field(..., description="Observed lift percentage")
    p_value: float = Field(..., ge=0, le=1, description="P-value from test")
    effect_size: float = Field(..., description="Effect size (Cohen's h or d)")
    ci_lower: float = Field(..., description="Lower bound of 95% CI")
    ci_upper: float = Field(..., description="Upper bound of 95% CI")
    bayesian_prob: float = Field(..., ge=0, le=1, description="Bayesian P(B > A)")
    sample_size_a: int = Field(..., gt=0, description="Sample size of group A")
    sample_size_b: int = Field(..., gt=0, description="Sample size of group B")


@app.get("/")
def root():
    return {"message": "A/B Testing Experiment Analyzer API"}


@app.post("/api/analyze")
async def analyze_ab_test(
    file: UploadFile = File(...),
    group_column: str = Form(...),
    metric_column: str = Form(...),
    metric_type: str = Form("binary"),
    group_a_name: str = Form(None),  # User-selected control group
    group_b_name: str = Form(None),   # User-selected treatment group
    enable_cuped: str = Form("false"),  # CUPED adjustment
    covariate_column: str = Form(None)  # Covariate for CUPED
):
    """
    Analyze A/B test data from uploaded CSV file
    """
    try:
        # Read CSV file
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        # Clean column names (strip whitespace)
        df.columns = df.columns.str.strip()
        group_column = group_column.strip()
        metric_column = metric_column.strip()
        
        # Validate columns exist
        if group_column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Group column '{group_column}' not found in CSV. Available columns: {', '.join(df.columns.tolist())}")
        if metric_column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Metric column '{metric_column}' not found in CSV. Available columns: {', '.join(df.columns.tolist())}")
        
        # Clean group column values (strip whitespace)
        df[group_column] = df[group_column].astype(str).str.strip()
        
        # Get unique groups
        unique_groups = df[group_column].unique().tolist()
        unique_groups = [g for g in unique_groups if str(g).strip() != '' and str(g).lower() != 'nan']
        
        # Check if we have exactly 2 groups
        if len(unique_groups) < 2:
            raise HTTPException(
                status_code=400,
                detail=f"CSV must have exactly 2 groups. Found {len(unique_groups)} group(s): {unique_groups}"
            )
        elif len(unique_groups) > 2:
            raise HTTPException(
                status_code=400,
                detail=f"CSV must have exactly 2 groups. Found {len(unique_groups)} groups: {unique_groups}. Please filter your data to only include 2 groups."
            )
        
        # User MUST explicitly select which group is A (control) and which is B (treatment)
        # Never auto-assign or sort - user selection is mandatory
        if not group_a_name or not group_b_name:
            raise HTTPException(
                status_code=400,
                detail=f"Please select which group is Control (A) and which is Treatment (B). Available groups: {unique_groups}"
            )
        
        # Validate user selections exist in data
        group_a_name = group_a_name.strip()
        group_b_name = group_b_name.strip()
        
        if group_a_name not in unique_groups:
            raise HTTPException(
                status_code=400,
                detail=f"Selected control group '{group_a_name}' not found in CSV. Available groups: {unique_groups}"
            )
        if group_b_name not in unique_groups:
            raise HTTPException(
                status_code=400,
                detail=f"Selected treatment group '{group_b_name}' not found in CSV. Available groups: {unique_groups}"
            )
        if group_a_name == group_b_name:
            raise HTTPException(
                status_code=400,
                detail="Control group and treatment group must be different"
            )
        
        # Split into groups using user-defined assignment
        group_a = df[df[group_column] == group_a_name]
        group_b = df[df[group_column] == group_b_name]
        
        if len(group_a) == 0 or len(group_b) == 0:
            raise HTTPException(
                status_code=400, 
                detail=f"Both groups must have data. Control group '{group_a_name}' has {len(group_a)} rows, Treatment group '{group_b_name}' has {len(group_b)} rows."
            )
        
        # Perform analysis based on metric type
        if metric_type == "binary":
            results = analyze_binary_metric(group_a, group_b, metric_column)
            # Calculate original SE for CUPED comparison
            p_pool = ((group_a[metric_column].sum() + group_b[metric_column].sum()) / 
                     (len(group_a) + len(group_b))) if (len(group_a) + len(group_b)) > 0 else 0
            original_se = np.sqrt(p_pool * (1 - p_pool) * (1/len(group_a) + 1/len(group_b))) if p_pool > 0 and p_pool < 1 else 0
        else:
            results = analyze_numeric_metric(group_a, group_b, metric_column)
            se_a = results['group_a']['std'] / np.sqrt(results['group_a']['size'])
            se_b = results['group_b']['std'] / np.sqrt(results['group_b']['size'])
            original_se = np.sqrt(se_a**2 + se_b**2)
        
        # Apply CUPED if requested
        use_cuped = enable_cuped.lower() == "true" if enable_cuped else False
        if use_cuped and covariate_column:
            cuped_results = apply_cuped_adjustment(
                group_a, group_b, metric_column, covariate_column, metric_type, original_se
            )
            results['cuped'] = cuped_results
        
        # Add Bayesian analysis
        bayesian_results = perform_bayesian_analysis(group_a, group_b, metric_column, metric_type)
        results.update(bayesian_results)
        
        # Add sample data for visualization
        results['sample_data'] = {
            'group_a': group_a[metric_column].tolist()[:100],  # Limit for response size
            'group_b': group_b[metric_column].tolist()[:100],
            'group_a_label': group_a_name,
            'group_b_label': group_b_name
        }
        
        # Store actual group names in results
        results['group_a_name'] = group_a_name
        results['group_b_name'] = group_b_name
        
        return JSONResponse(content=results)
    
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="CSV file is empty")
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error in analyze_ab_test: {str(e)}")
        print(f"Traceback: {error_trace}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


def apply_cuped_adjustment(
    group_a: pd.DataFrame,
    group_b: pd.DataFrame,
    metric_column: str,
    covariate_column: str,
    metric_type: str,
    original_se: float = None
) -> dict:
    """
    Apply CUPED (Controlled-experiment Using Pre-Experiment Data) variance reduction
    Θ = Cov(Y, X) / Var(X)
    Y_adj = Y - Θ(X - mean(X))
    """
    # Combine groups for overall statistics
    all_data = pd.concat([group_a, group_b])
    
    # Calculate overall mean of covariate
    covariate_mean = all_data[covariate_column].mean()
    
    # Calculate Θ using overall data (more stable)
    cov_overall = all_data[metric_column].cov(all_data[covariate_column])
    var_overall = all_data[covariate_column].var()
    theta = cov_overall / var_overall if var_overall > 0 else 0
    
    # Apply adjustment using the same theta for both groups
    group_a_adjusted = group_a[metric_column] - theta * (group_a[covariate_column] - covariate_mean)
    group_b_adjusted = group_b[metric_column] - theta * (group_b[covariate_column] - covariate_mean)
    
    # Recalculate metrics with adjusted values
    if metric_type == "binary":
        # For binary, recalculate conversion rates
        n_a = len(group_a_adjusted)
        n_b = len(group_b_adjusted)
        # Adjusted values might not be binary, so we use mean as rate
        p_a_adj = group_a_adjusted.mean()
        p_b_adj = group_b_adjusted.mean()
        
        # Recalculate lift
        lift_adj = ((p_b_adj - p_a_adj) / p_a_adj * 100) if p_a_adj > 0 else 0
        
        # Recalculate z-test with adjusted values
        # Use pooled variance for adjusted values
        pooled_var = ((n_a - 1) * group_a_adjusted.var() + (n_b - 1) * group_b_adjusted.var()) / (n_a + n_b - 2)
        se_adj = np.sqrt(pooled_var * (1/n_a + 1/n_b)) if pooled_var > 0 else 0
        diff_adj = p_b_adj - p_a_adj
        z_stat_adj = (diff_adj / se_adj) if se_adj > 0 else 0
        p_value_adj = 2 * stats.norm.sf(abs(z_stat_adj)) if se_adj > 0 else 1.0
        
        # Confidence interval
        z_critical = 1.96
        ci_lower_adj = diff_adj - z_critical * se_adj
        ci_upper_adj = diff_adj + z_critical * se_adj
        
        # Calculate variance reduction
        variance_reduction = (1 - se_adj / original_se * 100) if original_se and original_se > 0 else 0
    else:
        # For numeric metrics
        mean_a_adj = group_a_adjusted.mean()
        mean_b_adj = group_b_adjusted.mean()
        std_a_adj = group_a_adjusted.std()
        std_b_adj = group_b_adjusted.std()
        
        lift_adj = ((mean_b_adj - mean_a_adj) / mean_a_adj * 100) if mean_a_adj != 0 else 0
        
        # T-test with adjusted values
        se_a_adj = std_a_adj / np.sqrt(len(group_a_adjusted))
        se_b_adj = std_b_adj / np.sqrt(len(group_b_adjusted))
        se_diff_adj = np.sqrt(se_a_adj**2 + se_b_adj**2)
        diff_adj = mean_b_adj - mean_a_adj
        t_stat_adj = (diff_adj / se_diff_adj) if se_diff_adj > 0 else 0
        p_value_adj = 2 * (1 - stats.t.cdf(abs(t_stat_adj), df=len(group_a_adjusted) + len(group_b_adjusted) - 2)) if se_diff_adj > 0 else 1.0
        
        t_critical = stats.t.ppf(0.975, df=len(group_a_adjusted) + len(group_b_adjusted) - 2)
        ci_lower_adj = diff_adj - t_critical * se_diff_adj
        ci_upper_adj = diff_adj + t_critical * se_diff_adj
        
        # Calculate variance reduction
        variance_reduction = (1 - se_diff_adj / original_se * 100) if original_se and original_se > 0 else 0
    
    # Calculate diagnostics
    cov_a = group_a[metric_column].cov(group_a[covariate_column])
    cov_b = group_b[metric_column].cov(group_b[covariate_column])
    var_a = group_a[covariate_column].var()
    var_b = group_b[covariate_column].var()
    
    # Original SE for comparison
    if metric_type == "binary":
        n_a = len(group_a)
        n_b = len(group_b)
        p_a_orig = group_a[metric_column].mean()
        p_b_orig = group_b[metric_column].mean()
        p_pool_orig = (p_a_orig * n_a + p_b_orig * n_b) / (n_a + n_b) if (n_a + n_b) > 0 else 0
        se_orig = np.sqrt(p_pool_orig * (1 - p_pool_orig) * (1/n_a + 1/n_b)) if p_pool_orig > 0 and p_pool_orig < 1 else 0
        se_change = se_orig - se_adj if se_orig > 0 else 0
        diff_orig = p_b_orig - p_a_orig
        z_stat_orig = (diff_orig / se_orig) if se_orig > 0 else 0
        significance_before = 2 * stats.norm.sf(abs(z_stat_orig)) if se_orig > 0 else 1.0
        significance_after = p_value_adj
    else:
        se_orig = original_se if original_se else se_diff_adj
        se_change = se_orig - se_diff_adj
        # Approximate significance before (would need original test)
        significance_before = 1.0  # Placeholder
        significance_after = p_value_adj
    
    diagnostics = {
        'covariance_a': float(cov_a) if not np.isnan(cov_a) else 0.0,
        'covariance_b': float(cov_b) if not np.isnan(cov_b) else 0.0,
        'variance_covariate_a': float(var_a) if not np.isnan(var_a) else 0.0,
        'variance_covariate_b': float(var_b) if not np.isnan(var_b) else 0.0,
        'theta': float(theta),
        'variance_reduction_percent': float(variance_reduction),
        'adjusted_mean_difference': float(diff_adj),
        'se_change': float(se_change),
        'se_before': float(se_orig),
        'se_after': float(se_adj if metric_type == "binary" else se_diff_adj),
        'significance_before': float(significance_before),
        'significance_after': float(significance_after),
        'interpretation': f"CUPED {'helped reduce variance by' if variance_reduction > 0 else 'did not improve stability for'} this metric. Variance reduction: {abs(variance_reduction):.2f}%"
    }
    
    return {
        'enabled': True,
        'covariate_column': covariate_column,
        'theta': float(theta),
        'lift': float(lift_adj),
        'p_value': float(p_value_adj),
        'confidence_interval': {
            'lower': float(ci_lower_adj),
            'upper': float(ci_upper_adj),
            'level': 0.95
        },
        'difference': float(diff_adj),
        'variance_reduction_percent': float(variance_reduction),
        'diagnostics': diagnostics
    }


def analyze_binary_metric(group_a: pd.DataFrame, group_b: pd.DataFrame, metric_column: str) -> dict:
    """
    Analyze binary/conversion metric using two-proportion z-test
    Group A = Control, Group B = Treatment
    """
    # Calculate sample sizes
    n_a = len(group_a)
    n_b = len(group_b)
    
    # Calculate conversions
    conversions_a = group_a[metric_column].sum()
    conversions_b = group_b[metric_column].sum()
    
    # Calculate conversion rates (conversions / sample_size)
    p_a = conversions_a / n_a if n_a > 0 else 0
    p_b = conversions_b / n_b if n_b > 0 else 0
    
    # Calculate lift: (conversion_B - conversion_A) / conversion_A
    lift = ((p_b - p_a) / p_a * 100) if p_a > 0 else 0
    
    # Two-Proportion Z-Test
    # p_pool = (conversions_A + conversions_B) / (n_A + n_B)
    p_pool = (conversions_a + conversions_b) / (n_a + n_b) if (n_a + n_b) > 0 else 0
    
    # SE = sqrt(p_pool * (1 - p_pool) * (1/n_A + 1/n_B))
    se = np.sqrt(p_pool * (1 - p_pool) * (1/n_a + 1/n_b)) if p_pool > 0 and p_pool < 1 else 0
    
    # z = (conversion_B - conversion_A) / SE
    diff = p_b - p_a
    z_stat = (diff / se) if se > 0 else 0
    
    # Calculate p-value from z-statistic
    # Use survival function (sf) for better numerical stability with very small p-values
    if se > 0:
        p_value = 2 * stats.norm.sf(abs(z_stat))  # sf = 1 - cdf, more stable for large z
    else:
        p_value = 1.0
    
    # 95% Confidence Interval
    # CI_low = diff - 1.96 * SE
    # CI_high = diff + 1.96 * SE
    z_critical = 1.96  # For 95% CI
    ci_lower = diff - z_critical * se
    ci_upper = diff + z_critical * se
    
    # Effect Size - Cohen's h
    # h = 2 * (arcsin(sqrt(pB)) - arcsin(sqrt(pA)))
    h = 2 * (np.arcsin(np.sqrt(p_b)) - np.arcsin(np.sqrt(p_a)))
    
    return {
        'metric_type': 'binary',
        'group_a': {
            'size': int(n_a),
            'conversions': int(conversions_a),
            'rate': float(p_a),
            'mean': float(p_a)
        },
        'group_b': {
            'size': int(n_b),
            'conversions': int(conversions_b),
            'rate': float(p_b),
            'mean': float(p_b)
        },
        'lift': float(lift),
        'p_value': float(p_value),
        'test_statistic': float(z_stat),
        'test_type': 'two-proportion z-test',
        'effect_size': float(h),
        'effect_size_type': "Cohen's h",
        'confidence_interval': {
            'lower': float(ci_lower),
            'upper': float(ci_upper),
            'level': 0.95
        },
        'difference': float(diff)
    }


def analyze_numeric_metric(group_a: pd.DataFrame, group_b: pd.DataFrame, metric_column: str) -> dict:
    """
    Analyze numeric metric using two-sample t-test
    """
    data_a = group_a[metric_column].dropna()
    data_b = group_b[metric_column].dropna()
    
    n_a = len(data_a)
    n_b = len(data_b)
    mean_a = data_a.mean()
    mean_b = data_b.mean()
    std_a = data_a.std()
    std_b = data_b.std()
    
    # Calculate lift
    lift = ((mean_b - mean_a) / mean_a * 100) if mean_a != 0 else 0
    
    # Check normality (Shapiro-Wilk test on smaller sample)
    if n_a < 3 or n_b < 3:
        use_mann_whitney = True
    else:
        _, p_norm_a = stats.shapiro(data_a.sample(min(5000, n_a)))
        _, p_norm_b = stats.shapiro(data_b.sample(min(5000, n_b)))
        use_mann_whitney = p_norm_a < 0.05 or p_norm_b < 0.05
    
    # Perform appropriate test
    if use_mann_whitney:
        test_statistic, p_value = stats.mannwhitneyu(data_a, data_b, alternative='two-sided')
        test_type = 'Mann-Whitney U test'
    else:
        # Two-sample t-test
        test_statistic, p_value = stats.ttest_ind(data_a, data_b, equal_var=False)
        test_type = 'Welch\'s t-test'
    
    # Calculate 95% confidence interval
    se_a = std_a / np.sqrt(n_a)
    se_b = std_b / np.sqrt(n_b)
    se_diff = np.sqrt(se_a**2 + se_b**2)
    t_critical = stats.t.ppf(0.975, df=min(n_a-1, n_b-1))
    diff = mean_b - mean_a
    ci_lower = diff - t_critical * se_diff
    ci_upper = diff + t_critical * se_diff
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt(((n_a - 1) * std_a**2 + (n_b - 1) * std_b**2) / (n_a + n_b - 2))
    cohens_d = (mean_b - mean_a) / pooled_std if pooled_std > 0 else 0
    
    return {
        'metric_type': 'numeric',
        'group_a': {
            'size': int(n_a),
            'mean': float(mean_a),
            'std': float(std_a),
            'median': float(data_a.median())
        },
        'group_b': {
            'size': int(n_b),
            'mean': float(mean_b),
            'std': float(std_b),
            'median': float(data_b.median())
        },
        'lift': float(lift),
        'p_value': float(p_value),
        'test_statistic': float(test_statistic),
        'test_type': test_type,
        'effect_size': float(cohens_d),
        'effect_size_type': "Cohen's d",
        'confidence_interval': {
            'lower': float(ci_lower),
            'upper': float(ci_upper),
            'level': 0.95
        },
        'difference': float(diff)
    }


def perform_bayesian_analysis(
    group_a: pd.DataFrame, 
    group_b: pd.DataFrame, 
    metric_column: str,
    metric_type: str
) -> dict:
    """
    Perform Bayesian A/B testing analysis
    For binary: Beta-Binomial model
    For numeric: Normal model with conjugate priors
    """
    if metric_type == "binary":
        # Beta-Binomial model
        conversions_a = group_a[metric_column].sum()
        conversions_b = group_b[metric_column].sum()
        n_a = len(group_a)
        n_b = len(group_b)
        
        # Use uniform priors (Beta(1,1))
        alpha_a = 1 + conversions_a
        beta_a = 1 + (n_a - conversions_a)
        alpha_b = 1 + conversions_b
        beta_b = 1 + (n_b - conversions_b)
        
        # Sample from posterior distributions
        np.random.seed(42)
        samples_a = np.random.beta(alpha_a, beta_a, 10000)
        samples_b = np.random.beta(alpha_b, beta_b, 10000)
        
        # Calculate P(B > A)
        prob_b_better = np.mean(samples_b > samples_a)
        
        return {
            'bayesian': {
                'prob_b_better': float(prob_b_better),
                'posterior_a': {
                    'alpha': float(alpha_a),
                    'beta': float(beta_a),
                    'mean': float(alpha_a / (alpha_a + beta_a))
                },
                'posterior_b': {
                    'alpha': float(alpha_b),
                    'beta': float(beta_b),
                    'mean': float(alpha_b / (alpha_b + beta_b))
                },
                'samples_a': samples_a.tolist()[:1000],  # Limit for response
                'samples_b': samples_b.tolist()[:1000]
            }
        }
    else:
        # Normal model with conjugate priors
        data_a = group_a[metric_column].dropna()
        data_b = group_b[metric_column].dropna()
        
        mean_a = data_a.mean()
        mean_b = data_b.mean()
        std_a = data_a.std()
        std_b = data_b.std()
        n_a = len(data_a)
        n_b = len(data_b)
        
        # Use non-informative priors and sample from posterior
        np.random.seed(42)
        # Approximate posterior samples
        samples_a = np.random.normal(mean_a, std_a / np.sqrt(n_a), 10000)
        samples_b = np.random.normal(mean_b, std_b / np.sqrt(n_b), 10000)
        
        prob_b_better = np.mean(samples_b > samples_a)
        
        return {
            'bayesian': {
                'prob_b_better': float(prob_b_better),
                'posterior_a': {
                    'mean': float(mean_a),
                    'std': float(std_a / np.sqrt(n_a))
                },
                'posterior_b': {
                    'mean': float(mean_b),
                    'std': float(std_b / np.sqrt(n_b))
                },
                'samples_a': samples_a.tolist()[:1000],
                'samples_b': samples_b.tolist()[:1000]
            }
        }


@app.post("/api/generate-interpretation")
async def generate_interpretation(request: dict):
    """
    Generate AI-powered interpretation of A/B test results
    """
    try:
        # Extract metrics
        lift = request.get('lift', 0)
        p_value = request.get('p_value', 1)
        effect_size = request.get('effect_size', 0)
        ci_lower = request.get('ci_lower', 0)
        ci_upper = request.get('ci_upper', 0)
        prob_b_better = request.get('prob_b_better', 0.5)
        metric_type = request.get('metric_type', 'binary')
        test_type = request.get('test_type', '')
        
        # Generate interpretation
        interpretation = generate_ai_interpretation(
            lift, p_value, effect_size, ci_lower, ci_upper, 
            prob_b_better, metric_type, test_type
        )
        
        return JSONResponse(content={'interpretation': interpretation})
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating interpretation: {str(e)}")


def format_p_value(p_val):
    """Format p-value for display (preserve precision for very small values)"""
    if p_val == 0.0 or p_val < 1e-100:
        return "< 1e-100"
    elif p_val < 1e-10:
        exp_str = f"{p_val:.3e}"
        if 'e-' in exp_str:
            parts = exp_str.split('e-')
            if len(parts) == 2:
                exp_num = int(parts[1])
                return f"{parts[0]}e-{exp_num}"
        elif 'e+' in exp_str:
            parts = exp_str.split('e+')
            if len(parts) == 2:
                exp_num = int(parts[1])
                return f"{parts[0]}e+{exp_num}"
        return exp_str
    elif p_val < 0.0001:
        exp_str = f"{p_val:.3e}"
        if 'e-' in exp_str:
            parts = exp_str.split('e-')
            if len(parts) == 2:
                exp_num = int(parts[1])
                return f"{parts[0]}e-{exp_num}"
        return exp_str
    else:
        return f"{p_val:.4f}"


def generate_template_interpretation(
    lift: float, p_value: float, effect_size: float,
    ci_lower: float, ci_upper: float, prob_b_better: float,
    metric_type: str, test_type: str
) -> str:
    """Fallback template-based interpretation"""
    is_significant = p_value < 0.05
    p_value_str = format_p_value(p_value)
    
    if lift > 0 and p_value < 0.05:
        recommendation = "Choose B (Treatment)"
        recommendation_reason = f"Treatment group shows a statistically significant improvement of {lift:.2f}% (p = {p_value_str})"
    elif lift < 0 and p_value < 0.05:
        recommendation = "Choose A (Control)"
        recommendation_reason = f"Control group shows a statistically significant improvement of {abs(lift):.2f}% (p = {p_value_str})"
    elif p_value >= 0.05:
        recommendation = "Inconclusive – continue experiment"
        recommendation_reason = f"Results are not statistically significant (p = {p_value_str}). Consider running the experiment longer or increasing sample size."
    else:
        recommendation = "Inconclusive – continue experiment"
        recommendation_reason = "The difference between groups is minimal and not statistically significant."
    
    if metric_type == "binary":
        effect_interpretation = "small" if abs(effect_size) < 0.2 else "medium" if abs(effect_size) < 0.5 else "large"
    else:
        effect_interpretation = "small" if abs(effect_size) < 0.2 else "medium" if abs(effect_size) < 0.5 else "large" if abs(effect_size) < 0.8 else "very large"
    
    interpretation = f"""
## Results Summary

**Lift:** {lift:+.2f}%

**Statistical Significance:** {'Yes' if is_significant else 'No'} (p-value = {p_value_str})

**Effect Size:** {abs(effect_size):.3f} ({effect_interpretation} effect)

**95% Confidence Interval:** [{ci_lower:.4f}, {ci_upper:.4f}]

**Bayesian Probability (B > A):** {prob_b_better*100:.1f}%

## Interpretation

{recommendation_reason}. The {test_type} {'indicates' if is_significant else 'does not indicate'} a statistically significant difference between the two groups (p = {p_value_str}).

The effect size of {abs(effect_size):.3f} suggests a {effect_interpretation} practical effect. The 95% confidence interval [{ci_lower:.4f}, {ci_upper:.4f}] {'does not include zero' if (ci_lower > 0 and ci_upper > 0) or (ci_lower < 0 and ci_upper < 0) else 'includes zero'}, which {'supports' if is_significant else 'does not support'} the statistical conclusion.

From a Bayesian perspective, there is a {prob_b_better*100:.1f}% probability that Group B performs better than Group A.

## Recommendation

**{recommendation}**

This recommendation is based on both frequentist (p-value) and Bayesian statistical methods. {'The results are statistically significant' if is_significant else 'While not statistically significant, the Bayesian analysis provides additional insight'}.
"""
    return interpretation.strip()


def generate_ai_interpretation(
    lift: float, p_value: float, effect_size: float,
    ci_lower: float, ci_upper: float, prob_b_better: float,
    metric_type: str, test_type: str
) -> str:
    """
    Generate AI-powered interpretation of A/B test results using OpenAI
    Falls back to template-based interpretation if OpenAI is unavailable
    """
    # Try OpenAI first if available
    if openai_client:
        try:
            p_value_str = format_p_value(p_value)
            is_significant = p_value < 0.05
            
            # Effect size interpretation
            if metric_type == "binary":
                effect_interpretation = "small" if abs(effect_size) < 0.2 else "medium" if abs(effect_size) < 0.5 else "large"
            else:
                effect_interpretation = "small" if abs(effect_size) < 0.2 else "medium" if abs(effect_size) < 0.5 else "large" if abs(effect_size) < 0.8 else "very large"
            
            prompt = f"""You are a data scientist analyzing A/B test results. Provide a clear, professional interpretation of these results.

A/B Test Results:
- Lift: {lift:+.2f}%
- Statistical Significance: {'Yes' if is_significant else 'No'} (p-value = {p_value_str})
- Effect Size: {abs(effect_size):.3f} ({effect_interpretation} effect)
- 95% Confidence Interval: [{ci_lower:.4f}, {ci_upper:.4f}]
- Bayesian Probability (B > A): {prob_b_better*100:.1f}%
- Test Type: {test_type}
- Metric Type: {metric_type}

Please provide:
1. A Results Summary section with key metrics
2. An Interpretation section explaining what these results mean in practical terms
3. A Recommendation section with clear next steps

Format your response in Markdown with ## for main sections and ** for emphasis. Be concise but thorough."""

            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert data scientist specializing in A/B testing and statistical analysis. Provide clear, actionable insights."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=800
            )
            
            interpretation = response.choices[0].message.content.strip()
            return interpretation
            
        except Exception as e:
            # If OpenAI fails, fall back to template
            print(f"OpenAI API error: {str(e)}. Falling back to template interpretation.")
            return generate_template_interpretation(
                lift, p_value, effect_size, ci_lower, ci_upper, 
                prob_b_better, metric_type, test_type
            )
    else:
        # No OpenAI API key, use template
        return generate_template_interpretation(
            lift, p_value, effect_size, ci_lower, ci_upper, 
            prob_b_better, metric_type, test_type
        )


@app.post("/api/generate-interpretation-mode")
async def generate_interpretation_mode(request: dict):
    """
    Generate AI interpretation in different modes (executive, non-technical, slack/email, recommendations)
    Uses OpenAI if available, falls back to template-based responses
    """
    lift = request.get('lift', 0)
    p_value = request.get('p_value', 1.0)
    effect_size = request.get('effect_size', 0)
    ci_lower = request.get('ci_lower', 0)
    ci_upper = request.get('ci_upper', 0)
    prob_b_better = request.get('prob_b_better', 0.5)
    metric_type = request.get('metric_type', 'binary')
    test_type = request.get('test_type', 'two-proportion z-test')
    mode = request.get('mode', 'executive')
    group_a_name = request.get('group_a_name', 'Group A')
    group_b_name = request.get('group_b_name', 'Group B')
    
    is_significant = p_value < 0.05
    p_value_str = format_p_value(p_value)
    
    # Try OpenAI first if available
    if openai_client:
        try:
            mode_prompts = {
                'executive': f"""Provide a concise executive summary (3-4 bullet points) of these A/B test results:

- Control Group ({group_a_name}) vs Treatment Group ({group_b_name})
- Lift: {lift:+.2f}%
- Statistical Significance: {'Yes' if is_significant else 'No'} (p = {p_value_str})
- Effect Size: {abs(effect_size):.3f}
- Bayesian Probability (B > A): {prob_b_better*100:.1f}%

Format as bullet points with emojis. Include: key finding, confidence level, and recommendation.""",
                
                'non-technical': f"""Explain these A/B test results in simple, non-technical language:

- Control Group ({group_a_name}) vs Treatment Group ({group_b_name})
- Lift: {lift:+.2f}%
- Statistical Significance: {'Yes' if is_significant else 'No'} (p = {p_value_str})
- Effect Size: {abs(effect_size):.3f}
- Bayesian Probability (B > A): {prob_b_better*100:.1f}%

Use sections: "What This Means", "The Numbers", "What Should We Do?". Avoid jargon.""",
                
                'slack-email': f"""Create a concise Slack/Email summary of these A/B test results:

- Experiment: {group_a_name} (Control) vs {group_b_name} (Treatment)
- Lift: {lift:+.2f}%
- Statistical Significance: {'Yes' if is_significant else 'No'} (p = {p_value_str})
- Bayesian Confidence: {prob_b_better*100:.1f}%

Include: Key Findings, Recommendation, Next Steps. Use emojis and clear formatting.""",
                
                'recommendations': f"""Based on these A/B test results, provide actionable recommendations:

- Lift: {lift:+.2f}%
- Statistical Significance: {'Yes' if is_significant else 'No'} (p = {p_value_str})
- Effect Size: {abs(effect_size):.3f}
- Bayesian Probability (B > A): {prob_b_better*100:.1f}%

Provide 4-6 specific recommendations with emojis. Consider: deployment, sample size, next experiments, practical significance."""
            }
            
            if mode in mode_prompts:
                response = openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are an expert data scientist. Provide clear, actionable A/B test insights."},
                        {"role": "user", "content": mode_prompts[mode]}
                    ],
                    temperature=0.7,
                    max_tokens=500
                )
                
                interpretation = response.choices[0].message.content.strip()
                
                # For slack-email mode, also generate subject
                if mode == 'slack-email':
                    return JSONResponse(content={
                        "interpretation": interpretation,
                        "subject": f"A/B Test Results: {group_a_name} vs {group_b_name}"
                    })
                
                return JSONResponse(content={"interpretation": interpretation})
            
        except Exception as e:
            print(f"OpenAI API error: {str(e)}. Falling back to template interpretation.")
            # Fall through to template-based responses
    
    # Template-based fallback
    if mode == 'executive':
        bullets = []
        if lift > 0 and is_significant:
            bullets.append(f"✅ Treatment ({group_b_name}) shows a {lift:.2f}% improvement over control ({group_a_name})")
        elif lift < 0 and is_significant:
            bullets.append(f"❌ Control ({group_a_name}) performs {abs(lift):.2f}% better than treatment ({group_b_name})")
        else:
            bullets.append(f"⚠️ No statistically significant difference between groups (p = {p_value_str})")
        
        bullets.append(f"📊 Confidence: {prob_b_better*100:.1f}% probability that {group_b_name} > {group_a_name}")
        
        if is_significant:
            bullets.append(f"🎯 Recommendation: {'Deploy treatment' if lift > 0 else 'Keep control'}")
        else:
            bullets.append("🎯 Recommendation: Continue experiment or increase sample size")
        
        return JSONResponse(content={"interpretation": "\n".join(bullets)})
    
    elif mode == 'non-technical':
        explanation = f"""
## What This Means

{'🎉 Great news! ' if lift > 0 and is_significant else '⚠️ ' if not is_significant else '📉 '}
{'The new version (' + group_b_name + ') is performing better' if lift > 0 and is_significant 
 else 'The original version (' + group_a_name + ') is performing better' if lift < 0 and is_significant
 else 'We cannot confidently say which version is better yet'}

## The Numbers

• **Improvement**: {lift:+.2f}% {'(statistically significant)' if is_significant else '(not statistically significant)'}
• **Confidence Level**: {prob_b_better*100:.1f}% chance the treatment is better
• **Reliability**: {'High' if is_significant and prob_b_better > 0.95 else 'Medium' if is_significant else 'Low - need more data'}

## What Should We Do?

{'✅ Deploy the new version - it is clearly better' if lift > 0 and is_significant
 else '✅ Keep the original version' if lift < 0 and is_significant
 else '⏳ Run the test longer or get more users to participate'}
        """.strip()
        return JSONResponse(content={"interpretation": explanation})
    
    elif mode == 'slack-email':
        subject = f"A/B Test Results: {group_a_name} vs {group_b_name}"
        
        if lift > 0 and is_significant:
            recommendation = f"✅ Deploy {group_b_name} - statistically significant improvement"
        elif lift < 0 and is_significant:
            recommendation = f"✅ Keep {group_a_name} - control performs better"
        else:
            recommendation = "⏳ Continue experiment - results are not yet conclusive"
        
        if lift > 0 and is_significant:
            next_steps = f"• Plan rollout of {group_b_name}"
        elif lift < 0 and is_significant:
            next_steps = f"• Maintain current version ({group_a_name})"
        else:
            next_steps = "• Extend experiment duration" + "\n" + "• Consider increasing sample size"
        
        message = f"""
📊 A/B Test Results Summary

**Experiment**: {group_a_name} (Control) vs {group_b_name} (Treatment)

**Key Findings:**
• Lift: {lift:+.2f}%
• Statistical Significance: {'✅ Yes' if is_significant else '❌ No'} (p = {p_value_str})
• Bayesian Confidence: {prob_b_better*100:.1f}% probability treatment is better

**Recommendation:**
{recommendation}

**Next Steps:**
{next_steps}
        """.strip()
        return JSONResponse(content={"interpretation": message, "subject": subject})
    
    elif mode == 'recommendations':
        recommendations = []
        
        if not is_significant:
            recommendations.append("🔬 **Run Longer**: Extend experiment duration to collect more data")
            recommendations.append("👥 **Increase Sample Size**: More participants will improve statistical power")
        else:
            if lift > 0:
                recommendations.append("✅ **Deploy Treatment**: Statistically significant improvement detected")
                recommendations.append("📈 **Expected Impact**: " + f"{lift:.2f}% improvement in {metric_type} metric")
            else:
                recommendations.append("✅ **Keep Control**: Current version performs better")
        
        if abs(effect_size) < 0.2:
            recommendations.append("📊 **Effect Size**: Small effect detected - consider practical significance")
        
        if prob_b_better < 0.7 or prob_b_better > 0.3:
            recommendations.append("🎲 **Uncertainty**: Bayesian probability suggests moderate confidence - consider more data")
        
        recommendations.append("🔄 **Next Experiment**: Consider testing new variants or hypotheses")
        
        return JSONResponse(content={"interpretation": "\n\n".join(recommendations)})
    
    return JSONResponse(content={"interpretation": "Invalid mode"})


# ============================================================================
# NEW ADVANCED ANALYTICS ENDPOINTS
# ============================================================================

@app.post("/api/power-analysis")
async def power_analysis(request: PowerAnalysisRequest):
    """
    Calculate required sample size for a given power and effect size
    Uses Cohen's h for binary metrics
    """
    try:
        baseline_rate = request.baseline_rate
        expected_lift = request.expected_lift
        
        # Convert lift to absolute rate if needed (handle both percentage and decimal)
        if expected_lift > 1:
            # Assume percentage (e.g., 10 means 10%)
            expected_lift = expected_lift / 100
        
        expected_rate = baseline_rate * (1 + expected_lift)
        
        # Ensure valid rates
        if expected_rate <= 0 or expected_rate >= 1:
            raise HTTPException(status_code=400, detail="Invalid rate combination. Expected rate must be between 0 and 1.")
        
        # Calculate Cohen's h
        # h = 2 * (arcsin(sqrt(p2)) - arcsin(sqrt(p1)))
        h = 2 * (np.arcsin(np.sqrt(expected_rate)) - np.arcsin(np.sqrt(baseline_rate)))
        
        # Z-scores
        z_alpha = stats.norm.ppf(1 - request.alpha / 2)  # Two-tailed
        z_power = stats.norm.ppf(request.power)
        
        # Sample size calculation: n = 2 * (Z_{1−α/2} + Z_{power})^2 / h^2
        if abs(h) < 1e-10:
            raise HTTPException(status_code=400, detail="Effect size too small. Increase expected lift.")
        
        n_per_group = 2 * ((z_alpha + z_power) ** 2) / (h ** 2)
        n_per_group = int(np.ceil(n_per_group))
        
        interpretation = f"""
        To detect a {expected_lift*100:.1f}% lift from a baseline rate of {baseline_rate*100:.2f}% 
        with {request.power*100:.0f}% power and {request.alpha*100:.0f}% significance level, 
        you need {n_per_group:,} participants per group (total: {n_per_group*2:,}).
        The effect size (Cohen's h) is {abs(h):.3f}, which is {'small' if abs(h) < 0.2 else 'medium' if abs(h) < 0.5 else 'large'}.
        """
        
        return JSONResponse(content={
            "sample_size_per_group": n_per_group,
            "total_sample_size": n_per_group * 2,
            "effect_size": float(abs(h)),
            "effect_size_type": "Cohen's h",
            "interpretation": interpretation.strip()
        })
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in power analysis: {str(e)}")


@app.post("/api/mde-analysis")
async def mde_analysis(request: MDEAnalysisRequest):
    """
    Calculate Minimum Detectable Effect (MDE) for given sample size and power
    """
    try:
        baseline_rate = request.baseline_rate
        n = request.sample_size_per_group
        
        # Z-scores
        z_alpha = stats.norm.ppf(1 - request.alpha / 2)
        z_power = stats.norm.ppf(request.power)
        
        # Rearrange power equation to solve for h
        # n = 2 * (Z_{1−α/2} + Z_{power})^2 / h^2
        # h^2 = 2 * (Z_{1−α/2} + Z_{power})^2 / n
        h_squared = 2 * ((z_alpha + z_power) ** 2) / n
        h = np.sqrt(h_squared)
        
        # Convert h back to rate difference
        # h = 2 * (arcsin(sqrt(p2)) - arcsin(sqrt(p1)))
        # arcsin(sqrt(p2)) = h/2 + arcsin(sqrt(p1))
        # sqrt(p2) = sin(h/2 + arcsin(sqrt(p1)))
        # p2 = sin^2(h/2 + arcsin(sqrt(p1)))
        asin_p1 = np.arcsin(np.sqrt(baseline_rate))
        p2 = np.sin(h / 2 + asin_p1) ** 2
        
        # Ensure p2 is valid
        p2 = max(0.0, min(1.0, p2))
        
        # Calculate MDE as lift percentage
        mde_lift = ((p2 - baseline_rate) / baseline_rate * 100) if baseline_rate > 0 else 0
        
        interpretation = f"""
        With {n:,} participants per group, {request.power*100:.0f}% power, and {request.alpha*100:.0f}% significance level,
        you can detect a minimum lift of {mde_lift:.2f}% from a baseline rate of {baseline_rate*100:.2f}%.
        This corresponds to an effect size (Cohen's h) of {h:.3f}, which is {'small' if h < 0.2 else 'medium' if h < 0.5 else 'large'}.
        """
        
        return JSONResponse(content={
            "mde": float(mde_lift),
            "mde_absolute": float(p2 - baseline_rate),
            "effect_size_h": float(h),
            "interpretation": interpretation.strip()
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in MDE analysis: {str(e)}")


@app.post("/api/sequential-testing")
async def sequential_testing(
    file: UploadFile = File(...),
    timestamp_column: str = Form(...),
    metric_column: str = Form(...),
    group_column: str = Form(...),
    group_a_name: str = Form(...),
    group_b_name: str = Form(...),
    metric_type: str = Form("binary")
):
    """
    Monitor p-value over time to detect early stopping and p-hacking risks
    """
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        df.columns = df.columns.str.strip()
        
        # Validate columns
        for col in [timestamp_column, metric_column, group_column]:
            if col not in df.columns:
                raise HTTPException(status_code=400, detail=f"Column '{col}' not found in CSV")
        
        # Parse timestamps
        df[timestamp_column] = pd.to_datetime(df[timestamp_column], errors='coerce')
        df = df.dropna(subset=[timestamp_column]).sort_values(timestamp_column)
        
        # Filter groups
        df[group_column] = df[group_column].astype(str).str.strip()
        group_a = df[df[group_column] == group_a_name]
        group_b = df[df[group_column] == group_b_name]
        
        if len(group_a) == 0 or len(group_b) == 0:
            raise HTTPException(status_code=400, detail="One or both groups are empty")
        
        # Get unique timestamps (daily or hourly aggregation)
        timestamps = sorted(df[timestamp_column].unique())
        
        p_values_over_time = []
        cumulative_conversions_a = []
        cumulative_conversions_b = []
        cumulative_means_a = []
        cumulative_means_b = []
        significance_flags = []
        
        for ts in timestamps:
            # Cumulative data up to this timestamp
            group_a_cum = group_a[group_a[timestamp_column] <= ts]
            group_b_cum = group_b[group_b[timestamp_column] <= ts]
            
            if len(group_a_cum) < 10 or len(group_b_cum) < 10:
                # Too few data points
                p_values_over_time.append(1.0)
                significance_flags.append(False)
                if metric_type == "binary":
                    cumulative_conversions_a.append(0)
                    cumulative_conversions_b.append(0)
                else:
                    cumulative_means_a.append(0.0)
                    cumulative_means_b.append(0.0)
                continue
            
            # Calculate p-value
            if metric_type == "binary":
                conv_a = group_a_cum[metric_column].sum()
                conv_b = group_b_cum[metric_column].sum()
                n_a = len(group_a_cum)
                n_b = len(group_b_cum)
                
                cumulative_conversions_a.append(int(conv_a))
                cumulative_conversions_b.append(int(conv_b))
                
                p_a = conv_a / n_a if n_a > 0 else 0
                p_b = conv_b / n_b if n_b > 0 else 0
                p_pool = (conv_a + conv_b) / (n_a + n_b) if (n_a + n_b) > 0 else 0
                se = np.sqrt(p_pool * (1 - p_pool) * (1/n_a + 1/n_b)) if p_pool > 0 and p_pool < 1 else 1.0
                z_stat = (p_b - p_a) / se if se > 0 else 0
                p_value = 2 * stats.norm.sf(abs(z_stat)) if se > 0 else 1.0
            else:
                data_a = group_a_cum[metric_column].dropna()
                data_b = group_b_cum[metric_column].dropna()
                
                mean_a = data_a.mean()
                mean_b = data_b.mean()
                cumulative_means_a.append(float(mean_a))
                cumulative_means_b.append(float(mean_b))
                
                _, p_value = stats.ttest_ind(data_a, data_b, equal_var=False)
            
            p_values_over_time.append(float(p_value))
            significance_flags.append(p_value < 0.05)
        
        # Detect early stopping and p-hacking risks
        first_significant_idx = next((i for i, sig in enumerate(significance_flags) if sig), None)
        total_checks = len(significance_flags)
        significant_checks = sum(significance_flags)
        
        # Risk assessment
        if first_significant_idx is None:
            risk_level = "Safe"
            interpretation = "No significant results detected. Continue experiment."
        elif first_significant_idx < total_checks * 0.1:
            risk_level = "High False-Positive Risk"
            interpretation = f"Significant result appeared very early (at {first_significant_idx/total_checks*100:.1f}% of experiment). High risk of false positive. Consider extending experiment."
        elif significant_checks / total_checks > 0.5:
            risk_level = "Borderline"
            interpretation = f"Significant results appeared in {significant_checks/total_checks*100:.1f}% of checks. Moderate risk. Proceed with caution."
        else:
            risk_level = "Safe"
            interpretation = "Significant results appeared after sufficient data collection. Low risk of false positive."
        
        return JSONResponse(content={
            "timestamps": [ts.isoformat() for ts in timestamps],
            "cumulative_conversions_a": cumulative_conversions_a if metric_type == "binary" else None,
            "cumulative_conversions_b": cumulative_conversions_b if metric_type == "binary" else None,
            "cumulative_means_a": cumulative_means_a if metric_type == "numeric" else None,
            "cumulative_means_b": cumulative_means_b if metric_type == "numeric" else None,
            "p_values_over_time": p_values_over_time,
            "significance_flags": significance_flags,
            "risk_level": risk_level,
            "interpretation": interpretation,
            "first_significant_index": first_significant_idx,
            "total_checks": total_checks
        })
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in sequential testing: {str(e)}")


@app.post("/api/uplift-bootstrap")
async def uplift_bootstrap(
    file: UploadFile = File(...),
    metric_type: str = Form(...),
    metric_column: str = Form(...),
    group_column: str = Form(...),
    group_a_name: str = Form(...),
    group_b_name: str = Form(...),
    resamples: int = Form(5000)
):
    """
    Bootstrap resampling to estimate uplift distribution
    """
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        df.columns = df.columns.str.strip()
        
        # Validate columns
        for col in [metric_column, group_column]:
            if col not in df.columns:
                raise HTTPException(status_code=400, detail=f"Column '{col}' not found in CSV")
        
        df[group_column] = df[group_column].astype(str).str.strip()
        group_a = df[df[group_column] == group_a_name][metric_column].dropna()
        group_b = df[df[group_column] == group_b_name][metric_column].dropna()
        
        if len(group_a) == 0 or len(group_b) == 0:
            raise HTTPException(status_code=400, detail="One or both groups are empty")
        
        # Bootstrap resampling
        np.random.seed(42)
        bootstrap_differences = []
        
        for _ in range(resamples):
            # Resample with replacement
            sample_a = np.random.choice(group_a.values, size=len(group_a), replace=True)
            sample_b = np.random.choice(group_b.values, size=len(group_b), replace=True)
            
            if metric_type == "binary":
                diff = np.mean(sample_b) - np.mean(sample_a)
            else:
                diff = np.mean(sample_b) - np.mean(sample_a)
            
            bootstrap_differences.append(diff)
        
        bootstrap_differences = np.array(bootstrap_differences)
        
        # Calculate statistics
        mean_uplift = np.mean(bootstrap_differences)
        ci_lower = np.percentile(bootstrap_differences, 2.5)
        ci_upper = np.percentile(bootstrap_differences, 97.5)
        
        interpretation = f"""
        Bootstrap analysis with {resamples:,} resamples shows:
        Mean uplift: {mean_uplift:.4f}
        95% Confidence Interval: [{ci_lower:.4f}, {ci_upper:.4f}]
        {'The interval does not include zero, suggesting a significant effect.' if ci_lower > 0 or ci_upper < 0 else 'The interval includes zero, suggesting the effect may not be significant.'}
        """
        
        return JSONResponse(content={
            "simulated_uplift_distribution": bootstrap_differences.tolist()[:1000],  # Limit response size
            "mean_uplift": float(mean_uplift),
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
            "interpretation": interpretation.strip()
        })
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in bootstrap analysis: {str(e)}")


@app.post("/api/bayesian-grid")
async def bayesian_grid(request: BayesianGridRequest):
    """
    Calculate P(B > A) for different Beta prior parameters (sensitivity analysis)
    """
    try:
        if request.metric_type == "binary":
            if request.conversions_a is None or request.conversions_b is None or request.n_a is None or request.n_b is None:
                raise HTTPException(status_code=400, detail="For binary metrics, conversions_a, conversions_b, n_a, n_b are required")
            
            grid = []
            for alpha in request.alpha_range:
                for beta in request.beta_range:
                    # Posterior parameters
                    alpha_a = alpha + request.conversions_a
                    beta_a = beta + (request.n_a - request.conversions_a)
                    alpha_b = alpha + request.conversions_b
                    beta_b = beta + (request.n_b - request.conversions_b)
                    
                    # Sample from posteriors
                    np.random.seed(42)
                    samples_a = np.random.beta(alpha_a, beta_a, 10000)
                    samples_b = np.random.beta(alpha_b, beta_b, 10000)
                    
                    prob_b_better = np.mean(samples_b > samples_a)
                    
                    grid.append({
                        "alpha": float(alpha),
                        "beta": float(beta),
                        "prob_b_better": float(prob_b_better)
                    })
            
            # Create matrix for heatmap
            alpha_vals = sorted(set(request.alpha_range))
            beta_vals = sorted(set(request.beta_range))
            matrix = []
            
            for alpha in alpha_vals:
                row = []
                for beta in beta_vals:
                    item = next((g for g in grid if g["alpha"] == alpha and g["beta"] == beta), None)
                    row.append(item["prob_b_better"] if item else 0.5)
                matrix.append(row)
            
            return JSONResponse(content={
                "grid": grid,
                "matrix": matrix,
                "alpha_range": alpha_vals,
                "beta_range": beta_vals
            })
        
        else:
            raise HTTPException(status_code=400, detail="Bayesian grid currently only supports binary metrics")
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in Bayesian grid analysis: {str(e)}")


@app.post("/api/risk-assessment")
async def risk_assessment(request: RiskAssessmentRequest):
    """
    Assess experiment risk: false positive risk, chance of sign flip, etc.
    """
    try:
        p_value = request.p_value
        ci_lower = request.ci_lower
        ci_upper = request.ci_upper
        bayesian_prob = request.bayesian_prob
        
        # Risk level determination
        if p_value < 0.01 and (ci_lower > 0 and ci_upper > 0) or (ci_lower < 0 and ci_upper < 0):
            risk_level = "low"
        elif p_value < 0.05 and (ci_lower <= 0 <= ci_upper):
            risk_level = "medium"
        elif p_value >= 0.05:
            risk_level = "high"
        else:
            risk_level = "medium"
        
        # False positive risk (1 - Bayesian probability if significant)
        if p_value < 0.05:
            false_positive_risk = (1 - bayesian_prob) * 100
        else:
            false_positive_risk = 50.0  # High uncertainty
        
        # Chance of sign flip (bootstrap-based estimate)
        # If CI crosses zero, higher chance of flip
        if ci_lower < 0 < ci_upper:
            chance_sign_flip = abs(ci_lower) / (abs(ci_lower) + ci_upper) * 100
        elif ci_lower > 0:
            # Both positive, but if close to zero, some risk
            chance_sign_flip = max(0, (1 - ci_lower / ci_upper) * 20) if ci_upper > 0 else 0
        else:
            # Both negative
            chance_sign_flip = max(0, (1 - abs(ci_upper) / abs(ci_lower)) * 20) if ci_lower < 0 else 0
        
        # Required additional sample size (rough estimate)
        # If not significant, estimate needed sample size
        if p_value >= 0.05:
            # Rough estimate: need 4x current sample for 2x effect size detection
            current_total = request.sample_size_a + request.sample_size_b
            required_total = current_total * 4  # Conservative estimate
            required_additional = max(0, required_total - current_total)
        else:
            required_additional = 0
        
        interpretation = f"""
        Risk Assessment:
        - Risk Level: {risk_level.upper()}
        - False Positive Risk: {false_positive_risk:.1f}%
        - Chance of Sign Flip: {chance_sign_flip:.1f}%
        {'- Additional Sample Size Needed: ' + str(int(required_additional)) if required_additional > 0 else '- Current sample size is sufficient'}
        
        {'The results show low risk with strong statistical evidence.' if risk_level == 'low'
        else 'The results show moderate risk. Proceed with caution.' if risk_level == 'medium'
        else 'The results show high risk. Consider extending the experiment or increasing sample size.'}
        """
        
        return JSONResponse(content={
            "risk_level": risk_level,
            "false_positive_risk": float(false_positive_risk),
            "chance_of_sign_flip": float(chance_sign_flip),
            "required_additional_sample_size": int(required_additional),
            "interpretation": interpretation.strip()
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in risk assessment: {str(e)}")


@app.get("/api/glossary")
async def get_glossary():
    """
    Return glossary of statistical terms in Markdown format
    """
    glossary = {
        "p-value": """
**P-Value**

The probability of observing the current result (or more extreme) if there is no true difference between groups (null hypothesis is true).

- Lower values indicate stronger evidence against the null hypothesis
- Typically, p < 0.05 is considered statistically significant
- Very small p-values (< 0.001) indicate very strong evidence
        """,
        "lift": """
**Lift**

The percentage improvement of the treatment group over the control group.

Formula: Lift = (Conversion_B - Conversion_A) / Conversion_A × 100%

- Positive lift means treatment is better
- Negative lift means control is better
- Lift alone doesn't indicate statistical significance
        """,
        "effect_size": """
**Effect Size**

A standardized measure of the magnitude of the difference between groups, independent of sample size.

- **Cohen's h** (for proportions): h = 2 × (arcsin(√pB) - arcsin(√pA))
  - Small: < 0.2, Medium: 0.2-0.5, Large: > 0.5
- **Cohen's d** (for continuous): d = (mean_B - mean_A) / pooled_std
  - Small: < 0.2, Medium: 0.2-0.5, Large: 0.5-0.8, Very Large: > 0.8

Effect size helps assess practical significance beyond statistical significance.
        """,
        "confidence_intervals": """
**Confidence Intervals (CI)**

A range of values that, with a specified confidence level (typically 95%), contains the true difference between groups.

- If the 95% CI doesn't include zero, the result is statistically significant
- Narrower intervals indicate more precise estimates
- CI provides information about the magnitude and direction of the effect
        """,
        "bayesian_probability": """
**Bayesian Probability**

The probability that one variant is better than another, incorporating prior beliefs and observed data.

- P(B > A) = probability treatment is better than control
- Values close to 1 (e.g., > 0.95) indicate strong evidence treatment is better
- Values close to 0 (e.g., < 0.05) indicate strong evidence control is better
- Values near 0.5 indicate uncertainty

Uses Beta-Binomial (binary) or Normal (continuous) posterior distributions.
        """,
        "sequential_testing": """
**Sequential Testing**

Monitoring p-values over time as data accumulates during an experiment.

**Risks:**
- **Early Stopping**: Stopping when p < 0.05 first appears can inflate false positive rate
- **P-Hacking**: Repeatedly checking results and stopping when significant increases false positives

**Best Practices:**
- Pre-determine sample size and analysis time
- Use sequential testing methods (e.g., O'Brien-Fleming boundaries)
- Be cautious of early significant results
        """,
        "CUPED": """
**CUPED (Controlled-experiment Using Pre-Experiment Data)**

A variance reduction technique that uses a covariate (pre-experiment metric) to reduce variance and increase statistical power.

**Formula:**
- Θ = Cov(Y, X) / Var(X)
- Y_adj = Y - Θ(X - mean(X))

**Benefits:**
- Reduces variance in the metric
- Increases statistical power
- Can detect smaller effects with same sample size

**Requirements:**
- Need a pre-experiment covariate (e.g., pre-experiment conversion rate)
- Covariate should be correlated with the outcome metric
        """,
        "power": """
**Statistical Power**

The probability of correctly rejecting the null hypothesis when it is false (i.e., detecting a true effect).

- Typically set to 80% (0.8) or 90% (0.9)
- Higher power requires larger sample sizes
- Power = 1 - β (where β is Type II error rate)

**Factors affecting power:**
- Sample size (larger = more power)
- Effect size (larger = more power)
- Significance level (lower α = less power)
        """,
        "MDE": """
**MDE (Minimum Detectable Effect)**

The smallest effect size that can be detected with a given sample size, power, and significance level.

**Uses:**
- Experiment planning: "What's the smallest lift I can detect?"
- Sample size justification: "Do I have enough power to detect meaningful effects?"

**Formula:**
- Derived from power analysis equation
- For binary: Convert Cohen's h back to rate difference
- MDE helps set realistic expectations for experiment outcomes
        """
    }
    
    return JSONResponse(content=glossary)


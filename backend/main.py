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
import statsmodels.api as sm
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


# New models for advanced features
class MultiVariantAnalysisRequest(BaseModel):
    group_column: str = Field(..., description="Column name containing group labels")
    metric_column: str = Field(..., description="Column name containing metric values")
    metric_type: Literal["binary", "numeric"] = Field(..., description="Type of metric")
    control_group: str = Field(..., description="Control group name")
    treatment_groups: List[str] = Field(..., description="List of treatment group names")


class SegmentationAnalysisRequest(BaseModel):
    group_column: str = Field(..., description="Column name containing group labels")
    metric_column: str = Field(..., description="Column name containing metric values")
    segment_column: str = Field(..., description="Column name for segmentation (e.g., demographics)")
    metric_type: Literal["binary", "numeric"] = Field(..., description="Type of metric")
    group_a_name: str = Field(..., description="Control group name")
    group_b_name: str = Field(..., description="Treatment group name")


class TimeBasedAnalysisRequest(BaseModel):
    group_column: str = Field(..., description="Column name containing group labels")
    metric_column: str = Field(..., description="Column name containing metric values")
    timestamp_column: str = Field(..., description="Column name containing timestamps")
    metric_type: Literal["binary", "numeric"] = Field(..., description="Type of metric")
    group_a_name: str = Field(..., description="Control group name")
    group_b_name: str = Field(..., description="Treatment group name")
    time_granularity: Literal["hour", "day", "week"] = Field("day", description="Time granularity for analysis")


class RegressionAnalysisRequest(BaseModel):
    group_column: str = Field(..., description="Column name containing group labels")
    metric_column: str = Field(..., description="Column name containing metric values")
    covariate_columns: List[str] = Field(..., description="List of covariate column names")
    metric_type: Literal["binary", "numeric"] = Field(..., description="Type of metric")
    group_a_name: str = Field(..., description="Control group name")
    group_b_name: str = Field(..., description="Treatment group name")


class NonParametricTestRequest(BaseModel):
    group_column: str = Field(..., description="Column name containing group labels")
    metric_column: str = Field(..., description="Column name containing metric values")
    test_type: Literal["mann_whitney", "kruskal_wallis"] = Field(..., description="Type of non-parametric test")
    group_a_name: str = Field(..., description="Control group name (or first group)")
    group_b_name: Optional[str] = Field(None, description="Treatment group name (optional for Kruskal-Wallis)")
    additional_groups: Optional[List[str]] = Field(None, description="Additional groups for Kruskal-Wallis")


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


# ============================================================================
# ADVANCED STATISTICAL FEATURES ENDPOINTS
# ============================================================================

@app.post("/api/multi-variant-analysis")
async def multi_variant_analysis(
    file: UploadFile = File(...),
    group_column: str = Form(...),
    metric_column: str = Form(...),
    metric_type: str = Form("binary"),
    control_group: str = Form(...),
    treatment_groups: str = Form(...)  # Comma-separated list
):
    """
    Analyze multi-variant test (A/B/C/D) comparing multiple treatment groups to control
    """
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        df.columns = df.columns.str.strip()
        
        group_column = group_column.strip()
        metric_column = metric_column.strip()
        control_group = control_group.strip()
        treatment_groups_list = [g.strip() for g in treatment_groups.split(',')]
        
        if group_column not in df.columns or metric_column not in df.columns:
            raise HTTPException(status_code=400, detail="Invalid column names")
        
        df[group_column] = df[group_column].astype(str).str.strip()
        
        # Filter to only include control and treatment groups
        all_groups = [control_group] + treatment_groups_list
        df = df[df[group_column].isin(all_groups)]
        
        results = []
        control_data = df[df[group_column] == control_group][metric_column]
        
        for treatment in treatment_groups_list:
            treatment_data = df[df[group_column] == treatment][metric_column]
            
            if metric_type == "binary":
                analysis = analyze_binary_metric(
                    df[df[group_column] == control_group],
                    df[df[group_column] == treatment],
                    metric_column
                )
            else:
                analysis = analyze_numeric_metric(
                    df[df[group_column] == control_group],
                    df[df[group_column] == treatment],
                    metric_column
                )
            
            analysis['treatment_group'] = treatment
            analysis['control_group'] = control_group
            results.append(analysis)
        
        # Overall comparison (Kruskal-Wallis or Chi-square)
        if metric_type == "binary":
            # Chi-square test for multiple proportions
            contingency = pd.crosstab(df[group_column], df[metric_column])
            chi2, p_value_overall = stats.chi2_contingency(contingency)[:2]
            test_type_overall = "Chi-square test"
        else:
            # Kruskal-Wallis test
            groups_data = [df[df[group_column] == g][metric_column].dropna() for g in all_groups]
            h_stat, p_value_overall = stats.kruskal(*groups_data)
            test_type_overall = "Kruskal-Wallis test"
        
        return JSONResponse(content={
            "control_group": control_group,
            "treatment_groups": treatment_groups_list,
            "pairwise_comparisons": results,
            "overall_test": {
                "test_type": test_type_overall,
                "p_value": float(p_value_overall),
                "significant": bool(p_value_overall < 0.05)
            }
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in multi-variant analysis: {str(e)}")


@app.post("/api/segmentation-analysis")
async def segmentation_analysis(
    file: UploadFile = File(...),
    group_column: str = Form(...),
    metric_column: str = Form(...),
    segment_column: str = Form(...),
    metric_type: str = Form("binary"),
    group_a_name: str = Form(...),
    group_b_name: str = Form(...)
):
    """
    Analyze A/B test results segmented by user characteristics (demographics, cohorts, etc.)
    For each segment, computes control and treatment rates separately within that segment.
    """
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        df.columns = df.columns.str.strip()
        
        group_column = group_column.strip()
        metric_column = metric_column.strip()
        segment_column = segment_column.strip()
        control_group = group_a_name.strip()
        treatment_group = group_b_name.strip()
        
        if group_column not in df.columns or metric_column not in df.columns or segment_column not in df.columns:
            raise HTTPException(status_code=400, detail="Invalid column names")
        
        # Clean group and segment columns - ensure consistent string types
        df[group_column] = df[group_column].astype(str).str.strip()
        df[segment_column] = df[segment_column].astype(str).str.strip()
        
        # Also ensure metric column is numeric for binary (0/1) or float
        if metric_type == "binary":
            # Convert to numeric, coercing errors to NaN, then fill NaN with 0
            df[metric_column] = pd.to_numeric(df[metric_column], errors='coerce').fillna(0)
        else:
            df[metric_column] = pd.to_numeric(df[metric_column], errors='coerce')
        
        # Get unique segments
        unique_segments = df[segment_column].dropna().unique().tolist()
        
        # Debug: Check available group values
        unique_groups = df[group_column].dropna().unique().tolist()
        print(f"DEBUG: Available groups in data: {unique_groups}")
        print(f"DEBUG: Looking for control_group='{control_group}', treatment_group='{treatment_group}'")
        print(f"DEBUG: Segment column='{segment_column}', Group column='{group_column}'")
        
        segment_results = []
        for segment in unique_segments:
            # Filter to this segment
            segment_df = df[df[segment_column] == segment].copy()
            
            if len(segment_df) == 0:
                continue
            
            # Debug: Check what groups exist in this segment
            segment_groups = segment_df[group_column].dropna().unique().tolist()
            print(f"DEBUG: Segment '{segment}' has groups: {segment_groups}")
            
            # Within this segment, filter to control and treatment groups
            # Ensure we're comparing cleaned strings
            control_df = segment_df[segment_df[group_column].astype(str).str.strip() == control_group].copy()
            treatment_df = segment_df[segment_df[group_column].astype(str).str.strip() == treatment_group].copy()
            
            print(f"DEBUG: Segment '{segment}' - control_df size: {len(control_df)}, treatment_df size: {len(treatment_df)}")
            if len(control_df) > 0:
                print(f"DEBUG: Segment '{segment}' - control_df sample group values: {control_df[group_column].head(3).tolist()}")
            if len(treatment_df) > 0:
                print(f"DEBUG: Segment '{segment}' - treatment_df sample group values: {treatment_df[group_column].head(3).tolist()}")
            
            if len(control_df) == 0 or len(treatment_df) == 0:
                print(f"DEBUG: Skipping segment '{segment}' - missing control or treatment data")
                continue
            
            # Verify we have different dataframes
            if len(control_df) == len(treatment_df) and control_df.equals(treatment_df):
                print(f"WARNING: Segment '{segment}' - control and treatment dataframes are identical!")
                continue
            
            # Compute rates within the segment
            if metric_type == "binary":
                # For binary metrics, use mean() which gives conversion rate
                control_rate = float(control_df[metric_column].mean()) if len(control_df) > 0 else 0.0
                treatment_rate = float(treatment_df[metric_column].mean()) if len(treatment_df) > 0 else 0.0
                
                print(f"DEBUG: Segment '{segment}' - control_rate: {control_rate}, treatment_rate: {treatment_rate}")
                
                # Compute lift
                lift = ((treatment_rate - control_rate) / control_rate * 100) if control_rate > 0 else 0
                
                # Two-proportion z-test for statistical significance
                n_control = len(control_df)
                n_treatment = len(treatment_df)
                conversions_control = control_df[metric_column].sum()
                conversions_treatment = treatment_df[metric_column].sum()
                
                # Pooled proportion
                p_pool = (conversions_control + conversions_treatment) / (n_control + n_treatment) if (n_control + n_treatment) > 0 else 0
                
                # Standard error
                se = np.sqrt(p_pool * (1 - p_pool) * (1/n_control + 1/n_treatment)) if p_pool > 0 and p_pool < 1 and n_control > 0 and n_treatment > 0 else 0
                
                # Z-statistic
                diff = treatment_rate - control_rate
                z_stat = (diff / se) if se > 0 else 0
                
                # P-value
                if se > 0:
                    p_value = 2 * stats.norm.sf(abs(z_stat))
                else:
                    p_value = 1.0
                
                # 95% Confidence Interval
                z_critical = 1.96
                ci_lower = diff - z_critical * se
                ci_upper = diff + z_critical * se
                
                # Effect size (Cohen's h)
                h = 2 * (np.arcsin(np.sqrt(treatment_rate)) - np.arcsin(np.sqrt(control_rate))) if treatment_rate >= 0 and treatment_rate <= 1 and control_rate >= 0 and control_rate <= 1 else 0
                
                analysis = {
                    'metric_type': 'binary',
                    'group_a': {
                        'size': int(n_control),
                        'conversions': int(conversions_control),
                        'rate': float(control_rate),
                        'mean': float(control_rate)
                    },
                    'group_b': {
                        'size': int(n_treatment),
                        'conversions': int(conversions_treatment),
                        'rate': float(treatment_rate),
                        'mean': float(treatment_rate)
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
            else:
                # For numeric metrics, use mean()
                control_rate = control_df[metric_column].mean() if len(control_df) > 0 else 0
                treatment_rate = treatment_df[metric_column].mean() if len(treatment_df) > 0 else 0
                
                # Compute lift
                lift = ((treatment_rate - control_rate) / control_rate * 100) if control_rate != 0 else 0
                
                # Two-sample t-test for numeric metrics
                control_values = control_df[metric_column].dropna()
                treatment_values = treatment_df[metric_column].dropna()
                
                if len(control_values) == 0 or len(treatment_values) == 0:
                    continue
                
                # Perform t-test
                t_stat, p_value = stats.ttest_ind(treatment_values, control_values)
                
                # Standard errors
                se_control = control_values.std() / np.sqrt(len(control_values)) if len(control_values) > 0 else 0
                se_treatment = treatment_values.std() / np.sqrt(len(treatment_values)) if len(treatment_values) > 0 else 0
                se_diff = np.sqrt(se_control**2 + se_treatment**2)
                
                # 95% Confidence Interval
                t_critical = 1.96  # Approximate for large samples
                diff = treatment_rate - control_rate
                ci_lower = diff - t_critical * se_diff
                ci_upper = diff + t_critical * se_diff
                
                # Effect size (Cohen's d)
                pooled_std = np.sqrt(((len(control_values) - 1) * control_values.var() + (len(treatment_values) - 1) * treatment_values.var()) / (len(control_values) + len(treatment_values) - 2)) if (len(control_values) + len(treatment_values) > 2) else 0
                cohens_d = (treatment_rate - control_rate) / pooled_std if pooled_std > 0 else 0
                
                analysis = {
                    'metric_type': 'numeric',
                    'group_a': {
                        'size': int(len(control_values)),
                        'mean': float(control_rate),
                        'std': float(control_values.std()),
                        'rate': float(control_rate)  # For consistency
                    },
                    'group_b': {
                        'size': int(len(treatment_values)),
                        'mean': float(treatment_rate),
                        'std': float(treatment_values.std()),
                        'rate': float(treatment_rate)  # For consistency
                    },
                    'lift': float(lift),
                    'p_value': float(p_value),
                    'test_statistic': float(t_stat),
                    'test_type': 'two-sample t-test',
                    'effect_size': float(cohens_d),
                    'effect_size_type': "Cohen's d",
                    'confidence_interval': {
                        'lower': float(ci_lower),
                        'upper': float(ci_upper),
                        'level': 0.95
                    },
                    'difference': float(diff)
                }
            
            analysis['segment'] = segment
            analysis['segment_size'] = len(segment_df)
            segment_results.append(analysis)
        
        return JSONResponse(content={
            "segment_column": segment_column,
            "segments": segment_results,
            "total_segments": len(segment_results),
            "analysis_mode": "segmentation"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in segmentation analysis: {str(e)}")


@app.post("/api/time-based-analysis")
async def time_based_analysis(
    file: UploadFile = File(...),
    group_column: str = Form(...),
    metric_column: str = Form(...),
    timestamp_column: str = Form(...),
    metric_type: str = Form("binary"),
    group_a_name: str = Form(...),
    group_b_name: str = Form(...),
    time_granularity: str = Form("day")
):
    """
    Analyze A/B test results over time (day-of-week, hour-of-day patterns)
    """
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        df.columns = df.columns.str.strip()
        
        group_column = group_column.strip()
        metric_column = metric_column.strip()
        timestamp_column = timestamp_column.strip()
        
        if group_column not in df.columns or metric_column not in df.columns or timestamp_column not in df.columns:
            raise HTTPException(status_code=400, detail="Invalid column names")
        
        # Parse timestamps
        df[timestamp_column] = pd.to_datetime(df[timestamp_column], errors='coerce')
        df = df.dropna(subset=[timestamp_column])
        
        # Extract time features
        if time_granularity == "hour":
            df['time_period'] = df[timestamp_column].dt.hour
            period_name = "Hour of Day"
        elif time_granularity == "week":
            df['time_period'] = df[timestamp_column].dt.isocalendar().week
            period_name = "Week of Year"
        else:  # day
            df['time_period'] = df[timestamp_column].dt.dayofweek
            period_name = "Day of Week"
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        df[group_column] = df[group_column].astype(str).str.strip()
        
        time_results = []
        unique_periods = sorted(df['time_period'].dropna().unique())
        
        for period in unique_periods:
            period_df = df[df['time_period'] == period]
            group_a = period_df[period_df[group_column] == group_a_name]
            group_b = period_df[period_df[group_column] == group_b_name]
            
            if len(group_a) == 0 or len(group_b) == 0:
                continue
            
            if metric_type == "binary":
                analysis = analyze_binary_metric(group_a, group_b, metric_column)
            else:
                analysis = analyze_numeric_metric(group_a, group_b, metric_column)
            
            period_label = day_names[int(period)] if time_granularity == "day" and period < len(day_names) else str(period)
            analysis['time_period'] = int(period)
            analysis['time_period_label'] = period_label
            analysis['period_name'] = period_name
            time_results.append(analysis)
        
        return JSONResponse(content={
            "time_granularity": time_granularity,
            "period_name": period_name,
            "time_periods": time_results
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in time-based analysis: {str(e)}")


def extract_numeric_covariate_effects(coefficients, covariate_info):
    """
    Extract numeric covariate effects from regression coefficients.
    Returns a dictionary mapping variable names to their effects.
    """
    numeric_effects = {}
    
    for coef_data in coefficients:
        var_name = coef_data.get('name', '')
        
        # Skip treatment and intercept
        if var_name == 'treatment' or var_name == 'const':
            continue
        
        # Check if this is a numeric variable (not a dummy variable)
        is_numeric = True
        for info in covariate_info:
            if info.get('name') == var_name and info.get('type') == 'categorical':
                is_numeric = False
                break
        
        # Also check if it's a day effect (those are handled separately)
        if 'day' in var_name.lower() or 'most_ads_day' in var_name.lower():
            is_numeric = False
        
        if is_numeric:
            # Calculate odds ratio only if it exists (for logistic regression)
            odds_ratio = coef_data.get('odds_ratio')
            if odds_ratio is None:
                # For linear regression, calculate from coefficient if needed
                coef = coef_data.get('coef', 0)
                odds_ratio = np.exp(coef)  # Approximate for display purposes
            
            numeric_effects[var_name] = {
                'coef': coef_data.get('coef', 0),
                'odds_ratio': float(odds_ratio) if odds_ratio is not None else None,
                'p_value': coef_data.get('p_value', 1),
                'ci_lower': coef_data.get('ci_lower', 0),
                'ci_upper': coef_data.get('ci_upper', 0),
                'std_err': coef_data.get('std_err', 0),
                'significant': coef_data.get('p_value', 1) < 0.05,
                'original_name': coef_data.get('original_name', var_name)
            }
    
    return numeric_effects


def calculate_marginal_effects(model, X, X_column_names, numeric_effects, covariate_info):
    """
    Calculate marginal effects for numeric covariates.
    Returns data for plotting predicted probabilities vs covariate values.
    """
    marginal_effects_data = {}
    
    if not numeric_effects:
        return marginal_effects_data
    
    # Get mean values for all predictors (for holding constant)
    X_mean = X.mean(axis=0)
    X_mean_array = X_mean.values
    
    for var_name, effect_data in numeric_effects.items():
        if var_name not in X.columns:
            continue
        
        # Get column index
        var_idx = list(X.columns).index(var_name)
        
        # Get actual data range for this variable
        var_data = X[var_name]
        var_min = float(var_data.min())
        var_max = float(var_data.max())
        
        # Create range of values for plotting
        n_points = 50
        var_range = np.linspace(var_min, var_max, n_points)
        
        # Calculate predicted probabilities for each value
        predicted_probs = []
        ci_lower = []
        ci_upper = []
        
        for val in var_range:
            # Create prediction vector with this variable at val, others at mean
            X_pred = X_mean_array.copy()
            X_pred[var_idx] = val
            
            # Add intercept (first column)
            X_pred_with_const = np.insert(X_pred, 0, 1.0)
            
            # Predict
            pred_prob = model.predict(X_pred_with_const.reshape(1, -1))[0]
            predicted_probs.append(float(pred_prob))
            
            # Calculate CI (approximate)
            se = float(effect_data.get('std_err', 0))
            if se > 0:
                # Approximate CI on probability scale
                # Use delta method approximation
                logit_se = se * abs(val - X_mean_array[var_idx])
                prob_se = pred_prob * (1 - pred_prob) * logit_se
                ci_lower.append(max(0, float(pred_prob - 1.96 * prob_se)))
                ci_upper.append(min(1, float(pred_prob + 1.96 * prob_se)))
            else:
                ci_lower.append(float(pred_prob))
                ci_upper.append(float(pred_prob))
        
        marginal_effects_data[var_name] = {
            'variable_name': var_name,
            'original_name': effect_data.get('original_name', var_name),
            'x_values': var_range.tolist(),
            'predicted_probs': predicted_probs,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'coef': float(effect_data.get('coef', 0)),
            'odds_ratio': float(effect_data.get('odds_ratio', 1)),
            'p_value': float(effect_data.get('p_value', 1)),
            'significant': effect_data.get('significant', False)
        }
    
    return marginal_effects_data


def calculate_binned_effects(df, numeric_effects, covariate_info, metric_column):
    """
    Calculate effects for binned numeric variables (for categorical comparison view).
    """
    binned_effects = {}
    
    if not numeric_effects:
        return binned_effects
    
    for var_name, effect_data in numeric_effects.items():
        original_name = effect_data.get('original_name', var_name)
        
        # Skip if variable not in original dataframe
        if original_name not in df.columns:
            continue
        
        # Get variable data
        var_data = pd.to_numeric(df[original_name], errors='coerce').dropna()
        if len(var_data) == 0:
            continue
        
        # Determine binning strategy based on variable
        if 'hour' in var_name.lower() or 'time' in var_name.lower():
            # Time-based: Morning, Afternoon, Evening, Night
            bins = [0, 6, 12, 18, 24]
            labels = ['Night (0-6)', 'Morning (6-12)', 'Afternoon (12-18)', 'Evening (18-24)']
        else:
            # General numeric: Low, Medium, High (tertiles)
            q33 = float(var_data.quantile(0.33))
            q67 = float(var_data.quantile(0.67))
            bins = [-np.inf, q33, q67, np.inf]
            labels = ['Low', 'Medium', 'High']
        
        # Bin the data
        df_binned = df.copy()
        df_binned[f'{original_name}_binned'] = pd.cut(
            pd.to_numeric(df[original_name], errors='coerce'),
            bins=bins,
            labels=labels,
            include_lowest=True
        )
        
        # Calculate conversion rates by bin
        bin_effects = []
        baseline_rate = None
        
        for label in labels:
            bin_data = df_binned[df_binned[f'{original_name}_binned'] == label]
            if len(bin_data) == 0:
                continue
            
            # Calculate mean conversion rate in this bin
            if metric_column in bin_data.columns:
                conversion_rate = pd.to_numeric(bin_data[metric_column], errors='coerce').mean()
                n = len(bin_data)
                
                if pd.isna(conversion_rate):
                    continue
                
                # For comparison, use first bin as baseline
                if baseline_rate is None:
                    baseline_rate = conversion_rate
                    bin_effects.append({
                        'bin': str(label),
                        'conversion_rate': float(conversion_rate),
                        'n': int(n),
                        'odds_ratio': 1.0,
                        'lift': 0.0,
                        'is_baseline': True
                    })
                else:
                    # Calculate lift vs baseline
                    if baseline_rate > 0:
                        lift = ((conversion_rate - baseline_rate) / baseline_rate) * 100
                        odds_ratio = conversion_rate / baseline_rate if baseline_rate > 0 else 1.0
                    else:
                        lift = 0.0
                        odds_ratio = 1.0
                    
                    bin_effects.append({
                        'bin': str(label),
                        'conversion_rate': float(conversion_rate),
                        'n': int(n),
                        'odds_ratio': float(odds_ratio),
                        'lift': float(lift),
                        'is_baseline': False
                    })
        
        if bin_effects:
            binned_effects[var_name] = {
                'variable_name': var_name,
                'original_name': original_name,
                'bins': bin_effects
            }
    
    return binned_effects


def generate_all_covariates_insights(day_effects, numeric_effects, coefficients, covariate_info):
    """
    Generate unified insights summary explaining all covariate effects.
    """
    insights_parts = []
    
    # Numeric effects summary
    if numeric_effects:
        strong_numeric = [
            (name, data) for name, data in numeric_effects.items()
            if data.get('significant', False) and data.get('coef', 0) > 0
        ]
        strong_numeric.sort(key=lambda x: x[1].get('coef', 0), reverse=True)
        
        if strong_numeric:
            numeric_names = []
            for name, data in strong_numeric[:3]:
                original_name = data.get('original_name', name)
                # Make name readable
                readable_name = original_name.replace('_', ' ').title()
                numeric_names.append(readable_name)
            
            if len(numeric_names) == 1:
                insights_parts.append(f"Conversion likelihood increases steadily with {numeric_names[0].lower()}.")
            elif len(numeric_names) == 2:
                insights_parts.append(f"Conversion likelihood increases with {numeric_names[0].lower()} and {numeric_names[1].lower()}.")
            else:
                insights_parts.append(f"Conversion likelihood increases with {numeric_names[0].lower()}, {numeric_names[1].lower()}, and {numeric_names[2].lower()}.")
    
    # Day effects summary
    if day_effects and len(day_effects) > 0:
        significant_days = [d for d in day_effects if d.get('significant', False)]
        if significant_days:
            top_days = sorted(significant_days, key=lambda x: x.get('odds_ratio', 1), reverse=True)[:2]
            day_names = [d.get('day', '') for d in top_days]
            
            if len(day_names) == 1:
                insights_parts.append(f"Peaks during {day_names[0]} exposure.")
            elif len(day_names) == 2:
                insights_parts.append(f"Peaks during {day_names[0]} and {day_names[1]} exposure.")
            else:
                insights_parts.append(f"Peaks during early weekday exposure, especially {day_names[0]} and {day_names[1]}.")
    
    # Combine insights
    if insights_parts:
        return " ".join(insights_parts)
    else:
        return "Covariate effects are relatively uniform across all variables tested."


def extract_day_of_week_effects(coefficients, covariate_info):
    """
    Extract day-of-week effects from regression coefficients.
    Looks for dummy variables with patterns like 'day_Monday', 'day_Tuesday', 'most ads day_Monday', etc.
    """
    day_effects = []
    # Patterns to match (handle both underscore and space variations)
    day_patterns = ['day_', 'dayofweek_', 'weekday_', 'dow_', 'most_ads_day_', 'most ads day_']
    day_names = {
        'monday': 'Monday', 'mon': 'Monday',
        'tuesday': 'Tuesday', 'tue': 'Tuesday', 'tues': 'Tuesday',
        'wednesday': 'Wednesday', 'wed': 'Wednesday',
        'thursday': 'Thursday', 'thu': 'Thursday', 'thur': 'Thursday',
        'friday': 'Friday', 'fri': 'Friday',
        'saturday': 'Saturday', 'sat': 'Saturday',
        'sunday': 'Sunday', 'sun': 'Sunday'
    }
    
    for coef_data in coefficients:
        var_name = coef_data.get('name', '').lower()
        original_name = coef_data.get('original_name', '').lower()
        category = coef_data.get('category', '').lower() if coef_data.get('category') else ''
        
        # Normalize spaces to underscores for pattern matching
        var_name_normalized = var_name.replace(' ', '_')
        original_name_normalized = original_name.replace(' ', '_')
        
        # Check if this is a day-related variable
        is_day_var = False
        day_name = None
        
        # Check common day patterns (handle both underscore and space)
        for pattern in day_patterns:
            pattern_normalized = pattern.replace(' ', '_')
            if pattern_normalized in var_name_normalized or pattern_normalized in original_name_normalized:
                is_day_var = True
                # Extract day name from variable - search in all possible locations
                search_text = var_name_normalized + ' ' + original_name_normalized + ' ' + category
                for day_key, day_value in day_names.items():
                    if day_key in search_text:
                        day_name = day_value
                        break
                if day_name:
                    break
        
        # Also check if original covariate name or category suggests day-of-week
        if not is_day_var:
            # Check category field (from dummy encoding)
            if category:
                for day_key, day_value in day_names.items():
                    if day_key in category:
                        is_day_var = True
                        day_name = day_value
                        break
            
            # Check original name directly for day keywords
            if not is_day_var:
                search_text = original_name_normalized + ' ' + var_name_normalized
                for day_key, day_value in day_names.items():
                    if day_key in search_text and coef_data.get('type') == 'categorical':
                        is_day_var = True
                        day_name = day_value
                        break
        
        if is_day_var and day_name:
            day_effects.append({
                "day": day_name,
                "coef": coef_data.get('coef', 0),
                "odds_ratio": coef_data.get('odds_ratio', 1),
                "p_value": coef_data.get('p_value', 1),
                "ci_lower": coef_data.get('ci_lower', 0),
                "ci_upper": coef_data.get('ci_upper', 0),
                "significant": bool(coef_data.get('p_value', 1) < 0.05)
            })
    
    # Sort by odds ratio (highest first)
    day_effects.sort(key=lambda x: x['odds_ratio'], reverse=True)
    
    return day_effects


def generate_day_effect_interpretation(day_effects):
    """
    Generate human-readable interpretation of day-of-week effects.
    """
    if not day_effects or len(day_effects) == 0:
        return None
    
    # Correctly detect baseline day: the missing day from dummy encoding
    expected_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    returned_days = [d['day'] for d in day_effects]
    baseline_day = next((day for day in expected_days if day not in returned_days), None)
    
    # Get top and bottom performing days
    top_day = day_effects[0]
    bottom_day = day_effects[-1] if len(day_effects) > 1 else day_effects[0]
    
    # Count significant days
    significant_days = [d for d in day_effects if d['significant']]
    num_significant = len(significant_days)
    
    # Build interpretation
    interpretation_parts = []
    
    if top_day['significant']:
        lift_pct = (top_day['odds_ratio'] - 1) * 100
        p_val_str = f"p < {top_day['p_value']:.2e}" if top_day['p_value'] < 0.0001 else f"p = {top_day['p_value']:.4f}"
        interpretation_parts.append(
            f"Users who primarily saw ads on {top_day['day']} have {lift_pct:.1f}% higher odds of converting "
            f"compared to the baseline day ({baseline_day if baseline_day else 'reference'}). "
            f"This effect is statistically significant ({p_val_str})."
        )
    
    # Add second best if significant
    if len(day_effects) > 1 and day_effects[1]['significant']:
        second_day = day_effects[1]
        lift_pct = (second_day['odds_ratio'] - 1) * 100
        interpretation_parts.append(
            f"{second_day['day']} is also a strong performing day with a {lift_pct:.1f}% increase in odds."
        )
    
    # Mention non-significant days
    non_sig_days = [d for d in day_effects if not d['significant']]
    if non_sig_days:
        day_names = [d['day'] for d in non_sig_days]
        if len(day_names) == 1:
            interpretation_parts.append(f"{day_names[0]} shows no meaningful difference from baseline.")
        else:
            interpretation_parts.append(f"{', '.join(day_names[:-1])}, and {day_names[-1]} show no meaningful difference from baseline.")
    
    return " ".join(interpretation_parts)


def generate_targeting_recommendations(day_effects):
    """
    Generate practical targeting recommendations based on day-of-week effects.
    """
    if not day_effects or len(day_effects) == 0:
        return None
    
    # Get significant positive effects (odds ratio > 1 and significant)
    strong_days = [d for d in day_effects if d['significant'] and d['odds_ratio'] > 1]
    weak_days = [d for d in day_effects if d['significant'] and d['odds_ratio'] < 1]
    non_sig_days = [d for d in day_effects if not d['significant']]
    
    recommendations = []
    
    if strong_days:
        # Group strong days
        strong_day_names = [d['day'] for d in strong_days]
        if len(strong_day_names) == 1:
            recommendations.append(
                f"Consider allocating more budget or impressions on {strong_day_names[0]}, "
                f"as this day shows significantly higher conversion likelihood ({strong_days[0]['odds_ratio']:.2f}x odds ratio)."
            )
        elif len(strong_day_names) == 2:
            recommendations.append(
                f"Consider allocating more budget or impressions on {strong_day_names[0]} and {strong_day_names[1]}, "
                f"as these days show significantly higher conversion likelihood."
            )
        else:
            recommendations.append(
                f"Consider allocating more budget or impressions earlier in the week "
                f"({', '.join(strong_day_names[:2])}), as these days show significantly higher conversion likelihood."
            )
    
    if weak_days or non_sig_days:
        weak_day_names = [d['day'] for d in weak_days] + [d['day'] for d in non_sig_days if d['odds_ratio'] < 1]
        if weak_day_names:
            if len(weak_day_names) == 1:
                recommendations.append(
                    f"{weak_day_names[0]} appears less responsive, suggesting reduced ad spend "
                    f"or different creatives for this time period."
                )
            else:
                recommendations.append(
                    f"Weekend days ({', '.join(weak_day_names)}) appear less responsive, suggesting "
                    f"reduced ad spend or different creatives for those time periods."
                )
    
    if not recommendations:
        recommendations.append(
            "Day-of-week effects are relatively uniform. Consider testing other segmentation variables."
        )
    
    return " ".join(recommendations)


@app.post("/api/regression-analysis")
async def regression_analysis(
    file: UploadFile = File(...),
    group_column: str = Form(...),
    metric_column: str = Form(...),
    covariate_columns: str = Form(...),  # Comma-separated
    metric_type: str = Form("binary"),
    group_a_name: str = Form(...),
    group_b_name: str = Form(...)
):
    """
    Perform statistically valid regression analysis controlling for multiple covariates.
    Supports both numeric and categorical covariates (categoricals are one-hot encoded).
    Uses statsmodels for proper statistical inference.
    """
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        df.columns = df.columns.str.strip()
        
        group_column = group_column.strip()
        metric_column = metric_column.strip()
        covariate_list = [c.strip() for c in covariate_columns.split(',')]
        
        if group_column not in df.columns or metric_column not in df.columns:
            raise HTTPException(status_code=400, detail="Invalid column names")
        
        for col in covariate_list:
            if col not in df.columns:
                raise HTTPException(status_code=400, detail=f"Covariate column '{col}' not found")
        
        df[group_column] = df[group_column].astype(str).str.strip()
        
        # Filter to only include the two groups
        df = df[df[group_column].isin([group_a_name, group_b_name])]
        
        if len(df) < 10:
            raise HTTPException(status_code=400, detail="Not enough data after filtering groups")
        
        # Create binary treatment indicator: 1 for treatment (group B), 0 for control (group A)
        df['treatment'] = (df[group_column] == group_b_name).astype(int)
        
        # Prepare outcome variable
        if metric_type == "binary":
            # Ensure binary metric is 0/1
            y = pd.to_numeric(df[metric_column], errors='coerce')
            if y.isna().any():
                raise HTTPException(status_code=400, detail="Binary metric column contains non-numeric values")
            y = y.astype(int)
            if not set(y.unique()).issubset({0, 1}):
                raise HTTPException(status_code=400, detail="Binary metric must contain only 0 and 1 values")
        else:
            y = pd.to_numeric(df[metric_column], errors='coerce')
        
        # Prepare covariates: handle numeric and categorical
        warnings = []
        covariate_info = []
        
        # Start building X with treatment
        X = pd.DataFrame({'treatment': df['treatment']}, index=df.index)
        
        for cov in covariate_list:
            # Try to convert to numeric
            numeric_series = pd.to_numeric(df[cov], errors='coerce')
            
            # Check if column is fully numeric (no NaN after conversion)
            if numeric_series.isna().sum() == 0:
                # Fully numeric - use as is
                if numeric_series.nunique() == 1:
                    warnings.append(f"Warning: Covariate '{cov}' is constant (single value). It will be excluded.")
                    continue
                
                # Add numeric covariate to X
                X[cov] = numeric_series
                covariate_info.append({
                    "name": cov,
                    "type": "numeric",
                    "original_name": cov
                })
            else:
                # Categorical - one-hot encode with drop_first=True
                categorical_series = df[cov].astype(str)
                
                # Check for single category
                unique_cats = categorical_series.unique()
                if len(unique_cats) == 1:
                    warnings.append(f"Warning: Categorical covariate '{cov}' has only one category after filtering. It will be excluded.")
                    continue
                
                # One-hot encode with drop_first=True to avoid multicollinearity
                dummies = pd.get_dummies(categorical_series, prefix=cov, drop_first=True)
                
                # Add each dummy variable to X
                for dummy_col in dummies.columns:
                    dummy_series = dummies[dummy_col]
                    
                    # Check if dummy is constant
                    if dummy_series.nunique() == 1:
                        warnings.append(f"Warning: Dummy variable '{dummy_col}' is constant. It will be excluded.")
                        continue
                    
                    # Add dummy column to X
                    X[dummy_col] = dummy_series
                    covariate_info.append({
                        "name": dummy_col,
                        "type": "categorical",
                        "original_name": cov,
                        "category": dummy_col.replace(f"{cov}_", "")
                    })
        
        # Remove rows with missing values in covariates or outcome
        valid_mask = ~(y.isna() | X.isna().any(axis=1))
        X = X[valid_mask].copy()
        y = y[valid_mask].copy()
        
        if len(X) < 10:
            raise HTTPException(status_code=400, detail="Not enough valid data after removing missing values")
        
        # Remove any constant columns (shouldn't happen but double-check)
        constant_cols = []
        for col in X.columns:
            if X[col].nunique() == 1:
                constant_cols.append(col)
                X = X.drop(columns=[col])
                warnings.append(f"Warning: Column '{col}' is constant and was removed.")
        
        if len(X.columns) == 0:
            raise HTTPException(status_code=400, detail="No valid predictors after removing constant columns")
        
        # CRITICAL: Ensure all columns are numeric (convert to float64)
        # This fixes the "Pandas data cast to numpy dtype of object" error
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        
        # Remove any rows that became NaN after conversion
        valid_mask = ~(y.isna() | X.isna().any(axis=1))
        X = X[valid_mask].copy()
        y = y[valid_mask].copy()
        
        if len(X) < 10:
            raise HTTPException(status_code=400, detail="Not enough valid data after numeric conversion")
        
        # Check for multicollinearity (perfect correlation)
        if len(X.columns) > 1:
            correlation_matrix = X.corr().abs()
            high_corr_pairs = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    if correlation_matrix.iloc[i, j] > 0.99:  # Near-perfect correlation
                        high_corr_pairs.append((correlation_matrix.columns[i], correlation_matrix.columns[j]))
            
            if high_corr_pairs:
                warnings.append(f"Warning: Multicollinearity detected between: {', '.join([f'{p[0]}-{p[1]}' for p in high_corr_pairs])}")
        
        # Ensure y is also numeric
        y = pd.to_numeric(y, errors='coerce')
        valid_mask = ~(y.isna() | X.isna().any(axis=1))
        X = X[valid_mask].copy()
        y = y[valid_mask].copy()
        
        if len(X) < 10:
            raise HTTPException(status_code=400, detail="Not enough valid data after final cleaning")
        
        # Convert to numpy arrays to ensure proper dtype
        # This is the key fix for the convergence error
        X_array = X.values.astype(float)
        y_array = y.values.astype(float)
        
        # Store column names before adding constant
        X_column_names = list(X.columns)
        
        # Add intercept (must be last step before modeling)
        # sm.add_constant adds intercept as first column
        X_with_const = sm.add_constant(X_array, has_constant='add')
        
        if metric_type == "binary":
            # Logistic regression using statsmodels
            # Use numpy arrays to avoid dtype issues
            try:
                model = sm.Logit(y_array, X_with_const).fit(disp=0, maxiter=1000)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Logistic regression failed to converge: {str(e)}")
            
            # Extract results from summary table
            summary_table = model.summary2().tables[1]
            
            # Map variable names: const is first, then our X columns in order
            variable_names = ['const'] + X_column_names
            
            # Update summary table index with our variable names
            if len(summary_table) == len(variable_names):
                summary_table.index = variable_names
            else:
                # If lengths don't match, try to map what we can
                summary_table.index = variable_names[:len(summary_table)] + list(summary_table.index[len(summary_table):])
            
            # Get treatment coefficient
            if 'treatment' not in summary_table.index:
                raise HTTPException(status_code=500, detail=f"Treatment variable not found in model results. Available variables: {summary_table.index.tolist()}, Expected: {variable_names}")
            
            treatment_row = summary_table.loc['treatment']
            
            # Extract values with proper column name handling
            treatment_coef = float(treatment_row['Coef.'])
            treatment_std_err = float(treatment_row['Std.Err.'])
            treatment_z = float(treatment_row['z'])
            treatment_p_value = float(treatment_row['P>|z|'])
            
            # Handle CI column names (they might have different formats)
            ci_col_lower = '[0.025' if '[0.025' in summary_table.columns else '[0.025]' if '[0.025]' in summary_table.columns else None
            ci_col_upper = '0.975]' if '0.975]' in summary_table.columns else '[0.975]' if '[0.975]' in summary_table.columns else None
            
            if ci_col_lower and ci_col_upper:
                treatment_ci_lower = float(treatment_row[ci_col_lower])
                treatment_ci_upper = float(treatment_row[ci_col_upper])
            else:
                # Fallback: calculate CI from coef and std_err
                treatment_ci_lower = treatment_coef - 1.96 * treatment_std_err
                treatment_ci_upper = treatment_coef + 1.96 * treatment_std_err
            
            treatment_odds_ratio = np.exp(treatment_coef)
            
            # Build coefficients array
            coefficients = []
            for i, (idx, row) in enumerate(summary_table.iterrows()):
                if idx == 'const' or i == 0:  # Skip intercept
                    continue
                
                # Get the actual variable name from our X columns
                var_name = variable_names[i] if i < len(variable_names) else idx
                
                coef = float(row['Coef.'])
                std_err = float(row['Std.Err.'])
                z_val = float(row['z'])
                p_val = float(row['P>|z|'])
                
                # Handle CI column names
                if ci_col_lower and ci_col_upper:
                    ci_low = float(row[ci_col_lower])
                    ci_high = float(row[ci_col_upper])
                else:
                    # Fallback: calculate CI
                    ci_low = coef - 1.96 * std_err
                    ci_high = coef + 1.96 * std_err
                
                odds_ratio = np.exp(coef)
                
                # Find original covariate name
                original_name = var_name
                cov_type = "numeric"
                category = None
                
                for info in covariate_info:
                    if info['name'] == var_name:
                        original_name = info['original_name']
                        cov_type = info['type']
                        category = info.get('category')
                        break
                
                coefficients.append({
                    "name": var_name,
                    "original_name": original_name,
                    "type": cov_type,
                    "category": category,
                    "coef": coef,
                    "std_err": std_err,
                    "z": z_val,
                    "p_value": p_val,
                    "ci_lower": ci_low,
                    "ci_upper": ci_high,
                    "odds_ratio": odds_ratio
                })
            
            # Extract day-of-week effects if day-related covariates exist
            # Only for logistic regression
            day_effects = extract_day_of_week_effects(coefficients, covariate_info)
            day_effects_summary = None
            targeting_recommendations = None
            
            # Debug: log day effects extraction
            print(f"Day effects extracted: {len(day_effects) if day_effects else 0} days found")
            if day_effects:
                print(f"Day effects: {[d['day'] for d in day_effects]}")
            
            if day_effects and len(day_effects) > 0:
                day_effects_summary = generate_day_effect_interpretation(day_effects)
                targeting_recommendations = generate_targeting_recommendations(day_effects)
            
            # Extract numeric covariate effects
            numeric_effects = extract_numeric_covariate_effects(coefficients, covariate_info)
            
            # Calculate marginal effects for numeric variables
            marginal_effects_data = calculate_marginal_effects(
                model, X, X_column_names, numeric_effects, covariate_info
            )
            
            # Calculate binned effects (optional, for categorical comparison view)
            binned_effects = calculate_binned_effects(
                df, numeric_effects, covariate_info, metric_column
            )
            
            # Generate unified insights summary
            all_covariates_insights = generate_all_covariates_insights(
                day_effects, numeric_effects, coefficients, covariate_info
            )
            
            return JSONResponse(content={
                "model_type": "Logistic Regression",
                "treatment_effect": treatment_coef,
                "odds_ratio": treatment_odds_ratio,
                "p_value": treatment_p_value,
                "ci_lower": treatment_ci_lower,
                "ci_upper": treatment_ci_upper,
                "coefficients": coefficients,
                "day_effects": day_effects if day_effects else None,
                "day_effects_summary": day_effects_summary,
                "targeting_recommendations": targeting_recommendations,
                "numeric_effects": numeric_effects,
                "marginal_effects_data": marginal_effects_data,
                "binned_effects": binned_effects,
                "all_covariates_insights": all_covariates_insights,
                "warnings": warnings if warnings else None,
                "n_observations": int(len(X)),
                "model_summary": {
                    "llf": float(model.llf),  # Log-likelihood
                    "aic": float(model.aic),
                    "bic": float(model.bic),
                    "pseudo_r_squared": float(model.prsquared) if hasattr(model, 'prsquared') else None
                }
            })
        else:
            # Linear regression using statsmodels
            # Use numpy arrays to avoid dtype issues
            model = sm.OLS(y_array, X_with_const).fit()
            
            # Extract results from summary table
            summary_table = model.summary2().tables[1]
            
            # Map variable names: const is first, then our X columns in order
            variable_names = ['const'] + X_column_names
            
            # Update summary table index with our variable names
            if len(summary_table) == len(variable_names):
                summary_table.index = variable_names
            else:
                # If lengths don't match, try to map what we can
                summary_table.index = variable_names[:len(summary_table)] + list(summary_table.index[len(summary_table):])
            
            # Extract treatment coefficient
            if 'treatment' not in summary_table.index:
                raise HTTPException(status_code=500, detail=f"Treatment variable not found in model results. Available variables: {summary_table.index.tolist()}, Expected: {variable_names}")
            
            treatment_row = summary_table.loc['treatment']
            treatment_coef = float(treatment_row['Coef.'])
            treatment_std_err = float(treatment_row['Std.Err.'])
            treatment_t = float(treatment_row['t'])
            treatment_p_value = float(treatment_row['P>|t|'])
            
            # Handle CI column names
            ci_col_lower = '[0.025' if '[0.025' in summary_table.columns else '[0.025]' if '[0.025]' in summary_table.columns else None
            ci_col_upper = '0.975]' if '0.975]' in summary_table.columns else '[0.975]' if '[0.975]' in summary_table.columns else None
            
            if ci_col_lower and ci_col_upper:
                treatment_ci_lower = float(treatment_row[ci_col_lower])
                treatment_ci_upper = float(treatment_row[ci_col_upper])
            else:
                # Fallback: calculate CI
                treatment_ci_lower = treatment_coef - 1.96 * treatment_std_err
                treatment_ci_upper = treatment_coef + 1.96 * treatment_std_err
            
            # Build coefficients array
            coefficients = []
            for i, (idx, row) in enumerate(summary_table.iterrows()):
                if idx == 'const' or i == 0:  # Skip intercept
                    continue
                
                # Get the actual variable name from our X columns
                var_name = variable_names[i] if i < len(variable_names) else idx
                
                coef = float(row['Coef.'])
                std_err = float(row['Std.Err.'])
                t_val = float(row['t'])
                p_val = float(row['P>|t|'])
                
                # Handle CI column names
                if ci_col_lower and ci_col_upper:
                    ci_low = float(row[ci_col_lower])
                    ci_high = float(row[ci_col_upper])
                else:
                    # Fallback: calculate CI
                    ci_low = coef - 1.96 * std_err
                    ci_high = coef + 1.96 * std_err
                
                # Find original covariate name
                original_name = var_name
                cov_type = "numeric"
                category = None
                
                for info in covariate_info:
                    if info['name'] == var_name:
                        original_name = info['original_name']
                        cov_type = info['type']
                        category = info.get('category')
                        break
                
                coefficients.append({
                    "name": var_name,
                    "original_name": original_name,
                    "type": cov_type,
                    "category": category,
                    "coef": coef,
                    "std_err": std_err,
                    "t": t_val,
                    "p_value": p_val,
                    "ci_lower": ci_low,
                    "ci_upper": ci_high
                })
            
            # Extract numeric covariate effects for linear regression too
            numeric_effects = extract_numeric_covariate_effects(coefficients, covariate_info)
            
            # Calculate marginal effects for numeric variables (for linear regression, show predicted values)
            marginal_effects_data = calculate_marginal_effects(
                model, X, X_column_names, numeric_effects, covariate_info
            )
            
            # Generate unified insights summary
            all_covariates_insights = generate_all_covariates_insights(
                None, numeric_effects, coefficients, covariate_info
            )
            
            return JSONResponse(content={
                "model_type": "Linear Regression",
                "treatment_effect": treatment_coef,
                "p_value": treatment_p_value,
                "ci_lower": treatment_ci_lower,
                "ci_upper": treatment_ci_upper,
                "coefficients": coefficients,
                "numeric_effects": numeric_effects,
                "marginal_effects_data": marginal_effects_data,
                "all_covariates_insights": all_covariates_insights,
                "warnings": warnings if warnings else None,
                "n_observations": int(len(X)),
                "model_summary": {
                    "r_squared": float(model.rsquared),
                    "adj_r_squared": float(model.rsquared_adj),
                    "f_statistic": float(model.fvalue),
                    "f_p_value": float(model.f_pvalue)
                }
            })
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error in regression analysis: {str(e)}")
        print(f"Traceback: {error_trace}")
        raise HTTPException(status_code=500, detail=f"Error in regression analysis: {str(e)}")


@app.post("/api/non-parametric-test")
async def non_parametric_test(
    file: UploadFile = File(...),
    group_column: str = Form(...),
    metric_column: str = Form(...),
    test_type: str = Form("mann_whitney"),
    group_a_name: str = Form(...),
    group_b_name: str = Form(None),
    additional_groups: str = Form(None)  # Comma-separated
):
    """
    Perform non-parametric statistical tests (Mann-Whitney U, Kruskal-Wallis)
    """
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        df.columns = df.columns.str.strip()
        
        group_column = group_column.strip()
        metric_column = metric_column.strip()
        
        if group_column not in df.columns or metric_column not in df.columns:
            raise HTTPException(status_code=400, detail="Invalid column names")
        
        df[group_column] = df[group_column].astype(str).str.strip()
        data = df[metric_column].dropna()
        
        if test_type == "mann_whitney":
            if not group_b_name:
                raise HTTPException(status_code=400, detail="group_b_name required for Mann-Whitney test")
            
            group_a_data = df[df[group_column] == group_a_name][metric_column].dropna()
            group_b_data = df[df[group_column] == group_b_name][metric_column].dropna()
            
            if len(group_a_data) < 3 or len(group_b_data) < 3:
                raise HTTPException(status_code=400, detail="Need at least 3 observations per group")
            
            statistic, p_value = stats.mannwhitneyu(group_a_data, group_b_data, alternative='two-sided')
            
            # Calculate effect size (rank-biserial correlation)
            n1, n2 = len(group_a_data), len(group_b_data)
            z_score = stats.norm.ppf(p_value / 2) if p_value > 0 else 0
            r = abs(z_score) / np.sqrt(n1 + n2)
            
            return JSONResponse(content={
                "test_type": "Mann-Whitney U Test",
                "statistic": float(statistic),
                "p_value": float(p_value),
                "significant": bool(p_value < 0.05),
                "effect_size": float(r),
                "effect_size_interpretation": "small" if r < 0.3 else "medium" if r < 0.5 else "large",
                "group_a": {
                    "name": group_a_name,
                    "size": int(n1),
                    "median": float(group_a_data.median())
                },
                "group_b": {
                    "name": group_b_name,
                    "size": int(n2),
                    "median": float(group_b_data.median())
                }
            })
        else:  # kruskal_wallis
            groups = [group_a_name]
            if group_b_name:
                groups.append(group_b_name)
            if additional_groups:
                groups.extend([g.strip() for g in additional_groups.split(',')])
            
            groups_data = [df[df[group_column] == g][metric_column].dropna() for g in groups]
            
            if any(len(g) < 3 for g in groups_data):
                raise HTTPException(status_code=400, detail="Need at least 3 observations per group")
            
            statistic, p_value = stats.kruskal(*groups_data)
            
            return JSONResponse(content={
                "test_type": "Kruskal-Wallis Test",
                "statistic": float(statistic),
                "p_value": float(p_value),
                "significant": bool(p_value < 0.05),
                "groups": [
                    {
                        "name": groups[i],
                        "size": int(len(groups_data[i])),
                        "median": float(groups_data[i].median())
                    }
                    for i in range(len(groups))
                ]
            })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in non-parametric test: {str(e)}")


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


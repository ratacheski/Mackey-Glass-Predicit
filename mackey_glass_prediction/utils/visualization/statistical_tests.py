"""
Statistical tests: Kolmogorov-Smirnov and autocorrelation analysis
"""
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend that only saves files
import matplotlib.pyplot as plt
plt.ioff()  # Disable interactive mode

import numpy as np
from scipy import stats
from statsmodels.tsa.stattools import acf
from scipy.stats import gaussian_kde
from statsmodels.stats.diagnostic import acorr_ljungbox
import os

from .utils import ensure_output_dir, print_save_message, get_colors_and_styles


def plot_ks_test_analysis(actuals, predictions, save_path=None, title="Two-Sample Kolmogorov-Smirnov Test", alpha=0.05):
    """
    Visualize in detail the two-sample Kolmogorov-Smirnov test
    
    Args:
        actuals: Actual values
        predictions: Model predictions
        save_path: Path to save the plot
        title: Plot title
        alpha: Significance level (default: 0.05)
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # Ensure they are numpy arrays
    actuals = np.array(actuals).flatten()
    predictions = np.array(predictions).flatten()
    
    # Perform two-sample KS test
    ks_statistic, ks_pvalue = stats.ks_2samp(predictions, actuals)
    
    # Determine test result
    reject_h0 = ks_pvalue < alpha
    test_conclusion = "REJECT H₀" if reject_h0 else "DO NOT REJECT H₀"
    conclusion_color = "red" if reject_h0 else "green"
    
    # Range for CDFs
    combined_data = np.concatenate([actuals, predictions])
    x_range = np.linspace(np.min(combined_data), np.max(combined_data), 1000)
    
    # ===== SUBPLOT 1: CDFs and Maximum Difference =====
    ax1.set_title(f'{title}\nKS Statistic = {ks_statistic:.6f} | p-value = {ks_pvalue:.6f} | {test_conclusion}', 
                 fontsize=14, fontweight='bold', color=conclusion_color)
    
    # Calculate empirical CDFs
    cdf_actuals = np.array([np.mean(actuals <= x) for x in x_range])
    cdf_predictions = np.array([np.mean(predictions <= x) for x in x_range])
    
    # Plot CDFs
    ax1.plot(x_range, cdf_actuals, 'b-', linewidth=3, label='CDF - Actual Values', alpha=0.8)
    ax1.plot(x_range, cdf_predictions, 'r-', linewidth=3, label='CDF - Predictions', alpha=0.8)
    
    # Find maximum difference point
    diff = np.abs(cdf_actuals - cdf_predictions)
    max_diff_idx = np.argmax(diff)
    max_diff_x = x_range[max_diff_idx]
    max_diff_y1 = cdf_actuals[max_diff_idx]
    max_diff_y2 = cdf_predictions[max_diff_idx]
    
    # Highlight maximum difference
    ax1.plot([max_diff_x, max_diff_x], [max_diff_y1, max_diff_y2], 
             'k-', linewidth=4, alpha=0.8, label=f'Maximum Difference = {ks_statistic:.6f}')
    ax1.plot(max_diff_x, max_diff_y1, 'bo', markersize=10, markeredgecolor='black', markeredgewidth=2)
    ax1.plot(max_diff_x, max_diff_y2, 'ro', markersize=10, markeredgecolor='black', markeredgewidth=2)
    
    # Area between curves to visualize differences
    ax1.fill_between(x_range, cdf_actuals, cdf_predictions, alpha=0.2, color='gray', 
                    label='Differences between CDFs')
    
    # Vertical line at maximum difference point
    ax1.axvline(x=max_diff_x, color='black', linestyle='--', alpha=0.6, 
               label=f'x = {max_diff_x:.4f}')
    
    ax1.set_xlabel('Value', fontsize=12)
    ax1.set_ylabel('Cumulative Probability', fontsize=12)
    ax1.legend(fontsize=10, loc='center right')
    ax1.grid(True, alpha=0.3)
    
    # Add test information box
    info_text = (
        f"Kolmogorov-Smirnov Test (two samples)\n"
        f"H₀: Same probability distribution\n"
        f"H₁: Different distributions\n"
        f"Significance level (α): {alpha}\n"
        f"KS Statistic: {ks_statistic:.6f}\n"
        f"p-value: {ks_pvalue:.6f}\n"
        f"Maximum difference at x = {max_diff_x:.4f}\n"
        f"Actual sample size: {len(actuals)}\n"
        f"Prediction sample size: {len(predictions)}"
    )
    
    ax1.text(0.02, 0.98, info_text, transform=ax1.transAxes, 
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9),
             verticalalignment='top', fontsize=9, fontfamily='monospace')
    
    # ===== SUBPLOT 2: Comparative Histograms =====
    ax2.set_title('Sample Distributions (Normalized Histograms)', fontsize=12, fontweight='bold')
    
    # Calculate shared bins
    bins = np.linspace(np.min(combined_data), np.max(combined_data), 50)
    
    # Histograms
    ax2.hist(actuals, bins=bins, density=True, alpha=0.6, color='blue', 
            label=f'Actual Values (n={len(actuals)})', edgecolor='darkblue')
    ax2.hist(predictions, bins=bins, density=True, alpha=0.6, color='red',
            label=f'Predictions (n={len(predictions)})', edgecolor='darkred')
    
    # Add KDE for smoothing
    try:
        kde_actuals = gaussian_kde(actuals)
        kde_predictions = gaussian_kde(predictions)
        
        kde_x = np.linspace(np.min(combined_data), np.max(combined_data), 200)
        ax2.plot(kde_x, kde_actuals(kde_x), 'b-', linewidth=2, alpha=0.8, label='KDE - Actual')
        ax2.plot(kde_x, kde_predictions(kde_x), 'r-', linewidth=2, alpha=0.8, label='KDE - Predictions')
    except:
        pass
    
    # Vertical line at maximum difference point
    ax2.axvline(x=max_diff_x, color='black', linestyle='--', alpha=0.8, linewidth=2,
               label=f'Maximum difference (x = {max_diff_x:.4f})')
    
    ax2.set_xlabel('Value', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Result interpretation
    if reject_h0:
        interpretation = (
            f"CONCLUSION: We reject H₀ (p = {ks_pvalue:.6f} < α = {alpha})\n"
            f"There is SUFFICIENT statistical evidence that the distributions\n"
            f"of predictions and actual values are DIFFERENT.\n"
            f"The model does NOT adequately reproduce the data distribution."
        )
        interp_color = 'lightcoral'
    else:
        interpretation = (
            f"CONCLUSION: We do not reject H₀ (p = {ks_pvalue:.6f} ≥ α = {alpha})\n"
            f"There is NOT sufficient statistical evidence that the distributions\n"
            f"of predictions and actual values are different.\n"
            f"The model adequately reproduces the data distribution."
        )
        interp_color = 'lightgreen'
    
    ax2.text(0.02, 0.98, interpretation, transform=ax2.transAxes, 
             bbox=dict(boxstyle='round', facecolor=interp_color, alpha=0.9),
             verticalalignment='top', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        ensure_output_dir(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print_save_message(save_path, "Kolmogorov-Smirnov Test")
    
    plt.close()  # Close figure to free memory
    
    # Return test results
    return {
        'ks_statistic': ks_statistic,
        'p_value': ks_pvalue,
        'reject_h0': reject_h0,
        'max_diff_location': max_diff_x,
        'alpha': alpha,
        'conclusion': test_conclusion
    }


def plot_autocorrelation_analysis(actuals, predictions, save_path=None, 
                                  title="Autocorrelation Comparison", 
                                  max_lags=40, alpha=0.05):
    """
    Compare autocorrelation function between actual values and predictions
    
    Args:
        actuals: Actual values (time series)
        predictions: Model predictions (time series)
        save_path: Path to save the plot
        title: Plot title
        max_lags: Maximum number of lags to calculate autocorrelation
        alpha: Significance level for confidence intervals
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
    
    # Ensure they are numpy arrays
    actuals = np.array(actuals).flatten()
    predictions = np.array(predictions).flatten()
    
    # Check if series have sufficient size
    min_length = min(len(actuals), len(predictions))
    max_lags = min(max_lags, min_length // 4)  # Conservative rule for lags
    
    try:
        # ===== SUBPLOT 1: Autocorrelation of Actual Values =====
        ax1.set_title(f'Autocorrelation - Actual Values (n={len(actuals)})', 
                     fontsize=12, fontweight='bold')
        
        # Calculate autocorrelation of actual values
        acf_actuals = acf(actuals, nlags=max_lags, alpha=alpha, fft=True)
        lags = np.arange(len(acf_actuals[0]))
        
        # Plot autocorrelation
        ax1.plot(lags, acf_actuals[0], 'b-', linewidth=2, label='ACF - Actual Values', 
                marker='o', markersize=4, alpha=0.8)
        
        # Add confidence intervals
        if len(acf_actuals) > 1:  # If confidence intervals were calculated
            lower_conf = acf_actuals[1][:, 0] - acf_actuals[0]
            upper_conf = acf_actuals[1][:, 1] - acf_actuals[0]
            ax1.fill_between(lags, lower_conf, upper_conf, alpha=0.2, color='blue', 
                           label=f'CI {(1-alpha)*100:.0f}%')
        
        # Reference line at zero
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Statistical significance lines (approximate)
        significance_line = 1.96 / np.sqrt(len(actuals))
        ax1.axhline(y=significance_line, color='red', linestyle='--', alpha=0.6, 
                   label=f'Limit ±{significance_line:.3f}')
        ax1.axhline(y=-significance_line, color='red', linestyle='--', alpha=0.6)
        
        ax1.set_xlabel('Lag', fontsize=10)
        ax1.set_ylabel('Autocorrelation', fontsize=10)
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(-1, 1)
        
        # ===== SUBPLOT 2: Autocorrelation of Predictions =====
        ax2.set_title(f'Autocorrelation - Predictions (n={len(predictions)})', 
                     fontsize=12, fontweight='bold')
        
        # Calculate autocorrelation of predictions
        acf_predictions = acf(predictions, nlags=max_lags, alpha=alpha, fft=True)
        
        # Plot autocorrelation
        ax2.plot(lags, acf_predictions[0], 'r-', linewidth=2, label='ACF - Predictions', 
                marker='s', markersize=4, alpha=0.8)
        
        # Add confidence intervals
        if len(acf_predictions) > 1:
            lower_conf = acf_predictions[1][:, 0] - acf_predictions[0]
            upper_conf = acf_predictions[1][:, 1] - acf_predictions[0]
            ax2.fill_between(lags, lower_conf, upper_conf, alpha=0.2, color='red', 
                           label=f'CI {(1-alpha)*100:.0f}%')
        
        # Reference and significance lines
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        significance_line_pred = 1.96 / np.sqrt(len(predictions))
        ax2.axhline(y=significance_line_pred, color='red', linestyle='--', alpha=0.6, 
                   label=f'Limit ±{significance_line_pred:.3f}')
        ax2.axhline(y=-significance_line_pred, color='red', linestyle='--', alpha=0.6)
        
        ax2.set_xlabel('Lag', fontsize=10)
        ax2.set_ylabel('Autocorrelation', fontsize=10)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(-1, 1)
        
        # ===== SUBPLOT 3: Direct Comparison =====
        ax3.set_title('Direct Comparison of Autocorrelations', fontsize=12, fontweight='bold')
        
        # Plot both autocorrelations together
        ax3.plot(lags, acf_actuals[0], 'b-', linewidth=2.5, label='ACF - Actual Values', 
                marker='o', markersize=5, alpha=0.8)
        ax3.plot(lags, acf_predictions[0], 'r--', linewidth=2.5, label='ACF - Predictions', 
                marker='s', markersize=5, alpha=0.8)
        
        # Area between curves to show differences
        ax3.fill_between(lags, acf_actuals[0], acf_predictions[0], alpha=0.2, color='gray', 
                        label='Difference between ACFs')
        
        # Reference lines
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax3.axhline(y=significance_line, color='gray', linestyle=':', alpha=0.6, 
                   label=f'Significance limit ±{significance_line:.3f}')
        ax3.axhline(y=-significance_line, color='gray', linestyle=':', alpha=0.6)
        
        ax3.set_xlabel('Lag', fontsize=10)
        ax3.set_ylabel('Autocorrelation', fontsize=10)
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(-1, 1)
        
        # ===== COMPARATIVE METRICS CALCULATION =====
        # Mean squared difference between autocorrelations
        mse_acf = np.mean((acf_actuals[0] - acf_predictions[0])**2)
        
        # Correlation between the two autocorrelation functions
        corr_acf = np.corrcoef(acf_actuals[0], acf_predictions[0])[0, 1]
        
        # Maximum absolute difference
        max_diff_acf = np.max(np.abs(acf_actuals[0] - acf_predictions[0]))
        max_diff_lag = lags[np.argmax(np.abs(acf_actuals[0] - acf_predictions[0]))]
        
        # Ljung-Box test for residual autocorrelation
        residuals = predictions - actuals[:len(predictions)] if len(predictions) <= len(actuals) else predictions[:len(actuals)] - actuals
        try:
            ljung_box = acorr_ljungbox(residuals, lags=min(10, len(residuals)//5), return_df=True)
            ljung_box_pvalue = ljung_box['lb_pvalue'].iloc[-1]  # Last p-value
        except:
            ljung_box_pvalue = np.nan
        
        # Metrics box in subplot 3
        metrics_text = (
            f"COMPARATIVE METRICS:\n"
            f"ACF MSE: {mse_acf:.6f}\n"
            f"ACF Correlation: {corr_acf:.4f}\n"
            f"Maximum difference: {max_diff_acf:.4f}\n"
            f"Max diff lag: {max_diff_lag}\n"
            f"Ljung-Box p-value: {ljung_box_pvalue:.4f}" if not np.isnan(ljung_box_pvalue) else "Ljung-Box: N/A"
        )
        
        ax3.text(0.02, 0.98, metrics_text, transform=ax3.transAxes, 
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9),
                verticalalignment='top', fontsize=9, fontfamily='monospace')
        
        # ===== INTERPRETATION =====
        # Classify the quality of temporal structure reproduction
        if corr_acf > 0.9 and mse_acf < 0.1:
            interpretation = "EXCELLENT: Temporal structure very well reproduced"
            interp_color = 'lightgreen'
        elif corr_acf > 0.7 and mse_acf < 0.2:
            interpretation = "GOOD: Temporal structure well reproduced"
            interp_color = 'lightblue'
        elif corr_acf > 0.5 and mse_acf < 0.4:
            interpretation = "MODERATE: Temporal structure partially reproduced"
            interp_color = 'lightyellow'
        else:
            interpretation = "POOR: Temporal structure poorly reproduced"
            interp_color = 'lightcoral'
        
        ax3.text(0.02, 0.02, f"ASSESSMENT: {interpretation}", transform=ax3.transAxes, 
                bbox=dict(boxstyle='round', facecolor=interp_color, alpha=0.9),
                verticalalignment='bottom', fontsize=10, fontweight='bold')
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            ensure_output_dir(save_path)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print_save_message(save_path, "Autocorrelation analysis")
        
        plt.close()  # Close figure to free memory
        
        # Return metrics
        return {
            'acf_reals': acf_actuals[0],
            'acf_predictions': acf_predictions[0],
            'lags': lags,
            'mse_acf': mse_acf,
            'correlation_acf': corr_acf,
            'max_difference': max_diff_acf,
            'max_diff_lag': max_diff_lag,
            'ljung_box_pvalue': ljung_box_pvalue if not np.isnan(ljung_box_pvalue) else None
        }
        
    except Exception as e:
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, f'Error calculating autocorrelation:\n{str(e)}', 
                ha='center', va='center', transform=plt.gca().transAxes,
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8),
                fontsize=12)
        plt.title(f"Error in Autocorrelation Analysis - {title}")
        plt.axis('off')
        
        if save_path:
            ensure_output_dir(save_path)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()  # Close figure to free memory
        return None 
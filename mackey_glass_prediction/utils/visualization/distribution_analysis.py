"""
Distributional analyses: QQ-Plot, CDF and PDF
"""
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend that only saves files
import matplotlib.pyplot as plt
plt.ioff()  # Disable interactive mode

import numpy as np
from scipy import stats
from scipy.stats import gaussian_kde
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from .utils import ensure_output_dir, print_save_message, get_colors_and_styles


def plot_qq_analysis(actuals, predictions, save_path=None, 
                    title="QQ-Plot: Prediction Quantiles vs Actual Values"):
    """
    Create QQ-Plot (Quantile-Quantile) to compare distributions of predictions vs actual values
    
    Args:
        actuals: Actual values
        predictions: Model predictions
        save_path: Path to save the plot
        title: Plot title
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Ensure they are numpy arrays and flatten
    actuals = np.array(actuals).flatten()
    predictions = np.array(predictions).flatten()
    
    # ===== SUBPLOT 1: Main QQ-Plot =====
    ax1.set_title(title, fontsize=14, fontweight='bold')
    
    # Sort data to calculate quantiles
    actuals_sorted = np.sort(actuals)
    predictions_sorted = np.sort(predictions)
    
    # If sizes are different, interpolate to match
    if len(actuals_sorted) != len(predictions_sorted):
        # Use smallest size as reference
        min_size = min(len(actuals_sorted), len(predictions_sorted))
        
        # Create uniform quantiles
        quantiles = np.linspace(0, 1, min_size)
        
        # Interpolate to get corresponding quantiles
        actuals_quantiles = np.quantile(actuals_sorted, quantiles)
        predictions_quantiles = np.quantile(predictions_sorted, quantiles)
    else:
        # If same size, use directly
        actuals_quantiles = actuals_sorted
        predictions_quantiles = predictions_sorted
    
    # Plot QQ-plot
    ax1.scatter(actuals_quantiles, predictions_quantiles, alpha=0.6, s=30, color='blue', edgecolors='darkblue')
    
    # Perfect reference line (y=x)
    min_val = min(np.min(actuals_quantiles), np.min(predictions_quantiles))
    max_val = max(np.max(actuals_quantiles), np.max(predictions_quantiles))
    ax1.plot([min_val, max_val], [min_val, max_val], 'r-', linewidth=3, 
             label='Perfect Line (identical distributions)', alpha=0.8)
    
    # Regression line through quantiles
    try:
        slope, intercept, r_value, p_value, std_err = stats.linregress(actuals_quantiles, predictions_quantiles)
        regression_line = slope * actuals_quantiles + intercept
        ax1.plot(actuals_quantiles, regression_line, 'g--', linewidth=2, 
                 label=f'Regression (R²={r_value**2:.4f})', alpha=0.8)
        
        # Add line equation
        ax1.text(0.05, 0.95, f'y = {slope:.3f}x + {intercept:.3f}\nR² = {r_value**2:.4f}', 
                transform=ax1.transAxes, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7),
                verticalalignment='top', fontsize=10, fontfamily='monospace')
        
    except Exception as e:
        print(f"Warning: Error calculating regression: {e}")
    
    ax1.set_xlabel('Quantiles - Actual Values', fontsize=12)
    ax1.set_ylabel('Quantiles - Predictions', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Add metrics to plot
    r2 = r2_score(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mae = mean_absolute_error(actuals, predictions)
    
    # Normality test on quantile residuals
    residuals_qq = predictions_quantiles - actuals_quantiles
    shapiro_stat, shapiro_p = stats.shapiro(residuals_qq) if len(residuals_qq) <= 5000 else (np.nan, np.nan)
    
    metrics_text = (
        f'Distributional Metrics:\n'
        f'Global R²: {r2:.4f}\n'
        f'RMSE: {rmse:.4f}\n'
        f'MAE: {mae:.4f}\n'
        f'Shapiro-Wilk (QQ residuals):\n'
        f'  Statistic: {shapiro_stat:.4f}\n'
        f'  p-value: {shapiro_p:.4f}' if not np.isnan(shapiro_stat) else 'Shapiro-Wilk: N/A (large sample)'
    )
    
    ax1.text(0.02, 0.02, metrics_text, transform=ax1.transAxes, 
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
             verticalalignment='bottom', fontsize=9, fontfamily='monospace')
    
    # ===== SUBPLOT 2: Deviation Analysis by Quantile =====
    ax2.set_title('Deviation Analysis by Quantile', fontsize=14, fontweight='bold')
    
    # Calculate relative deviations
    desvios = (predictions_quantiles - actuals_quantiles) / actuals_quantiles * 100
    quantil_positions = np.linspace(0, 100, len(desvios))
    
    # Plot deviations
    ax2.plot(quantil_positions, desvios, 'b-', linewidth=2, alpha=0.8, label='Relative Deviation (%)')
    ax2.axhline(y=0, color='red', linestyle='-', linewidth=2, alpha=0.8, label='Zero (perfect)')
    ax2.fill_between(quantil_positions, desvios, 0, alpha=0.3, color='blue')
    
    # Highlight extreme quantiles (10% and 90%)
    q10_idx = int(len(desvios) * 0.1)
    q90_idx = int(len(desvios) * 0.9)
    
    ax2.scatter([10, 90], [desvios[q10_idx], desvios[q90_idx]], 
               color='red', s=100, zorder=5, label='10% and 90% Quantiles')
    
    ax2.set_xlabel('Percentile (%)', fontsize=12)
    ax2.set_ylabel('Relative Deviation (%)', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Deviation analysis
    desvio_medio = np.mean(np.abs(desvios))
    desvio_max = np.max(np.abs(desvios))
    desvio_std = np.std(desvios)
    
    # Identify where the largest deviations are
    worst_quantile = quantil_positions[np.argmax(np.abs(desvios))]
    
    analysis_text = (
        f'Deviation Analysis:\n'
        f'Mean deviation: ±{desvio_medio:.2f}%\n'
        f'Max deviation: ±{desvio_max:.2f}%\n'
        f'Std deviation: {desvio_std:.2f}%\n'
        f'Worst quantile: {worst_quantile:.0f}%\n'
        f'Q10 deviation: {desvios[q10_idx]:+.2f}%\n'
        f'Q90 deviation: {desvios[q90_idx]:+.2f}%'
    )
    
    ax2.text(0.02, 0.98, analysis_text, transform=ax2.transAxes, 
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
             verticalalignment='top', fontsize=9, fontfamily='monospace')
    
    # Interpretation
    if desvio_medio < 5:
        interpretation = "EXCELLENT: Very similar distributions"
        interp_color = 'lightgreen'
    elif desvio_medio < 10:
        interpretation = "GOOD: Similar distributions"
        interp_color = 'lightblue'
    elif desvio_medio < 20:
        interpretation = "MODERATE: Some distributional differences"
        interp_color = 'lightyellow'
    else:
        interpretation = "POOR: Significant distributional differences"
        interp_color = 'lightcoral'
    
    ax2.text(0.02, 0.02, f'INTERPRETATION: {interpretation}', transform=ax2.transAxes, 
             bbox=dict(boxstyle='round', facecolor=interp_color, alpha=0.9),
             verticalalignment='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        ensure_output_dir(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print_save_message(save_path, "Q-Q Analysis")
    
    plt.close()  # Close figure to free memory
    
    # Return QQ-plot metrics
    return {
        'qq_r2': r_value**2 if 'r_value' in locals() else np.nan,
        'qq_slope': slope if 'slope' in locals() else np.nan,
        'qq_intercept': intercept if 'intercept' in locals() else np.nan,
        'mean_relative_deviation': desvio_medio,
        'max_relative_deviation': desvio_max,
        'worst_quantile': worst_quantile,
        'shapiro_stat': shapiro_stat if not np.isnan(shapiro_stat) else None,
        'shapiro_pvalue': shapiro_p if not np.isnan(shapiro_p) else None
    }


def plot_cdf_comparison(actuals, predictions, save_path=None, 
                       title="CDF Comparison - Cumulative Distribution Function"):
    """
    Plot comparison of Cumulative Distribution Functions (CDF)
    
    Args:
        actuals: Actual values
        predictions: Model predictions
        save_path: Path to save the plot
        title: Plot title
    """
    plt.figure(figsize=(12, 8))
    
    # Ensure they are numpy arrays
    actuals = np.array(actuals).flatten()
    predictions = np.array(predictions).flatten()
    
    # Combined range for evaluation
    combined_range = np.linspace(
        min(np.min(predictions), np.min(actuals)),
        max(np.max(predictions), np.max(actuals)),
        1000
    )
    
    # Calculate empirical CDFs
    cdf_actuals = np.array([np.mean(actuals <= x) for x in combined_range])
    cdf_predictions = np.array([np.mean(predictions <= x) for x in combined_range])
    
    # Plot CDFs
    plt.plot(combined_range, cdf_actuals, 'b-', linewidth=2.5, label='Actual Values', alpha=0.8)
    plt.plot(combined_range, cdf_predictions, 'r--', linewidth=2.5, label='Predictions', alpha=0.8)
    
    # Add area between curves
    plt.fill_between(combined_range, cdf_actuals, cdf_predictions, alpha=0.2, color='gray', label='Difference')
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Value', fontsize=12)
    plt.ylabel('Cumulative Probability', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Add metrics to plot
    ks_statistic, ks_pvalue = stats.ks_2samp(predictions, actuals)
    fda_distance = np.mean(np.abs(cdf_predictions - cdf_actuals))
    
    metrics_text = f'KS Statistic: {ks_statistic:.4f}\nKS p-value: {ks_pvalue:.4f}\nMean Distance: {fda_distance:.4f}'
    plt.text(0.02, 0.98, metrics_text, transform=plt.gca().transAxes, 
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
             verticalalignment='top', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        ensure_output_dir(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print_save_message(save_path, "CDF Comparison")
    
    plt.close()  # Close figure to free memory


def plot_pdf_comparison(actuals, predictions, save_path=None, 
                       title="PDF Comparison - Probability Density Function"):
    """
    Plot comparison of Probability Density Functions (PDF) using KDE
    
    Args:
        actuals: Actual values
        predictions: Model predictions
        save_path: Path to save the plot
        title: Plot title
    """
    plt.figure(figsize=(12, 8))
    
    # Ensure they are numpy arrays
    actuals = np.array(actuals).flatten()
    predictions = np.array(predictions).flatten()
    
    try:
        # Combined range for evaluation
        combined_range = np.linspace(
            min(np.min(predictions), np.min(actuals)),
            max(np.max(predictions), np.max(actuals)),
            1000
        )
        
        # Kernel Density Estimation
        kde_actuals = gaussian_kde(actuals)
        kde_predictions = gaussian_kde(predictions)
        
        # Evaluate PDFs
        pdf_actuals = kde_actuals(combined_range)
        pdf_predictions = kde_predictions(combined_range)
        
        # Plot PDFs
        plt.plot(combined_range, pdf_actuals, 'b-', linewidth=2.5, label='Actual Values', alpha=0.8)
        plt.plot(combined_range, pdf_predictions, 'r--', linewidth=2.5, label='Predictions', alpha=0.8)
        
        # Add area between curves
        plt.fill_between(combined_range, pdf_actuals, pdf_predictions, alpha=0.2, color='gray', label='Difference')
        
        # Add normalized histograms for context
        plt.hist(actuals, bins=50, density=True, alpha=0.3, color='blue', label='Actual Hist.')
        plt.hist(predictions, bins=50, density=True, alpha=0.3, color='red', label='Prediction Hist.')
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Value', fontsize=12)
        plt.ylabel('Probability Density', fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        
        # Calculate PDF metrics
        fdp_l2_distance = np.sqrt(np.trapz((pdf_predictions - pdf_actuals)**2, combined_range))
        
        # Normalize PDFs for JS divergence
        pdf_pred_norm = pdf_predictions / np.trapz(pdf_predictions, combined_range)
        pdf_actual_norm = pdf_actuals / np.trapz(pdf_actuals, combined_range)
        pdf_mean = 0.5 * (pdf_pred_norm + pdf_actual_norm)
        
        # JS divergence with protection
        epsilon = 1e-10
        kl_pred_mean = np.trapz(pdf_pred_norm * np.log((pdf_pred_norm + epsilon) / (pdf_mean + epsilon)), combined_range)
        kl_actual_mean = np.trapz(pdf_actual_norm * np.log((pdf_actual_norm + epsilon) / (pdf_mean + epsilon)), combined_range)
        js_divergence = 0.5 * (kl_pred_mean + kl_actual_mean)
        
        # Add metrics to plot
        metrics_text = f'L2 Distance: {fdp_l2_distance:.4f}\nJS Divergence: {js_divergence:.4f}'
        plt.text(0.02, 0.98, metrics_text, transform=plt.gca().transAxes, 
                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
                 verticalalignment='top', fontsize=10)
        
    except Exception as e:
        plt.text(0.5, 0.5, f'Error calculating KDE: {str(e)}', 
                transform=plt.gca().transAxes, ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        print(f"Warning: Error plotting PDF: {e}")
    
    plt.tight_layout()
    
    if save_path:
        ensure_output_dir(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print_save_message(save_path, "PDF Comparison")
    
    plt.close()  # Close figure to free memory


def plot_distribution_analysis(actuals, predictions, save_path=None, 
                              title_prefix="Distributional Analysis"):
    """
    Create complete distribution analysis (CDF + PDF) in a single figure
    
    Args:
        actuals: Actual values
        predictions: Model predictions
        save_path: Path to save the plot
        title_prefix: Title prefix
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Ensure they are numpy arrays
    actuals = np.array(actuals).flatten()
    predictions = np.array(predictions).flatten()
    
    # Combined range
    combined_range = np.linspace(
        min(np.min(predictions), np.min(actuals)),
        max(np.max(predictions), np.max(actuals)),
        1000
    )
    
    # ===== SUBPLOT 1: CDF =====
    ax1.set_title(f'{title_prefix} - CDF (Cumulative Distribution Function)', fontsize=12, fontweight='bold')
    
    # Calculate empirical CDFs
    cdf_actuals = np.array([np.mean(actuals <= x) for x in combined_range])
    cdf_predictions = np.array([np.mean(predictions <= x) for x in combined_range])
    
    # Plot CDFs
    ax1.plot(combined_range, cdf_actuals, 'b-', linewidth=2.5, label='Actual Values', alpha=0.8)
    ax1.plot(combined_range, cdf_predictions, 'r--', linewidth=2.5, label='Predictions', alpha=0.8)
    ax1.fill_between(combined_range, cdf_actuals, cdf_predictions, alpha=0.2, color='gray', label='Difference')
    
    ax1.set_xlabel('Value', fontsize=10)
    ax1.set_ylabel('Cumulative Probability', fontsize=10)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # CDF metrics
    ks_statistic, ks_pvalue = stats.ks_2samp(predictions, actuals)
    fda_distance = np.mean(np.abs(cdf_predictions - cdf_actuals))
    metrics_text1 = f'KS: {ks_statistic:.4f}\np-val: {ks_pvalue:.4f}\nDist: {fda_distance:.4f}'
    ax1.text(0.02, 0.98, metrics_text1, transform=ax1.transAxes, 
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
             verticalalignment='top', fontsize=9)
    
    # ===== SUBPLOT 2: PDF =====
    ax2.set_title(f'{title_prefix} - PDF (Probability Density Function)', fontsize=12, fontweight='bold')
    
    try:
        # Kernel Density Estimation
        kde_actuals = gaussian_kde(actuals)
        kde_predictions = gaussian_kde(predictions)
        
        # Evaluate PDFs
        pdf_actuals = kde_actuals(combined_range)
        pdf_predictions = kde_predictions(combined_range)
        
        # Plot PDFs
        ax2.plot(combined_range, pdf_actuals, 'b-', linewidth=2.5, label='Actual Values', alpha=0.8)
        ax2.plot(combined_range, pdf_predictions, 'r--', linewidth=2.5, label='Predictions', alpha=0.8)
        ax2.fill_between(combined_range, pdf_actuals, pdf_predictions, alpha=0.2, color='gray', label='Difference')
        
        # Context histograms
        ax2.hist(actuals, bins=30, density=True, alpha=0.3, color='blue', label='Actual Hist.')
        ax2.hist(predictions, bins=30, density=True, alpha=0.3, color='red', label='Prediction Hist.')
        
        ax2.set_xlabel('Value', fontsize=10)
        ax2.set_ylabel('Probability Density', fontsize=10)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        # PDF metrics
        fdp_l2_distance = np.sqrt(np.trapz((pdf_predictions - pdf_actuals)**2, combined_range))
        
        # JS divergence
        pdf_pred_norm = pdf_predictions / np.trapz(pdf_predictions, combined_range)
        pdf_actual_norm = pdf_actuals / np.trapz(pdf_actuals, combined_range)
        pdf_mean = 0.5 * (pdf_pred_norm + pdf_actual_norm)
        
        epsilon = 1e-10
        kl_pred_mean = np.trapz(pdf_pred_norm * np.log((pdf_pred_norm + epsilon) / (pdf_mean + epsilon)), combined_range)
        kl_actual_mean = np.trapz(pdf_actual_norm * np.log((pdf_actual_norm + epsilon) / (pdf_mean + epsilon)), combined_range)
        js_divergence = 0.5 * (kl_pred_mean + kl_actual_mean)
        
        metrics_text2 = f'L2: {fdp_l2_distance:.4f}\nJS: {js_divergence:.4f}'
        ax2.text(0.02, 0.98, metrics_text2, transform=ax2.transAxes, 
                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
                 verticalalignment='top', fontsize=9)
        
    except Exception as e:
        ax2.text(0.5, 0.5, f'KDE Error: {str(e)}', transform=ax2.transAxes, ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        ensure_output_dir(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print_save_message(save_path, "Distribution analysis")
    
    plt.close()  # Close figure to free memory


def plot_multi_model_cdf_comparison(results_dict, save_path=None, 
                                   title="CDF Comparison - All Models"):
    """
    Compare Cumulative Distribution Functions (CDF) of multiple models
    
    Args:
        results_dict: Dictionary with model results
        save_path: Path to save the plot
        title: Plot title
    """
    plt.figure(figsize=(15, 10))
    
    # Colors for different models
    colors, linestyles = get_colors_and_styles(len(results_dict))
    
    # Find global range
    all_actuals = []
    all_predictions = []
    
    for model_name, results in results_dict.items():
        if 'actuals' in results and 'predictions' in results:
            all_actuals.extend(results['actuals'].flatten())
            all_predictions.extend(results['predictions'].flatten())
    
    global_range = np.linspace(min(all_actuals + all_predictions), 
                              max(all_actuals + all_predictions), 1000)
    
    # Plot CDF of actual values (common for all)
    actuals_combined = np.array(all_actuals)
    cdf_actuals = np.array([np.mean(actuals_combined <= x) for x in global_range])
    plt.plot(global_range, cdf_actuals, 'black', linewidth=3, 
             label='Actual Values', alpha=0.8, zorder=10)
    
    # Plot CDF of predictions from each model
    for i, (model_name, results) in enumerate(results_dict.items()):
        if 'predictions' in results:
            predictions = np.array(results['predictions']).flatten()
            cdf_predictions = np.array([np.mean(predictions <= x) for x in global_range])
            
            plt.plot(global_range, cdf_predictions, color=colors[i], linestyle=linestyles[i],
                    linewidth=2.5, label=f'{model_name}', alpha=0.8)
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Value', fontsize=12)
    plt.ylabel('Cumulative Probability', fontsize=12)
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3)
    
    # Add informative text
    plt.text(0.02, 0.02, f'Comparison based on {len(results_dict)} models\nBlack line: actual distribution', 
             transform=plt.gca().transAxes, 
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
             fontsize=10, verticalalignment='bottom')
    
    plt.tight_layout()
    
    if save_path:
        ensure_output_dir(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print_save_message(save_path, "CDF comparison multiple models")
    
    plt.close()  # Close figure to free memory


def plot_multi_model_pdf_comparison(results_dict, save_path=None, 
                                   title="PDF Comparison - All Models"):
    """
    Compare Probability Density Functions (PDF) of multiple models
    
    Args:
        results_dict: Dictionary with model results
        save_path: Path to save the plot
        title: Plot title
    """
    plt.figure(figsize=(15, 10))
    
    # Colors for different models
    colors, linestyles = get_colors_and_styles(len(results_dict))
    
    # Find global range
    all_actuals = []
    all_predictions = []
    
    for model_name, results in results_dict.items():
        if 'actuals' in results and 'predictions' in results:
            all_actuals.extend(results['actuals'].flatten())
            all_predictions.extend(results['predictions'].flatten())
    
    global_range = np.linspace(min(all_actuals + all_predictions), 
                              max(all_actuals + all_predictions), 1000)
    
    try:
        # Plot PDF of actual values (common for all)
        actuals_combined = np.array(all_actuals)
        kde_actuals = gaussian_kde(actuals_combined)
        pdf_actuals = kde_actuals(global_range)
        plt.plot(global_range, pdf_actuals, 'black', linewidth=3, 
                 label='Actual Values', alpha=0.8, zorder=10)
        
        # Plot PDF of predictions from each model
        for i, (model_name, results) in enumerate(results_dict.items()):
            if 'predictions' in results:
                try:
                    predictions = np.array(results['predictions']).flatten()
                    kde_predictions = gaussian_kde(predictions)
                    pdf_predictions = kde_predictions(global_range)
                    
                    plt.plot(global_range, pdf_predictions, color=colors[i], linestyle=linestyles[i],
                            linewidth=2.5, label=f'{model_name}', alpha=0.8)
                except Exception as e:
                    print(f"Warning: Error calculating KDE for {model_name}: {e}")
                    continue
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Value', fontsize=12)
        plt.ylabel('Probability Density', fontsize=12)
        plt.legend(fontsize=10, loc='best')
        plt.grid(True, alpha=0.3)
        
        # Add histogram of actual values for context
        plt.hist(actuals_combined, bins=50, density=True, alpha=0.2, color='black', 
                label='Actual Hist.', zorder=1)
        
        # Add informative text
        plt.text(0.02, 0.98, f'Comparison based on {len(results_dict)} models\nBlack line: actual distribution', 
                 transform=plt.gca().transAxes, 
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                 fontsize=10, verticalalignment='top')
        
    except Exception as e:
        plt.text(0.5, 0.5, f'Error generating PDF comparison: {str(e)}', 
                transform=plt.gca().transAxes, ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
        print(f"Error plotting PDF comparison: {e}")
    
    plt.tight_layout()
    
    if save_path:
        ensure_output_dir(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print_save_message(save_path, "PDF comparison multiple models")
    
    plt.close()  # Close figure to free memory 
"""
Comprehensive report generation with multiple visualizations
"""
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend that only saves files
import matplotlib.pyplot as plt
plt.ioff()  # Disable interactive mode

import os
from datetime import datetime
import numpy as np
import pandas as pd
from io import StringIO

from .utils import ensure_output_dir, print_save_message, validate_and_clean_metrics, format_metric_value, get_medal_emoji, get_status_emoji
from .basic_plots import plot_training_history, plot_predictions, plot_metrics_comparison, save_metrics_table
from .distribution_analysis import plot_qq_analysis, plot_cdf_comparison, plot_pdf_comparison
from .statistical_tests import plot_ks_test_analysis, plot_autocorrelation_analysis
from .comparison_plots import plot_models_comparison_overview
from .interactive_html import generate_interactive_html_report


def generate_comprehensive_report(results_dict, output_dir, model_name=None):
    """
    Generate comprehensive report with all available analyses
    
    Args:
        results_dict: Dictionary with model results (or single model results)
        output_dir: Directory to save the report
        model_name: Model name (used when results_dict contains single model data)
    
    Returns:
        dict: Dictionary with paths of generated files
    """
    ensure_output_dir(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    generated_files = {}
    
    # Determine if it's single model analysis or comparison
    if model_name and isinstance(results_dict, dict) and 'actuals' in results_dict:
        # Single model analysis
        single_model_results = {model_name: results_dict}
        report_type = "single"
        report_title = f"Comprehensive Report - {model_name}"
    else:
        # Multiple model comparison
        single_model_results = results_dict
        report_type = "comparison"
        report_title = "Model Comparison Report"
    
    print(f"\n🔄 Generating comprehensive report ({report_type})...")
    print(f"📁 Output directory: {output_dir}")
    
    # ========== GENERAL ANALYSIS ==========
    if report_type == "comparison":
        print("\n📊 Generating comparative overview...")
        overview_path = os.path.join(output_dir, f"01_overview_{timestamp}.png")
        try:
            plot_models_comparison_overview(single_model_results, save_path=overview_path)
            generated_files['overview'] = overview_path
        except Exception as e:
            print(f"⚠️ Error in overview: {e}")
    
    # ========== BASIC PLOTS ==========
    print("\n📈 Generating basic plots...")
    
    # For each model individually
    for i, (model_name, results) in enumerate(single_model_results.items(), 1):
        model_prefix = f"{i:02d}_{model_name.replace(' ', '_')}"
        
        # Training history
        if 'train_losses' in results and 'val_losses' in results:
            training_path = os.path.join(output_dir, f"{model_prefix}_training_{timestamp}.png")
            try:
                plot_training_history(results['train_losses'], results['val_losses'], 
                                    save_path=training_path, title=f"Training History - {model_name}")
                generated_files[f'training_{model_name}'] = training_path
            except Exception as e:
                print(f"⚠️ Error in training history for {model_name}: {e}")
        
        # Predictions vs Actual
        if 'actuals' in results and 'predictions' in results:
            predictions_path = os.path.join(output_dir, f"{model_prefix}_predictions_{timestamp}.png")
            try:
                plot_predictions(results['actuals'], results['predictions'], 
                               save_path=predictions_path, title=f"Predictions vs Actual Values - {model_name}")
                generated_files[f'predictions_{model_name}'] = predictions_path
            except Exception as e:
                print(f"⚠️ Error in predictions for {model_name}: {e}")
    
    # ========== STATISTICAL ANALYSES ==========
    print("\n🔬 Generating statistical analyses...")
    
    for i, (model_name, results) in enumerate(single_model_results.items(), 1):
        if 'actuals' not in results or 'predictions' not in results:
            continue
            
        model_prefix = f"{i:02d}_{model_name.replace(' ', '_')}"
        
        # QQ-Plot
        qq_path = os.path.join(output_dir, f"{model_prefix}_qq_plot_{timestamp}.png")
        try:
            plot_qq_analysis(results['actuals'], results['predictions'], 
                           save_path=qq_path, title=f"QQ-Plot - {model_name}")
            generated_files[f'qq_{model_name}'] = qq_path
        except Exception as e:
            print(f"⚠️ Error in Q-Q plot for {model_name}: {e}")
        
        # Distribution analysis (CDF and PDF)
        cdf_path = os.path.join(output_dir, f"{model_prefix}_cdf_{timestamp}.png")
        try:
            plot_cdf_comparison(results['actuals'], results['predictions'], 
                              save_path=cdf_path, title=f"CDF Comparison - {model_name}")
            generated_files[f'cdf_{model_name}'] = cdf_path
        except Exception as e:
            print(f"⚠️ Error in CDF analysis for {model_name}: {e}")
        
        pdf_path = os.path.join(output_dir, f"{model_prefix}_pdf_{timestamp}.png")
        try:
            plot_pdf_comparison(results['actuals'], results['predictions'], 
                              save_path=pdf_path, title=f"PDF Comparison - {model_name}")
            generated_files[f'pdf_{model_name}'] = pdf_path
        except Exception as e:
            print(f"⚠️ Error in PDF analysis for {model_name}: {e}")
        
        # Kolmogorov-Smirnov test
        ks_path = os.path.join(output_dir, f"{model_prefix}_ks_test_{timestamp}.png")
        try:
            plot_ks_test_analysis(results['actuals'], results['predictions'], 
                                 save_path=ks_path, title=f"Kolmogorov-Smirnov Test - {model_name}")
            generated_files[f'ks_{model_name}'] = ks_path
        except Exception as e:
            print(f"⚠️ Error in KS test for {model_name}: {e}")
        
        # Autocorrelation analysis (if time series exists)
        if 'actuals' in results and 'predictions' in results:
            autocorr_path = os.path.join(output_dir, f"{model_prefix}_autocorrelation_{timestamp}.png")
            try:
                plot_autocorrelation_analysis(results['actuals'], results['predictions'], 
                                             save_path=autocorr_path, title=f"Autocorrelation Analysis - {model_name}")
                generated_files[f'autocorr_{model_name}'] = autocorr_path
            except Exception as e:
                print(f"⚠️ Error in autocorrelation analysis for {model_name}: {e}")
    
    # ========== COMPARATIVE ANALYSES ==========
    if report_type == "comparison" and len(single_model_results) > 1:
        print("\n🆚 Generating comparative analyses...")
        
        # Metrics table
        metrics_table_path = os.path.join(output_dir, f"99_metrics_table_{timestamp}.png")
        metrics_table_comparison = os.path.join(output_dir, f"99_metrics_comparison_{timestamp}.png")
        try:
            plot_metrics_comparison(single_model_results, save_path=metrics_table_comparison)
            save_metrics_table(single_model_results, metrics_table_path)
            generated_files['metrics_table'] = metrics_table_path
            generated_files['metrics_comparison'] = metrics_table_comparison
        except Exception as e:
            print(f"⚠️ Error in metrics table: {e}")
    
    # ========== TEXT REPORT ==========
    print("\n📄 Generating text report...")
    text_report_path = os.path.join(output_dir, f"text_report.txt")
    try:
        generate_text_report(single_model_results, text_report_path, report_type)
        generated_files['text_report'] = text_report_path
    except Exception as e:
        print(f"⚠️ Error in text report: {e}")
    
    # ========== INTERACTIVE HTML REPORT ==========
    print("\n🌐 Generating interactive HTML report...")
    interactive_html_path = os.path.join(output_dir, f"report.html")
    try:
        generate_interactive_html_report(single_model_results, generated_files, interactive_html_path, report_type)
        generated_files['interactive_html_report'] = interactive_html_path
    except Exception as e:
        print(f"⚠️ Error in interactive HTML report: {e}")
    
    print(f"\n✅ Comprehensive report generated successfully!")
    print(f"📁 {len(generated_files)} files generated in: {output_dir}")
    
    return generated_files


def generate_text_report(results_dict, save_path, report_type="comparison"):
    """
    Generate detailed text report
    
    Args:
        results_dict: Dictionary with model results
        save_path: Path to save the report
        report_type: Report type ("single" or "comparison")
    """
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    from scipy import stats
    
    clean_results = validate_and_clean_metrics(results_dict)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("DETAILED REPORT OF MODEL ANALYSIS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated on: {datetime.now().strftime('%d/%m/%Y at %H:%M:%S')}\n")
        f.write(f"Analysis type: {'Multiple models comparison' if report_type == 'comparison' else 'Single model analysis'}\n")
        f.write(f"Number of analyzed models: {len(clean_results)}\n\n")
        
        # ========== EXECUTIVE SUMMARY ==========
        f.write("EXECUTIVE SUMMARY\n")
        f.write("-" * 50 + "\n\n")
        
        if not clean_results:
            f.write("❌ ERROR: Insufficient data for analysis.\n")
            return
        
        # Calculate metrics for all models
        model_metrics = {}
        for model_name, results in clean_results.items():
            if 'actuals' in results and 'predictions' in results:
                actuals = np.array(results['actuals']).flatten()
                predictions = np.array(results['predictions']).flatten()
                
                # Performance metrics
                r2 = r2_score(actuals, predictions)
                rmse = np.sqrt(mean_squared_error(actuals, predictions))
                mae = mean_absolute_error(actuals, predictions)
                mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100 if np.all(actuals != 0) else np.nan
                
                # Residual analysis
                residuals = predictions - actuals
                residuals_mean = np.mean(residuals)
                residuals_std = np.std(residuals)
                residuals_skew = stats.skew(residuals)
                residuals_kurt = stats.kurtosis(residuals)
                
                # Normality test
                try:
                    if len(residuals) <= 5000:
                        shapiro_stat, shapiro_p = stats.shapiro(residuals)
                    else:
                        shapiro_stat, shapiro_p = np.nan, np.nan
                except:
                    shapiro_stat, shapiro_p = np.nan, np.nan
                
                model_metrics[model_name] = {
                    'r2': r2, 'rmse': rmse, 'mae': mae, 'mape': mape,
                    'residuals_mean': residuals_mean, 'residuals_std': residuals_std,
                    'residuals_skew': residuals_skew, 'residuals_kurt': residuals_kurt,
                    'shapiro_stat': shapiro_stat, 'shapiro_p': shapiro_p,
                    'n_samples': len(actuals)
                }
        
        if report_type == "comparison" and len(model_metrics) > 1:
            # General ranking
            rankings = []
            for name, metrics in model_metrics.items():
                score = 0
                # R² (weight 3)
                score += metrics['r2'] * 3
                # Inverted RMSE (weight 2)
                max_rmse = max([m['rmse'] for m in model_metrics.values()])
                score += (1 - metrics['rmse'] / max_rmse) * 2 if max_rmse > 0 else 0
                # Inverted MAE (weight 2)
                max_mae = max([m['mae'] for m in model_metrics.values()])
                score += (1 - metrics['mae'] / max_mae) * 2 if max_mae > 0 else 0
                
                rankings.append((name, score, metrics))
            
            rankings.sort(key=lambda x: x[1], reverse=True)
            
            f.write("🏆 GENERAL RANKING OF MODELS:\n\n")
            for rank, (name, score, metrics) in enumerate(rankings, 1):
                medal = get_medal_emoji(rank)
                f.write(f"{medal} {name}\n")
                f.write(f"    Total Score: {score:.4f}\n")
                f.write(f"    R²: {metrics['r2']:.6f}\n")
                f.write(f"    RMSE: {metrics['rmse']:.6f}\n")
                f.write(f"    MAE: {metrics['mae']:.6f}\n")
                if not np.isnan(metrics['mape']):
                    f.write(f"    MAPE: {metrics['mape']:.2f}%\n")
                f.write("\n")
            
            # Best model analysis
            best_name, best_score, best_metrics = rankings[0]
            f.write(f"🎯 BEST MODEL ANALYSIS ({best_name}):\n")
            if best_metrics['r2'] > 0.95:
                f.write("   ✅ EXCEPTIONAL PERFORMANCE (R² > 0.95)\n")
            elif best_metrics['r2'] > 0.9:
                f.write("   ✅ EXCELLENT PERFORMANCE (R² > 0.9)\n")
            elif best_metrics['r2'] > 0.8:
                f.write("   ✅ GOOD PERFORMANCE (R² > 0.8)\n")
            elif best_metrics['r2'] > 0.6:
                f.write("   ⚠️ MODERATED PERFORMANCE (R² > 0.6)\n")
            else:
                f.write("   ❌ LOW PERFORMANCE (R² ≤ 0.6)\n")
            
            # Best model residual analysis
            if abs(best_metrics['residuals_mean']) < best_metrics['residuals_std'] * 0.1:
                f.write("   ✅ Well-centered residuals (no bias)\n")
            else:
                f.write("   ⚠️ Residuals with possible bias\n")
            
            if abs(best_metrics['residuals_skew']) < 0.5:
                f.write("   ✅ Approximately symmetric residuals\n")
            else:
                f.write("   ⚠️ Residuals with significant asymmetry\n")
        
        f.write("\n")
        
        # ========== DETAILED ANALYSIS BY MODEL ==========
        f.write("DETAILED ANALYSIS BY MODEL\n")
        f.write("-" * 50 + "\n\n")
        
        for model_name, metrics in model_metrics.items():
            f.write(f"📊 MODEL: {model_name}\n")
            f.write("=" * 40 + "\n")
            
            # Basic information
            f.write(f"Model: {model_name}\n")
            f.write(f"Dataset Size: {metrics['n_samples']} samples\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Performance metrics
            f.write("🎯 PERFORMANCE METRICS:\n")
            f.write(f"   R² (Coefficient of Determination): {metrics['r2']:.6f}\n")
            f.write(f"   RMSE (Root Mean Squared Error): {metrics['rmse']:.6f}\n")
            f.write(f"   MAE (Mean Absolute Error): {metrics['mae']:.6f}\n")
            if not np.isnan(metrics['mape']):
                f.write(f"   MAPE (Mean Absolute Percentage Error): {metrics['mape']:.2f}%\n")
            f.write("\n")
            
            # Metric interpretation
            f.write("METRIC INTERPRETATION:\n")
            f.write("=" * 50 + "\n")
            
            if metrics['r2'] > 0.9:
                f.write("   • Excellent predictive capability\n")
            elif metrics['r2'] > 0.8:
                f.write("   • Good predictive capability\n")
            elif metrics['r2'] > 0.6:
                f.write("   • Moderate predictive capability\n")
            else:
                f.write("   • Poor predictive capability\n")
            
            if not np.isnan(metrics['mape']):
                if metrics['mape'] < 5:
                    f.write("   • Very low percentage error (< 5%)\n")
                elif metrics['mape'] < 10:
                    f.write("   • Low percentage error (< 10%)\n")
                elif metrics['mape'] < 20:
                    f.write("   • Moderate percentage error (< 20%)\n")
                else:
                    f.write("   • High percentage error (≥ 20%)\n")
            
            f.write("\n")
            
            # Residual analysis
            f.write("🔬 RESIDUAL ANALYSIS:\n")
            f.write(f"   Mean: {metrics['residuals_mean']:.6f}\n")
            f.write(f"   Standard Deviation: {metrics['residuals_std']:.6f}\n")
            f.write(f"   Skewness: {metrics['residuals_skew']:.4f}\n")
            f.write(f"   Kurtosis: {metrics['residuals_kurt']:.4f}\n")
            
            if not np.isnan(metrics['shapiro_stat']):
                f.write(f"   Shapiro-Wilk Test: {metrics['shapiro_stat']:.4f} (p-value: {metrics['shapiro_p']:.4f})\n")
                if metrics['shapiro_p'] > 0.05:
                    f.write("   ✅ Residuals follow normal distribution (p > 0.05)\n")
                else:
                    f.write("   ⚠️ Residuals do not follow normal distribution (p ≤ 0.05)\n")
            
            # Residual diagnosis
            f.write("\n🩺 RESIDUAL DIAGNOSIS:\n")
            
            # Bias
            if abs(metrics['residuals_mean']) < metrics['residuals_std'] * 0.1:
                f.write("   ✅ No significant bias\n")
            else:
                f.write("   ⚠️ Possible bias in residuals\n")
            
            # Symmetry
            if abs(metrics['residuals_skew']) < 0.5:
                f.write("   ✅ Approximately symmetric distribution\n")
            elif abs(metrics['residuals_skew']) < 1:
                f.write("   ⚠️ Mild asymmetry\n")
            else:
                f.write("   ❌ Significant asymmetry\n")
            
            # Kurtosis
            if abs(metrics['residuals_kurt']) < 1:
                f.write("   ✅ Normal kurtosis\n")
            elif abs(metrics['residuals_kurt']) < 2:
                f.write("   ⚠️ Moderately high kurtosis\n")
            else:
                f.write("   ❌ Excessively high kurtosis\n")
            
            f.write("\n" + "─" * 40 + "\n\n")
        
        # ========== RECOMMENDATIONS ==========
        f.write("RECOMMENDATIONS\n")
        f.write("=" * 50 + "\n")
        
        if report_type == "comparison" and len(model_metrics) > 1:
            best_name = rankings[0][0]
            f.write(f"🏆 RECOMMENDED MODEL: {best_name}\n")
            
            f.write("💡 JUSTIFICATIONS:\n")
            best_metrics = model_metrics[best_name]
            
            if best_metrics['r2'] > 0.8:
                f.write(f"   • High R² ({best_metrics['r2']:.4f}) indicates good predictive capability\n")
            
            if best_metrics['rmse'] == min([m['rmse'] for m in model_metrics.values()]):
                f.write("   • Lowest RMSE among all models\n")
            
            if abs(best_metrics['residuals_mean']) < best_metrics['residuals_std'] * 0.1:
                f.write("   • Residuals well centered (no bias)\n")
            
            f.write("\n🚀 NEXT STEPS:\n")
            if best_metrics['r2'] < 0.9:
                f.write("   • Consider hyperparameter adjustments\n")
                f.write("   • Evaluate feature engineering\n")
            
            if abs(best_metrics['residuals_skew']) > 0.5:
                f.write("   • Investigate outliers in data\n")
                f.write("   • Consider data transformations\n")
            
            if not np.isnan(best_metrics['shapiro_p']) and best_metrics['shapiro_p'] <= 0.05:
                f.write("   • Verify model assumptions\n")
                f.write("   • Consider non-parametric models\n")
        else:
            # Single model analysis
            model_name = list(model_metrics.keys())[0]
            metrics = model_metrics[model_name]
            
            f.write(f"📊 MODEL EVALUATION: {model_name}\n\n")
            
            if metrics['r2'] > 0.9:
                f.write("✅ VERDICT: EXCELLENT model for production use\n")
            elif metrics['r2'] > 0.8:
                f.write("✅ VERDICT: GOOD model, suitable for use\n")
            elif metrics['r2'] > 0.6:
                f.write("⚠️ VERDICT: MODERATE model, needs improvements\n")
            else:
                f.write("❌ VERDICT: INADEQUATE model, requires complete revision\n")
            
            f.write("\n🎯 STRENGTHS:\n")
            if metrics['r2'] > 0.8:
                f.write("   • Good capability to explain variability\n")
            if abs(metrics['residuals_mean']) < metrics['residuals_std'] * 0.1:
                f.write("   • Residuals without significant bias\n")
            if abs(metrics['residuals_skew']) < 0.5:
                f.write("   • Symmetric residual distribution\n")
            
            f.write("\n⚠️ AREAS OF ATTENTION:\n")
            if metrics['r2'] < 0.8:
                f.write("   • Relatively low R²\n")
            if abs(metrics['residuals_mean']) >= metrics['residuals_std'] * 0.1:
                f.write("   • Possible bias in residuals\n")
            if abs(metrics['residuals_skew']) >= 0.5:
                f.write("   • Asymmetry in residuals\n")
        
        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")
    
    print_save_message(save_path, "Text report")


def generate_quick_summary(results_dict, model_name=None):
    """
    Generate quick summary for console visualization
    
    Args:
        results_dict: Dictionary with model results
        model_name: Model name (for single model analysis)
    
    Returns:
        str: Formatted summary
    """
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    
    if model_name and isinstance(results_dict, dict) and 'actuals' in results_dict:
        # Single model analysis
        single_model_results = {model_name: results_dict}
    else:
        # Multiple models
        single_model_results = results_dict
    
    clean_results = validate_and_clean_metrics(single_model_results)
    
    if not clean_results:
        return "❌ Insufficient data to generate summary."
    
    summary = "\n" + "="*60 + "\n"
    summary += "📊 QUICK PERFORMANCE SUMMARY\n"
    summary += "="*60 + "\n"
    
    model_scores = []
    
    for model_name, results in clean_results.items():
        if 'actuals' in results and 'predictions' in results:
            actuals = np.array(results['actuals']).flatten()
            predictions = np.array(results['predictions']).flatten()
            
            r2 = r2_score(actuals, predictions)
            rmse = np.sqrt(mean_squared_error(actuals, predictions))
            mae = mean_absolute_error(actuals, predictions)
            
            # Simple score for ranking
            score = r2 * 100 - (rmse + mae) * 10
            model_scores.append((model_name, score, r2, rmse, mae))
            
            # Status based on R²
            if r2 > 0.9:
                status = "🟢 EXCELLENT"
            elif r2 > 0.8:
                status = "🟢 GOOD"
            elif r2 > 0.6:
                status = "🟡 MODERATE"
            else:
                status = "🔴 LOW"
            
            summary += f"\n📈 {model_name}:\n"
            summary += f"   Status: {status}\n"
            summary += f"   R²: {r2:.6f} | RMSE: {rmse:.6f} | MAE: {mae:.6f}\n"
    
    # Ranking if multiple models
    if len(model_scores) > 1:
        model_scores.sort(key=lambda x: x[1], reverse=True)
        summary += "\n🏆 RANKING:\n"
        for rank, (name, score, r2, rmse, mae) in enumerate(model_scores, 1):
            medal = get_medal_emoji(rank)
            summary += f"   {medal} {name} (Score: {score:.2f})\n"
        
        # Recommendation
        best_model = model_scores[0][0]
        summary += f"\n🎯 RECOMMENDED MODEL: {best_model}\n"
    
    summary += "\n" + "="*60 + "\n"
    
    return summary 
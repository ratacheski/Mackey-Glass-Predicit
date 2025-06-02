"""
Model comparison plots
"""
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend that only saves files
import matplotlib.pyplot as plt
plt.ioff()  # Disable interactive mode

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from .utils import (ensure_output_dir, print_save_message, get_colors_and_styles, 
                   validate_and_clean_metrics, format_metric_value, get_medal_emoji, get_status_emoji)


def plot_models_comparison_overview(results_dict, save_path=None,
                                   title="Overview - Model Comparison"):
    """
    Comparative overview of multiple models with main metrics
    
    Args:
        results_dict: Dictionary with model results
        save_path: Path to save the plot
        title: Plot title
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # Validate and clean data
    clean_results = validate_and_clean_metrics(results_dict)
    
    if not clean_results:
        fig.text(0.5, 0.5, 'Insufficient data for comparison', 
                ha='center', va='center', fontsize=16,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        plt.tight_layout()
        if save_path:
            ensure_output_dir(save_path)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print_save_message(save_path, "Model comparison")
        plt.close()  # Close figure to free memory
        return
    
    colors, _ = get_colors_and_styles(len(clean_results))
    model_names = list(clean_results.keys())
    
    # Calculate main metrics
    metrics_data = {}
    for model_name, results in clean_results.items():
        if 'actuals' in results and 'predictions' in results:
            actuals = np.array(results['actuals']).flatten()
            predictions = np.array(results['predictions']).flatten()
            
            metrics_data[model_name] = {
                'R²': r2_score(actuals, predictions),
                'RMSE': np.sqrt(mean_squared_error(actuals, predictions)),
                'MAE': mean_absolute_error(actuals, predictions),
                'MAPE': np.mean(np.abs((actuals - predictions) / actuals)) * 100 if np.all(actuals != 0) else np.nan
            }
    
    # ===== SUBPLOT 1: R² =====
    ax1.set_title('Coefficient of Determination (R²)', fontsize=14, fontweight='bold')
    r2_values = [metrics_data[name]['R²'] for name in model_names if name in metrics_data]
    valid_names = [name for name in model_names if name in metrics_data]
    
    bars1 = ax1.bar(valid_names, r2_values, color=colors[:len(valid_names)], alpha=0.7)
    ax1.set_ylabel('R²', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, 1)
    
    # Add reference lines
    ax1.axhline(y=0.8, color='green', linestyle='--', alpha=0.7, label='Good (0.8)')
    ax1.axhline(y=0.9, color='darkgreen', linestyle='--', alpha=0.7, label='Excellent (0.9)')
    ax1.legend(fontsize=10)
    
    # Values on bars
    for bar, val in zip(bars1, r2_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # ===== SUBPLOT 2: RMSE =====
    ax2.set_title('Root Mean Square Error (RMSE)', fontsize=14, fontweight='bold')
    rmse_values = [metrics_data[name]['RMSE'] for name in valid_names]
    
    bars2 = ax2.bar(valid_names, rmse_values, color=colors[:len(valid_names)], alpha=0.7)
    ax2.set_ylabel('RMSE', fontsize=12)
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Values on bars
    for bar, val in zip(bars2, rmse_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + max(rmse_values)*0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # ===== SUBPLOT 3: MAE =====
    ax3.set_title('Mean Absolute Error (MAE)', fontsize=14, fontweight='bold')
    mae_values = [metrics_data[name]['MAE'] for name in valid_names]
    
    bars3 = ax3.bar(valid_names, mae_values, color=colors[:len(valid_names)], alpha=0.7)
    ax3.set_ylabel('MAE', fontsize=12)
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Values on bars
    for bar, val in zip(bars3, mae_values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + max(mae_values)*0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # ===== SUBPLOT 4: Ranking and Summary =====
    ax4.set_title('Model Ranking', fontsize=14, fontweight='bold')
    ax4.axis('off')
    
    # Calculate ranking based on multiple metrics
    model_scores = []
    for name in valid_names:
        score = 0
        metrics = metrics_data[name]
        
        # R² (higher is better) - weight 3
        r2_rank = sorted(valid_names, key=lambda x: metrics_data[x]['R²'], reverse=True).index(name)
        score += (len(valid_names) - r2_rank) * 3
        
        # RMSE (lower is better) - weight 2
        rmse_rank = sorted(valid_names, key=lambda x: metrics_data[x]['RMSE']).index(name)
        score += (len(valid_names) - rmse_rank) * 2
        
        # MAE (lower is better) - weight 2
        mae_rank = sorted(valid_names, key=lambda x: metrics_data[x]['MAE']).index(name)
        score += (len(valid_names) - mae_rank) * 2
        
        model_scores.append((name, score, metrics))
    
    # Sort by score
    model_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Create ranking text
    ranking_text = "GENERAL RANKING:\n\n"
    
    for rank, (name, score, metrics) in enumerate(model_scores, 1):
        medal = get_medal_emoji(rank)
        if rank == 1:
            color_bg = 'gold'
        elif rank == 2:
            color_bg = 'silver'
        elif rank == 3:
            color_bg = '#CD7F32'
        else:
            color_bg = 'lightgray'
        
        ranking_text += f"{medal} {name}\n"
        ranking_text += f"   Score: {score}\n"
        ranking_text += f"   R²: {metrics['R²']:.4f}\n"
        ranking_text += f"   RMSE: {metrics['RMSE']:.4f}\n"
        ranking_text += f"   MAE: {metrics['MAE']:.4f}\n"
        if not np.isnan(metrics['MAPE']):
            ranking_text += f"   MAPE: {metrics['MAPE']:.2f}%\n"
        ranking_text += "\n"
    
    # Best model analysis
    if model_scores:
        best_model, best_score, best_metrics = model_scores[0]
        ranking_text += f"BEST MODEL: {best_model}\n"
        if best_metrics['R²'] > 0.9:
            ranking_text += "• Excellent fit (R² > 0.9)\n"
        elif best_metrics['R²'] > 0.8:
            ranking_text += "• Good fit (R² > 0.8)\n"
        else:
            ranking_text += "• Moderate fit\n"
        
        # Comparison with second best
        if len(model_scores) > 1:
            second_best = model_scores[1]
            r2_diff = best_metrics['R²'] - second_best[2]['R²']
            ranking_text += f"• Advantage over 2nd: +{r2_diff:.4f} in R²\n"
    
    ax4.text(0.05, 0.95, ranking_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        ensure_output_dir(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print_save_message(save_path, "Model comparison")
    
    plt.close()  # Close figure to free memory
    
    return {
        'rankings': model_scores,
        'metrics_summary': metrics_data
    }


def plot_predictions_comparison(results_dict, n_show=500, save_path=None,
                               title="Predictions Comparison"):
    """
    Compare predictions from multiple models against actual values
    
    Args:
        results_dict: Dictionary with model results
        n_show: Number of points to show
        save_path: Path to save the plot
        title: Plot title
    """
    # Validate data
    clean_results = validate_and_clean_metrics(results_dict)
    
    if not clean_results:
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, 'Insufficient data for comparison', 
                ha='center', va='center', fontsize=16,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        plt.close()  # Close figure to free memory
        return
    
    n_models = len(clean_results)
    
    # Create subplots
    if n_models <= 2:
        fig, axes = plt.subplots(1, n_models, figsize=(10*n_models, 8))
    elif n_models <= 4:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    else:
        n_cols = 3
        n_rows = (n_models + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 6*n_rows))
    
    # Ensure axes is always an array
    if n_models == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_models > 1 else [axes]
    
    colors, _ = get_colors_and_styles(n_models)
    
    for i, (model_name, results) in enumerate(clean_results.items()):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        if 'actuals' in results and 'predictions' in results:
            actuals = np.array(results['actuals']).flatten()
            predictions = np.array(results['predictions']).flatten()
            
            # Limit number of points shown
            if len(actuals) > n_show:
                indices = np.random.choice(len(actuals), n_show, replace=False)
                actuals_show = actuals[indices]
                predictions_show = predictions[indices]
            else:
                actuals_show = actuals
                predictions_show = predictions
            
            # Scatter plot
            ax.scatter(actuals_show, predictions_show, alpha=0.6, s=20, 
                      color=colors[i], edgecolors='darker', linewidth=0.5)
            
            # Perfect line
            min_val = min(np.min(actuals_show), np.min(predictions_show))
            max_val = max(np.max(actuals_show), np.max(predictions_show))
            ax.plot([min_val, max_val], [min_val, max_val], 'r-', linewidth=2, 
                   alpha=0.8, label='Perfect Prediction')
            
            # Calculate metrics
            r2 = r2_score(actuals, predictions)
            rmse = np.sqrt(mean_squared_error(actuals, predictions))
            mae = mean_absolute_error(actuals, predictions)
            
            ax.set_title(f'{model_name}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Actual Values', fontsize=10)
            ax.set_ylabel('Predictions', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)
            
            # Add metrics
            metrics_text = f'R² = {r2:.4f}\nRMSE = {rmse:.4f}\nMAE = {mae:.4f}'
            ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, 
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                   verticalalignment='top', fontsize=9, fontfamily='monospace')
        else:
            ax.text(0.5, 0.5, f'Data not available\nfor {model_name}', 
                   transform=ax.transAxes, ha='center', va='center',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # Hide extra axes
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    
    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        ensure_output_dir(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print_save_message(save_path, "Predictions comparison")
    
    plt.close()  # Close figure to free memory


def plot_residuals_comparison(results_dict, save_path=None,
                             title="Residuals Comparison"):
    """
    Compare residual distributions from multiple models
    
    Args:
        results_dict: Dictionary with model results
        save_path: Path to save the plot
        title: Plot title
    """
    clean_results = validate_and_clean_metrics(results_dict)
    
    if not clean_results:
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, 'Insufficient data for comparison', 
                ha='center', va='center', fontsize=16,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        plt.close()  # Close figure to free memory
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    colors, _ = get_colors_and_styles(len(clean_results))
    
    # Calculate residuals for all models
    residuals_data = {}
    for model_name, results in clean_results.items():
        if 'actuals' in results and 'predictions' in results:
            actuals = np.array(results['actuals']).flatten()
            predictions = np.array(results['predictions']).flatten()
            residuals = predictions - actuals
            residuals_data[model_name] = residuals
    
    # ===== SUBPLOT 1: Residuals Histograms =====
    ax1.set_title('Residuals Distribution', fontsize=14, fontweight='bold')
    
    for i, (model_name, residuals) in enumerate(residuals_data.items()):
        ax1.hist(residuals, bins=50, alpha=0.6, color=colors[i], 
                label=model_name, density=True, edgecolor='darker')
    
    ax1.axvline(x=0, color='black', linestyle='--', alpha=0.8, label='Zero')
    ax1.set_xlabel('Residual', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # ===== SUBPLOT 2: Residuals Box Plots =====
    ax2.set_title('Residuals Box Plots', fontsize=14, fontweight='bold')
    
    residuals_list = list(residuals_data.values())
    model_names = list(residuals_data.keys())
    
    box_plot = ax2.boxplot(residuals_list, labels=model_names, patch_artist=True)
    
    # Color the boxes
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.8, label='Zero')
    ax2.set_ylabel('Residual', fontsize=12)
    ax2.tick_params(axis='x', rotation=45)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # ===== SUBPLOT 3: Q-Q Plots =====
    ax3.set_title('Q-Q Plots vs Normal Distribution', fontsize=14, fontweight='bold')
    
    from scipy import stats
    
    for i, (model_name, residuals) in enumerate(residuals_data.items()):
        # Calculate theoretical and empirical quantiles
        theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(residuals)))
        sample_quantiles = np.sort(residuals)
        
        ax3.scatter(theoretical_quantiles, sample_quantiles, alpha=0.6, 
                   color=colors[i], label=model_name, s=10)
    
    # Reference line
    all_residuals = np.concatenate(list(residuals_data.values()))
    min_q, max_q = np.min(all_residuals), np.max(all_residuals)
    ax3.plot([min_q, max_q], [min_q, max_q], 'r-', linewidth=2, alpha=0.8, 
             label='Normal Distribution')
    
    ax3.set_xlabel('Theoretical Quantiles (Normal)', fontsize=12)
    ax3.set_ylabel('Empirical Quantiles', fontsize=12)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # ===== SUBPLOT 4: Residuals Statistics =====
    ax4.set_title('Residuals Statistics', fontsize=14, fontweight='bold')
    ax4.axis('off')
    
    # Calculate statistics for each model
    stats_text = "RESIDUALS STATISTICS:\n\n"
    
    for i, (model_name, residuals) in enumerate(residuals_data.items()):
        mean_res = np.mean(residuals)
        std_res = np.std(residuals)
        skew_res = stats.skew(residuals)
        kurt_res = stats.kurtosis(residuals)
        
        # Normality test
        try:
            if len(residuals) <= 5000:
                shapiro_stat, shapiro_p = stats.shapiro(residuals)
                shapiro_text = f"Shapiro: {shapiro_stat:.4f} (p={shapiro_p:.4f})"
            else:
                shapiro_text = "Shapiro: N/A (large sample)"
        except:
            shapiro_text = "Shapiro: Calculation error"
        
        # Qualitative analysis
        if abs(mean_res) < std_res * 0.1 and abs(skew_res) < 0.5 and abs(kurt_res) < 1:
            quality = f"{get_status_emoji('good')} EXCELLENT"
            quality_color = get_status_emoji('excellent')
        elif abs(mean_res) < std_res * 0.2 and abs(skew_res) < 1 and abs(kurt_res) < 2:
            quality = f"{get_status_emoji('warning')} GOOD"
            quality_color = get_status_emoji('moderate')
        else:
            quality = f"{get_status_emoji('bad')} PROBLEMATIC"
            quality_color = get_status_emoji('poor')
        
        stats_text += f"{quality_color} {model_name}:\n"
        stats_text += f"  Mean: {mean_res:.6f}\n"
        stats_text += f"  Std Dev: {std_res:.6f}\n"
        stats_text += f"  Skewness: {skew_res:.4f}\n"
        stats_text += f"  Kurtosis: {kurt_res:.4f}\n"
        stats_text += f"  {shapiro_text}\n"
        stats_text += f"  Quality: {quality}\n\n"
    
    # Residuals ranking
    residual_scores = []
    for model_name, residuals in residuals_data.items():
        score = 0
        mean_res = abs(np.mean(residuals))
        std_res = np.std(residuals)
        skew_res = abs(stats.skew(residuals))
        kurt_res = abs(stats.kurtosis(residuals))
        
        # Score based on residuals quality
        if mean_res < std_res * 0.1: score += 3
        elif mean_res < std_res * 0.2: score += 2
        else: score += 1
        
        if skew_res < 0.5: score += 2
        elif skew_res < 1: score += 1
        
        if kurt_res < 1: score += 2
        elif kurt_res < 2: score += 1
        
        residual_scores.append((model_name, score))
    
    residual_scores.sort(key=lambda x: x[1], reverse=True)
    
    stats_text += "RANKING (Residuals Quality):\n"
    for rank, (name, score) in enumerate(residual_scores, 1):
        medal = get_medal_emoji(rank)
        stats_text += f"{medal} {name} (Score: {score}/7)\n"
    
    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        ensure_output_dir(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print_save_message(save_path, "Residuals comparison")
    
    plt.close()  # Close figure to free memory
    
    return {
        'residuals_statistics': {name: {
            'mean': float(np.mean(res)),
            'std': float(np.std(res)),
            'skewness': float(stats.skew(res)),
            'kurtosis': float(stats.kurtosis(res))
        } for name, res in residuals_data.items()},
        'residuals_ranking': residual_scores
    }


def plot_training_comparison(results_dict, save_path=None,
                            title="Training History Comparison"):
    """
    Compare training histories from multiple models
    
    Args:
        results_dict: Dictionary with model results
        save_path: Path to save the plot
        title: Plot title
    """
    # Filter models that have training data
    training_data = {}
    for model_name, results in results_dict.items():
        if 'train_losses' in results and 'val_losses' in results:
            training_data[model_name] = results
    
    if not training_data:
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, 'Training data not available', 
                ha='center', va='center', fontsize=16,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        plt.close()  # Close figure to free memory
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    colors, linestyles = get_colors_and_styles(len(training_data))
    
    # ===== SUBPLOT 1: Training Loss =====
    ax1.set_title('Training Loss', fontsize=14, fontweight='bold')
    
    for i, (model_name, results) in enumerate(training_data.items()):
        train_losses = results['train_losses']
        epochs = range(1, len(train_losses) + 1)
        ax1.plot(epochs, train_losses, color=colors[i], linestyle=linestyles[i],
                linewidth=2, label=model_name, alpha=0.8)
    
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Training Loss', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')  # Log scale for better visualization
    
    # ===== SUBPLOT 2: Validation Loss =====
    ax2.set_title('Validation Loss', fontsize=14, fontweight='bold')
    
    for i, (model_name, results) in enumerate(training_data.items()):
        val_losses = results['val_losses']
        epochs = range(1, len(val_losses) + 1)
        ax2.plot(epochs, val_losses, color=colors[i], linestyle=linestyles[i],
                linewidth=2, label=model_name, alpha=0.8)
    
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Validation Loss', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # ===== SUBPLOT 3: Combined Learning Curves =====
    ax3.set_title('Learning Curves (Train vs Val)', fontsize=14, fontweight='bold')
    
    for i, (model_name, results) in enumerate(training_data.items()):
        train_losses = results['train_losses']
        val_losses = results['val_losses']
        epochs = range(1, len(train_losses) + 1)
        
        # Training (solid line)
        ax3.plot(epochs, train_losses, color=colors[i], linestyle='-',
                linewidth=2, label=f'{model_name} (Train)', alpha=0.8)
        
        # Validation (dashed line)
        ax3.plot(epochs, val_losses, color=colors[i], linestyle='--',
                linewidth=2, label=f'{model_name} (Val)', alpha=0.8)
    
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Loss', fontsize=12)
    ax3.legend(fontsize=9, ncol=2)
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    # ===== SUBPLOT 4: Convergence Analysis =====
    ax4.set_title('Convergence Analysis', fontsize=14, fontweight='bold')
    ax4.axis('off')
    
    # Analyze convergence for each model
    convergence_text = "CONVERGENCE ANALYSIS:\n\n"
    
    convergence_scores = []
    
    for model_name, results in training_data.items():
        train_losses = np.array(results['train_losses'])
        val_losses = np.array(results['val_losses'])
        
        # Convergence metrics
        final_train_loss = train_losses[-1]
        final_val_loss = val_losses[-1]
        min_val_loss = np.min(val_losses)
        min_val_epoch = np.argmin(val_losses) + 1
        
        # Check overfitting
        overfitting_gap = final_val_loss - final_train_loss
        overfitting_ratio = final_val_loss / final_train_loss if final_train_loss > 0 else np.inf
        
        # Stability (variation in last 10% of epochs)
        last_10_percent = max(1, len(val_losses) // 10)
        val_stability = np.std(val_losses[-last_10_percent:])
        
        # Convergence speed (epochs to reach 90% of best loss)
        target_loss = min_val_loss * 1.1  # 110% of best loss
        convergence_epoch = len(val_losses)  # Default: last epoch
        for i, loss in enumerate(val_losses):
            if loss <= target_loss:
                convergence_epoch = i + 1
                break
        
        # Convergence score
        score = 0
        
        # Best loss (weight 3)
        if min_val_loss < 0.01: score += 3
        elif min_val_loss < 0.1: score += 2
        else: score += 1
        
        # Overfitting (weight 2)
        if overfitting_ratio < 1.1: score += 2
        elif overfitting_ratio < 1.5: score += 1
        
        # Speed (weight 1)
        if convergence_epoch < len(val_losses) * 0.5: score += 1
        
        convergence_scores.append((model_name, score, {
            'final_train_loss': final_train_loss,
            'final_val_loss': final_val_loss,
            'min_val_loss': min_val_loss,
            'min_val_epoch': min_val_epoch,
            'overfitting_ratio': overfitting_ratio,
            'convergence_epoch': convergence_epoch,
            'total_epochs': len(val_losses)
        }))
        
        # Overfitting status
        if overfitting_ratio < 1.1:
            overfitting_status = "✓ No overfitting"
            status_color = "🟢"
        elif overfitting_ratio < 1.5:
            overfitting_status = "⚠ Mild overfitting"
            status_color = "🟡"
        else:
            overfitting_status = "✗ Severe overfitting"
            status_color = "🔴"
        
        convergence_text += f"{status_color} {model_name}:\n"
        convergence_text += f"  Final loss (train): {final_train_loss:.6f}\n"
        convergence_text += f"  Final loss (val): {final_val_loss:.6f}\n"
        convergence_text += f"  Best val loss: {min_val_loss:.6f} (epoch {min_val_epoch})\n"
        convergence_text += f"  Overfitting ratio: {overfitting_ratio:.2f}\n"
        convergence_text += f"  Convergence at: {convergence_epoch}/{len(val_losses)} epochs\n"
        convergence_text += f"  Status: {overfitting_status}\n\n"
    
    # Convergence ranking
    convergence_scores.sort(key=lambda x: x[1], reverse=True)
    
    convergence_text += "RANKING (Convergence Quality):\n"
    for rank, (name, score, _) in enumerate(convergence_scores, 1):
        medal = get_medal_emoji(rank)
        convergence_text += f"{medal} {name} (Score: {score}/6)\n"
    
    ax4.text(0.05, 0.95, convergence_text, transform=ax4.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        ensure_output_dir(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print_save_message(save_path, "Training comparison")
    
    plt.close()  # Close figure to free memory
    
    return {
        'convergence_analysis': {name: metrics for name, score, metrics in convergence_scores},
        'convergence_ranking': [(name, score) for name, score, _ in convergence_scores]
    } 
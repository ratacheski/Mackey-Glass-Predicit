"""
Basic plots for training and prediction visualization
"""
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend that only saves files
import matplotlib.pyplot as plt
plt.ioff()  # Disable interactive mode

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from .utils import (format_metric_value, validate_and_clean_metrics, 
                   ensure_output_dir, print_save_message)


def plot_training_history(train_losses, val_losses, save_path=None, 
                         title="Training History"):
    """
    Plot training and validation loss history
    
    Args:
        train_losses: List with training losses
        val_losses: List with validation losses
        save_path: Path to save the plot
        title: Plot title
    """
    plt.figure(figsize=(12, 5))
    
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        ensure_output_dir(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print_save_message(save_path, "Training plot")
    
    plt.close()  # Close figure to free memory


def plot_predictions(actuals, predictions, n_show=500, save_path=None, 
                    title="Predictions vs Actual Values"):
    """
    Plot predictions vs actual values
    
    Args:
        actuals: Actual values
        predictions: Model predictions
        n_show: Maximum number of points to show
        save_path: Path to save the plot
        title: Plot title
    """
    plt.figure(figsize=(15, 8))
    
    # Limit number of points for visualization
    if len(actuals) > n_show:
        indices = np.linspace(0, len(actuals)-1, n_show, dtype=int)
        actuals_plot = actuals[indices]
        predictions_plot = predictions[indices]
        x_axis = indices
    else:
        actuals_plot = actuals
        predictions_plot = predictions
        x_axis = range(len(actuals))
    
    plt.plot(x_axis, actuals_plot, 'b-', label='Actual Values', alpha=0.7, linewidth=1.5)
    plt.plot(x_axis, predictions_plot, 'r-', label='Predictions', alpha=0.8, linewidth=1.5)
    
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        ensure_output_dir(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print_save_message(save_path, "Predictions plot")
    
    plt.close()  # Close figure to free memory


def plot_metrics_comparison(results_dict, save_path=None, 
                           title="Metrics Comparison Between Models"):
    """
    Plot metrics comparison between different models
    
    Args:
        results_dict: Dictionary with model results
        save_path: Path to save the plot
        title: Plot title
    """
    # Validate and clean data
    results_dict = validate_and_clean_metrics(results_dict)
    
    # Prepare data
    models = list(results_dict.keys())
    # Metrics to be plotted
    metrics = ['MSE', 'EQMN1', 'EQMN2', 'R²', 'D2_PINBALL_SCORE', 'MEAN_PINBALL_LOSS']
    
    # Check which metrics are available in the data
    available_metrics = []
    if models:
        first_model_metrics = results_dict[models[0]].get('metrics', {})
        for metric in metrics:
            if metric in first_model_metrics:
                available_metrics.append(metric)
    
    if not available_metrics:
        print("No metrics found to plot.")
        return
    
    # Calculate number of subplots needed
    n_metrics = len(available_metrics)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    # Create subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for i, metric in enumerate(available_metrics):
        values = [results_dict[model]['metrics'][metric] for model in models]
        
        bars = axes[i].bar(models, values, alpha=0.7)
        axes[i].set_title(f'{metric}', fontsize=12, fontweight='bold')
        axes[i].set_ylabel(metric)
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].grid(True, alpha=0.3)
        
        # Add values on bars with improved formatting
        for bar, value in zip(bars, values):
            height = bar.get_height()
            formatted_value = format_metric_value(value, metric, context='display')
            axes[i].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        formatted_value, ha='center', va='bottom', 
                        fontsize=9, fontweight='bold')
        
        # Highlight the best bar (only if there are valid values)
        valid_values = [v for v in values if not (np.isnan(v) or np.isinf(v))]
        if valid_values:    
            # For R² and d2 pinball loss higher is better, for other metrics lower is better
            if metric in ['R²', 'D2_PINBALL_SCORE']:
                best_value = max(valid_values)
                best_idx = values.index(best_value)
            else:
                best_value = min(valid_values)
                best_idx = values.index(best_value)
            
            bars[best_idx].set_color('#28a745')
            bars[best_idx].set_alpha(0.8)
    
    # Remove extra subplots
    for i in range(len(available_metrics), len(axes)):
        fig.delaxes(axes[i])
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        ensure_output_dir(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print_save_message(save_path, "Metrics comparison")
    
    plt.close()  # Close figure to free memory


def save_metrics_table(results_dict, save_path):
    """
    Save metrics table to CSV and create visualization
    
    Args:
        results_dict: Dictionary with model results
        save_path: Base path to save files
        
    Returns:
        DataFrame with the metrics
    """
    # Validate and clean data
    results_dict = validate_and_clean_metrics(results_dict)
    
    # Metrics to be plotted (same as plot_metrics_comparison)
    target_metrics = ['MSE', 'EQMN1', 'EQMN2', 'R²', 'D2_PINBALL_SCORE', 'MEAN_PINBALL_LOSS']
    
    # Check if there are available models
    if not results_dict:
        print("No models found to save the metrics table.")
        return pd.DataFrame()
    
    # Create DataFrame only with specified metrics
    data = []
    for model_name, results in results_dict.items():
        all_metrics = results['metrics']
        row = {'Model': model_name}
        
        # Add only specified metrics
        for metric in target_metrics:
            if metric in all_metrics:
                row[metric] = all_metrics[metric]
            else:
                row[metric] = np.nan  # If metric is not available
        
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # Save CSV with proper formatting
    csv_path = save_path.replace('.png', '.csv') if save_path.endswith('.png') else save_path + '.csv'
    
    # Format data for CSV
    df_formatted = df.copy()
    for col in df.columns:
        if col != 'Model' and df[col].dtype in ['float64', 'float32']:
            df_formatted[col] = df[col].round(6)
    
    df_formatted.to_csv(csv_path, index=False)
    print_save_message(csv_path, "Metrics table")
    
    # Create table visualization
    plt.figure(figsize=(14, 8))
    
    # Prepare table data with custom formatting
    table_data = []
    col_labels = df.columns.tolist()
    
    for _, row in df.iterrows():
        formatted_row = []
        for col in col_labels:
            if col == 'Model':
                # Truncate very long names
                model_name = str(row[col])
                if len(model_name) > 15:
                    model_name = model_name[:12] + '...'
                formatted_row.append(model_name)
            else:
                formatted_row.append(format_metric_value(row[col], col, context='table'))
        table_data.append(formatted_row)
    
    # Create table
    table = plt.table(cellText=table_data,
                     colLabels=col_labels,
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    
    # Style table
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.3, 2.2)
    
    # Highlight header
    for i in range(len(col_labels)):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
        
    # Style data cells
    for i in range(1, len(table_data) + 1):
        for j in range(len(col_labels)):
            # Alternate row colors
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f8f9fa')
            else:
                table[(i, j)].set_facecolor('#ffffff')
            
            # Adjust spacing
            table[(i, j)].set_text_props(fontsize=9)
            
            # Highlight best model (lower value for error metrics; higher for R² and d2 pinball loss)
            if j > 0:  # Don't apply to model name
                metric_name = col_labels[j]
                values = df[metric_name].values
                
                # Filter valid values
                valid_mask = ~(np.isnan(values) | np.isinf(values))
                if valid_mask.any():
                    valid_values = values[valid_mask]
                    
                    # Find best value
                    # R² and d2 pinball loss are better when higher; other metrics are better when lower
                    if metric_name in ['R²', 'D2_PINBALL_SCORE']:
                        best_value = np.max(valid_values)
                        is_best = abs(df.iloc[i-1][metric_name] - best_value) < 1e-6
                    else:
                        best_value = np.min(valid_values)
                        is_best = abs(df.iloc[i-1][metric_name] - best_value) < 1e-6
                    
                    if is_best and not (np.isnan(df.iloc[i-1][metric_name]) or np.isinf(df.iloc[i-1][metric_name])):
                        table[(i, j)].set_facecolor('#d4edda')
                        table[(i, j)].set_text_props(weight='bold', color='#155724')
    
    # Remove axes
    plt.axis('off')
    plt.title('Metrics Comparison - All Models', 
              fontsize=16, fontweight='bold', pad=30)
    
    # Add legend
    plt.figtext(0.5, 0.02, 'Green highlighted cells indicate the best performance for each metric', 
                ha='center', fontsize=10, style='italic')
    
    if save_path:
        png_path = save_path.replace('.csv', '.png') if save_path.endswith('.csv') else (save_path if save_path.endswith('.png') else save_path + '.png')
        ensure_output_dir(png_path)
        plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.2)
        print_save_message(png_path, "Table visualization")
    
    plt.close()  # Close figure to free memory
    
    return df 
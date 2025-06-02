"""
Visualization utilities
"""
# Configure matplotlib BEFORE any other import
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
plt.ioff()  # Disable interactive mode

import numpy as np
import pandas as pd
import seaborn as sns
import os
import platform
import matplotlib.font_manager as fm

# Configure plot style
matplotlib.style.use('ggplot')
sns.set_palette("husl")

# Configure plot style
plt.style.use('default')


def format_metric_value(value, metric_name, context='table'):
    """
    Utility function for consistent formatting of metric values
    
    Args:
        value: Numeric value to be formatted
        metric_name: Metric name (MSE, EQMN1, EQMN2, RMSE, MAE, MAPE, R², FDA_*, FDP_*, D2_PINBALL_SCORE, MEAN_PINBALL_LOSS)
        context: Formatting context ('table' or 'display')
    """
    if pd.isna(value) or value is None:
        return 'N/A'
    
    # Check extreme values
    if np.isinf(value):
        return '∞' if value > 0 else '-∞'
    
    # Special treatment for different metrics
    if metric_name == 'R²':
        return f'{value:.4f}'
    elif metric_name == 'D2_PINBALL_SCORE':
        # d2 pinball loss is similar to R² (can be negative, typically varies between -∞ and 1)
        if value < 0:
            return f'{value:.4f}'
        else:
            return f'{value:.4f}'
    elif metric_name == 'MEAN_PINBALL_LOSS':
        # mean pinball loss is an error metric (lower values are better)
        if abs(value) < 1e-4:
            return f'{value:.2e}'
        elif abs(value) < 0.01:
            return f'{value:.6f}'
        elif abs(value) < 1:
            return f'{value:.5f}'
        elif abs(value) < 10:
            return f'{value:.4f}'
        else:
            return f'{value:.3f}'
    elif metric_name == 'MAPE':
        if context == 'table':
            return f'{value:.2f}%' if value < 100 else f'{value:.1f}%'
        else:
            return f'{value:.1f}%'
    elif metric_name in ['MSE', 'RMSE', 'MAE', 'EQMN1', 'EQMN2']:
        # For very small values, use scientific notation
        if abs(value) < 1e-4:
            return f'{value:.2e}'
        elif abs(value) < 0.01:
            return f'{value:.6f}'
        elif abs(value) < 1:
            return f'{value:.5f}'
        elif abs(value) < 10:
            return f'{value:.4f}'
        elif abs(value) < 100:
            return f'{value:.3f}'
        else:
            return f'{value:.2f}'
    elif metric_name.startswith('FDA_'):
        # CDF metrics (Cumulative Distribution Function)
        if metric_name in ['ks_statistic', 'ks_pvalue', 'fda_distance']:
            # P-values with more precision
            if 'pvalue' in metric_name:
                if value < 0.001:
                    return f'{value:.2e}'
                else:
                    return f'{value:.4f}'
            else:
                return f'{value:.4f}'
        elif metric_name in ['FDA_KS_Statistic', 'FDA_Distance']:
            # Distances and divergences with scientific precision if very small
            if abs(value) < 0.001:
                return f'{value:.2e}'
            else:
                return f'{value:.6f}'
    elif metric_name.startswith('FDP_'):
        # PDF metrics (Probability Density Function)
        if metric_name in ['FDP_L2_Distance', 'FDP_JS_Divergence']:
            # Distances and divergences with scientific precision if very small
            if abs(value) < 1e-4:
                return f'{value:.2e}'
            elif abs(value) < 0.01:
                return f'{value:.6f}'
            else:
                return f'{value:.4f}'
    else:
        # For very small values, use scientific notation
        if abs(value) < 1e-6 and value != 0:
            return f'{value:.2e}'
        elif abs(value) < 1e-3:
            return f'{value:.6f}'
        elif abs(value) < 1:
            return f'{value:.4f}'
        else:
            return f'{value:.3f}'


def validate_and_clean_metrics(results_dict):
    """
    Validate and clean metrics data to avoid formatting issues
    
    Args:
        results_dict: Dictionary with model results
        
    Returns:
        Cleaned and validated dictionary
    """
    cleaned_dict = {}
    
    for model_name, results in results_dict.items():
        cleaned_results = results.copy()
        
        if 'metrics' in results:
            cleaned_metrics = {}
            for metric_name, value in results['metrics'].items():
                # Clean problematic values
                if pd.isna(value) or value is None:
                    cleaned_value = np.nan
                elif np.isinf(value):
                    # For infinites, use a very large but finite value
                    cleaned_value = 1e10 if value > 0 else -1e10
                else:
                    cleaned_value = float(value)
                
                cleaned_metrics[metric_name] = cleaned_value
            
            cleaned_results['metrics'] = cleaned_metrics
        
        cleaned_dict[model_name] = cleaned_results
    
    return cleaned_dict


def ensure_output_dir(save_path):
    """
    Ensure that the output directory exists
    
    Args:
        save_path: Path to the output file
    """
    if save_path:
        dir_path = os.path.dirname(save_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)


def get_colors_and_styles(n_items):
    """
    Return colors and line styles for multiple items
    
    Args:
        n_items: Number of items that need colors/styles
        
    Returns:
        tuple: (colors, linestyles)
    """
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':']
    
    # Repeat if necessary
    colors = (colors * ((n_items // len(colors)) + 1))[:n_items]
    linestyles = (linestyles * ((n_items // len(linestyles)) + 1))[:n_items]
    
    return colors, linestyles


def add_metrics_text_box(ax, metrics_dict, title="Metrics", 
                        position=(0.02, 0.98), box_color='lightblue',
                        fontsize=9, family='monospace'):
    """
    Add a text box with metrics to the plot
    
    Args:
        ax: Matplotlib axis
        metrics_dict: Dictionary with metrics
        title: Box title
        position: Box position (x, y) in relative coordinates
        box_color: Box background color
        fontsize: Font size
        family: Font family
    """
    text_lines = [f"{title}:"]
    for key, value in metrics_dict.items():
        if isinstance(value, float):
            text_lines.append(f"{key}: {value:.4f}")
        else:
            text_lines.append(f"{key}: {value}")
    
    metrics_text = "\n".join(text_lines)
    
    ax.text(position[0], position[1], metrics_text, transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor=box_color, alpha=0.8),
            verticalalignment='top', fontsize=fontsize, fontfamily=family)


def print_save_message(save_path, description="Plot"):
    """
    Print save confirmation message
    
    Args:
        save_path: Path where it was saved
        description: Description of what was saved
    """
    if save_path:
        print(f"{description} saved at: {save_path}")


def setup_emoji_font():
    """
    Configure font that supports emojis in matplotlib
    
    Returns:
        bool: True if successfully configured emoji font, False otherwise
    """
    system = platform.system()
    
    # List of fonts that support emojis by system
    emoji_fonts = {
        'Windows': ['Segoe UI Emoji', 'Microsoft YaHei', 'Malgun Gothic'],
        'Darwin': ['Apple Color Emoji', 'Arial Unicode MS', 'Menlo'],  # macOS
        'Linux': ['Noto Color Emoji', 'Noto Emoji', 'DejaVu Sans', 'Liberation Sans']
    }
    
    # Get list of available fonts
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    # Try to find a compatible font
    fonts_to_try = emoji_fonts.get(system, emoji_fonts['Linux'])  # Linux as fallback
    
    for font_name in fonts_to_try:
        if font_name in available_fonts:
            try:
                # Configure font
                plt.rcParams['font.family'] = [font_name]
                
                # Test if font supports emojis
                fig, ax = plt.subplots(figsize=(1, 1))
                ax.text(0.5, 0.5, '🥇', fontsize=12, ha='center', va='center')
                plt.close(fig)
                
                print(f"✅ Font configured: {font_name} (emoji support)")
                return True
            except Exception:
                continue
    
    # If reached here, didn't find compatible font
    return False


def get_medal_emoji(rank):
    """
    Return medal emoji or text based on font support
    
    Args:
        rank: Position in ranking (1, 2, 3, ...)
    
    Returns:
        str: Emoji or text representing the medal
    """
    if hasattr(get_medal_emoji, '_emoji_supported'):
        emoji_supported = get_medal_emoji._emoji_supported
    else:
        emoji_supported = setup_emoji_font()
        get_medal_emoji._emoji_supported = emoji_supported
    
    if emoji_supported:
        if rank == 1:
            return "🥇"
        elif rank == 2:
            return "🥈"
        elif rank == 3:
            return "🥉"
        else:
            return f"{rank}º"
    else:
        # Fallback to text
        if rank == 1:
            return "[1º]"
        elif rank == 2:
            return "[2º]"
        elif rank == 3:
            return "[3º]"
        else:
            return f"[{rank}º]"


def get_status_emoji(status_type, emoji_supported=None):
    """
    Return status emoji or text based on font support
    
    Args:
        status_type: Status type ('good', 'warning', 'bad', 'info')
        emoji_supported: If None, detects automatically
    
    Returns:
        str: Emoji or text representing the status
    """
    if emoji_supported is None:
        if hasattr(get_status_emoji, '_emoji_supported'):
            emoji_supported = get_status_emoji._emoji_supported
        else:
            emoji_supported = setup_emoji_font()
            get_status_emoji._emoji_supported = emoji_supported
    
    if emoji_supported:
        status_map = {
            'good': '✅',
            'warning': '⚠️',
            'bad': '❌',
            'info': 'ℹ️',
            'excellent': '🟢',
            'moderate': '🟡',
            'poor': '🔴'
        }
        return status_map.get(status_type, '•')
    else:
        # Fallback to text
        status_map = {
            'good': '[OK]',
            'warning': '[!]',
            'bad': '[X]',
            'info': '[i]',
            'excellent': '[G]',
            'moderate': '[Y]',
            'poor': '[R]'
        }
        return status_map.get(status_type, '•')

# Configure font
if setup_emoji_font():
    plt.rcParams['font.family'] = ['Noto Color Emoji', 'DejaVu Sans'] 
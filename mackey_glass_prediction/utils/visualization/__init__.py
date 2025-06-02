"""
Modular visualization package for machine learning model analysis

This package contains specialized modules for different types of visualizations:
- utils: Utility functions and formatting
- basic_plots: Basic plots (training, predictions)
- distribution_analysis: Distribution analysis (QQ-plot, CDF, PDF)
- statistical_tests: Statistical tests (KS, autocorrelation)
- comparison_plots: Model comparisons
- reports: Comprehensive report generation
"""

# Configure matplotlib to not display plots on screen
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend that only saves files
import matplotlib.pyplot as plt

# Additional configurations for better quality and performance
plt.ioff()  # Disable interactive mode

# Main imports from each module
from .utils import (
    format_metric_value,
    validate_and_clean_metrics,
    ensure_output_dir,
    get_colors_and_styles,
    add_metrics_text_box,
    print_save_message
)

from .basic_plots import (
    plot_training_history,
    plot_predictions,
    plot_metrics_comparison,
    save_metrics_table
)

from .distribution_analysis import (
    plot_qq_analysis,
    plot_cdf_comparison,
    plot_pdf_comparison,
    plot_distribution_analysis,
    plot_multi_model_cdf_comparison,
    plot_multi_model_pdf_comparison
)

from .statistical_tests import (
    plot_ks_test_analysis,
    plot_autocorrelation_analysis
)

from .comparison_plots import (
    plot_models_comparison_overview,
    plot_predictions_comparison,
    plot_residuals_comparison,
    plot_training_comparison
)

from .reports import (
    generate_comprehensive_report,
    generate_text_report,
    generate_quick_summary
)

# Package version
__version__ = "1.0.0"

# List of all available functions
__all__ = [
    # Utils
    'format_metric_value',
    'validate_and_clean_metrics',
    'ensure_output_dir',
    'get_colors_and_styles',
    'add_metrics_text_box',
    'print_save_message',
    
    # Basic plots
    'plot_training_history',
    'plot_predictions',
    'plot_metrics_comparison',
    'save_metrics_table',
    
    # Distribution analysis
    'plot_qq_analysis',
    'plot_cdf_comparison',
    'plot_pdf_comparison',
    'plot_distribution_analysis',
    'plot_multi_model_cdf_comparison',
    'plot_multi_model_pdf_comparison',
    
    # Statistical tests
    'plot_ks_test_analysis',
    'plot_autocorrelation_analysis',
    
    # Comparison plots
    'plot_models_comparison_overview',
    'plot_predictions_comparison',
    'plot_residuals_comparison',
    'plot_training_comparison',
    
    # Reports
    'generate_comprehensive_report',
    'generate_text_report',
    'generate_quick_summary'
]


def show_available_functions():
    """
    Show all available functions in the visualization package
    """
    print("📊 VISUALIZATION PACKAGE - AVAILABLE FUNCTIONS")
    print("=" * 60)
    
    print("\n🛠️  UTILITIES:")
    print("   • format_metric_value() - Format metric values")
    print("   • validate_and_clean_metrics() - Validate input data")
    print("   • ensure_output_dir() - Ensure output directory")
    print("   • get_colors_and_styles() - Get colors and styles")
    print("   • add_metrics_text_box() - Add metrics text box")
    print("   • print_save_message() - Confirmation message")
    
    print("\n📈 BASIC PLOTS:")
    print("   • plot_training_history() - Training history")
    print("   • plot_predictions() - Predictions vs actual values")
    print("   • plot_metrics_comparison() - Metrics comparison")
    print("   • save_metrics_table() - Save metrics table")
    
    print("\n📊 DISTRIBUTION ANALYSIS:")
    print("   • plot_qq_analysis() - Q-Q Plot")
    print("   • plot_cdf_comparison() - CDF comparison")
    print("   • plot_pdf_comparison() - PDF comparison")
    print("   • plot_distribution_analysis() - Complete analysis")
    print("   • plot_multi_model_cdf_comparison() - Multi-model CDF")
    print("   • plot_multi_model_pdf_comparison() - Multi-model PDF")
    
    print("\n🔬 STATISTICAL TESTS:")
    print("   • plot_ks_test_analysis() - Kolmogorov-Smirnov test")
    print("   • plot_autocorrelation_analysis() - Autocorrelation analysis")
    
    print("\n🆚 MODEL COMPARISON:")
    print("   • plot_models_comparison_overview() - Comparative overview")
    print("   • plot_predictions_comparison() - Compare predictions")
    print("   • plot_residuals_comparison() - Compare residuals")
    print("   • plot_training_comparison() - Compare training")
    
    print("\n📄 REPORTS:")
    print("   • generate_comprehensive_report() - Complete report")
    print("   • generate_text_report() - Text report")
    print("   • generate_quick_summary() - Quick summary")
    
    print("\n" + "=" * 60)
    print("🚀 For more information, use help(function)")


def quick_start_guide():
    """
    Quick start guide for the package
    """
    guide = """
    🚀 QUICK START GUIDE - VISUALIZATION PACKAGE
    ===========================================
    
    📋 DATA PREPARATION:
    -------------------
    Data should be in dictionary format:
    
    # For a single model:
    results = {
        'actuals': [actual_values],
        'predictions': [predictions],
        'train_losses': [training_losses],  # optional
        'val_losses': [validation_losses]  # optional
    }
    
    # For multiple models:
    results_dict = {
        'Model_1': {
            'actuals': [...],
            'predictions': [...],
            ...
        },
        'Model_2': {
            'actuals': [...],
            'predictions': [...],
            ...
        }
    }
    
    ⚡ QUICK USAGE:
    --------------
    
    # 1. Import the package
    from mackey_glass_prediction.utils import visualization as viz
    
    # 2. Basic prediction plot
    viz.plot_predictions(actuals, predictions, save_path="predictions.png")
    
    # 3. Q-Q analysis
    viz.plot_qq_analysis(actuals, predictions, save_path="qq_plot.png")
    
    # 4. Model comparison
    viz.plot_models_comparison_overview(results_dict, save_path="comparison.png")
    
    # 5. Complete report
    generated_files = viz.generate_comprehensive_report(
        results_dict, 
        output_dir="report/"
    )
    
    # 6. Quick summary in console
    print(viz.generate_quick_summary(results_dict))
    
    🎯 TIPS:
    --------
    • Use save_path=None to show plots without saving
    • All plots have customizable titles and parameters
    • Comprehensive report generates all types of analysis
    • Use show_available_functions() to see all options
    
    📚 DOCUMENTATION:
    -----------------
    • Each function has detailed docstring
    • Use help(viz.function) for more information
    • Examples available in docstrings
    """
    
    print(guide)


# Convenience function for quick analysis
def quick_analysis(results_dict, output_dir="analysis_output", model_name=None):
    """
    Quick analysis with the most important plots
    
    Args:
        results_dict: Model data
        output_dir: Output directory
        model_name: Model name (for single analysis)
    
    Returns:
        dict: Generated files
    """
    ensure_output_dir(output_dir)
    
    print("🚀 Starting quick analysis...")
    
    # If it's a single model
    if model_name and isinstance(results_dict, dict) and 'actuals' in results_dict:
        single_model = {model_name: results_dict}
        
        generated = {}
        
        # Basic plots
        if 'actuals' in results_dict and 'predictions' in results_dict:
            pred_path = f"{output_dir}/predictions_{model_name}.png"
            plot_predictions(results_dict['actuals'], results_dict['predictions'], 
                           save_path=pred_path)
            generated['predictions'] = pred_path
            
            qq_path = f"{output_dir}/qq_plot_{model_name}.png"
            plot_qq_analysis(results_dict['actuals'], results_dict['predictions'], 
                           save_path=qq_path)
            generated['qq_plot'] = qq_path
        
        if 'train_losses' in results_dict and 'val_losses' in results_dict:
            train_path = f"{output_dir}/training_{model_name}.png"
            plot_training_history(results_dict['train_losses'], results_dict['val_losses'], 
                                save_path=train_path)
            generated['training'] = train_path
        
        print(f"✅ Quick analysis completed! {len(generated)} files generated.")
        return generated
    
    else:
        # Multiple models
        generated = {}
        
        # Comparative overview
        overview_path = f"{output_dir}/models_overview.png"
        plot_models_comparison_overview(results_dict, save_path=overview_path)
        generated['overview'] = overview_path
        
        # Predictions comparison
        pred_comp_path = f"{output_dir}/predictions_comparison.png"
        plot_predictions_comparison(results_dict, save_path=pred_comp_path)
        generated['predictions_comparison'] = pred_comp_path
        
        print(f"✅ Quick analysis completed! {len(generated)} files generated.")
        return generated 
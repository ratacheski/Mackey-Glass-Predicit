#!/usr/bin/env python3
"""
Script to generate interactive and didactic HTML report
"""

import os
import sys
import numpy as np
from datetime import datetime

# Add current directory to PYTHONPATH
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.visualization.interactive_html import generate_interactive_html_report

def create_sample_results():
    """Creates sample results based on project models"""
    
    # Simulate time series data (Mackey-Glass)
    np.random.seed(42)
    n_samples = 1000
    
    # "Real" values (simulated)
    t = np.linspace(0, 100, n_samples)
    actuals = np.sin(0.1 * t) + 0.1 * np.sin(0.5 * t) + 0.05 * np.random.randn(n_samples)
    
    # Simulate predictions for different models with different qualities
    models_data = {
        'LSTM Medium': {
            'actuals': actuals,
            'predictions': actuals + 0.02 * np.random.randn(n_samples),  # Very good
            'train_losses': np.exp(-np.linspace(0, 5, 100)) + 0.01 * np.random.randn(100),
            'val_losses': np.exp(-np.linspace(0, 4.5, 100)) + 0.015 * np.random.randn(100),
            'model_info': {
                'total_parameters': 50113,
                'architecture': 'LSTM',
                'hidden_size': 64
            }
        },
        'GRU Medium': {
            'actuals': actuals,
            'predictions': actuals + 0.015 * np.random.randn(n_samples) + 0.01,  # Excellent
            'train_losses': np.exp(-np.linspace(0, 5.2, 100)) + 0.008 * np.random.randn(100),
            'val_losses': np.exp(-np.linspace(0, 4.8, 100)) + 0.012 * np.random.randn(100),
            'model_info': {
                'total_parameters': 37633,
                'architecture': 'GRU',
                'hidden_size': 64
            }
        },
        'MLP Medium': {
            'actuals': actuals,
            'predictions': actuals + 0.08 * np.random.randn(n_samples) + 0.05,  # Moderate
            'train_losses': np.exp(-np.linspace(0, 3.5, 100)) + 0.02 * np.random.randn(100),
            'val_losses': np.exp(-np.linspace(0, 3, 100)) + 0.025 * np.random.randn(100),
            'model_info': {
                'total_parameters': 14625,
                'architecture': 'MLP',
                'hidden_layers': 3
            }
        },
        'RNN Basic': {
            'actuals': actuals,
            'predictions': actuals + 0.12 * np.random.randn(n_samples) + 0.08,  # Low
            'train_losses': np.exp(-np.linspace(0, 2.8, 100)) + 0.03 * np.random.randn(100),
            'val_losses': np.exp(-np.linspace(0, 2.5, 100)) + 0.035 * np.random.randn(100),
            'model_info': {
                'total_parameters': 8421,
                'architecture': 'RNN',
                'hidden_size': 32
            }
        }
    }
    
    return models_data

def create_sample_files():
    """Creates list of sample files that would be generated"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    sample_files = {
        'overview': f'01_overview_{timestamp}.png',
        'training_LSTM_Medium': f'02_LSTM_Medium_training_{timestamp}.png',
        'predictions_LSTM_Medium': f'02_LSTM_Medium_predictions_{timestamp}.png',
        'training_GRU_Medium': f'03_GRU_Medium_training_{timestamp}.png',
        'predictions_GRU_Medium': f'03_GRU_Medium_predictions_{timestamp}.png',
        'training_MLP_Medium': f'04_MLP_Medium_training_{timestamp}.png',
        'predictions_MLP_Medium': f'04_MLP_Medium_predictions_{timestamp}.png',
        'qq_LSTM_Medium': f'02_LSTM_Medium_qq_plot_{timestamp}.png',
        'cdf_LSTM_Medium': f'02_LSTM_Medium_cdf_{timestamp}.png',
        'metrics_table': f'99_metrics_table_{timestamp}.png',
        'metrics_comparison': f'99_metrics_comparison_{timestamp}.png'
    }
    
    return sample_files

def main():
    """Main function to generate the interactive report"""
    print("🚀 Generating Interactive and Didactic HTML Report...")
    print("=" * 60)
    print("📝 Developed by: Rafael Ratacheski de Sousa Raulino")
    print("🎓 Master's Student in Electrical and Computer Engineering - UFG")
    print("📚 Course: Deep Neural Networks - 2025/1")
    print("=" * 60)
    
    # Create output directory
    output_dir = "output_reports"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create sample data
    print("📊 Creating sample data...")
    sample_results = create_sample_results()
    sample_files = create_sample_files()
    
    # Show preview of calculated metrics
    print("\n📈 Preview of Calculated Metrics:")
    print("-" * 50)
    
    from utils.visualization.interactive_html import calculate_metrics
    
    for model_name, data in sample_results.items():
        if 'actuals' in data and 'predictions' in data:
            metrics = calculate_metrics(data['actuals'], data['predictions'])
            print(f"\n🤖 {model_name}:")
            print(f"   R²: {metrics['r2']:.6f}")
            print(f"   RMSE: {metrics['rmse']:.6f}")
            print(f"   MAE: {metrics['mae']:.6f}")
            print(f"   MSE: {metrics['mse']:.6f}")
            if not np.isnan(metrics['mape']):
                print(f"   MAPE: {metrics['mape']:.2f}%")
            if not np.isnan(metrics['eqmn1']):
                print(f"   EQMN1: {metrics['eqmn1']:.6f}")
            if not np.isnan(metrics['eqmn2']):
                print(f"   EQMN2: {metrics['eqmn2']:.6f}")
    
    # Generate interactive HTML report
    html_path = os.path.join(output_dir, f"report.html")
    
    print(f"\n🌐 Generating HTML report...")
    
    try:
        generate_interactive_html_report(
            results_dict=sample_results,
            generated_files=sample_files,
            save_path=html_path,
            report_type="comparison"
        )
        
        print("\n✅ Report generated successfully!")
        print(f"📁 File: {html_path}")
        print(f"🌐 To view, open the file in a web browser")
        print("\n🔧 Included features:")
        print("   • 📊 Detailed metrics (R², RMSE, MAE, MSE, MAPE, EQMN1, EQMN2)")
        print("   • 🖼️  Full-screen image visualization")
        print("   • 📈 Graphs organized by model")
        print("   • 📋 Interactive comparisons")
        print("   • 👨‍🎓 Author information")
        
        # Try to open automatically in browser (Linux)
        try:
            import webbrowser
            webbrowser.open(f'file://{os.path.abspath(html_path)}')
            print("🚀 Opening report in browser...")
        except:
            print("💡 Manually open the file in a browser to view")
            
    except Exception as e:
        print(f"❌ Error generating report: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 
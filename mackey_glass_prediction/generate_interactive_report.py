#!/usr/bin/env python3
"""
Script para gerar relatório HTML interativo e didático
"""

import os
import sys
import numpy as np
from datetime import datetime

# Adicionar o diretório atual ao PYTHONPATH
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.visualization.interactive_html import generate_interactive_html_report

def create_sample_results():
    """Cria resultados de exemplo baseados nos modelos do projeto"""
    
    # Simular dados de séries temporais (Mackey-Glass)
    np.random.seed(42)
    n_samples = 1000
    
    # Valores "reais" (simulados)
    t = np.linspace(0, 100, n_samples)
    actuals = np.sin(0.1 * t) + 0.1 * np.sin(0.5 * t) + 0.05 * np.random.randn(n_samples)
    
    # Simular predições para diferentes modelos com diferentes qualidades
    models_data = {
        'LSTM Medium': {
            'actuals': actuals,
            'predictions': actuals + 0.02 * np.random.randn(n_samples),  # Muito bom
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
            'predictions': actuals + 0.015 * np.random.randn(n_samples) + 0.01,  # Excelente
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
            'predictions': actuals + 0.08 * np.random.randn(n_samples) + 0.05,  # Moderado
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
            'predictions': actuals + 0.12 * np.random.randn(n_samples) + 0.08,  # Baixo
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
    """Cria lista de arquivos de exemplo que seriam gerados"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    sample_files = {
        'overview': f'01_visao_geral_{timestamp}.png',
        'training_LSTM_Medium': f'02_LSTM_Medium_treinamento_{timestamp}.png',
        'predictions_LSTM_Medium': f'02_LSTM_Medium_predicoes_{timestamp}.png',
        'training_GRU_Medium': f'03_GRU_Medium_treinamento_{timestamp}.png',
        'predictions_GRU_Medium': f'03_GRU_Medium_predicoes_{timestamp}.png',
        'training_MLP_Medium': f'04_MLP_Medium_treinamento_{timestamp}.png',
        'predictions_MLP_Medium': f'04_MLP_Medium_predicoes_{timestamp}.png',
        'qq_LSTM_Medium': f'02_LSTM_Medium_qq_plot_{timestamp}.png',
        'cdf_LSTM_Medium': f'02_LSTM_Medium_cdf_{timestamp}.png',
        'metrics_table': f'99_tabela_metricas_{timestamp}.png',
        'metrics_comparison': f'99_comparacao_metricas_{timestamp}.png'
    }
    
    return sample_files

def main():
    """Função principal para gerar o relatório interativo"""
    print("🚀 Gerando Relatório HTML Interativo e Didático...")
    print("=" * 60)
    print("📝 Desenvolvido por: Rafael Ratacheski de Sousa Raulino")
    print("🎓 Mestrando em Engenharia Elétrica e de Computação - UFG")
    print("📚 Disciplina: Redes Neurais Profundas - 2025/1")
    print("=" * 60)
    
    # Criar diretório de saída
    output_dir = "output_reports"
    os.makedirs(output_dir, exist_ok=True)
    
    # Criar dados de exemplo
    print("📊 Criando dados de exemplo...")
    sample_results = create_sample_results()
    sample_files = create_sample_files()
    
    # Mostrar preview das métricas calculadas
    print("\n📈 Preview das Métricas Calculadas:")
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
    
    # Gerar relatório HTML interativo
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    html_path = os.path.join(output_dir, f"relatorio_interativo_{timestamp}.html")
    
    print(f"\n🌐 Gerando relatório HTML...")
    
    try:
        generate_interactive_html_report(
            results_dict=sample_results,
            generated_files=sample_files,
            save_path=html_path,
            report_type="comparison"
        )
        
        print("\n✅ Relatório gerado com sucesso!")
        print(f"📁 Arquivo: {html_path}")
        print(f"🌐 Para visualizar, abra o arquivo em um navegador web")
        print("\n🔧 Funcionalidades incluídas:")
        print("   • 📊 Métricas detalhadas (R², RMSE, MAE, MSE, MAPE, EQMN1, EQMN2)")
        print("   • 🖼️  Visualização de imagens em tela cheia")
        print("   • 📈 Gráficos organizados por modelo")
        print("   • 📋 Comparações interativas")
        print("   • 👨‍🎓 Informações do autor")
        
        # Tentar abrir automaticamente no navegador (Linux)
        try:
            import webbrowser
            webbrowser.open(f'file://{os.path.abspath(html_path)}')
            print("🚀 Abrindo relatório no navegador...")
        except:
            print("💡 Abra manualmente o arquivo no navegador para visualizar")
            
    except Exception as e:
        print(f"❌ Erro ao gerar relatório: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 
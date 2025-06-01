"""
Pacote de visualização modular para análise de modelos de machine learning

Este pacote contém módulos especializados para diferentes tipos de visualizações:
- utils: Funções utilitárias e formatação
- basic_plots: Gráficos básicos (treinamento, predições)
- distribution_analysis: Análises de distribuição (QQ-plot, FDA, FDP)
- statistical_tests: Testes estatísticos (KS, autocorrelação)
- comparison_plots: Comparações entre modelos
- reports: Geração de relatórios abrangentes
"""

# Configurar matplotlib para não exibir gráficos na tela
import matplotlib
matplotlib.use('Agg')  # Backend não-interativo que apenas salva arquivos
import matplotlib.pyplot as plt

# Configurações adicionais para melhor qualidade e desempenho
plt.ioff()  # Desabilitar modo interativo

# Imports principais de cada módulo
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
    plot_training_comparison,
    plot_performance_radar
)

from .reports import (
    generate_comprehensive_report,
    generate_text_report,
    generate_html_report,
    generate_quick_summary
)

# Versão do pacote
__version__ = "1.0.0"

# Funções principais (atalhos)
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
    'plot_performance_radar',
    
    # Reports
    'generate_comprehensive_report',
    'generate_text_report',
    'generate_html_report',
    'generate_quick_summary'
]


def show_available_functions():
    """
    Mostra todas as funções disponíveis no pacote de visualização
    """
    print("📊 PACOTE DE VISUALIZAÇÃO - FUNÇÕES DISPONÍVEIS")
    print("=" * 60)
    
    print("\n🛠️  UTILITÁRIOS:")
    print("   • format_metric_value() - Formatar valores de métricas")
    print("   • validate_and_clean_metrics() - Validar dados de entrada")
    print("   • ensure_output_dir() - Garantir diretório de saída")
    print("   • get_colors_and_styles() - Obter cores e estilos")
    print("   • add_metrics_text_box() - Adicionar caixa de métricas")
    print("   • print_save_message() - Mensagem de confirmação")
    
    print("\n📈 GRÁFICOS BÁSICOS:")
    print("   • plot_training_history() - Histórico de treinamento")
    print("   • plot_predictions() - Predições vs valores reais")
    print("   • plot_metrics_comparison() - Comparação de métricas")
    print("   • save_metrics_table() - Salvar tabela de métricas")
    
    print("\n📊 ANÁLISE DE DISTRIBUIÇÕES:")
    print("   • plot_qq_analysis() - Q-Q Plot")
    print("   • plot_cdf_comparison() - Comparação FDA")
    print("   • plot_pdf_comparison() - Comparação FDP")
    print("   • plot_distribution_analysis() - Análise completa")
    print("   • plot_multi_model_cdf_comparison() - FDA múltiplos modelos")
    print("   • plot_multi_model_pdf_comparison() - FDP múltiplos modelos")
    
    print("\n🔬 TESTES ESTATÍSTICOS:")
    print("   • plot_ks_test_analysis() - Teste Kolmogorov-Smirnov")
    print("   • plot_autocorrelation_analysis() - Análise autocorrelação")
    
    print("\n🆚 COMPARAÇÃO DE MODELOS:")
    print("   • plot_models_comparison_overview() - Visão geral comparativa")
    print("   • plot_predictions_comparison() - Comparar predições")
    print("   • plot_residuals_comparison() - Comparar resíduos")
    print("   • plot_training_comparison() - Comparar treinamento")
    print("   • plot_performance_radar() - Gráfico radar")
    
    print("\n📄 RELATÓRIOS:")
    print("   • generate_comprehensive_report() - Relatório completo")
    print("   • generate_text_report() - Relatório textual")
    print("   • generate_html_report() - Relatório HTML")
    print("   • generate_quick_summary() - Resumo rápido")
    
    print("\n" + "=" * 60)
    print("🚀 Para mais informações, use help(função)")


def quick_start_guide():
    """
    Guia rápido de uso do pacote
    """
    guide = """
    🚀 GUIA RÁPIDO - PACOTE DE VISUALIZAÇÃO
    ======================================
    
    📋 PREPARAÇÃO DOS DADOS:
    -----------------------
    Os dados devem estar no formato de dicionário:
    
    # Para um único modelo:
    results = {
        'actuals': [valores_reais],
        'predictions': [predições],
        'train_losses': [losses_treino],  # opcional
        'val_losses': [losses_validação]  # opcional
    }
    
    # Para múltiplos modelos:
    results_dict = {
        'Modelo_1': {
            'actuals': [...],
            'predictions': [...],
            ...
        },
        'Modelo_2': {
            'actuals': [...],
            'predictions': [...],
            ...
        }
    }
    
    ⚡ USO RÁPIDO:
    -------------
    
    # 1. Importar o pacote
    from mackey_glass_prediction.utils import visualization as viz
    
    # 2. Gráfico básico de predições
    viz.plot_predictions(actuals, predictions, save_path="predicoes.png")
    
    # 3. Análise Q-Q
    viz.plot_qq_analysis(actuals, predictions, save_path="qq_plot.png")
    
    # 4. Comparação de modelos
    viz.plot_models_comparison_overview(results_dict, save_path="comparacao.png")
    
    # 5. Relatório completo
    generated_files = viz.generate_comprehensive_report(
        results_dict, 
        output_dir="relatorio/"
    )
    
    # 6. Resumo rápido no console
    print(viz.generate_quick_summary(results_dict))
    
    🎯 DICAS:
    ---------
    • Use save_path=None para mostrar gráficos sem salvar
    • Todos os gráficos têm títulos e parâmetros customizáveis
    • O relatório abrangente gera todos os tipos de análise
    • Use show_available_functions() para ver todas as opções
    
    📚 DOCUMENTAÇÃO:
    ----------------
    • Cada função tem docstring detalhada
    • Use help(viz.função) para mais informações
    • Exemplos disponíveis nos docstrings
    """
    
    print(guide)


# Função de conveniência para análise rápida
def quick_analysis(results_dict, output_dir="analysis_output", model_name=None):
    """
    Análise rápida com os gráficos mais importantes
    
    Args:
        results_dict: Dados dos modelos
        output_dir: Diretório de saída
        model_name: Nome do modelo (para análise única)
    
    Returns:
        dict: Arquivos gerados
    """
    ensure_output_dir(output_dir)
    
    print("🚀 Iniciando análise rápida...")
    
    # Se for um único modelo
    if model_name and isinstance(results_dict, dict) and 'actuals' in results_dict:
        single_model = {model_name: results_dict}
        
        generated = {}
        
        # Gráficos básicos
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
        
        print(f"✅ Análise rápida concluída! {len(generated)} arquivos gerados.")
        return generated
    
    else:
        # Múltiplos modelos
        generated = {}
        
        # Visão geral comparativa
        overview_path = f"{output_dir}/models_overview.png"
        plot_models_comparison_overview(results_dict, save_path=overview_path)
        generated['overview'] = overview_path
        
        # Gráfico radar
        radar_path = f"{output_dir}/performance_radar.png"
        plot_performance_radar(results_dict, save_path=radar_path)
        generated['radar'] = radar_path
        
        # Comparação de predições
        pred_comp_path = f"{output_dir}/predictions_comparison.png"
        plot_predictions_comparison(results_dict, save_path=pred_comp_path)
        generated['predictions_comparison'] = pred_comp_path
        
        print(f"✅ Análise rápida concluída! {len(generated)} arquivos gerados.")
        return generated 
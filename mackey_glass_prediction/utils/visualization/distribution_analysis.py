"""
Análises distribucionais: QQ-Plot, FDA (CDF) e FDP (PDF)
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.stats import gaussian_kde
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from .utils import ensure_output_dir, print_save_message, get_colors_and_styles


def plot_qq_analysis(actuals, predictions, save_path=None, 
                    title="QQ-Plot: Quantis Predições vs Valores Reais"):
    """
    Cria QQ-Plot (Quantile-Quantile) para comparar distribuições das predições vs valores reais
    
    Args:
        actuals: Valores reais
        predictions: Predições do modelo
        save_path: Caminho para salvar o gráfico
        title: Título do gráfico
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Garantir que são arrays numpy e achatar
    actuals = np.array(actuals).flatten()
    predictions = np.array(predictions).flatten()
    
    # ===== SUBPLOT 1: QQ-Plot Principal =====
    ax1.set_title(title, fontsize=14, fontweight='bold')
    
    # Ordenar os dados para calcular quantis
    actuals_sorted = np.sort(actuals)
    predictions_sorted = np.sort(predictions)
    
    # Se os tamanhos forem diferentes, interpolar para igualar
    if len(actuals_sorted) != len(predictions_sorted):
        # Usar o menor tamanho como referência
        min_size = min(len(actuals_sorted), len(predictions_sorted))
        
        # Criar quantis uniformes
        quantiles = np.linspace(0, 1, min_size)
        
        # Interpolar para obter quantis correspondentes
        actuals_quantiles = np.quantile(actuals_sorted, quantiles)
        predictions_quantiles = np.quantile(predictions_sorted, quantiles)
    else:
        # Se têm o mesmo tamanho, usar diretamente
        actuals_quantiles = actuals_sorted
        predictions_quantiles = predictions_sorted
    
    # Plotar QQ-plot
    ax1.scatter(actuals_quantiles, predictions_quantiles, alpha=0.6, s=30, color='blue', edgecolors='darkblue')
    
    # Linha de referência perfeita (y=x)
    min_val = min(np.min(actuals_quantiles), np.min(predictions_quantiles))
    max_val = max(np.max(actuals_quantiles), np.max(predictions_quantiles))
    ax1.plot([min_val, max_val], [min_val, max_val], 'r-', linewidth=3, 
             label='Linha Perfeita (distribuições idênticas)', alpha=0.8)
    
    # Linha de regressão através dos quantis
    try:
        slope, intercept, r_value, p_value, std_err = stats.linregress(actuals_quantiles, predictions_quantiles)
        regression_line = slope * actuals_quantiles + intercept
        ax1.plot(actuals_quantiles, regression_line, 'g--', linewidth=2, 
                 label=f'Regressão (R²={r_value**2:.4f})', alpha=0.8)
        
        # Adicionar equação da reta
        ax1.text(0.05, 0.95, f'y = {slope:.3f}x + {intercept:.3f}\nR² = {r_value**2:.4f}', 
                transform=ax1.transAxes, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7),
                verticalalignment='top', fontsize=10, fontfamily='monospace')
        
    except Exception as e:
        print(f"Aviso: Erro ao calcular regressão: {e}")
    
    ax1.set_xlabel('Quantis - Valores Reais', fontsize=12)
    ax1.set_ylabel('Quantis - Predições', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Adicionar métricas no gráfico
    r2 = r2_score(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mae = mean_absolute_error(actuals, predictions)
    
    # Teste de normalidade nos resíduos dos quantis
    residuals_qq = predictions_quantiles - actuals_quantiles
    shapiro_stat, shapiro_p = stats.shapiro(residuals_qq) if len(residuals_qq) <= 5000 else (np.nan, np.nan)
    
    metrics_text = (
        f'Métricas Distribucionais:\n'
        f'R² global: {r2:.4f}\n'
        f'RMSE: {rmse:.4f}\n'
        f'MAE: {mae:.4f}\n'
        f'Shapiro-Wilk (resíduos QQ):\n'
        f'  Estatística: {shapiro_stat:.4f}\n'
        f'  p-value: {shapiro_p:.4f}' if not np.isnan(shapiro_stat) else 'Shapiro-Wilk: N/A (amostra grande)'
    )
    
    ax1.text(0.02, 0.02, metrics_text, transform=ax1.transAxes, 
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
             verticalalignment='bottom', fontsize=9, fontfamily='monospace')
    
    # ===== SUBPLOT 2: Análise de Desvios por Quantil =====
    ax2.set_title('Análise de Desvios por Quantil', fontsize=14, fontweight='bold')
    
    # Calcular desvios relativos
    desvios = (predictions_quantiles - actuals_quantiles) / actuals_quantiles * 100
    quantil_positions = np.linspace(0, 100, len(desvios))
    
    # Plotar desvios
    ax2.plot(quantil_positions, desvios, 'b-', linewidth=2, alpha=0.8, label='Desvio Relativo (%)')
    ax2.axhline(y=0, color='red', linestyle='-', linewidth=2, alpha=0.8, label='Zero (perfeito)')
    ax2.fill_between(quantil_positions, desvios, 0, alpha=0.3, color='blue')
    
    # Destacar quantis extremos (10% e 90%)
    q10_idx = int(len(desvios) * 0.1)
    q90_idx = int(len(desvios) * 0.9)
    
    ax2.scatter([10, 90], [desvios[q10_idx], desvios[q90_idx]], 
               color='red', s=100, zorder=5, label='Quantis 10% e 90%')
    
    ax2.set_xlabel('Percentil (%)', fontsize=12)
    ax2.set_ylabel('Desvio Relativo (%)', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Análise dos desvios
    desvio_medio = np.mean(np.abs(desvios))
    desvio_max = np.max(np.abs(desvios))
    desvio_std = np.std(desvios)
    
    # Identificar onde estão os maiores desvios
    worst_quantile = quantil_positions[np.argmax(np.abs(desvios))]
    
    analysis_text = (
        f'Análise de Desvios:\n'
        f'Desvio médio: ±{desvio_medio:.2f}%\n'
        f'Desvio máximo: ±{desvio_max:.2f}%\n'
        f'Desvio padrão: {desvio_std:.2f}%\n'
        f'Pior quantil: {worst_quantile:.0f}%\n'
        f'Desvio Q10: {desvios[q10_idx]:+.2f}%\n'
        f'Desvio Q90: {desvios[q90_idx]:+.2f}%'
    )
    
    ax2.text(0.02, 0.98, analysis_text, transform=ax2.transAxes, 
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
             verticalalignment='top', fontsize=9, fontfamily='monospace')
    
    # Interpretação
    if desvio_medio < 5:
        interpretation = "EXCELENTE: Distribuições muito similares"
        interp_color = 'lightgreen'
    elif desvio_medio < 10:
        interpretation = "BOM: Distribuições similares"
        interp_color = 'lightblue'
    elif desvio_medio < 20:
        interpretation = "MODERADO: Algumas diferenças distribucionais"
        interp_color = 'lightyellow'
    else:
        interpretation = "RUIM: Diferenças significativas nas distribuições"
        interp_color = 'lightcoral'
    
    ax2.text(0.02, 0.02, f'INTERPRETAÇÃO: {interpretation}', transform=ax2.transAxes, 
             bbox=dict(boxstyle='round', facecolor=interp_color, alpha=0.9),
             verticalalignment='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        ensure_output_dir(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print_save_message(save_path, "QQ-Plot")
    
    plt.show()
    
    # Retornar métricas do QQ-plot
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
                       title="Comparação FDA - Função Distribuição Acumulada"):
    """
    Plota comparação das Funções de Distribuição Acumulada (FDA/CDF)
    
    Args:
        actuals: Valores reais
        predictions: Predições do modelo
        save_path: Caminho para salvar o gráfico
        title: Título do gráfico
    """
    plt.figure(figsize=(12, 8))
    
    # Garantir que são arrays numpy
    actuals = np.array(actuals).flatten()
    predictions = np.array(predictions).flatten()
    
    # Range combinado para avaliação
    combined_range = np.linspace(
        min(np.min(predictions), np.min(actuals)),
        max(np.max(predictions), np.max(actuals)),
        1000
    )
    
    # Calcular CDFs empíricas
    cdf_actuals = np.array([np.mean(actuals <= x) for x in combined_range])
    cdf_predictions = np.array([np.mean(predictions <= x) for x in combined_range])
    
    # Plotar CDFs
    plt.plot(combined_range, cdf_actuals, 'b-', linewidth=2.5, label='Valores Reais', alpha=0.8)
    plt.plot(combined_range, cdf_predictions, 'r--', linewidth=2.5, label='Predições', alpha=0.8)
    
    # Adicionar área entre as curvas
    plt.fill_between(combined_range, cdf_actuals, cdf_predictions, alpha=0.2, color='gray', label='Diferença')
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Valor', fontsize=12)
    plt.ylabel('Probabilidade Acumulada', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Adicionar métricas no gráfico
    ks_statistic, ks_pvalue = stats.ks_2samp(predictions, actuals)
    fda_distance = np.mean(np.abs(cdf_predictions - cdf_actuals))
    
    metrics_text = f'KS Statistic: {ks_statistic:.4f}\nKS p-value: {ks_pvalue:.4f}\nDistância Média: {fda_distance:.4f}'
    plt.text(0.02, 0.98, metrics_text, transform=plt.gca().transAxes, 
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
             verticalalignment='top', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        ensure_output_dir(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print_save_message(save_path, "Gráfico FDA")
    
    plt.show()


def plot_pdf_comparison(actuals, predictions, save_path=None, 
                       title="Comparação FDP - Função de Distribuição de Probabilidade"):
    """
    Plota comparação das Funções de Distribuição de Probabilidade (FDP/PDF) usando KDE
    
    Args:
        actuals: Valores reais
        predictions: Predições do modelo
        save_path: Caminho para salvar o gráfico
        title: Título do gráfico
    """
    plt.figure(figsize=(12, 8))
    
    # Garantir que são arrays numpy
    actuals = np.array(actuals).flatten()
    predictions = np.array(predictions).flatten()
    
    try:
        # Range combinado para avaliação
        combined_range = np.linspace(
            min(np.min(predictions), np.min(actuals)),
            max(np.max(predictions), np.max(actuals)),
            1000
        )
        
        # Kernel Density Estimation
        kde_actuals = gaussian_kde(actuals)
        kde_predictions = gaussian_kde(predictions)
        
        # Avaliar PDFs
        pdf_actuals = kde_actuals(combined_range)
        pdf_predictions = kde_predictions(combined_range)
        
        # Plotar PDFs
        plt.plot(combined_range, pdf_actuals, 'b-', linewidth=2.5, label='Valores Reais', alpha=0.8)
        plt.plot(combined_range, pdf_predictions, 'r--', linewidth=2.5, label='Predições', alpha=0.8)
        
        # Adicionar área entre as curvas
        plt.fill_between(combined_range, pdf_actuals, pdf_predictions, alpha=0.2, color='gray', label='Diferença')
        
        # Adicionar histogramas normalizados para contexto
        plt.hist(actuals, bins=50, density=True, alpha=0.3, color='blue', label='Hist. Reais')
        plt.hist(predictions, bins=50, density=True, alpha=0.3, color='red', label='Hist. Predições')
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Valor', fontsize=12)
        plt.ylabel('Densidade de Probabilidade', fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        
        # Calcular métricas FDP
        fdp_l2_distance = np.sqrt(np.trapz((pdf_predictions - pdf_actuals)**2, combined_range))
        
        # Normalizar PDFs para JS divergence
        pdf_pred_norm = pdf_predictions / np.trapz(pdf_predictions, combined_range)
        pdf_actual_norm = pdf_actuals / np.trapz(pdf_actuals, combined_range)
        pdf_mean = 0.5 * (pdf_pred_norm + pdf_actual_norm)
        
        # JS divergence com proteção
        epsilon = 1e-10
        kl_pred_mean = np.trapz(pdf_pred_norm * np.log((pdf_pred_norm + epsilon) / (pdf_mean + epsilon)), combined_range)
        kl_actual_mean = np.trapz(pdf_actual_norm * np.log((pdf_actual_norm + epsilon) / (pdf_mean + epsilon)), combined_range)
        js_divergence = 0.5 * (kl_pred_mean + kl_actual_mean)
        
        # Adicionar métricas no gráfico
        metrics_text = f'Distância L2: {fdp_l2_distance:.4f}\nJS Divergence: {js_divergence:.4f}'
        plt.text(0.02, 0.98, metrics_text, transform=plt.gca().transAxes, 
                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
                 verticalalignment='top', fontsize=10)
        
    except Exception as e:
        plt.text(0.5, 0.5, f'Erro ao calcular KDE: {str(e)}', 
                transform=plt.gca().transAxes, ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        print(f"Aviso: Erro ao plotar FDP: {e}")
    
    plt.tight_layout()
    
    if save_path:
        ensure_output_dir(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print_save_message(save_path, "Gráfico FDP")
    
    plt.show()


def plot_distribution_analysis(actuals, predictions, save_path=None, 
                              title_prefix="Análise Distribucional"):
    """
    Cria análise completa das distribuições (FDA + FDP) em uma única figura
    
    Args:
        actuals: Valores reais
        predictions: Predições do modelo
        save_path: Caminho para salvar o gráfico
        title_prefix: Prefixo para o título
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Garantir que são arrays numpy
    actuals = np.array(actuals).flatten()
    predictions = np.array(predictions).flatten()
    
    # Range combinado
    combined_range = np.linspace(
        min(np.min(predictions), np.min(actuals)),
        max(np.max(predictions), np.max(actuals)),
        1000
    )
    
    # ===== SUBPLOT 1: FDA/CDF =====
    ax1.set_title(f'{title_prefix} - FDA (Função Distribuição Acumulada)', fontsize=12, fontweight='bold')
    
    # Calcular CDFs empíricas
    cdf_actuals = np.array([np.mean(actuals <= x) for x in combined_range])
    cdf_predictions = np.array([np.mean(predictions <= x) for x in combined_range])
    
    # Plotar CDFs
    ax1.plot(combined_range, cdf_actuals, 'b-', linewidth=2.5, label='Valores Reais', alpha=0.8)
    ax1.plot(combined_range, cdf_predictions, 'r--', linewidth=2.5, label='Predições', alpha=0.8)
    ax1.fill_between(combined_range, cdf_actuals, cdf_predictions, alpha=0.2, color='gray', label='Diferença')
    
    ax1.set_xlabel('Valor', fontsize=10)
    ax1.set_ylabel('Probabilidade Acumulada', fontsize=10)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Métricas FDA
    ks_statistic, ks_pvalue = stats.ks_2samp(predictions, actuals)
    fda_distance = np.mean(np.abs(cdf_predictions - cdf_actuals))
    metrics_text1 = f'KS: {ks_statistic:.4f}\np-val: {ks_pvalue:.4f}\nDist: {fda_distance:.4f}'
    ax1.text(0.02, 0.98, metrics_text1, transform=ax1.transAxes, 
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
             verticalalignment='top', fontsize=9)
    
    # ===== SUBPLOT 2: FDP/PDF =====
    ax2.set_title(f'{title_prefix} - FDP (Função de Distribuição de Probabilidade)', fontsize=12, fontweight='bold')
    
    try:
        # Kernel Density Estimation
        kde_actuals = gaussian_kde(actuals)
        kde_predictions = gaussian_kde(predictions)
        
        # Avaliar PDFs
        pdf_actuals = kde_actuals(combined_range)
        pdf_predictions = kde_predictions(combined_range)
        
        # Plotar PDFs
        ax2.plot(combined_range, pdf_actuals, 'b-', linewidth=2.5, label='Valores Reais', alpha=0.8)
        ax2.plot(combined_range, pdf_predictions, 'r--', linewidth=2.5, label='Predições', alpha=0.8)
        ax2.fill_between(combined_range, pdf_actuals, pdf_predictions, alpha=0.2, color='gray', label='Diferença')
        
        # Histogramas de contexto
        ax2.hist(actuals, bins=30, density=True, alpha=0.3, color='blue', label='Hist. Reais')
        ax2.hist(predictions, bins=30, density=True, alpha=0.3, color='red', label='Hist. Predições')
        
        ax2.set_xlabel('Valor', fontsize=10)
        ax2.set_ylabel('Densidade de Probabilidade', fontsize=10)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        # Métricas FDP
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
        ax2.text(0.5, 0.5, f'Erro KDE: {str(e)}', transform=ax2.transAxes, ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        ensure_output_dir(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print_save_message(save_path, "Análise distribucional")
    
    plt.show()


def plot_multi_model_cdf_comparison(results_dict, save_path=None, 
                                   title="Comparação FDA - Todos os Modelos"):
    """
    Compara Funções de Distribuição Acumulada (FDA/CDF) de múltiplos modelos
    
    Args:
        results_dict: Dicionário com resultados dos modelos
        save_path: Caminho para salvar o gráfico
        title: Título do gráfico
    """
    plt.figure(figsize=(15, 10))
    
    # Cores para diferentes modelos
    colors, linestyles = get_colors_and_styles(len(results_dict))
    
    # Encontrar range global
    all_actuals = []
    all_predictions = []
    
    for model_name, results in results_dict.items():
        if 'actuals' in results and 'predictions' in results:
            all_actuals.extend(results['actuals'].flatten())
            all_predictions.extend(results['predictions'].flatten())
    
    global_range = np.linspace(min(all_actuals + all_predictions), 
                              max(all_actuals + all_predictions), 1000)
    
    # Plotar CDF dos valores reais (comum para todos)
    actuals_combined = np.array(all_actuals)
    cdf_actuals = np.array([np.mean(actuals_combined <= x) for x in global_range])
    plt.plot(global_range, cdf_actuals, 'black', linewidth=3, 
             label='Valores Reais', alpha=0.8, zorder=10)
    
    # Plotar CDF das predições de cada modelo
    for i, (model_name, results) in enumerate(results_dict.items()):
        if 'predictions' in results:
            predictions = np.array(results['predictions']).flatten()
            cdf_predictions = np.array([np.mean(predictions <= x) for x in global_range])
            
            plt.plot(global_range, cdf_predictions, color=colors[i], linestyle=linestyles[i],
                    linewidth=2.5, label=f'{model_name}', alpha=0.8)
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Valor', fontsize=12)
    plt.ylabel('Probabilidade Acumulada', fontsize=12)
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3)
    
    # Adicionar texto informativo
    plt.text(0.02, 0.02, f'Comparação baseada em {len(results_dict)} modelos\nLinha preta: distribuição real', 
             transform=plt.gca().transAxes, 
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
             fontsize=10, verticalalignment='bottom')
    
    plt.tight_layout()
    
    if save_path:
        ensure_output_dir(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print_save_message(save_path, "Comparação FDA multi-modelo")
    
    plt.show()


def plot_multi_model_pdf_comparison(results_dict, save_path=None, 
                                   title="Comparação FDP - Todos os Modelos"):
    """
    Compara Funções de Distribuição de Probabilidade (FDP/PDF) de múltiplos modelos
    
    Args:
        results_dict: Dicionário com resultados dos modelos
        save_path: Caminho para salvar o gráfico
        title: Título do gráfico
    """
    plt.figure(figsize=(15, 10))
    
    # Cores para diferentes modelos
    colors, linestyles = get_colors_and_styles(len(results_dict))
    
    # Encontrar range global
    all_actuals = []
    all_predictions = []
    
    for model_name, results in results_dict.items():
        if 'actuals' in results and 'predictions' in results:
            all_actuals.extend(results['actuals'].flatten())
            all_predictions.extend(results['predictions'].flatten())
    
    global_range = np.linspace(min(all_actuals + all_predictions), 
                              max(all_actuals + all_predictions), 1000)
    
    try:
        # Plotar PDF dos valores reais (comum para todos)
        actuals_combined = np.array(all_actuals)
        kde_actuals = gaussian_kde(actuals_combined)
        pdf_actuals = kde_actuals(global_range)
        plt.plot(global_range, pdf_actuals, 'black', linewidth=3, 
                 label='Valores Reais', alpha=0.8, zorder=10)
        
        # Plotar PDF das predições de cada modelo
        for i, (model_name, results) in enumerate(results_dict.items()):
            if 'predictions' in results:
                try:
                    predictions = np.array(results['predictions']).flatten()
                    kde_predictions = gaussian_kde(predictions)
                    pdf_predictions = kde_predictions(global_range)
                    
                    plt.plot(global_range, pdf_predictions, color=colors[i], linestyle=linestyles[i],
                            linewidth=2.5, label=f'{model_name}', alpha=0.8)
                except Exception as e:
                    print(f"Aviso: Erro ao calcular KDE para {model_name}: {e}")
                    continue
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Valor', fontsize=12)
        plt.ylabel('Densidade de Probabilidade', fontsize=12)
        plt.legend(fontsize=10, loc='best')
        plt.grid(True, alpha=0.3)
        
        # Adicionar histograma dos valores reais para contexto
        plt.hist(actuals_combined, bins=50, density=True, alpha=0.2, color='black', 
                label='Hist. Reais', zorder=1)
        
        # Adicionar texto informativo
        plt.text(0.02, 0.98, f'Comparação baseada em {len(results_dict)} modelos\nLinha preta: distribuição real', 
                 transform=plt.gca().transAxes, 
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                 fontsize=10, verticalalignment='top')
        
    except Exception as e:
        plt.text(0.5, 0.5, f'Erro ao gerar comparação FDP: {str(e)}', 
                transform=plt.gca().transAxes, ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
        print(f"Erro ao plotar comparação FDP: {e}")
    
    plt.tight_layout()
    
    if save_path:
        ensure_output_dir(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print_save_message(save_path, "Comparação FDP multi-modelo")
    
    plt.show() 
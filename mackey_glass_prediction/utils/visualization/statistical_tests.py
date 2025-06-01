"""
Testes estatísticos: Kolmogorov-Smirnov e análise de autocorrelação
"""
import matplotlib
matplotlib.use('Agg')  # Backend não-interativo que apenas salva arquivos
import matplotlib.pyplot as plt
plt.ioff()  # Desabilitar modo interativo

import numpy as np
from scipy import stats
from statsmodels.tsa.stattools import acf
from scipy.stats import gaussian_kde
from statsmodels.stats.diagnostic import acorr_ljungbox
import os

from .utils import ensure_output_dir, print_save_message, get_colors_and_styles


def plot_ks_test_analysis(actuals, predictions, save_path=None, title="Teste de Kolmogorov-Smirnov de Duas Amostras", alpha=0.05):
    """
    Visualiza em detalhes o teste de Kolmogorov-Smirnov de duas amostras
    
    Args:
        actuals: Valores reais
        predictions: Predições do modelo
        save_path: Caminho para salvar o gráfico
        title: Título do gráfico
        alpha: Nível de significância (padrão: 0.05)
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # Garantir que são arrays numpy
    actuals = np.array(actuals).flatten()
    predictions = np.array(predictions).flatten()
    
    # Realizar teste KS de duas amostras
    ks_statistic, ks_pvalue = stats.ks_2samp(predictions, actuals)
    
    # Determinar resultado do teste
    reject_h0 = ks_pvalue < alpha
    test_conclusion = "REJEITAR H₀" if reject_h0 else "NÃO REJEITAR H₀"
    conclusion_color = "red" if reject_h0 else "green"
    
    # Range para CDFs
    combined_data = np.concatenate([actuals, predictions])
    x_range = np.linspace(np.min(combined_data), np.max(combined_data), 1000)
    
    # ===== SUBPLOT 1: CDFs e Diferença Máxima =====
    ax1.set_title(f'{title}\nEstatística KS = {ks_statistic:.6f} | p-value = {ks_pvalue:.6f} | {test_conclusion}', 
                 fontsize=14, fontweight='bold', color=conclusion_color)
    
    # Calcular CDFs empíricas
    cdf_actuals = np.array([np.mean(actuals <= x) for x in x_range])
    cdf_predictions = np.array([np.mean(predictions <= x) for x in x_range])
    
    # Plotar CDFs
    ax1.plot(x_range, cdf_actuals, 'b-', linewidth=3, label='CDF - Valores Reais', alpha=0.8)
    ax1.plot(x_range, cdf_predictions, 'r-', linewidth=3, label='CDF - Predições', alpha=0.8)
    
    # Encontrar ponto de diferença máxima
    diff = np.abs(cdf_actuals - cdf_predictions)
    max_diff_idx = np.argmax(diff)
    max_diff_x = x_range[max_diff_idx]
    max_diff_y1 = cdf_actuals[max_diff_idx]
    max_diff_y2 = cdf_predictions[max_diff_idx]
    
    # Destacar diferença máxima
    ax1.plot([max_diff_x, max_diff_x], [max_diff_y1, max_diff_y2], 
             'k-', linewidth=4, alpha=0.8, label=f'Diferença Máxima = {ks_statistic:.6f}')
    ax1.plot(max_diff_x, max_diff_y1, 'bo', markersize=10, markeredgecolor='black', markeredgewidth=2)
    ax1.plot(max_diff_x, max_diff_y2, 'ro', markersize=10, markeredgecolor='black', markeredgewidth=2)
    
    # Área entre as curvas para visualizar diferenças
    ax1.fill_between(x_range, cdf_actuals, cdf_predictions, alpha=0.2, color='gray', 
                    label='Diferenças entre CDFs')
    
    # Linha vertical no ponto de diferença máxima
    ax1.axvline(x=max_diff_x, color='black', linestyle='--', alpha=0.6, 
               label=f'x = {max_diff_x:.4f}')
    
    ax1.set_xlabel('Valor', fontsize=12)
    ax1.set_ylabel('Probabilidade Acumulada', fontsize=12)
    ax1.legend(fontsize=10, loc='center right')
    ax1.grid(True, alpha=0.3)
    
    # Adicionar caixa de informações do teste
    info_text = (
        f"Teste de Kolmogorov-Smirnov (duas amostras)\n"
        f"H₀: Mesma distribuição de probabilidade\n"
        f"H₁: Distribuições diferentes\n"
        f"Nível de significância (α): {alpha}\n"
        f"Estatística KS: {ks_statistic:.6f}\n"
        f"p-value: {ks_pvalue:.6f}\n"
        f"Diferença máxima em x = {max_diff_x:.4f}\n"
        f"Tamanho amostra real: {len(actuals)}\n"
        f"Tamanho amostra predição: {len(predictions)}"
    )
    
    ax1.text(0.02, 0.98, info_text, transform=ax1.transAxes, 
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9),
             verticalalignment='top', fontsize=9, fontfamily='monospace')
    
    # ===== SUBPLOT 2: Histogramas Comparativos =====
    ax2.set_title('Distribuições das Amostras (Histogramas Normalizados)', fontsize=12, fontweight='bold')
    
    # Calcular bins compartilhados
    bins = np.linspace(np.min(combined_data), np.max(combined_data), 50)
    
    # Histogramas
    ax2.hist(actuals, bins=bins, density=True, alpha=0.6, color='blue', 
            label=f'Valores Reais (n={len(actuals)})', edgecolor='darkblue')
    ax2.hist(predictions, bins=bins, density=True, alpha=0.6, color='red',
            label=f'Predições (n={len(predictions)})', edgecolor='darkred')
    
    # Adicionar KDE para suavização
    try:
        kde_actuals = gaussian_kde(actuals)
        kde_predictions = gaussian_kde(predictions)
        
        kde_x = np.linspace(np.min(combined_data), np.max(combined_data), 200)
        ax2.plot(kde_x, kde_actuals(kde_x), 'b-', linewidth=2, alpha=0.8, label='KDE - Reais')
        ax2.plot(kde_x, kde_predictions(kde_x), 'r-', linewidth=2, alpha=0.8, label='KDE - Predições')
    except:
        pass
    
    # Linha vertical no ponto de diferença máxima
    ax2.axvline(x=max_diff_x, color='black', linestyle='--', alpha=0.8, linewidth=2,
               label=f'Diferença máxima (x = {max_diff_x:.4f})')
    
    ax2.set_xlabel('Valor', fontsize=12)
    ax2.set_ylabel('Densidade', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Interpretação do resultado
    if reject_h0:
        interpretation = (
            f"CONCLUSÃO: Rejeitamos H₀ (p = {ks_pvalue:.6f} < α = {alpha})\n"
            f"Há evidência estatística SUFICIENTE de que as distribuições\n"
            f"das predições e valores reais são DIFERENTES.\n"
            f"O modelo NÃO reproduz adequadamente a distribuição dos dados."
        )
        interp_color = 'lightcoral'
    else:
        interpretation = (
            f"CONCLUSÃO: Não rejeitamos H₀ (p = {ks_pvalue:.6f} ≥ α = {alpha})\n"
            f"NÃO há evidência estatística suficiente de que as distribuições\n"
            f"das predições e valores reais sejam diferentes.\n"
            f"O modelo reproduz adequadamente a distribuição dos dados."
        )
        interp_color = 'lightgreen'
    
    ax2.text(0.02, 0.98, interpretation, transform=ax2.transAxes, 
             bbox=dict(boxstyle='round', facecolor=interp_color, alpha=0.9),
             verticalalignment='top', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        ensure_output_dir(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print_save_message(save_path, "Teste Kolmogorov-Smirnov")
    
    plt.close()  # Fechar figura para liberar memória
    
    # Retornar resultados do teste
    return {
        'ks_statistic': ks_statistic,
        'p_value': ks_pvalue,
        'reject_h0': reject_h0,
        'max_diff_location': max_diff_x,
        'alpha': alpha,
        'conclusion': test_conclusion
    }


def plot_autocorrelation_analysis(actuals, predictions, save_path=None, 
                                  title="Comparação de Autocorrelação", 
                                  max_lags=40, alpha=0.05):
    """
    Compara a função de autocorrelação entre valores reais e predições
    
    Args:
        actuals: Valores reais (série temporal)
        predictions: Predições do modelo (série temporal)
        save_path: Caminho para salvar o gráfico
        title: Título do gráfico
        max_lags: Número máximo de lags para calcular a autocorrelação
        alpha: Nível de significância para intervalos de confiança
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
    
    # Garantir que são arrays numpy
    actuals = np.array(actuals).flatten()
    predictions = np.array(predictions).flatten()
    
    # Verificar se as séries têm tamanho suficiente
    min_length = min(len(actuals), len(predictions))
    max_lags = min(max_lags, min_length // 4)  # Regra conservadora para lags
    
    try:
        # ===== SUBPLOT 1: Autocorrelação dos Valores Reais =====
        ax1.set_title(f'Autocorrelação - Valores Reais (n={len(actuals)})', 
                     fontsize=12, fontweight='bold')
        
        # Calcular autocorrelação dos valores reais
        acf_actuals = acf(actuals, nlags=max_lags, alpha=alpha, fft=True)
        lags = np.arange(len(acf_actuals[0]))
        
        # Plotar autocorrelação
        ax1.plot(lags, acf_actuals[0], 'b-', linewidth=2, label='ACF - Valores Reais', 
                marker='o', markersize=4, alpha=0.8)
        
        # Adicionar intervalos de confiança
        if len(acf_actuals) > 1:  # Se intervalos de confiança foram calculados
            lower_conf = acf_actuals[1][:, 0] - acf_actuals[0]
            upper_conf = acf_actuals[1][:, 1] - acf_actuals[0]
            ax1.fill_between(lags, lower_conf, upper_conf, alpha=0.2, color='blue', 
                           label=f'IC {(1-alpha)*100:.0f}%')
        
        # Linha de referência em zero
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Linhas de significância estatística (aproximada)
        significance_line = 1.96 / np.sqrt(len(actuals))
        ax1.axhline(y=significance_line, color='red', linestyle='--', alpha=0.6, 
                   label=f'Limite ±{significance_line:.3f}')
        ax1.axhline(y=-significance_line, color='red', linestyle='--', alpha=0.6)
        
        ax1.set_xlabel('Lag', fontsize=10)
        ax1.set_ylabel('Autocorrelação', fontsize=10)
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(-1, 1)
        
        # ===== SUBPLOT 2: Autocorrelação das Predições =====
        ax2.set_title(f'Autocorrelação - Predições (n={len(predictions)})', 
                     fontsize=12, fontweight='bold')
        
        # Calcular autocorrelação das predições
        acf_predictions = acf(predictions, nlags=max_lags, alpha=alpha, fft=True)
        
        # Plotar autocorrelação
        ax2.plot(lags, acf_predictions[0], 'r-', linewidth=2, label='ACF - Predições', 
                marker='s', markersize=4, alpha=0.8)
        
        # Adicionar intervalos de confiança
        if len(acf_predictions) > 1:
            lower_conf = acf_predictions[1][:, 0] - acf_predictions[0]
            upper_conf = acf_predictions[1][:, 1] - acf_predictions[0]
            ax2.fill_between(lags, lower_conf, upper_conf, alpha=0.2, color='red', 
                           label=f'IC {(1-alpha)*100:.0f}%')
        
        # Linha de referência e significância
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        significance_line_pred = 1.96 / np.sqrt(len(predictions))
        ax2.axhline(y=significance_line_pred, color='red', linestyle='--', alpha=0.6, 
                   label=f'Limite ±{significance_line_pred:.3f}')
        ax2.axhline(y=-significance_line_pred, color='red', linestyle='--', alpha=0.6)
        
        ax2.set_xlabel('Lag', fontsize=10)
        ax2.set_ylabel('Autocorrelação', fontsize=10)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(-1, 1)
        
        # ===== SUBPLOT 3: Comparação Direta =====
        ax3.set_title('Comparação Direta das Autocorrelações', fontsize=12, fontweight='bold')
        
        # Plotar ambas as autocorrelações juntas
        ax3.plot(lags, acf_actuals[0], 'b-', linewidth=2.5, label='ACF - Valores Reais', 
                marker='o', markersize=5, alpha=0.8)
        ax3.plot(lags, acf_predictions[0], 'r--', linewidth=2.5, label='ACF - Predições', 
                marker='s', markersize=5, alpha=0.8)
        
        # Área entre as curvas para mostrar diferenças
        ax3.fill_between(lags, acf_actuals[0], acf_predictions[0], alpha=0.2, color='gray', 
                        label='Diferença entre ACFs')
        
        # Linhas de referência
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax3.axhline(y=significance_line, color='gray', linestyle=':', alpha=0.6, 
                   label=f'Limite significância ±{significance_line:.3f}')
        ax3.axhline(y=-significance_line, color='gray', linestyle=':', alpha=0.6)
        
        ax3.set_xlabel('Lag', fontsize=10)
        ax3.set_ylabel('Autocorrelação', fontsize=10)
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(-1, 1)
        
        # ===== CÁLCULO DE MÉTRICAS COMPARATIVAS =====
        # Diferença quadrática média entre autocorrelações
        mse_acf = np.mean((acf_actuals[0] - acf_predictions[0])**2)
        
        # Correlação entre as duas funções de autocorrelação
        corr_acf = np.corrcoef(acf_actuals[0], acf_predictions[0])[0, 1]
        
        # Diferença máxima absoluta
        max_diff_acf = np.max(np.abs(acf_actuals[0] - acf_predictions[0]))
        max_diff_lag = lags[np.argmax(np.abs(acf_actuals[0] - acf_predictions[0]))]
        
        # Teste de Ljung-Box para autocorrelação dos resíduos
        residuals = predictions - actuals[:len(predictions)] if len(predictions) <= len(actuals) else predictions[:len(actuals)] - actuals
        try:
            ljung_box = acorr_ljungbox(residuals, lags=min(10, len(residuals)//5), return_df=True)
            ljung_box_pvalue = ljung_box['lb_pvalue'].iloc[-1]  # Último p-value
        except:
            ljung_box_pvalue = np.nan
        
        # Caixa de métricas no subplot 3
        metrics_text = (
            f"MÉTRICAS COMPARATIVAS:\n"
            f"MSE das ACFs: {mse_acf:.6f}\n"
            f"Correlação ACFs: {corr_acf:.4f}\n"
            f"Diferença máxima: {max_diff_acf:.4f}\n"
            f"Lag da dif. máxima: {max_diff_lag}\n"
            f"Ljung-Box p-value: {ljung_box_pvalue:.4f}" if not np.isnan(ljung_box_pvalue) else "Ljung-Box: N/A"
        )
        
        ax3.text(0.02, 0.98, metrics_text, transform=ax3.transAxes, 
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9),
                verticalalignment='top', fontsize=9, fontfamily='monospace')
        
        # ===== INTERPRETAÇÃO =====
        # Classificar a qualidade da reprodução da estrutura temporal
        if corr_acf > 0.9 and mse_acf < 0.1:
            interpretation = "EXCELENTE: Estrutura temporal muito bem reproduzida"
            interp_color = 'lightgreen'
        elif corr_acf > 0.7 and mse_acf < 0.2:
            interpretation = "BOM: Estrutura temporal bem reproduzida"
            interp_color = 'lightblue'
        elif corr_acf > 0.5 and mse_acf < 0.4:
            interpretation = "MODERADO: Estrutura temporal parcialmente reproduzida"
            interp_color = 'lightyellow'
        else:
            interpretation = "RUIM: Estrutura temporal mal reproduzida"
            interp_color = 'lightcoral'
        
        ax3.text(0.02, 0.02, f"AVALIAÇÃO: {interpretation}", transform=ax3.transAxes, 
                bbox=dict(boxstyle='round', facecolor=interp_color, alpha=0.9),
                verticalalignment='bottom', fontsize=10, fontweight='bold')
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            ensure_output_dir(save_path)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print_save_message(save_path, "Análise de autocorrelação")
        
        plt.close()  # Fechar figura para liberar memória
        
        # Retornar métricas
        return {
            'acf_reals': acf_actuals[0],
            'acf_predictions': acf_predictions[0],
            'lags': lags,
            'mse_acf': mse_acf,
            'correlation_acf': corr_acf,
            'max_difference': max_diff_acf,
            'max_diff_lag': max_diff_lag,
            'ljung_box_pvalue': ljung_box_pvalue if not np.isnan(ljung_box_pvalue) else None
        }
        
    except Exception as e:
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, f'Erro ao calcular autocorrelação:\n{str(e)}', 
                ha='center', va='center', transform=plt.gca().transAxes,
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8),
                fontsize=12)
        plt.title(f"Erro na Análise de Autocorrelação - {title}")
        plt.axis('off')
        
        if save_path:
            ensure_output_dir(save_path)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()  # Fechar figura para liberar memória
        return None 
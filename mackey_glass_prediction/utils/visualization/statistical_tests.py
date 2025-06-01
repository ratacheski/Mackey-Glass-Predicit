"""
Testes estatísticos: Kolmogorov-Smirnov e análise de autocorrelação
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.stats import jarque_bera, shapiro, normaltest, anderson
from statsmodels.tsa.stattools import acf, pacf

from .utils import ensure_output_dir, print_save_message, get_colors_and_styles


def plot_ks_test_analysis(actuals, predictions, save_path=None,
                         title="Análise Completa - Teste Kolmogorov-Smirnov"):
    """
    Análise completa do Teste Kolmogorov-Smirnov com visualizações detalhadas
    
    Args:
        actuals: Valores reais
        predictions: Predições do modelo
        save_path: Caminho para salvar o gráfico
        title: Título do gráfico
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # Garantir que são arrays numpy
    actuals = np.array(actuals).flatten()
    predictions = np.array(predictions).flatten()
    
    # ===== SUBPLOT 1: Comparação das Distribuições Empíricas =====
    ax1.set_title('Funções de Distribuição Empíricas (FDA)', fontsize=14, fontweight='bold')
    
    # Calcular CDFs empíricas
    combined_range = np.linspace(
        min(np.min(predictions), np.min(actuals)),
        max(np.max(predictions), np.max(actuals)),
        1000
    )
    
    cdf_actuals = np.array([np.mean(actuals <= x) for x in combined_range])
    cdf_predictions = np.array([np.mean(predictions <= x) for x in combined_range])
    
    # Plotar CDFs
    ax1.plot(combined_range, cdf_actuals, 'b-', linewidth=3, label='Valores Reais', alpha=0.8)
    ax1.plot(combined_range, cdf_predictions, 'r--', linewidth=3, label='Predições', alpha=0.8)
    
    # Destacar ponto de máxima diferença
    max_diff_idx = np.argmax(np.abs(cdf_predictions - cdf_actuals))
    max_diff_x = combined_range[max_diff_idx]
    max_diff_val = np.abs(cdf_predictions[max_diff_idx] - cdf_actuals[max_diff_idx])
    
    ax1.vlines(max_diff_x, min(cdf_actuals[max_diff_idx], cdf_predictions[max_diff_idx]),
               max(cdf_actuals[max_diff_idx], cdf_predictions[max_diff_idx]),
               colors='red', linewidth=4, alpha=0.7, 
               label=f'Máx. Diferença = {max_diff_val:.4f}')
    
    ax1.scatter([max_diff_x, max_diff_x], 
               [cdf_actuals[max_diff_idx], cdf_predictions[max_diff_idx]],
               color='red', s=100, zorder=5)
    
    ax1.set_xlabel('Valor', fontsize=12)
    ax1.set_ylabel('Probabilidade Acumulada', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # ===== SUBPLOT 2: Histogramas Comparativos =====
    ax2.set_title('Distribuições de Frequência', fontsize=14, fontweight='bold')
    
    # Criar bins comuns
    bins = np.linspace(min(np.min(actuals), np.min(predictions)), 
                      max(np.max(actuals), np.max(predictions)), 50)
    
    # Plotar histogramas
    ax2.hist(actuals, bins=bins, alpha=0.6, color='blue', density=True, 
             label=f'Valores Reais (n={len(actuals)})', edgecolor='darkblue')
    ax2.hist(predictions, bins=bins, alpha=0.6, color='red', density=True, 
             label=f'Predições (n={len(predictions)})', edgecolor='darkred')
    
    ax2.set_xlabel('Valor', fontsize=12)
    ax2.set_ylabel('Densidade', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # ===== SUBPLOT 3: Análise dos Resíduos =====
    ax3.set_title('Análise de Resíduos', fontsize=14, fontweight='bold')
    
    residuals = predictions - actuals
    
    # Q-Q plot dos resíduos vs distribuição normal
    stats.probplot(residuals, dist="norm", plot=ax3)
    ax3.set_title('Q-Q Plot dos Resíduos vs Normal', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # ===== SUBPLOT 4: Estatísticas KS e Testes de Normalidade =====
    ax4.set_title('Estatísticas dos Testes', fontsize=14, fontweight='bold')
    ax4.axis('off')  # Remover eixos para texto
    
    # Teste Kolmogorov-Smirnov
    ks_statistic, ks_pvalue = stats.ks_2samp(predictions, actuals)
    
    # Testes de normalidade nos resíduos
    try:
        jb_stat, jb_pvalue = jarque_bera(residuals)
    except:
        jb_stat, jb_pvalue = np.nan, np.nan
    
    try:
        if len(residuals) <= 5000:
            shapiro_stat, shapiro_pvalue = shapiro(residuals)
        else:
            shapiro_stat, shapiro_pvalue = np.nan, np.nan
    except:
        shapiro_stat, shapiro_pvalue = np.nan, np.nan
    
    try:
        dagostino_stat, dagostino_pvalue = normaltest(residuals)
    except:
        dagostino_stat, dagostino_pvalue = np.nan, np.nan
    
    # Teste Anderson-Darling
    try:
        anderson_result = anderson(residuals, dist='norm')
        anderson_stat = anderson_result.statistic
        anderson_critical = anderson_result.critical_values[2]  # 5% level
    except:
        anderson_stat, anderson_critical = np.nan, np.nan
    
    # Estatísticas descritivas
    mean_residuals = np.mean(residuals)
    std_residuals = np.std(residuals)
    skew_residuals = stats.skew(residuals)
    kurt_residuals = stats.kurtosis(residuals)
    
    # Interpretação do KS
    if ks_pvalue > 0.05:
        ks_interpretation = "NÃO rejeitamos H₀: distribuições são iguais"
        ks_color = 'lightgreen'
    elif ks_pvalue > 0.01:
        ks_interpretation = "Evidência FRACA contra H₀"
        ks_color = 'lightyellow'
    else:
        ks_interpretation = "REJEITAMOS H₀: distribuições são diferentes"
        ks_color = 'lightcoral'
    
    # Texto com resultados
    results_text = f"""
TESTE KOLMOGOROV-SMIRNOV:
• Estatística KS: {ks_statistic:.6f}
• p-value: {ks_pvalue:.6f}
• Interpretação: {ks_interpretation}

ANÁLISE DE RESÍDUOS:
• Média: {mean_residuals:.6f}
• Desvio Padrão: {std_residuals:.6f}
• Assimetria: {skew_residuals:.4f}
• Curtose: {kurt_residuals:.4f}

TESTES DE NORMALIDADE (resíduos):
• Jarque-Bera: {jb_stat:.4f} (p={jb_pvalue:.4f})
• Shapiro-Wilk: {shapiro_stat:.4f} (p={shapiro_pvalue:.4f})
• D'Agostino: {dagostino_stat:.4f} (p={dagostino_pvalue:.4f})
• Anderson-Darling: {anderson_stat:.4f} (crit={anderson_critical:.4f})

RECOMENDAÇÕES:
• KS p-value < 0.05: Evidência de diferenças distribucionais
• |Assimetria| > 0.5: Distribuição assimétrica
• |Curtose| > 0.5: Caudas mais pesadas/leves que normal
    """
    
    ax4.text(0.05, 0.95, results_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor=ks_color, alpha=0.8))
    
    # Box com resumo interpretativo
    if ks_pvalue > 0.05 and abs(skew_residuals) < 0.5 and abs(kurt_residuals) < 0.5:
        summary = "✓ EXCELENTE: Distribuições estatisticamente similares"
        summary_color = 'lightgreen'
    elif ks_pvalue > 0.01:
        summary = "⚠ MODERADO: Pequenas diferenças distribucionais"
        summary_color = 'lightyellow'
    else:
        summary = "✗ PROBLEMÁTICO: Diferenças significativas"
        summary_color = 'lightcoral'
    
    ax4.text(0.05, 0.05, f'RESUMO: {summary}', transform=ax4.transAxes, fontsize=12,
             verticalalignment='bottom', fontweight='bold',
             bbox=dict(boxstyle='round', facecolor=summary_color, alpha=0.9))
    
    plt.tight_layout()
    
    if save_path:
        ensure_output_dir(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print_save_message(save_path, "Análise KS")
    
    plt.show()
    
    # Retornar resultados
    return {
        'ks_statistic': ks_statistic,
        'ks_pvalue': ks_pvalue,
        'max_difference': max_diff_val,
        'residuals_mean': mean_residuals,
        'residuals_std': std_residuals,
        'residuals_skewness': skew_residuals,
        'residuals_kurtosis': kurt_residuals,
        'jarque_bera_stat': jb_stat,
        'jarque_bera_pvalue': jb_pvalue,
        'shapiro_stat': shapiro_stat,
        'shapiro_pvalue': shapiro_pvalue,
        'anderson_stat': anderson_stat
    }


def plot_autocorrelation_analysis(series, lags=50, save_path=None,
                                 title="Análise de Autocorrelação"):
    """
    Análise completa de autocorrelação e autocorrelação parcial
    
    Args:
        series: Série temporal para análise
        lags: Número de lags para calcular
        save_path: Caminho para salvar o gráfico
        title: Título do gráfico
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # Garantir que é array numpy
    series = np.array(series).flatten()
    
    # Limitar lags se necessário
    max_lags = min(lags, len(series) // 4)
    
    # ===== SUBPLOT 1: Série Temporal =====
    ax1.set_title(f'{title} - Série Temporal', fontsize=14, fontweight='bold')
    ax1.plot(series, 'b-', linewidth=1, alpha=0.8)
    ax1.set_xlabel('Tempo', fontsize=12)
    ax1.set_ylabel('Valor', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Adicionar estatísticas básicas
    mean_val = np.mean(series)
    std_val = np.std(series)
    ax1.axhline(y=mean_val, color='red', linestyle='--', alpha=0.7, label=f'Média = {mean_val:.4f}')
    ax1.fill_between(range(len(series)), mean_val - std_val, mean_val + std_val, 
                     alpha=0.2, color='red', label=f'±1σ = {std_val:.4f}')
    ax1.legend(fontsize=10)
    
    # ===== SUBPLOT 2: Autocorrelação (ACF) =====
    ax2.set_title('Função de Autocorrelação (ACF)', fontsize=14, fontweight='bold')
    
    try:
        # Calcular ACF
        autocorr_values = acf(series, nlags=max_lags, fft=True)
        lags_range = range(len(autocorr_values))
        
        # Plotar ACF
        ax2.stem(lags_range, autocorr_values, basefmt=' ')
        
        # Intervalos de confiança (aproximadamente ±1.96/√n)
        n = len(series)
        confidence_interval = 1.96 / np.sqrt(n)
        ax2.axhline(y=confidence_interval, color='red', linestyle='--', alpha=0.7, 
                   label=f'IC 95% = ±{confidence_interval:.4f}')
        ax2.axhline(y=-confidence_interval, color='red', linestyle='--', alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        ax2.set_xlabel('Lag', fontsize=12)
        ax2.set_ylabel('Autocorrelação', fontsize=12)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # Identificar lags significativos
        significant_lags_acf = np.where(np.abs(autocorr_values[1:]) > confidence_interval)[0] + 1
        
    except Exception as e:
        ax2.text(0.5, 0.5, f'Erro ao calcular ACF: {str(e)}', 
                transform=ax2.transAxes, ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        autocorr_values = np.array([])
        significant_lags_acf = np.array([])
    
    # ===== SUBPLOT 3: Autocorrelação Parcial (PACF) =====
    ax3.set_title('Função de Autocorrelação Parcial (PACF)', fontsize=14, fontweight='bold')
    
    try:
        # Calcular PACF
        partial_autocorr_values = pacf(series, nlags=max_lags, method='ols')
        lags_range_pacf = range(len(partial_autocorr_values))
        
        # Plotar PACF
        ax3.stem(lags_range_pacf, partial_autocorr_values, basefmt=' ')
        
        # Intervalos de confiança
        ax3.axhline(y=confidence_interval, color='red', linestyle='--', alpha=0.7,
                   label=f'IC 95% = ±{confidence_interval:.4f}')
        ax3.axhline(y=-confidence_interval, color='red', linestyle='--', alpha=0.7)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        ax3.set_xlabel('Lag', fontsize=12)
        ax3.set_ylabel('Autocorrelação Parcial', fontsize=12)
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        # Identificar lags significativos
        significant_lags_pacf = np.where(np.abs(partial_autocorr_values[1:]) > confidence_interval)[0] + 1
        
    except Exception as e:
        ax3.text(0.5, 0.5, f'Erro ao calcular PACF: {str(e)}', 
                transform=ax3.transAxes, ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        partial_autocorr_values = np.array([])
        significant_lags_pacf = np.array([])
    
    # ===== SUBPLOT 4: Análise e Interpretação =====
    ax4.set_title('Análise dos Resultados', fontsize=14, fontweight='bold')
    ax4.axis('off')
    
    # Estatísticas da série
    stationarity_stats = {
        'mean': np.mean(series),
        'std': np.std(series),
        'min': np.min(series),
        'max': np.max(series),
        'range': np.max(series) - np.min(series),
        'skewness': stats.skew(series),
        'kurtosis': stats.kurtosis(series)
    }
    
    # Teste de estacionariedade (Augmented Dickey-Fuller)
    try:
        from statsmodels.tsa.stattools import adfuller
        adf_result = adfuller(series, autolag='AIC')
        adf_statistic = adf_result[0]
        adf_pvalue = adf_result[1]
        adf_critical_1 = adf_result[4]['1%']
        adf_critical_5 = adf_result[4]['5%']
        
        if adf_pvalue < 0.05:
            stationarity_result = "ESTACIONÁRIA (p < 0.05)"
            stationarity_color = 'lightgreen'
        else:
            stationarity_result = "NÃO ESTACIONÁRIA (p ≥ 0.05)"
            stationarity_color = 'lightcoral'
    except:
        adf_statistic, adf_pvalue = np.nan, np.nan
        adf_critical_1, adf_critical_5 = np.nan, np.nan
        stationarity_result = "Teste ADF indisponível"
        stationarity_color = 'lightyellow'
    
    # Análise de autocorrelação
    if len(autocorr_values) > 1:
        # Detectar padrões de autocorrelação
        if len(significant_lags_acf) == 0:
            acf_pattern = "Ruído branco - sem autocorrelação significativa"
        elif len(significant_lags_acf) < 5:
            acf_pattern = f"Autocorrelação em poucos lags: {list(significant_lags_acf[:5])}"
        else:
            acf_pattern = f"Autocorrelação persistente em {len(significant_lags_acf)} lags"
    else:
        acf_pattern = "Análise ACF indisponível"
    
    if len(partial_autocorr_values) > 1:
        if len(significant_lags_pacf) == 0:
            pacf_pattern = "Sem autocorrelação parcial significativa"
        elif len(significant_lags_pacf) < 3:
            pacf_pattern = f"PACF significativa em: {list(significant_lags_pacf[:3])}"
        else:
            pacf_pattern = f"PACF complexa com {len(significant_lags_pacf)} lags"
    else:
        pacf_pattern = "Análise PACF indisponível"
    
    # Sugestão de modelo ARIMA
    p = len(significant_lags_pacf) if len(significant_lags_pacf) <= 3 else 3
    d = 0 if adf_pvalue < 0.05 else 1 if not np.isnan(adf_pvalue) else 1
    q = len(significant_lags_acf) if len(significant_lags_acf) <= 3 else 3
    
    analysis_text = f"""
ESTATÍSTICAS DA SÉRIE:
• Tamanho: {len(series):,} observações
• Média: {stationarity_stats['mean']:.6f}
• Desvio Padrão: {stationarity_stats['std']:.6f}
• Amplitude: {stationarity_stats['range']:.6f}
• Assimetria: {stationarity_stats['skewness']:.4f}
• Curtose: {stationarity_stats['kurtosis']:.4f}

TESTE DE ESTACIONARIEDADE (ADF):
• Estatística: {adf_statistic:.6f}
• p-value: {adf_pvalue:.6f}
• Crítico 5%: {adf_critical_5:.6f}
• Resultado: {stationarity_result}

ANÁLISE DE AUTOCORRELAÇÃO:
• ACF: {acf_pattern}
• PACF: {pacf_pattern}
• Lags significativos ACF: {len(significant_lags_acf)}
• Lags significativos PACF: {len(significant_lags_pacf)}

SUGESTÃO MODELO ARIMA({p},{d},{q}):
• p (AR): {p} (baseado em PACF)
• d (diferenciação): {d} (baseado em ADF)
• q (MA): {q} (baseado em ACF)
    """
    
    ax4.text(0.05, 0.95, analysis_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor=stationarity_color, alpha=0.8))
    
    # Interpretação final
    if len(significant_lags_acf) == 0 and len(significant_lags_pacf) == 0:
        interpretation = "✓ RUÍDO BRANCO: Série sem dependência temporal"
        interp_color = 'lightgreen'
    elif len(significant_lags_acf) > 10 or len(significant_lags_pacf) > 10:
        interpretation = "⚠ ALTA DEPENDÊNCIA: Série com forte estrutura temporal"
        interp_color = 'lightyellow'
    else:
        interpretation = "📊 ESTRUTURADA: Série com dependência temporal moderada"
        interp_color = 'lightblue'
    
    ax4.text(0.05, 0.05, f'INTERPRETAÇÃO: {interpretation}', 
             transform=ax4.transAxes, fontsize=12,
             verticalalignment='bottom', fontweight='bold',
             bbox=dict(boxstyle='round', facecolor=interp_color, alpha=0.9))
    
    plt.tight_layout()
    
    if save_path:
        ensure_output_dir(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print_save_message(save_path, "Análise de autocorrelação")
    
    plt.show()
    
    # Retornar resultados
    return {
        'autocorrelation_values': autocorr_values.tolist() if len(autocorr_values) > 0 else [],
        'partial_autocorr_values': partial_autocorr_values.tolist() if len(partial_autocorr_values) > 0 else [],
        'significant_lags_acf': significant_lags_acf.tolist() if len(significant_lags_acf) > 0 else [],
        'significant_lags_pacf': significant_lags_pacf.tolist() if len(significant_lags_pacf) > 0 else [],
        'adf_statistic': adf_statistic if not np.isnan(adf_statistic) else None,
        'adf_pvalue': adf_pvalue if not np.isnan(adf_pvalue) else None,
        'is_stationary': adf_pvalue < 0.05 if not np.isnan(adf_pvalue) else None,
        'suggested_arima': (p, d, q),
        'series_stats': stationarity_stats
    }


def plot_residuals_autocorrelation(residuals, lags=40, save_path=None,
                                  title="Análise de Autocorrelação dos Resíduos"):
    """
    Análise específica de autocorrelação dos resíduos do modelo
    
    Args:
        residuals: Resíduos do modelo (predictions - actuals)
        lags: Número de lags para analisar
        save_path: Caminho para salvar o gráfico
        title: Título do gráfico
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Garantir que são arrays numpy
    residuals = np.array(residuals).flatten()
    
    # Limitar lags
    max_lags = min(lags, len(residuals) // 4)
    
    # ===== SUBPLOT 1: Resíduos no Tempo =====
    ax1.set_title('Resíduos ao Longo do Tempo', fontsize=12, fontweight='bold')
    ax1.plot(residuals, 'b-', linewidth=1, alpha=0.7)
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.8)
    
    # Adicionar bandas de confiança (±2σ)
    std_residuals = np.std(residuals)
    ax1.axhline(y=2*std_residuals, color='orange', linestyle=':', alpha=0.7, label='±2σ')
    ax1.axhline(y=-2*std_residuals, color='orange', linestyle=':', alpha=0.7)
    
    ax1.set_xlabel('Índice', fontsize=10)
    ax1.set_ylabel('Resíduo', fontsize=10)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # ===== SUBPLOT 2: ACF dos Resíduos =====
    ax2.set_title('ACF dos Resíduos', fontsize=12, fontweight='bold')
    
    try:
        # Calcular ACF
        autocorr_residuals = acf(residuals, nlags=max_lags, fft=True)
        lags_range = range(len(autocorr_residuals))
        
        # Plotar ACF
        ax2.stem(lags_range, autocorr_residuals, basefmt=' ')
        
        # Intervalos de confiança
        n = len(residuals)
        confidence_interval = 1.96 / np.sqrt(n)
        ax2.axhline(y=confidence_interval, color='red', linestyle='--', alpha=0.7,
                   label=f'IC 95%')
        ax2.axhline(y=-confidence_interval, color='red', linestyle='--', alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        ax2.set_xlabel('Lag', fontsize=10)
        ax2.set_ylabel('Autocorrelação', fontsize=10)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        # Verificar autocorrelação significativa
        significant_autocorr = np.sum(np.abs(autocorr_residuals[1:]) > confidence_interval)
        
    except Exception as e:
        ax2.text(0.5, 0.5, f'Erro ACF: {str(e)}', transform=ax2.transAxes, ha='center', va='center')
        significant_autocorr = np.nan
        autocorr_residuals = np.array([])
    
    # ===== SUBPLOT 3: Teste Ljung-Box =====
    ax3.set_title('Teste Ljung-Box', fontsize=12, fontweight='bold')
    ax3.axis('off')
    
    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox
        
        # Teste Ljung-Box para diferentes lags
        test_lags = [5, 10, 15, 20] if max_lags >= 20 else [max(1, max_lags//4), max(1, max_lags//2), max_lags]
        lb_results = []
        
        for lag in test_lags:
            if lag <= max_lags:
                try:
                    lb_result = acorr_ljungbox(residuals, lags=lag, return_df=True)
                    lb_stat = lb_result['lb_stat'].iloc[-1]
                    lb_pvalue = lb_result['lb_pvalue'].iloc[-1]
                    lb_results.append((lag, lb_stat, lb_pvalue))
                except:
                    continue
        
        # Mostrar resultados
        ljung_text = "TESTE LJUNG-BOX:\n(H₀: resíduos são independentes)\n\n"
        
        for lag, stat, pvalue in lb_results:
            if pvalue > 0.05:
                interpretation = "✓ Independentes"
                color_marker = "🟢"
            elif pvalue > 0.01:
                interpretation = "⚠ Duvidoso"
                color_marker = "🟡"
            else:
                interpretation = "✗ Dependentes"
                color_marker = "🔴"
            
            ljung_text += f"Lag {lag}: {color_marker}\n"
            ljung_text += f"  Estat: {stat:.4f}\n"
            ljung_text += f"  p-val: {pvalue:.4f}\n"
            ljung_text += f"  {interpretation}\n\n"
        
        # Resumo geral
        if len(lb_results) > 0:
            min_pvalue = min([p for _, _, p in lb_results])
            if min_pvalue > 0.05:
                overall = "✅ RESÍDUOS INDEPENDENTES\nModelo adequado"
                overall_color = 'lightgreen'
            elif min_pvalue > 0.01:
                overall = "⚠ POSSÍVEL DEPENDÊNCIA\nVerificar modelo"
                overall_color = 'lightyellow'
            else:
                overall = "❌ RESÍDUOS DEPENDENTES\nModelo inadequado"
                overall_color = 'lightcoral'
        else:
            overall = "Teste indisponível"
            overall_color = 'lightgray'
        
        ljung_text += f"CONCLUSÃO:\n{overall}"
        
    except Exception as e:
        ljung_text = f"Erro no teste Ljung-Box:\n{str(e)}\n\nVerifique se statsmodels\nestá instalado corretamente."
        overall_color = 'lightyellow'
    
    # Adicionar informações sobre autocorrelação
    if not np.isnan(significant_autocorr):
        ljung_text += f"\n\nAUTOCORRELAÇÃO:\n• Lags significativos: {significant_autocorr}/{max_lags}"
        if significant_autocorr == 0:
            ljung_text += "\n• ✓ Sem autocorrelação detectada"
        elif significant_autocorr < max_lags * 0.05:  # Menos de 5% dos lags
            ljung_text += "\n• ⚠ Pouca autocorrelação"
        else:
            ljung_text += "\n• ✗ Autocorrelação excessiva"
    
    ax3.text(0.05, 0.95, ljung_text, transform=ax3.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor=overall_color, alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        ensure_output_dir(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print_save_message(save_path, "Autocorrelação dos resíduos")
    
    plt.show()
    
    # Retornar resultados
    return {
        'autocorrelation_residuals': autocorr_residuals.tolist() if len(autocorr_residuals) > 0 else [],
        'significant_autocorr_count': int(significant_autocorr) if not np.isnan(significant_autocorr) else None,
        'ljungbox_results': lb_results if 'lb_results' in locals() else [],
        'residuals_std': float(std_residuals),
        'residuals_mean': float(np.mean(residuals))
    }


def compare_statistical_tests(results_dict, save_path=None,
                             title="Comparação de Testes Estatísticos"):
    """
    Compara testes estatísticos entre múltiplos modelos
    
    Args:
        results_dict: Dicionário com resultados dos modelos
        save_path: Caminho para salvar o gráfico
        title: Título do gráfico
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    colors, _ = get_colors_and_styles(len(results_dict))
    
    # Coletar dados para comparação
    ks_stats = []
    ks_pvalues = []
    normality_tests = []
    model_names = []
    
    for model_name, results in results_dict.items():
        if 'actuals' in results and 'predictions' in results:
            actuals = np.array(results['actuals']).flatten()
            predictions = np.array(results['predictions']).flatten()
            residuals = predictions - actuals
            
            # Teste KS
            ks_stat, ks_pval = stats.ks_2samp(predictions, actuals)
            ks_stats.append(ks_stat)
            ks_pvalues.append(ks_pval)
            
            # Teste de normalidade nos resíduos
            try:
                _, jb_pval = jarque_bera(residuals)
                normality_tests.append(jb_pval)
            except:
                normality_tests.append(np.nan)
            
            model_names.append(model_name)
    
    # ===== SUBPLOT 1: Estatísticas KS =====
    ax1.set_title('Estatísticas Kolmogorov-Smirnov', fontsize=14, fontweight='bold')
    bars1 = ax1.bar(model_names, ks_stats, color=colors[:len(model_names)], alpha=0.7)
    ax1.set_ylabel('Estatística KS', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Adicionar valores nas barras
    for bar, val in zip(bars1, ks_stats):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    
    # ===== SUBPLOT 2: p-values KS =====
    ax2.set_title('p-values Kolmogorov-Smirnov', fontsize=14, fontweight='bold')
    bars2 = ax2.bar(model_names, ks_pvalues, color=colors[:len(model_names)], alpha=0.7)
    ax2.set_ylabel('p-value', fontsize=12)
    ax2.tick_params(axis='x', rotation=45)
    ax2.axhline(y=0.05, color='red', linestyle='--', alpha=0.8, label='α = 0.05')
    ax2.axhline(y=0.01, color='darkred', linestyle='--', alpha=0.8, label='α = 0.01')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Adicionar valores nas barras
    for bar, val in zip(bars2, ks_pvalues):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + max(ks_pvalues)*0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    
    # ===== SUBPLOT 3: Testes de Normalidade =====
    ax3.set_title('p-values Teste de Normalidade (Jarque-Bera)', fontsize=14, fontweight='bold')
    valid_normality = [p for p in normality_tests if not np.isnan(p)]
    valid_names = [name for name, p in zip(model_names, normality_tests) if not np.isnan(p)]
    valid_colors = [colors[i] for i, p in enumerate(normality_tests) if not np.isnan(p)]
    
    if valid_normality:
        bars3 = ax3.bar(valid_names, valid_normality, color=valid_colors, alpha=0.7)
        ax3.set_ylabel('p-value (Jarque-Bera)', fontsize=12)
        ax3.tick_params(axis='x', rotation=45)
        ax3.axhline(y=0.05, color='red', linestyle='--', alpha=0.8, label='α = 0.05')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Adicionar valores nas barras
        for bar, val in zip(bars3, valid_normality):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + max(valid_normality)*0.01,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    else:
        ax3.text(0.5, 0.5, 'Dados de normalidade indisponíveis', 
                transform=ax3.transAxes, ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # ===== SUBPLOT 4: Ranking dos Modelos =====
    ax4.set_title('Ranking Estatístico dos Modelos', fontsize=14, fontweight='bold')
    ax4.axis('off')
    
    # Calcular scores
    model_scores = []
    for i, name in enumerate(model_names):
        score = 0
        
        # KS p-value (quanto maior, melhor)
        if ks_pvalues[i] > 0.05:
            score += 3
        elif ks_pvalues[i] > 0.01:
            score += 2
        else:
            score += 1
        
        # Normalidade (quanto maior p-value, melhor)
        if i < len(normality_tests) and not np.isnan(normality_tests[i]):
            if normality_tests[i] > 0.05:
                score += 2
            elif normality_tests[i] > 0.01:
                score += 1
        
        model_scores.append((name, score, ks_pvalues[i], 
                           normality_tests[i] if i < len(normality_tests) else np.nan))
    
    # Ordenar por score
    model_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Criar texto do ranking
    ranking_text = "RANKING ESTATÍSTICO:\n\n"
    
    for rank, (name, score, ks_pval, norm_pval) in enumerate(model_scores, 1):
        if rank == 1:
            medal = "🥇"
        elif rank == 2:
            medal = "🥈"
        elif rank == 3:
            medal = "🥉"
        else:
            medal = f"{rank}º"
        
        # Interpretação KS
        if ks_pval > 0.05:
            ks_status = "✓"
        elif ks_pval > 0.01:
            ks_status = "⚠"
        else:
            ks_status = "✗"
        
        # Interpretação normalidade
        if not np.isnan(norm_pval):
            if norm_pval > 0.05:
                norm_status = "✓"
            else:
                norm_status = "✗"
        else:
            norm_status = "-"
        
        ranking_text += f"{medal} {name}\n"
        ranking_text += f"   Score: {score}/5\n"
        ranking_text += f"   KS: {ks_status} (p={ks_pval:.4f})\n"
        ranking_text += f"   Norm: {norm_status} (p={norm_pval:.4f})\n\n"
    
    ranking_text += "\nLEGENDA:\n"
    ranking_text += "✓ = Bom (p > 0.05)\n"
    ranking_text += "⚠ = Moderado (0.01 < p ≤ 0.05)\n"
    ranking_text += "✗ = Ruim (p ≤ 0.01)\n"
    ranking_text += "- = Indisponível"
    
    ax4.text(0.05, 0.95, ranking_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        ensure_output_dir(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print_save_message(save_path, "Comparação de testes estatísticos")
    
    plt.show()
    
    return {
        'model_rankings': model_scores,
        'ks_statistics': list(zip(model_names, ks_stats, ks_pvalues)),
        'normality_tests': list(zip(model_names, normality_tests))
    } 
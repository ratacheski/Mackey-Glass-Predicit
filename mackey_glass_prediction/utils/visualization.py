import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import os
from scipy import stats
from scipy.stats import gaussian_kde


# Configurar estilo dos gráficos
matplotlib.style.use('ggplot')
sns.set_palette("husl")


def format_metric_value(value, metric_name, context='table'):
    """
    Função utilitária para formatação consistente de valores de métricas
    
    Args:
        value: Valor numérico a ser formatado
        metric_name: Nome da métrica (MSE, RMSE, MAE, MAPE, R², FDA_*, FDP_*)
        context: Contexto da formatação ('table' ou 'display')
    """
    if pd.isna(value) or value is None:
        return 'N/A'
    
    # Verificar valores extremos
    if np.isinf(value):
        return '∞' if value > 0 else '-∞'
    
    # Tratamento especial para diferentes métricas
    if metric_name == 'R²':
        return f'{value:.4f}'
    elif metric_name == 'MAPE':
        if context == 'table':
            return f'{value:.2f}%' if value < 100 else f'{value:.1f}%'
        else:
            return f'{value:.1f}%'
    elif metric_name in ['MSE', 'RMSE', 'MAE']:
        # Para valores muito pequenos, usar notação científica
        if abs(value) < 1e-4:
            return f'{value:.2e}'
        # Para valores pequenos, mais casas decimais
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
        # Métricas FDA (Função Distribuição Acumulada)
        if metric_name == 'FDA_KS_PValue':
            # P-values com mais precisão
            if value < 0.001:
                return f'{value:.2e}'
            else:
                return f'{value:.4f}'
        elif metric_name in ['FDA_KS_Statistic', 'FDA_Distance']:
            # Estatísticas KS e distâncias com 4-6 casas decimais
            if abs(value) < 0.001:
                return f'{value:.2e}'
            else:
                return f'{value:.6f}'
    elif metric_name.startswith('FDP_'):
        # Métricas FDP (Função de Distribuição de Probabilidade)
        if metric_name in ['FDP_L2_Distance', 'FDP_JS_Divergence']:
            # Distâncias e divergências com precisão científica se muito pequenas
            if abs(value) < 1e-4:
                return f'{value:.2e}'
            elif abs(value) < 0.01:
                return f'{value:.6f}'
            else:
                return f'{value:.4f}'
    else:
        # Para outras métricas
        if abs(value) < 0.001:
            return f'{value:.2e}'
        elif abs(value) < 1:
            return f'{value:.4f}'
        else:
            return f'{value:.3f}'


def validate_and_clean_metrics(results_dict):
    """
    Valida e limpa dados de métricas para evitar problemas de formatação
    
    Args:
        results_dict: Dicionário com resultados dos modelos
        
    Returns:
        Dicionário limpo e validado
    """
    cleaned_dict = {}
    
    for model_name, results in results_dict.items():
        cleaned_results = results.copy()
        
        if 'metrics' in results:
            cleaned_metrics = {}
            for metric_name, value in results['metrics'].items():
                # Limpar valores problemáticos
                if pd.isna(value) or value is None:
                    cleaned_value = np.nan
                elif np.isinf(value):
                    # Para infinitos, usar um valor muito grande mas finito
                    cleaned_value = 1e10 if value > 0 else -1e10
                else:
                    cleaned_value = float(value)
                
                cleaned_metrics[metric_name] = cleaned_value
            
            cleaned_results['metrics'] = cleaned_metrics
        
        cleaned_dict[model_name] = cleaned_results
    
    return cleaned_dict


def plot_training_history(train_losses, val_losses, save_path=None, title="Histórico de Treinamento"):
    """
    Plota o histórico de loss de treinamento e validação
    """
    plt.figure(figsize=(12, 5))
    
    # Plot de loss
    plt.subplot(1, 1, 1)
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Loss de Treinamento', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Loss de Validação', linewidth=2)
    plt.title(title)
    plt.xlabel('Épocas')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Gráfico de treinamento salvo em: {save_path}")
    
    plt.show()


def plot_predictions(actuals, predictions, n_show=500, save_path=None, 
                    title="Predições vs Valores Reais"):
    """
    Plota predições vs valores reais
    """
    plt.figure(figsize=(15, 8))
    
    # Limitar número de pontos para visualização
    if len(actuals) > n_show:
        indices = np.linspace(0, len(actuals)-1, n_show, dtype=int)
        actuals_plot = actuals[indices]
        predictions_plot = predictions[indices]
        x_axis = indices
    else:
        actuals_plot = actuals
        predictions_plot = predictions
        x_axis = range(len(actuals))
    
    plt.plot(x_axis, actuals_plot, 'b-', label='Valores Reais', alpha=0.7, linewidth=1.5)
    plt.plot(x_axis, predictions_plot, 'r-', label='Predições', alpha=0.8, linewidth=1.5)
    
    plt.title(title)
    plt.xlabel('Tempo')
    plt.ylabel('Valor')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Gráfico de predições salvo em: {save_path}")
    
    plt.show()


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
        from scipy import stats
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
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
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
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"QQ-Plot salvo em: {save_path}")
    
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


def plot_metrics_comparison(results_dict, save_path=None, 
                           title="Comparação de Métricas entre Modelos"):
    """
    Plota comparação de métricas entre diferentes modelos
    """
    # Validar e limpar dados
    results_dict = validate_and_clean_metrics(results_dict)
    
    # Preparar dados
    models = list(results_dict.keys())
    # Métricas principais (removendo FDA e FDP que agora têm gráficos dedicados)
    metrics = ['MSE', 'RMSE', 'MAE', 'MAPE', 'R²']
    
    # Verificar quais métricas estão disponíveis nos dados
    available_metrics = []
    if models:
        first_model_metrics = results_dict[models[0]].get('metrics', {})
        for metric in metrics:
            if metric in first_model_metrics:
                available_metrics.append(metric)
    
    if not available_metrics:
        print("Nenhuma métrica encontrada para plotar.")
        return
    
    # Calcular número de subplots necessários
    n_metrics = len(available_metrics)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    # Criar subplots
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
        
        # Adicionar valores nas barras com formatação melhorada
        for bar, value in zip(bars, values):
            height = bar.get_height()
            formatted_value = format_metric_value(value, metric, context='display')
            axes[i].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        formatted_value, ha='center', va='bottom', 
                        fontsize=9, fontweight='bold')
        
        # Destacar a melhor barra (apenas se houver valores válidos)
        valid_values = [v for v in values if not (np.isnan(v) or np.isinf(v))]
        if valid_values:
            # Para R² maior é melhor, para todas as outras métricas menor é melhor
            if metric == 'R²':
                best_value = max(valid_values)
                best_idx = values.index(best_value)
            else:
                best_value = min(valid_values)
                best_idx = values.index(best_value)
            
            bars[best_idx].set_color('#28a745')
            bars[best_idx].set_alpha(0.8)
    
    # Remover subplots extras
    for i in range(len(available_metrics), len(axes)):
        fig.delaxes(axes[i])
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparação de métricas salva em: {save_path}")
    
    plt.show()


def plot_sequential_predictions(original_series, train_size, predictions, 
                               actual_test, save_path=None,
                               title="Predições Sequenciais"):
    """
    Plota série original com predições sequenciais
    """
    plt.figure(figsize=(15, 8))
    
    # Série de treinamento
    train_series = original_series[:train_size]
    plt.plot(range(len(train_series)), train_series, 'b-', 
             label='Dados de Treinamento', alpha=0.7, linewidth=1.5)
    
    # Série de teste real
    test_start = train_size
    test_indices = range(test_start, test_start + len(actual_test))
    plt.plot(test_indices, actual_test, 'g-', 
             label='Valores Reais (Teste)', alpha=0.8, linewidth=2)
    
    # Predições
    pred_indices = range(test_start, test_start + len(predictions))
    plt.plot(pred_indices, predictions, 'r--', 
             label='Predições Sequenciais', alpha=0.8, linewidth=2)
    
    # Linha divisória
    plt.axvline(x=train_size, color='black', linestyle=':', alpha=0.8, 
                label='Início das Predições')
    
    plt.title(title)
    plt.xlabel('Tempo')
    plt.ylabel('Valor')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Gráfico de predições sequenciais salvo em: {save_path}")
    
    plt.show()


def save_metrics_table(results_dict, save_path):
    """
    Salva tabela de métricas em CSV e cria visualização
    """
    # Validar e limpar dados
    results_dict = validate_and_clean_metrics(results_dict)
    
    # Criar DataFrame
    data = []
    for model_name, results in results_dict.items():
        metrics = results['metrics']
        row = {'Modelo': model_name}
        row.update(metrics)
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # Salvar CSV com formatação adequada
    csv_path = save_path.replace('.png', '.csv') if save_path.endswith('.png') else save_path + '.csv'
    
    # Formatar dados para CSV
    df_formatted = df.copy()
    for col in df.columns:
        if col != 'Modelo' and df[col].dtype in ['float64', 'float32']:
            df_formatted[col] = df[col].round(6)
    
    df_formatted.to_csv(csv_path, index=False)
    print(f"Tabela de métricas salva em: {csv_path}")
    
    # Criar visualização da tabela
    plt.figure(figsize=(14, 8))
    
    # Preparar dados da tabela com formatação personalizada
    table_data = []
    col_labels = df.columns.tolist()
    
    for _, row in df.iterrows():
        formatted_row = []
        for col in col_labels:
            if col == 'Modelo':
                # Truncar nomes muito longos
                model_name = str(row[col])
                if len(model_name) > 15:
                    model_name = model_name[:12] + '...'
                formatted_row.append(model_name)
            else:
                formatted_row.append(format_metric_value(row[col], col, context='table'))
        table_data.append(formatted_row)
    
    # Criar tabela
    table = plt.table(cellText=table_data,
                     colLabels=col_labels,
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    
    # Estilizar tabela
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.3, 2.2)
    
    # Destacar cabeçalho
    for i in range(len(col_labels)):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
        
    # Estilizar células de dados
    for i in range(1, len(table_data) + 1):
        for j in range(len(col_labels)):
            # Alternar cores das linhas
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f8f9fa')
            else:
                table[(i, j)].set_facecolor('#ffffff')
            
            # Ajustar espaçamento
            table[(i, j)].set_text_props(fontsize=9)
            
            # Destacar melhor modelo (menor valor para métricas de erro; maior para R²)
            if j > 0:  # Não aplicar ao nome do modelo
                metric_name = col_labels[j]
                values = df[metric_name].values
                
                # Filtrar valores válidos
                valid_mask = ~(np.isnan(values) | np.isinf(values))
                if valid_mask.any():
                    valid_values = values[valid_mask]
                    
                    # Encontrar melhor valor
                    # R² é melhor quando maior; todas as outras métricas são melhores quando menores
                    if metric_name == 'R²':
                        best_value = np.max(valid_values)
                        is_best = abs(df.iloc[i-1][metric_name] - best_value) < 1e-6
                    else:
                        best_value = np.min(valid_values)
                        is_best = abs(df.iloc[i-1][metric_name] - best_value) < 1e-6
                    
                    if is_best and not (np.isnan(df.iloc[i-1][metric_name]) or np.isinf(df.iloc[i-1][metric_name])):
                        table[(i, j)].set_facecolor('#d4edda')
                        table[(i, j)].set_text_props(weight='bold', color='#155724')
    
    # Remover eixos
    plt.axis('off')
    plt.title('Comparação de Métricas - Todos os Modelos', 
              fontsize=16, fontweight='bold', pad=30)
    
    # Adicionar legenda
    plt.figtext(0.5, 0.02, 'Células destacadas em verde indicam o melhor desempenho para cada métrica', 
                ha='center', fontsize=10, style='italic')
    
    if save_path:
        png_path = save_path.replace('.csv', '.png') if save_path.endswith('.csv') else save_path + '.png'
        plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.2)
        print(f"Visualização da tabela salva em: {png_path}")
    
    plt.show()
    
    return df


def create_comprehensive_report(results_dict, output_dir):
    """
    Cria relatório abrangente com todas as visualizações
    """
    # Validar e limpar dados
    results_dict = validate_and_clean_metrics(results_dict)
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Gerando relatório abrangente...")
    
    # 1. Comparação de métricas
    plot_metrics_comparison(results_dict, 
                          save_path=os.path.join(output_dir, 'metrics_comparison.png'))
    
    # 2. Tabela de métricas
    df_metrics = save_metrics_table(results_dict, 
                                  save_path=os.path.join(output_dir, 'metrics_table'))
    
    # 3. Gráficos individuais para cada modelo
    for model_name, results in results_dict.items():
        model_dir = os.path.join(output_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        # Histórico de treinamento
        if 'train_losses' in results and 'val_losses' in results:
            plot_training_history(results['train_losses'], results['val_losses'],
                                save_path=os.path.join(model_dir, 'training_history.png'),
                                title=f'Histórico de Treinamento - {model_name}')
        
        # Predições vs Reais
        if 'predictions' in results and 'actuals' in results:
            plot_predictions(results['actuals'], results['predictions'],
                           save_path=os.path.join(model_dir, 'predictions.png'),
                           title=f'Predições vs Reais - {model_name}')
            
            plot_qq_analysis(results['actuals'], results['predictions'],
                           save_path=os.path.join(model_dir, 'qq_analysis.png'),
                           title=f'QQ-Plot - {model_name}')
            
            # ===== NOVOS GRÁFICOS FDA e FDP =====
            # Análise distribucional completa (FDA + FDP em uma figura)
            plot_distribution_analysis(results['actuals'], results['predictions'],
                                     save_path=os.path.join(model_dir, 'distribution_analysis.png'),
                                     title_prefix=f'{model_name}')
            
            # FDA separado
            plot_cdf_comparison(results['actuals'], results['predictions'],
                              save_path=os.path.join(model_dir, 'fda_comparison.png'),
                              title=f'FDA - {model_name}')
            
            # FDP separado
            plot_pdf_comparison(results['actuals'], results['predictions'],
                              save_path=os.path.join(model_dir, 'fdp_comparison.png'),
                              title=f'FDP - {model_name}')
            
            # ===== NOVO: TESTE DE KOLMOGOROV-SMIRNOV =====
            # Análise detalhada do teste KS de duas amostras
            ks_results = plot_ks_test_analysis(results['actuals'], results['predictions'],
                                             save_path=os.path.join(model_dir, 'ks_test_analysis.png'),
                                             title=f'Teste Kolmogorov-Smirnov - {model_name}')
            
            # Salvar resultados do teste KS em arquivo texto
            ks_summary_path = os.path.join(model_dir, 'ks_test_summary.txt')
            with open(ks_summary_path, 'w', encoding='utf-8') as f:
                f.write(f"TESTE DE KOLMOGOROV-SMIRNOV - {model_name}\n")
                f.write("=" * 50 + "\n\n")
                f.write("HIPÓTESES:\n")
                f.write("H₀: As predições e valores reais seguem a mesma distribuição\n")
                f.write("H₁: As predições e valores reais seguem distribuições diferentes\n\n")
                f.write("RESULTADOS:\n")
                f.write(f"Estatística KS: {ks_results['ks_statistic']:.6f}\n")
                f.write(f"p-value: {ks_results['p_value']:.6f}\n")
                f.write(f"Nível de significância (α): {ks_results['alpha']}\n")
                f.write(f"Localização da diferença máxima: {ks_results['max_diff_location']:.6f}\n\n")
                f.write("CONCLUSÃO:\n")
                f.write(f"{ks_results['conclusion']}\n\n")
                if ks_results['reject_h0']:
                    f.write("INTERPRETAÇÃO: Há evidência estatística significativa de que\n")
                    f.write("as distribuições das predições e valores reais são diferentes.\n")
                    f.write("O modelo NÃO reproduz adequadamente a distribuição dos dados.\n")
                else:
                    f.write("INTERPRETAÇÃO: NÃO há evidência estatística suficiente para\n")
                    f.write("afirmar que as distribuições sejam diferentes.\n")
                    f.write("O modelo reproduz adequadamente a distribuição dos dados.\n")
            
            print(f"Resumo do teste KS salvo em: {ks_summary_path}")
    
    # 4. Comparação FDA/FDP entre todos os modelos
    print("Gerando comparações distribucionais entre modelos...")
    
    # Criar diretório para comparações
    comparison_dir = os.path.join(output_dir, 'distribution_comparisons')
    os.makedirs(comparison_dir, exist_ok=True)
    
    # FDA comparativo
    plot_multi_model_cdf_comparison(results_dict, 
                                   save_path=os.path.join(comparison_dir, 'all_models_fda.png'))
    
    # FDP comparativo
    plot_multi_model_pdf_comparison(results_dict,
                                   save_path=os.path.join(comparison_dir, 'all_models_fdp.png'))
    
    # ===== NOVO: COMPARAÇÃO TESTE KS ENTRE MODELOS =====
    # Comparação dos testes de Kolmogorov-Smirnov
    ks_comparison_results = plot_multi_model_ks_comparison(results_dict,
                                                         save_path=os.path.join(comparison_dir, 'all_models_ks_comparison.png'))
    
    # Salvar resumo geral dos testes KS
    ks_general_summary_path = os.path.join(comparison_dir, 'ks_general_summary.txt')
    with open(ks_general_summary_path, 'w', encoding='utf-8') as f:
        f.write("RESUMO GERAL - TESTES DE KOLMOGOROV-SMIRNOV\n")
        f.write("=" * 60 + "\n\n")
        f.write("OBJETIVO: Avaliar se as predições de cada modelo seguem a mesma\n")
        f.write("distribuição de probabilidade dos valores reais.\n\n")
        f.write("HIPÓTESES:\n")
        f.write("H₀: Predições e valores reais seguem a mesma distribuição\n")
        f.write("H₁: Predições e valores reais seguem distribuições diferentes\n\n")
        f.write(f"NÍVEL DE SIGNIFICÂNCIA: α = {ks_comparison_results['alpha']}\n\n")
        f.write("RESULTADOS POR MODELO:\n")
        f.write("-" * 40 + "\n")
        
        for i, model in enumerate(ks_comparison_results['models']):
            f.write(f"\n{model}:\n")
            f.write(f"  Estatística KS: {ks_comparison_results['ks_statistics'][i]:.6f}\n")
            f.write(f"  p-value: {ks_comparison_results['p_values'][i]:.6f}\n")
            f.write(f"  Resultado: {'REJEITA H₀' if ks_comparison_results['rejections'][i] else 'NÃO REJEITA H₀'}\n")
            if ks_comparison_results['rejections'][i]:
                f.write(f"  Interpretação: Distribuições DIFERENTES - Modelo não adequado\n")
            else:
                f.write(f"  Interpretação: Distribuições SIMILARES - Modelo adequado\n")
        
        f.write(f"\n\nRESUMO ESTATÍSTICO:\n")
        f.write("-" * 30 + "\n")
        summary = ks_comparison_results['summary']
        f.write(f"Total de modelos avaliados: {summary['total_models']}\n")
        f.write(f"Modelos com distribuição adequada: {summary['models_with_good_distribution']}\n")
        f.write(f"Modelos com distribuição inadequada: {summary['models_with_different_distribution']}\n")
        f.write(f"Taxa de adequação: {(summary['models_with_good_distribution']/summary['total_models']*100):.1f}%\n\n")
        
        f.write("INTERPRETAÇÃO GERAL:\n")
        f.write("-" * 25 + "\n")
        if summary['models_with_good_distribution'] > summary['models_with_different_distribution']:
            f.write("A MAIORIA dos modelos reproduz adequadamente a distribuição dos dados.\n")
            f.write("Isso indica que as predições seguem padrões estatísticos similares\n")
            f.write("aos valores reais, o que é um bom indicativo de qualidade dos modelos.\n")
        elif summary['models_with_different_distribution'] > summary['models_with_good_distribution']:
            f.write("A MAIORIA dos modelos NÃO reproduz adequadamente a distribuição dos dados.\n")
            f.write("Isso pode indicar problemas de overfitting, underfitting ou\n")
            f.write("inadequação dos modelos para capturar as características estatísticas\n")
            f.write("dos dados reais.\n")
        else:
            f.write("Há um EMPATE entre modelos adequados e inadequados.\n")
            f.write("Recomenda-se análise mais detalhada de cada modelo individualmente.\n")
    
    print(f"Resumo geral dos testes KS salvo em: {ks_general_summary_path}")
    
    print(f"Relatório completo gerado em: {output_dir}")
    return df_metrics


def plot_cdf_comparison(actuals, predictions, save_path=None, title="Comparação FDA - Função Distribuição Acumulada"):
    """
    Plota comparação das Funções de Distribuição Acumulada (FDA/CDF)
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
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Gráfico FDA salvo em: {save_path}")
    
    plt.show()


def plot_pdf_comparison(actuals, predictions, save_path=None, title="Comparação FDP - Função de Distribuição de Probabilidade"):
    """
    Plota comparação das Funções de Distribuição de Probabilidade (FDP/PDF) usando KDE
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
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Gráfico FDP salvo em: {save_path}")
    
    plt.show()


def plot_distribution_analysis(actuals, predictions, save_path=None, title_prefix="Análise Distribucional"):
    """
    Cria análise completa das distribuições (FDA + FDP) em uma única figura
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
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Análise distribucional salva em: {save_path}")
    
    plt.show()


def plot_multi_model_cdf_comparison(results_dict, save_path=None, title="Comparação FDA - Todos os Modelos"):
    """
    Compara Funções de Distribuição Acumulada (FDA/CDF) de múltiplos modelos
    """
    plt.figure(figsize=(15, 10))
    
    # Cores para diferentes modelos
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':']
    
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
            
            color = colors[i % len(colors)]
            linestyle = linestyles[i % len(linestyles)]
            
            plt.plot(global_range, cdf_predictions, color=color, linestyle=linestyle,
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
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparação FDA multi-modelo salva em: {save_path}")
    
    plt.show()


def plot_multi_model_pdf_comparison(results_dict, save_path=None, title="Comparação FDP - Todos os Modelos"):
    """
    Compara Funções de Distribuição de Probabilidade (FDP/PDF) de múltiplos modelos
    """
    plt.figure(figsize=(15, 10))
    
    # Cores para diferentes modelos
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':']
    
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
                    
                    color = colors[i % len(colors)]
                    linestyle = linestyles[i % len(linestyles)]
                    
                    plt.plot(global_range, pdf_predictions, color=color, linestyle=linestyle,
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
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparação FDP multi-modelo salva em: {save_path}")
    
    plt.show()


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
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Análise do teste KS salva em: {save_path}")
    
    plt.show()
    
    # Retornar resultados do teste
    return {
        'ks_statistic': ks_statistic,
        'p_value': ks_pvalue,
        'reject_h0': reject_h0,
        'max_diff_location': max_diff_x,
        'alpha': alpha,
        'conclusion': test_conclusion
    }


def plot_multi_model_ks_comparison(results_dict, save_path=None, title="Comparação Teste Kolmogorov-Smirnov - Todos os Modelos", alpha=0.05):
    """
    Compara resultados do teste de Kolmogorov-Smirnov entre múltiplos modelos
    """
    plt.figure(figsize=(16, 10))
    
    # Coletar dados dos testes KS
    models = []
    ks_statistics = []
    p_values = []
    rejections = []
    
    for model_name, results in results_dict.items():
        if 'actuals' in results and 'predictions' in results:
            actuals = np.array(results['actuals']).flatten()
            predictions = np.array(results['predictions']).flatten()
            
            ks_stat, ks_pval = stats.ks_2samp(predictions, actuals)
            reject = ks_pval < alpha
            
            models.append(model_name)
            ks_statistics.append(ks_stat)
            p_values.append(ks_pval)
            rejections.append(reject)
    
    if not models:
        print("Nenhum modelo encontrado para comparação KS")
        return
    
    # Configurar subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))
    
    # ===== SUBPLOT 1: Estatísticas KS =====
    bars1 = ax1.bar(models, ks_statistics, color=['red' if r else 'green' for r in rejections], alpha=0.7)
    ax1.set_title('Estatísticas KS por Modelo', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Estatística KS', fontsize=10)
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Adicionar valores nas barras
    for bar, stat in zip(bars1, ks_statistics):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{stat:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # ===== SUBPLOT 2: P-values =====
    bars2 = ax2.bar(models, p_values, color=['red' if r else 'green' for r in rejections], alpha=0.7)
    ax2.axhline(y=alpha, color='black', linestyle='--', linewidth=2, label=f'α = {alpha}')
    ax2.set_title('P-values por Modelo', fontsize=12, fontweight='bold')
    ax2.set_ylabel('p-value', fontsize=10)
    ax2.tick_params(axis='x', rotation=45)
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Adicionar valores nas barras
    for bar, pval in zip(bars2, p_values):
        height = bar.get_height()
        if pval < 0.001:
            text = f'{pval:.2e}'
        else:
            text = f'{pval:.4f}'
        ax2.text(bar.get_x() + bar.get_width()/2., height * 1.2,
                text, ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # ===== SUBPLOT 3: Resultados dos Testes =====
    reject_counts = [sum(rejections), len(rejections) - sum(rejections)]
    labels = ['Rejeitam H₀\n(Distribuições Diferentes)', 'Não Rejeitam H₀\n(Distribuições Similares)']
    colors = ['lightcoral', 'lightgreen']
    
    wedges, texts, autotexts = ax3.pie(reject_counts, labels=labels, colors=colors, autopct='%1.1f%%',
                                      startangle=90)
    ax3.set_title('Distribuição dos Resultados dos Testes', fontsize=12, fontweight='bold')
    
    # ===== SUBPLOT 4: Tabela Resumo =====
    ax4.axis('off')
    
    # Criar dados da tabela
    table_data = []
    for i, model in enumerate(models):
        status = "REJEITA H₀" if rejections[i] else "NÃO REJEITA H₀"
        status_color = "red" if rejections[i] else "green"
        table_data.append([
            model,
            f'{ks_statistics[i]:.4f}',
            f'{p_values[i]:.4f}' if p_values[i] >= 0.001 else f'{p_values[i]:.2e}',
            status
        ])
    
    # Criar tabela
    col_labels = ['Modelo', 'Estatística KS', 'p-value', 'Resultado']
    table = ax4.table(cellText=table_data, colLabels=col_labels,
                     cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    
    # Estilizar tabela
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)
    
    # Estilizar cabeçalho
    for i in range(len(col_labels)):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Estilizar células
    for i in range(1, len(table_data) + 1):
        for j in range(len(col_labels)):
            if j == 3:  # Coluna de resultado
                if rejections[i-1]:
                    table[(i, j)].set_facecolor('#ffcccb')  # Vermelho claro
                    table[(i, j)].set_text_props(color='darkred', weight='bold')
                else:
                    table[(i, j)].set_facecolor('#d4edda')  # Verde claro
                    table[(i, j)].set_text_props(color='darkgreen', weight='bold')
            else:
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f8f9fa')
                else:
                    table[(i, j)].set_facecolor('#ffffff')
    
    ax4.set_title('Resumo Detalhado dos Testes KS', fontsize=12, fontweight='bold', pad=20)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Adicionar interpretação geral
    total_models = len(models)
    rejected_models = sum(rejections)
    good_models = total_models - rejected_models
    
    interpretation = (
        f"INTERPRETAÇÃO GERAL:\n"
        f"• {good_models}/{total_models} modelos reproduzem adequadamente a distribuição dos dados\n"
        f"• {rejected_models}/{total_models} modelos apresentam distribuições significativamente diferentes\n"
        f"• Nível de significância utilizado: α = {alpha}"
    )
    
    plt.figtext(0.5, 0.02, interpretation, ha='center', va='bottom', 
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                fontsize=11, fontweight='bold')
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparação KS multi-modelo salva em: {save_path}")
    
    plt.show()
    
    return {
        'models': models,
        'ks_statistics': ks_statistics,
        'p_values': p_values,
        'rejections': rejections,
        'alpha': alpha,
        'summary': {
            'total_models': total_models,
            'models_with_good_distribution': good_models,
            'models_with_different_distribution': rejected_models
        }
    } 
"""
Gráficos de comparação entre modelos
"""
import matplotlib
matplotlib.use('Agg')  # Backend não-interativo que apenas salva arquivos
import matplotlib.pyplot as plt
plt.ioff()  # Desabilitar modo interativo

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from .utils import (ensure_output_dir, print_save_message, get_colors_and_styles, 
                   validate_and_clean_metrics, format_metric_value, get_medal_emoji, get_status_emoji)


def plot_models_comparison_overview(results_dict, save_path=None,
                                   title="Visão Geral - Comparação de Modelos"):
    """
    Visão geral comparativa de múltiplos modelos com métricas principais
    
    Args:
        results_dict: Dicionário com resultados dos modelos
        save_path: Caminho para salvar o gráfico
        title: Título do gráfico
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # Validar e limpar dados
    clean_results = validate_and_clean_metrics(results_dict)
    
    if not clean_results:
        fig.text(0.5, 0.5, 'Dados insuficientes para comparação', 
                ha='center', va='center', fontsize=16,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        plt.tight_layout()
        if save_path:
            ensure_output_dir(save_path)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print_save_message(save_path, "Comparação de modelos")
        plt.close()  # Fechar figura para liberar memória
        return
    
    colors, _ = get_colors_and_styles(len(clean_results))
    model_names = list(clean_results.keys())
    
    # Calcular métricas principais
    metrics_data = {}
    for model_name, results in clean_results.items():
        if 'actuals' in results and 'predictions' in results:
            actuals = np.array(results['actuals']).flatten()
            predictions = np.array(results['predictions']).flatten()
            
            metrics_data[model_name] = {
                'R²': r2_score(actuals, predictions),
                'RMSE': np.sqrt(mean_squared_error(actuals, predictions)),
                'MAE': mean_absolute_error(actuals, predictions),
                'MAPE': np.mean(np.abs((actuals - predictions) / actuals)) * 100 if np.all(actuals != 0) else np.nan
            }
    
    # ===== SUBPLOT 1: R² =====
    ax1.set_title('Coeficiente de Determinação (R²)', fontsize=14, fontweight='bold')
    r2_values = [metrics_data[name]['R²'] for name in model_names if name in metrics_data]
    valid_names = [name for name in model_names if name in metrics_data]
    
    bars1 = ax1.bar(valid_names, r2_values, color=colors[:len(valid_names)], alpha=0.7)
    ax1.set_ylabel('R²', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, 1)
    
    # Adicionar linha de referência
    ax1.axhline(y=0.8, color='green', linestyle='--', alpha=0.7, label='Bom (0.8)')
    ax1.axhline(y=0.9, color='darkgreen', linestyle='--', alpha=0.7, label='Excelente (0.9)')
    ax1.legend(fontsize=10)
    
    # Valores nas barras
    for bar, val in zip(bars1, r2_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # ===== SUBPLOT 2: RMSE =====
    ax2.set_title('Raiz do Erro Quadrático Médio (RMSE)', fontsize=14, fontweight='bold')
    rmse_values = [metrics_data[name]['RMSE'] for name in valid_names]
    
    bars2 = ax2.bar(valid_names, rmse_values, color=colors[:len(valid_names)], alpha=0.7)
    ax2.set_ylabel('RMSE', fontsize=12)
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Valores nas barras
    for bar, val in zip(bars2, rmse_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + max(rmse_values)*0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # ===== SUBPLOT 3: MAE =====
    ax3.set_title('Erro Absoluto Médio (MAE)', fontsize=14, fontweight='bold')
    mae_values = [metrics_data[name]['MAE'] for name in valid_names]
    
    bars3 = ax3.bar(valid_names, mae_values, color=colors[:len(valid_names)], alpha=0.7)
    ax3.set_ylabel('MAE', fontsize=12)
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Valores nas barras
    for bar, val in zip(bars3, mae_values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + max(mae_values)*0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # ===== SUBPLOT 4: Ranking e Resumo =====
    ax4.set_title('Ranking dos Modelos', fontsize=14, fontweight='bold')
    ax4.axis('off')
    
    # Calcular ranking baseado em múltiplas métricas
    model_scores = []
    for name in valid_names:
        score = 0
        metrics = metrics_data[name]
        
        # R² (quanto maior, melhor) - peso 3
        r2_rank = sorted(valid_names, key=lambda x: metrics_data[x]['R²'], reverse=True).index(name)
        score += (len(valid_names) - r2_rank) * 3
        
        # RMSE (quanto menor, melhor) - peso 2
        rmse_rank = sorted(valid_names, key=lambda x: metrics_data[x]['RMSE']).index(name)
        score += (len(valid_names) - rmse_rank) * 2
        
        # MAE (quanto menor, melhor) - peso 2
        mae_rank = sorted(valid_names, key=lambda x: metrics_data[x]['MAE']).index(name)
        score += (len(valid_names) - mae_rank) * 2
        
        model_scores.append((name, score, metrics))
    
    # Ordenar por score
    model_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Criar texto do ranking
    ranking_text = "RANKING GERAL:\n\n"
    
    for rank, (name, score, metrics) in enumerate(model_scores, 1):
        medal = get_medal_emoji(rank)
        if rank == 1:
            color_bg = 'gold'
        elif rank == 2:
            color_bg = 'silver'
        elif rank == 3:
            color_bg = '#CD7F32'
        else:
            color_bg = 'lightgray'
        
        ranking_text += f"{medal} {name}\n"
        ranking_text += f"   Score: {score}\n"
        ranking_text += f"   R²: {metrics['R²']:.4f}\n"
        ranking_text += f"   RMSE: {metrics['RMSE']:.4f}\n"
        ranking_text += f"   MAE: {metrics['MAE']:.4f}\n"
        if not np.isnan(metrics['MAPE']):
            ranking_text += f"   MAPE: {metrics['MAPE']:.2f}%\n"
        ranking_text += "\n"
    
    # Análise do melhor modelo
    if model_scores:
        best_model, best_score, best_metrics = model_scores[0]
        ranking_text += f"MELHOR MODELO: {best_model}\n"
        if best_metrics['R²'] > 0.9:
            ranking_text += "• Excelente ajuste (R² > 0.9)\n"
        elif best_metrics['R²'] > 0.8:
            ranking_text += "• Bom ajuste (R² > 0.8)\n"
        else:
            ranking_text += "• Ajuste moderado\n"
        
        # Comparação com o segundo melhor
        if len(model_scores) > 1:
            second_best = model_scores[1]
            r2_diff = best_metrics['R²'] - second_best[2]['R²']
            ranking_text += f"• Vantagem sobre 2º: +{r2_diff:.4f} em R²\n"
    
    ax4.text(0.05, 0.95, ranking_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        ensure_output_dir(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print_save_message(save_path, "Comparação de modelos")
    
    plt.close()  # Fechar figura para liberar memória
    
    return {
        'rankings': model_scores,
        'metrics_summary': metrics_data
    }


def plot_predictions_comparison(results_dict, n_show=500, save_path=None,
                               title="Comparação de Predições"):
    """
    Compara predições de múltiplos modelos contra valores reais
    
    Args:
        results_dict: Dicionário com resultados dos modelos
        n_show: Número de pontos a mostrar
        save_path: Caminho para salvar o gráfico
        title: Título do gráfico
    """
    # Validar dados
    clean_results = validate_and_clean_metrics(results_dict)
    
    if not clean_results:
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, 'Dados insuficientes para comparação', 
                ha='center', va='center', fontsize=16,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        plt.close()  # Fechar figura para liberar memória
        return
    
    n_models = len(clean_results)
    
    # Criar subplots
    if n_models <= 2:
        fig, axes = plt.subplots(1, n_models, figsize=(10*n_models, 8))
    elif n_models <= 4:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    else:
        n_cols = 3
        n_rows = (n_models + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 6*n_rows))
    
    # Garantir que axes seja sempre um array
    if n_models == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_models > 1 else [axes]
    
    colors, _ = get_colors_and_styles(n_models)
    
    for i, (model_name, results) in enumerate(clean_results.items()):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        if 'actuals' in results and 'predictions' in results:
            actuals = np.array(results['actuals']).flatten()
            predictions = np.array(results['predictions']).flatten()
            
            # Limitar número de pontos mostrados
            if len(actuals) > n_show:
                indices = np.random.choice(len(actuals), n_show, replace=False)
                actuals_show = actuals[indices]
                predictions_show = predictions[indices]
            else:
                actuals_show = actuals
                predictions_show = predictions
            
            # Scatter plot
            ax.scatter(actuals_show, predictions_show, alpha=0.6, s=20, 
                      color=colors[i], edgecolors='darker', linewidth=0.5)
            
            # Linha perfeita
            min_val = min(np.min(actuals_show), np.min(predictions_show))
            max_val = max(np.max(actuals_show), np.max(predictions_show))
            ax.plot([min_val, max_val], [min_val, max_val], 'r-', linewidth=2, 
                   alpha=0.8, label='Predição Perfeita')
            
            # Calcular métricas
            r2 = r2_score(actuals, predictions)
            rmse = np.sqrt(mean_squared_error(actuals, predictions))
            mae = mean_absolute_error(actuals, predictions)
            
            ax.set_title(f'{model_name}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Valores Reais', fontsize=10)
            ax.set_ylabel('Predições', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)
            
            # Adicionar métricas
            metrics_text = f'R² = {r2:.4f}\nRMSE = {rmse:.4f}\nMAE = {mae:.4f}'
            ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, 
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                   verticalalignment='top', fontsize=9, fontfamily='monospace')
        else:
            ax.text(0.5, 0.5, f'Dados não disponíveis\npara {model_name}', 
                   transform=ax.transAxes, ha='center', va='center',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # Ocultar axes extras
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    
    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        ensure_output_dir(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print_save_message(save_path, "Comparação de predições")
    
    plt.close()  # Fechar figura para liberar memória


def plot_residuals_comparison(results_dict, save_path=None,
                             title="Comparação de Resíduos"):
    """
    Compara distribuições de resíduos de múltiplos modelos
    
    Args:
        results_dict: Dicionário com resultados dos modelos
        save_path: Caminho para salvar o gráfico
        title: Título do gráfico
    """
    clean_results = validate_and_clean_metrics(results_dict)
    
    if not clean_results:
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, 'Dados insuficientes para comparação', 
                ha='center', va='center', fontsize=16,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        plt.close()  # Fechar figura para liberar memória
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    colors, _ = get_colors_and_styles(len(clean_results))
    
    # Calcular resíduos para todos os modelos
    residuals_data = {}
    for model_name, results in clean_results.items():
        if 'actuals' in results and 'predictions' in results:
            actuals = np.array(results['actuals']).flatten()
            predictions = np.array(results['predictions']).flatten()
            residuals = predictions - actuals
            residuals_data[model_name] = residuals
    
    # ===== SUBPLOT 1: Histogramas dos Resíduos =====
    ax1.set_title('Distribuição dos Resíduos', fontsize=14, fontweight='bold')
    
    for i, (model_name, residuals) in enumerate(residuals_data.items()):
        ax1.hist(residuals, bins=50, alpha=0.6, color=colors[i], 
                label=model_name, density=True, edgecolor='darker')
    
    ax1.axvline(x=0, color='black', linestyle='--', alpha=0.8, label='Zero')
    ax1.set_xlabel('Resíduo', fontsize=12)
    ax1.set_ylabel('Densidade', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # ===== SUBPLOT 2: Box Plots dos Resíduos =====
    ax2.set_title('Box Plots dos Resíduos', fontsize=14, fontweight='bold')
    
    residuals_list = list(residuals_data.values())
    model_names = list(residuals_data.keys())
    
    box_plot = ax2.boxplot(residuals_list, labels=model_names, patch_artist=True)
    
    # Colorir as caixas
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.8, label='Zero')
    ax2.set_ylabel('Resíduo', fontsize=12)
    ax2.tick_params(axis='x', rotation=45)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # ===== SUBPLOT 3: Q-Q Plots =====
    ax3.set_title('Q-Q Plots vs Distribuição Normal', fontsize=14, fontweight='bold')
    
    from scipy import stats
    
    for i, (model_name, residuals) in enumerate(residuals_data.items()):
        # Calcular quantis teóricos e empíricos
        theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(residuals)))
        sample_quantiles = np.sort(residuals)
        
        ax3.scatter(theoretical_quantiles, sample_quantiles, alpha=0.6, 
                   color=colors[i], label=model_name, s=10)
    
    # Linha de referência
    all_residuals = np.concatenate(list(residuals_data.values()))
    min_q, max_q = np.min(all_residuals), np.max(all_residuals)
    ax3.plot([min_q, max_q], [min_q, max_q], 'r-', linewidth=2, alpha=0.8, 
             label='Distribuição Normal')
    
    ax3.set_xlabel('Quantis Teóricos (Normal)', fontsize=12)
    ax3.set_ylabel('Quantis Empíricos', fontsize=12)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # ===== SUBPLOT 4: Estatísticas dos Resíduos =====
    ax4.set_title('Estatísticas dos Resíduos', fontsize=14, fontweight='bold')
    ax4.axis('off')
    
    # Calcular estatísticas para cada modelo
    stats_text = "ESTATÍSTICAS DOS RESÍDUOS:\n\n"
    
    for i, (model_name, residuals) in enumerate(residuals_data.items()):
        mean_res = np.mean(residuals)
        std_res = np.std(residuals)
        skew_res = stats.skew(residuals)
        kurt_res = stats.kurtosis(residuals)
        
        # Teste de normalidade
        try:
            if len(residuals) <= 5000:
                shapiro_stat, shapiro_p = stats.shapiro(residuals)
                shapiro_text = f"Shapiro: {shapiro_stat:.4f} (p={shapiro_p:.4f})"
            else:
                shapiro_text = "Shapiro: N/A (amostra grande)"
        except:
            shapiro_text = "Shapiro: Erro no cálculo"
        
        # Análise qualitativa
        if abs(mean_res) < std_res * 0.1 and abs(skew_res) < 0.5 and abs(kurt_res) < 1:
            quality = f"{get_status_emoji('good')} EXCELENTE"
            quality_color = get_status_emoji('excellent')
        elif abs(mean_res) < std_res * 0.2 and abs(skew_res) < 1 and abs(kurt_res) < 2:
            quality = f"{get_status_emoji('warning')} BOM"
            quality_color = get_status_emoji('moderate')
        else:
            quality = f"{get_status_emoji('bad')} PROBLEMÁTICO"
            quality_color = get_status_emoji('poor')
        
        stats_text += f"{quality_color} {model_name}:\n"
        stats_text += f"  Média: {mean_res:.6f}\n"
        stats_text += f"  Desvio: {std_res:.6f}\n"
        stats_text += f"  Assimetria: {skew_res:.4f}\n"
        stats_text += f"  Curtose: {kurt_res:.4f}\n"
        stats_text += f"  {shapiro_text}\n"
        stats_text += f"  Qualidade: {quality}\n\n"
    
    # Ranking dos resíduos
    residual_scores = []
    for model_name, residuals in residuals_data.items():
        score = 0
        mean_res = abs(np.mean(residuals))
        std_res = np.std(residuals)
        skew_res = abs(stats.skew(residuals))
        kurt_res = abs(stats.kurtosis(residuals))
        
        # Pontuação baseada na qualidade dos resíduos
        if mean_res < std_res * 0.1: score += 3
        elif mean_res < std_res * 0.2: score += 2
        else: score += 1
        
        if skew_res < 0.5: score += 2
        elif skew_res < 1: score += 1
        
        if kurt_res < 1: score += 2
        elif kurt_res < 2: score += 1
        
        residual_scores.append((model_name, score))
    
    residual_scores.sort(key=lambda x: x[1], reverse=True)
    
    stats_text += "RANKING (Qualidade dos Resíduos):\n"
    for rank, (name, score) in enumerate(residual_scores, 1):
        medal = get_medal_emoji(rank)
        stats_text += f"{medal} {name} (Score: {score}/7)\n"
    
    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        ensure_output_dir(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print_save_message(save_path, "Comparação de resíduos")
    
    plt.close()  # Fechar figura para liberar memória
    
    return {
        'residuals_statistics': {name: {
            'mean': float(np.mean(res)),
            'std': float(np.std(res)),
            'skewness': float(stats.skew(res)),
            'kurtosis': float(stats.kurtosis(res))
        } for name, res in residuals_data.items()},
        'residuals_ranking': residual_scores
    }


def plot_training_comparison(results_dict, save_path=None,
                            title="Comparação do Histórico de Treinamento"):
    """
    Compara históricos de treinamento de múltiplos modelos
    
    Args:
        results_dict: Dicionário com resultados dos modelos
        save_path: Caminho para salvar o gráfico
        title: Título do gráfico
    """
    # Filtrar modelos que têm dados de treinamento
    training_data = {}
    for model_name, results in results_dict.items():
        if 'train_losses' in results and 'val_losses' in results:
            training_data[model_name] = results
    
    if not training_data:
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, 'Dados de treinamento não disponíveis', 
                ha='center', va='center', fontsize=16,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        plt.close()  # Fechar figura para liberar memória
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    colors, linestyles = get_colors_and_styles(len(training_data))
    
    # ===== SUBPLOT 1: Loss de Treinamento =====
    ax1.set_title('Loss de Treinamento', fontsize=14, fontweight='bold')
    
    for i, (model_name, results) in enumerate(training_data.items()):
        train_losses = results['train_losses']
        epochs = range(1, len(train_losses) + 1)
        ax1.plot(epochs, train_losses, color=colors[i], linestyle=linestyles[i],
                linewidth=2, label=model_name, alpha=0.8)
    
    ax1.set_xlabel('Época', fontsize=12)
    ax1.set_ylabel('Loss de Treinamento', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')  # Escala logarítmica para melhor visualização
    
    # ===== SUBPLOT 2: Loss de Validação =====
    ax2.set_title('Loss de Validação', fontsize=14, fontweight='bold')
    
    for i, (model_name, results) in enumerate(training_data.items()):
        val_losses = results['val_losses']
        epochs = range(1, len(val_losses) + 1)
        ax2.plot(epochs, val_losses, color=colors[i], linestyle=linestyles[i],
                linewidth=2, label=model_name, alpha=0.8)
    
    ax2.set_xlabel('Época', fontsize=12)
    ax2.set_ylabel('Loss de Validação', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # ===== SUBPLOT 3: Curvas de Aprendizado Combinadas =====
    ax3.set_title('Curvas de Aprendizado (Train vs Val)', fontsize=14, fontweight='bold')
    
    for i, (model_name, results) in enumerate(training_data.items()):
        train_losses = results['train_losses']
        val_losses = results['val_losses']
        epochs = range(1, len(train_losses) + 1)
        
        # Treino (linha sólida)
        ax3.plot(epochs, train_losses, color=colors[i], linestyle='-',
                linewidth=2, label=f'{model_name} (Train)', alpha=0.8)
        
        # Validação (linha tracejada)
        ax3.plot(epochs, val_losses, color=colors[i], linestyle='--',
                linewidth=2, label=f'{model_name} (Val)', alpha=0.8)
    
    ax3.set_xlabel('Época', fontsize=12)
    ax3.set_ylabel('Loss', fontsize=12)
    ax3.legend(fontsize=9, ncol=2)
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    # ===== SUBPLOT 4: Análise de Convergência =====
    ax4.set_title('Análise de Convergência', fontsize=14, fontweight='bold')
    ax4.axis('off')
    
    # Analisar convergência para cada modelo
    convergence_text = "ANÁLISE DE CONVERGÊNCIA:\n\n"
    
    convergence_scores = []
    
    for model_name, results in training_data.items():
        train_losses = np.array(results['train_losses'])
        val_losses = np.array(results['val_losses'])
        
        # Métricas de convergência
        final_train_loss = train_losses[-1]
        final_val_loss = val_losses[-1]
        min_val_loss = np.min(val_losses)
        min_val_epoch = np.argmin(val_losses) + 1
        
        # Verificar overfitting
        overfitting_gap = final_val_loss - final_train_loss
        overfitting_ratio = final_val_loss / final_train_loss if final_train_loss > 0 else np.inf
        
        # Estabilidade (variação nos últimos 10% das épocas)
        last_10_percent = max(1, len(val_losses) // 10)
        val_stability = np.std(val_losses[-last_10_percent:])
        
        # Velocidade de convergência (épocas para atingir 90% do melhor loss)
        target_loss = min_val_loss * 1.1  # 110% do melhor loss
        convergence_epoch = len(val_losses)  # Default: última época
        for i, loss in enumerate(val_losses):
            if loss <= target_loss:
                convergence_epoch = i + 1
                break
        
        # Score de convergência
        score = 0
        
        # Melhor loss (peso 3)
        if min_val_loss < 0.01: score += 3
        elif min_val_loss < 0.1: score += 2
        else: score += 1
        
        # Overfitting (peso 2)
        if overfitting_ratio < 1.1: score += 2
        elif overfitting_ratio < 1.5: score += 1
        
        # Velocidade (peso 1)
        if convergence_epoch < len(val_losses) * 0.5: score += 1
        
        convergence_scores.append((model_name, score, {
            'final_train_loss': final_train_loss,
            'final_val_loss': final_val_loss,
            'min_val_loss': min_val_loss,
            'min_val_epoch': min_val_epoch,
            'overfitting_ratio': overfitting_ratio,
            'convergence_epoch': convergence_epoch,
            'total_epochs': len(val_losses)
        }))
        
        # Status de overfitting
        if overfitting_ratio < 1.1:
            overfitting_status = "✓ Sem overfitting"
            status_color = "🟢"
        elif overfitting_ratio < 1.5:
            overfitting_status = "⚠ Overfitting leve"
            status_color = "🟡"
        else:
            overfitting_status = "✗ Overfitting severo"
            status_color = "🔴"
        
        convergence_text += f"{status_color} {model_name}:\n"
        convergence_text += f"  Loss final (train): {final_train_loss:.6f}\n"
        convergence_text += f"  Loss final (val): {final_val_loss:.6f}\n"
        convergence_text += f"  Melhor val loss: {min_val_loss:.6f} (época {min_val_epoch})\n"
        convergence_text += f"  Razão overfitting: {overfitting_ratio:.2f}\n"
        convergence_text += f"  Convergência em: {convergence_epoch}/{len(val_losses)} épocas\n"
        convergence_text += f"  Status: {overfitting_status}\n\n"
    
    # Ranking de convergência
    convergence_scores.sort(key=lambda x: x[1], reverse=True)
    
    convergence_text += "RANKING (Qualidade de Convergência):\n"
    for rank, (name, score, _) in enumerate(convergence_scores, 1):
        medal = get_medal_emoji(rank)
        convergence_text += f"{medal} {name} (Score: {score}/6)\n"
    
    ax4.text(0.05, 0.95, convergence_text, transform=ax4.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        ensure_output_dir(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print_save_message(save_path, "Comparação de treinamento")
    
    plt.close()  # Fechar figura para liberar memória
    
    return {
        'convergence_analysis': {name: metrics for name, score, metrics in convergence_scores},
        'convergence_ranking': [(name, score) for name, score, _ in convergence_scores]
    }


def plot_performance_radar(results_dict, save_path=None,
                          title="Gráfico Radar - Performance dos Modelos"):
    """
    Cria gráfico radar comparando múltiplas métricas de performance
    
    Args:
        results_dict: Dicionário com resultados dos modelos
        save_path: Caminho para salvar o gráfico
        title: Título do gráfico
    """
    clean_results = validate_and_clean_metrics(results_dict)
    
    if not clean_results:
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, 'Dados insuficientes para gráfico radar', 
                ha='center', va='center', fontsize=16,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        plt.close()  # Fechar figura para liberar memória
        return
    
    # Calcular métricas normalizadas
    all_metrics = {}
    metric_names = ['R²', 'RMSE_inv', 'MAE_inv', 'MAPE_inv']  # _inv significa invertido (1-metric) para normalização
    
    for model_name, results in clean_results.items():
        if 'actuals' in results and 'predictions' in results:
            actuals = np.array(results['actuals']).flatten()
            predictions = np.array(results['predictions']).flatten()
            
            r2 = r2_score(actuals, predictions)
            rmse = np.sqrt(mean_squared_error(actuals, predictions))
            mae = mean_absolute_error(actuals, predictions)
            mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100 if np.all(actuals != 0) else 50
            
            all_metrics[model_name] = {
                'R²': max(0, r2),  # Garantir que R² não seja negativo
                'RMSE': rmse,
                'MAE': mae,
                'MAPE': mape
            }
    
    if not all_metrics:
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, 'Não foi possível calcular métricas', 
                ha='center', va='center', fontsize=16,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        plt.close()  # Fechar figura para liberar memória
        return
    
    # Normalizar métricas (0-1, onde 1 é melhor)
    r2_values = [metrics['R²'] for metrics in all_metrics.values()]
    rmse_values = [metrics['RMSE'] for metrics in all_metrics.values()]
    mae_values = [metrics['MAE'] for metrics in all_metrics.values()]
    mape_values = [metrics['MAPE'] for metrics in all_metrics.values()]
    
    # Para métricas onde menor é melhor, inverter
    max_rmse = max(rmse_values) if rmse_values else 1
    max_mae = max(mae_values) if mae_values else 1
    max_mape = max(mape_values) if mape_values else 100
    
    normalized_metrics = {}
    for model_name, metrics in all_metrics.items():
        normalized_metrics[model_name] = [
            metrics['R²'],  # R² já está 0-1, maior é melhor
            1 - (metrics['RMSE'] / max_rmse) if max_rmse > 0 else 0,  # Inverter RMSE
            1 - (metrics['MAE'] / max_mae) if max_mae > 0 else 0,    # Inverter MAE
            1 - (metrics['MAPE'] / max_mape) if max_mape > 0 else 0  # Inverter MAPE
        ]
    
    # Criar gráfico radar
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))
    
    # Configurar ângulos
    angles = np.linspace(0, 2 * np.pi, len(metric_names), endpoint=False).tolist()
    angles += angles[:1]  # Fechar o círculo
    
    # Labels das métricas
    metric_labels = ['R²', 'RMSE (inv)', 'MAE (inv)', 'MAPE (inv)']
    
    colors, _ = get_colors_and_styles(len(normalized_metrics))
    
    # Plotar cada modelo
    for i, (model_name, values) in enumerate(normalized_metrics.items()):
        values += values[:1]  # Fechar o círculo
        
        ax.plot(angles, values, color=colors[i], linewidth=2, 
               label=model_name, alpha=0.8)
        ax.fill(angles, values, color=colors[i], alpha=0.2)
    
    # Configurar gráfico
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.title(title, size=16, fontweight='bold', y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=11)
    
    # Adicionar explicação
    explanation_text = (
        "Interpretação:\n"
        "• Valores próximos a 1.0 = Melhor performance\n"
        "• R²: Coeficiente de determinação\n"
        "• RMSE (inv): 1 - RMSE normalizado\n"
        "• MAE (inv): 1 - MAE normalizado\n"
        "• MAPE (inv): 1 - MAPE normalizado\n"
        "• Área maior = Modelo melhor"
    )
    
    fig.text(0.02, 0.02, explanation_text, fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8),
             verticalalignment='bottom')
    
    plt.tight_layout()
    
    if save_path:
        ensure_output_dir(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print_save_message(save_path, "Gráfico radar")
    
    plt.close()  # Fechar figura para liberar memória
    
    # Calcular score total de cada modelo
    total_scores = {}
    for model_name, values in normalized_metrics.items():
        total_scores[model_name] = np.mean(values[:-1])  # Excluir o último elemento (duplicado)
    
    # Ranking
    ranking = sorted(total_scores.items(), key=lambda x: x[1], reverse=True)
    
    return {
        'normalized_metrics': normalized_metrics,
        'total_scores': total_scores,
        'ranking': ranking,
        'raw_metrics': all_metrics
    } 
"""
Utilitários para visualização e formatação de métricas
"""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import os

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


def ensure_output_dir(save_path):
    """
    Garante que o diretório de saída existe
    
    Args:
        save_path: Caminho para o arquivo de saída
    """
    if save_path:
        dir_path = os.path.dirname(save_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)


def get_colors_and_styles(n_items):
    """
    Retorna cores e estilos de linha para múltiplos itens
    
    Args:
        n_items: Número de itens que precisam de cores/estilos
        
    Returns:
        tuple: (colors, linestyles)
    """
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':']
    
    # Repetir se necessário
    colors = (colors * ((n_items // len(colors)) + 1))[:n_items]
    linestyles = (linestyles * ((n_items // len(linestyles)) + 1))[:n_items]
    
    return colors, linestyles


def add_metrics_text_box(ax, metrics_dict, title="Métricas", 
                        position=(0.02, 0.98), box_color='lightblue',
                        fontsize=9, family='monospace'):
    """
    Adiciona uma caixa de texto com métricas ao gráfico
    
    Args:
        ax: Eixo do matplotlib
        metrics_dict: Dicionário com métricas
        title: Título da caixa
        position: Posição da caixa (x, y) em coordenadas relativas
        box_color: Cor de fundo da caixa
        fontsize: Tamanho da fonte
        family: Família da fonte
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


def print_save_message(save_path, description="Gráfico"):
    """
    Imprime mensagem de confirmação de salvamento
    
    Args:
        save_path: Caminho onde foi salvo
        description: Descrição do que foi salvo
    """
    if save_path:
        print(f"{description} salvo em: {save_path}") 
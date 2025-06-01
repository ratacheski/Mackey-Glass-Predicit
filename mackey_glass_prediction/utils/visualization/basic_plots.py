"""
Gráficos básicos para visualização de treinamento e predições
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from .utils import (format_metric_value, validate_and_clean_metrics, 
                   ensure_output_dir, print_save_message)


def plot_training_history(train_losses, val_losses, save_path=None, 
                         title="Histórico de Treinamento"):
    """
    Plota o histórico de loss de treinamento e validação
    
    Args:
        train_losses: Lista com losses de treinamento
        val_losses: Lista com losses de validação
        save_path: Caminho para salvar o gráfico
        title: Título do gráfico
    """
    plt.figure(figsize=(12, 5))
    
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
        ensure_output_dir(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print_save_message(save_path, "Gráfico de treinamento")
    
    plt.show()


def plot_predictions(actuals, predictions, n_show=500, save_path=None, 
                    title="Predições vs Valores Reais"):
    """
    Plota predições vs valores reais
    
    Args:
        actuals: Valores reais
        predictions: Predições do modelo
        n_show: Número máximo de pontos a mostrar
        save_path: Caminho para salvar o gráfico
        title: Título do gráfico
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
        ensure_output_dir(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print_save_message(save_path, "Gráfico de predições")
    
    plt.show()


def plot_sequential_predictions(original_series, train_size, predictions, 
                               actual_test, save_path=None,
                               title="Predições Sequenciais"):
    """
    Plota série original com predições sequenciais
    
    Args:
        original_series: Série temporal original completa
        train_size: Tamanho do conjunto de treinamento
        predictions: Predições sequenciais
        actual_test: Valores reais do conjunto de teste
        save_path: Caminho para salvar o gráfico
        title: Título do gráfico
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
        ensure_output_dir(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print_save_message(save_path, "Gráfico de predições sequenciais")
    
    plt.show()


def plot_metrics_comparison(results_dict, save_path=None, 
                           title="Comparação de Métricas entre Modelos"):
    """
    Plota comparação de métricas entre diferentes modelos
    
    Args:
        results_dict: Dicionário com resultados dos modelos
        save_path: Caminho para salvar o gráfico
        title: Título do gráfico
    """
    # Validar e limpar dados
    results_dict = validate_and_clean_metrics(results_dict)
    
    # Preparar dados
    models = list(results_dict.keys())
    # Métricas que serão plotadas
    metrics = ['MSE', 'R²']
    
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
        ensure_output_dir(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print_save_message(save_path, "Comparação de métricas")
    
    plt.show()


def save_metrics_table(results_dict, save_path):
    """
    Salva tabela de métricas em CSV e cria visualização
    
    Args:
        results_dict: Dicionário com resultados dos modelos
        save_path: Caminho base para salvar arquivos
        
    Returns:
        DataFrame com as métricas
    """
    # Validar e limpar dados
    results_dict = validate_and_clean_metrics(results_dict)
    
    # Métricas que serão plotadas (mesmas da plot_metrics_comparison)
    target_metrics = ['MSE', 'R²']
    
    # Verificar se há modelos disponíveis
    if not results_dict:
        print("Nenhum modelo encontrado para salvar a tabela de métricas.")
        return pd.DataFrame()
    
    # Criar DataFrame apenas com as métricas especificadas
    data = []
    for model_name, results in results_dict.items():
        all_metrics = results['metrics']
        row = {'Modelo': model_name}
        
        # Adicionar apenas as métricas especificadas
        for metric in target_metrics:
            if metric in all_metrics:
                row[metric] = all_metrics[metric]
            else:
                row[metric] = np.nan  # Se métrica não estiver disponível
        
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
    print_save_message(csv_path, "Tabela de métricas")
    
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
        ensure_output_dir(png_path)
        plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.2)
        print_save_message(png_path, "Visualização da tabela")
    
    plt.show()
    
    return df 
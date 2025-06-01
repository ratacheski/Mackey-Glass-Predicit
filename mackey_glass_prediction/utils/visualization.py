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
        metric_name: Nome da métrica (MSE, RMSE, MAE, MAPE, R²)
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


def plot_prediction_scatter(actuals, predictions, save_path=None, 
                           title="Scatter Plot: Predições vs Valores Reais"):
    """
    Cria scatter plot das predições vs valores reais
    """
    plt.figure(figsize=(10, 8))
    
    # Scatter plot
    plt.scatter(actuals, predictions, alpha=0.6, s=20)
    
    # Linha perfeita (y=x)
    min_val = min(min(actuals), min(predictions))
    max_val = max(max(actuals), max(predictions))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Linha Perfeita')
    
    plt.title(title)
    plt.xlabel('Valores Reais')
    plt.ylabel('Predições')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Adicionar R²
    from sklearn.metrics import r2_score
    r2 = r2_score(actuals, predictions)
    r2_formatted = format_metric_value(r2, 'R²')
    plt.text(0.05, 0.95, f'R² = {r2_formatted}', transform=plt.gca().transAxes, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=10)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Scatter plot salvo em: {save_path}")
    
    plt.show()


def plot_metrics_comparison(results_dict, save_path=None, 
                           title="Comparação de Métricas entre Modelos"):
    """
    Plota comparação de métricas entre diferentes modelos
    """
    # Validar e limpar dados
    results_dict = validate_and_clean_metrics(results_dict)
    
    # Preparar dados
    models = list(results_dict.keys())
    metrics = ['MSE', 'RMSE', 'MAE', 'MAPE', 'R²']
    
    # Criar subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
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
            if metric == 'R²':
                best_value = max(valid_values)
                best_idx = values.index(best_value)
            else:
                best_value = min(valid_values)
                best_idx = values.index(best_value)
            
            bars[best_idx].set_color('#28a745')
            bars[best_idx].set_alpha(0.8)
    
    # Remover subplot extra
    fig.delaxes(axes[5])
    
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
            
            # Destacar melhor modelo (menor valor para MSE, RMSE, MAE, MAPE; maior para R²)
            if j > 0:  # Não aplicar ao nome do modelo
                metric_name = col_labels[j]
                values = df[metric_name].values
                
                # Filtrar valores válidos
                valid_mask = ~(np.isnan(values) | np.isinf(values))
                if valid_mask.any():
                    valid_values = values[valid_mask]
                    
                    # Encontrar melhor valor
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
            
            plot_prediction_scatter(results['actuals'], results['predictions'],
                                  save_path=os.path.join(model_dir, 'scatter.png'),
                                  title=f'Scatter Plot - {model_name}')
    
    print(f"Relatório completo gerado em: {output_dir}")
    return df_metrics 
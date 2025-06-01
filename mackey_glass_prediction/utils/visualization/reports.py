"""
Geração de relatórios abrangentes
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
import os
from io import StringIO

from .utils import ensure_output_dir, print_save_message, validate_and_clean_metrics, format_metric_value
from .basic_plots import plot_training_history, plot_predictions, plot_metrics_comparison
from .distribution_analysis import plot_qq_analysis, plot_cdf_comparison, plot_pdf_comparison
from .statistical_tests import plot_ks_test_analysis, plot_autocorrelation_analysis
from .comparison_plots import plot_models_comparison_overview, plot_performance_radar


def generate_comprehensive_report(results_dict, output_dir, model_name=None):
    """
    Gera relatório abrangente com todas as análises disponíveis
    
    Args:
        results_dict: Dicionário com resultados dos modelos (ou resultados de um único modelo)
        output_dir: Diretório para salvar o relatório
        model_name: Nome do modelo (usado quando results_dict contém dados de um único modelo)
    
    Returns:
        dict: Dicionário com caminhos dos arquivos gerados
    """
    ensure_output_dir(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    generated_files = {}
    
    # Determinar se é análise de um modelo ou comparação
    if model_name and isinstance(results_dict, dict) and 'actuals' in results_dict:
        # Análise de um único modelo
        single_model_results = {model_name: results_dict}
        report_type = "single"
        report_title = f"Relatório Abrangente - {model_name}"
    else:
        # Comparação de múltiplos modelos
        single_model_results = results_dict
        report_type = "comparison"
        report_title = "Relatório Comparativo de Modelos"
    
    print(f"\n🔄 Gerando relatório abrangente ({report_type})...")
    print(f"📁 Diretório de saída: {output_dir}")
    
    # ========== ANÁLISE GERAL ==========
    if report_type == "comparison":
        print("\n📊 Gerando visão geral comparativa...")
        overview_path = os.path.join(output_dir, f"01_visao_geral_{timestamp}.png")
        try:
            plot_models_comparison_overview(single_model_results, save_path=overview_path)
            generated_files['overview'] = overview_path
        except Exception as e:
            print(f"⚠️ Erro na visão geral: {e}")
    
    # ========== GRÁFICOS BÁSICOS ==========
    print("\n📈 Gerando gráficos básicos...")
    
    # Para cada modelo individualmente
    for i, (model_name, results) in enumerate(single_model_results.items(), 1):
        model_prefix = f"{i:02d}_{model_name.replace(' ', '_')}"
        
        # Histórico de treinamento
        if 'train_losses' in results and 'val_losses' in results:
            training_path = os.path.join(output_dir, f"{model_prefix}_treinamento_{timestamp}.png")
            try:
                plot_training_history(results['train_losses'], results['val_losses'], 
                                    save_path=training_path, title=f"Histórico de Treinamento - {model_name}")
                generated_files[f'training_{model_name}'] = training_path
            except Exception as e:
                print(f"⚠️ Erro no histórico de treinamento para {model_name}: {e}")
        
        # Predições vs Reais
        if 'actuals' in results and 'predictions' in results:
            predictions_path = os.path.join(output_dir, f"{model_prefix}_predicoes_{timestamp}.png")
            try:
                plot_predictions(results['actuals'], results['predictions'], 
                               save_path=predictions_path, title=f"Predições vs Valores Reais - {model_name}")
                generated_files[f'predictions_{model_name}'] = predictions_path
            except Exception as e:
                print(f"⚠️ Erro nas predições para {model_name}: {e}")
    
    # ========== ANÁLISES ESTATÍSTICAS ==========
    print("\n🔬 Gerando análises estatísticas...")
    
    for i, (model_name, results) in enumerate(single_model_results.items(), 1):
        if 'actuals' not in results or 'predictions' not in results:
            continue
            
        model_prefix = f"{i:02d}_{model_name.replace(' ', '_')}"
        
        # QQ-Plot
        qq_path = os.path.join(output_dir, f"{model_prefix}_qq_plot_{timestamp}.png")
        try:
            plot_qq_analysis(results['actuals'], results['predictions'], 
                           save_path=qq_path, title=f"QQ-Plot - {model_name}")
            generated_files[f'qq_{model_name}'] = qq_path
        except Exception as e:
            print(f"⚠️ Erro no Q-Q plot para {model_name}: {e}")
        
        # Análise de distribuições (CDF e FDP)
        cdf_path = os.path.join(output_dir, f"{model_prefix}_cdf_{timestamp}.png")
        try:
            plot_cdf_comparison(results['actuals'], results['predictions'], 
                              save_path=cdf_path, title=f"Comparação FDA - {model_name}")
            generated_files[f'cdf_{model_name}'] = cdf_path
        except Exception as e:
            print(f"⚠️ Erro na análise CDF para {model_name}: {e}")
        
        pdf_path = os.path.join(output_dir, f"{model_prefix}_pdf_{timestamp}.png")
        try:
            plot_pdf_comparison(results['actuals'], results['predictions'], 
                              save_path=pdf_path, title=f"Comparação FDP - {model_name}")
            generated_files[f'pdf_{model_name}'] = pdf_path
        except Exception as e:
            print(f"⚠️ Erro na análise FDP para {model_name}: {e}")
        
        # Teste de Kolmogorov-Smirnov
        ks_path = os.path.join(output_dir, f"{model_prefix}_ks_test_{timestamp}.png")
        try:
            plot_ks_test_analysis(results['actuals'], results['predictions'], 
                                 save_path=ks_path, title=f"Teste Kolmogorov-Smirnov - {model_name}")
            generated_files[f'ks_{model_name}'] = ks_path
        except Exception as e:
            print(f"⚠️ Erro no teste KS para {model_name}: {e}")
        
        # Análise de autocorrelação (se existir série temporal)
        if 'series' in results:
            autocorr_path = os.path.join(output_dir, f"{model_prefix}_autocorrelacao_{timestamp}.png")
            try:
                plot_autocorrelation_analysis(results['series'], 
                                             save_path=autocorr_path, title=f"Análise de Autocorrelação - {model_name}")
                generated_files[f'autocorr_{model_name}'] = autocorr_path
            except Exception as e:
                print(f"⚠️ Erro na análise de autocorrelação para {model_name}: {e}")
    
    # ========== ANÁLISES COMPARATIVAS ==========
    if report_type == "comparison" and len(single_model_results) > 1:
        print("\n🆚 Gerando análises comparativas...")
        
        # Gráfico radar
        radar_path = os.path.join(output_dir, f"99_radar_performance_{timestamp}.png")
        try:
            plot_performance_radar(single_model_results, save_path=radar_path)
            generated_files['radar'] = radar_path
        except Exception as e:
            print(f"⚠️ Erro no gráfico radar: {e}")
        
        # Tabela de métricas
        metrics_path = os.path.join(output_dir, f"99_tabela_metricas_{timestamp}.png")
        try:
            plot_metrics_comparison(single_model_results, save_path=metrics_path)
            generated_files['metrics_table'] = metrics_path
        except Exception as e:
            print(f"⚠️ Erro na tabela de métricas: {e}")
    
    # ========== RELATÓRIO TEXTUAL ==========
    print("\n📄 Gerando relatório textual...")
    text_report_path = os.path.join(output_dir, f"relatorio_textual_{timestamp}.txt")
    try:
        generate_text_report(single_model_results, text_report_path, report_type)
        generated_files['text_report'] = text_report_path
    except Exception as e:
        print(f"⚠️ Erro no relatório textual: {e}")
    
    # ========== SUMÁRIO HTML ==========
    print("\n🌐 Gerando sumário HTML...")
    html_report_path = os.path.join(output_dir, f"relatorio_html_{timestamp}.html")
    try:
        generate_html_report(single_model_results, generated_files, html_report_path, report_type)
        generated_files['html_report'] = html_report_path
    except Exception as e:
        print(f"⚠️ Erro no relatório HTML: {e}")
    
    print(f"\n✅ Relatório abrangente gerado com sucesso!")
    print(f"📁 {len(generated_files)} arquivos gerados em: {output_dir}")
    
    return generated_files


def generate_text_report(results_dict, save_path, report_type="comparison"):
    """
    Gera relatório textual detalhado
    
    Args:
        results_dict: Dicionário com resultados dos modelos
        save_path: Caminho para salvar o relatório
        report_type: Tipo do relatório ("single" ou "comparison")
    """
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    from scipy import stats
    
    clean_results = validate_and_clean_metrics(results_dict)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("RELATÓRIO DETALHADO DE ANÁLISE DE MODELOS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Gerado em: {datetime.now().strftime('%d/%m/%Y às %H:%M:%S')}\n")
        f.write(f"Tipo de análise: {'Comparação de múltiplos modelos' if report_type == 'comparison' else 'Análise de modelo único'}\n")
        f.write(f"Número de modelos analisados: {len(clean_results)}\n\n")
        
        # ========== RESUMO EXECUTIVO ==========
        f.write("RESUMO EXECUTIVO\n")
        f.write("-" * 50 + "\n\n")
        
        if not clean_results:
            f.write("❌ ERRO: Dados insuficientes para análise.\n")
            return
        
        # Calcular métricas para todos os modelos
        model_metrics = {}
        for model_name, results in clean_results.items():
            if 'actuals' in results and 'predictions' in results:
                actuals = np.array(results['actuals']).flatten()
                predictions = np.array(results['predictions']).flatten()
                
                # Métricas de performance
                r2 = r2_score(actuals, predictions)
                rmse = np.sqrt(mean_squared_error(actuals, predictions))
                mae = mean_absolute_error(actuals, predictions)
                mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100 if np.all(actuals != 0) else np.nan
                
                # Análise de resíduos
                residuals = predictions - actuals
                residuals_mean = np.mean(residuals)
                residuals_std = np.std(residuals)
                residuals_skew = stats.skew(residuals)
                residuals_kurt = stats.kurtosis(residuals)
                
                # Teste de normalidade
                try:
                    if len(residuals) <= 5000:
                        shapiro_stat, shapiro_p = stats.shapiro(residuals)
                    else:
                        shapiro_stat, shapiro_p = np.nan, np.nan
                except:
                    shapiro_stat, shapiro_p = np.nan, np.nan
                
                model_metrics[model_name] = {
                    'r2': r2, 'rmse': rmse, 'mae': mae, 'mape': mape,
                    'residuals_mean': residuals_mean, 'residuals_std': residuals_std,
                    'residuals_skew': residuals_skew, 'residuals_kurt': residuals_kurt,
                    'shapiro_stat': shapiro_stat, 'shapiro_p': shapiro_p,
                    'n_samples': len(actuals)
                }
        
        if report_type == "comparison" and len(model_metrics) > 1:
            # Ranking geral
            rankings = []
            for name, metrics in model_metrics.items():
                score = 0
                # R² (peso 3)
                score += metrics['r2'] * 3
                # RMSE invertido (peso 2)
                max_rmse = max([m['rmse'] for m in model_metrics.values()])
                score += (1 - metrics['rmse'] / max_rmse) * 2 if max_rmse > 0 else 0
                # MAE invertido (peso 2)
                max_mae = max([m['mae'] for m in model_metrics.values()])
                score += (1 - metrics['mae'] / max_mae) * 2 if max_mae > 0 else 0
                
                rankings.append((name, score, metrics))
            
            rankings.sort(key=lambda x: x[1], reverse=True)
            
            f.write("🏆 RANKING GERAL DOS MODELOS:\n\n")
            for rank, (name, score, metrics) in enumerate(rankings, 1):
                medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank}º"
                f.write(f"{medal} {name}\n")
                f.write(f"    Score total: {score:.4f}\n")
                f.write(f"    R²: {metrics['r2']:.6f}\n")
                f.write(f"    RMSE: {metrics['rmse']:.6f}\n")
                f.write(f"    MAE: {metrics['mae']:.6f}\n")
                if not np.isnan(metrics['mape']):
                    f.write(f"    MAPE: {metrics['mape']:.2f}%\n")
                f.write("\n")
            
            # Análise do melhor modelo
            best_name, best_score, best_metrics = rankings[0]
            f.write(f"🎯 ANÁLISE DO MELHOR MODELO ({best_name}):\n")
            if best_metrics['r2'] > 0.95:
                f.write("   ✅ Performance EXCEPCIONAL (R² > 0.95)\n")
            elif best_metrics['r2'] > 0.9:
                f.write("   ✅ Performance EXCELENTE (R² > 0.9)\n")
            elif best_metrics['r2'] > 0.8:
                f.write("   ✅ Performance BOA (R² > 0.8)\n")
            elif best_metrics['r2'] > 0.6:
                f.write("   ⚠️ Performance MODERADA (R² > 0.6)\n")
            else:
                f.write("   ❌ Performance BAIXA (R² ≤ 0.6)\n")
            
            # Análise de resíduos do melhor modelo
            if abs(best_metrics['residuals_mean']) < best_metrics['residuals_std'] * 0.1:
                f.write("   ✅ Resíduos bem centrados (sem viés)\n")
            else:
                f.write("   ⚠️ Resíduos com possível viés\n")
            
            if abs(best_metrics['residuals_skew']) < 0.5:
                f.write("   ✅ Resíduos aproximadamente simétricos\n")
            else:
                f.write("   ⚠️ Resíduos com assimetria significativa\n")
        
        f.write("\n")
        
        # ========== ANÁLISE DETALHADA POR MODELO ==========
        f.write("ANÁLISE DETALHADA POR MODELO\n")
        f.write("-" * 50 + "\n\n")
        
        for model_name, metrics in model_metrics.items():
            f.write(f"📊 MODELO: {model_name}\n")
            f.write("=" * 40 + "\n")
            
            # Informações básicas
            f.write(f"Tamanho da amostra: {metrics['n_samples']:,} pontos\n\n")
            
            # Métricas de performance
            f.write("🎯 MÉTRICAS DE PERFORMANCE:\n")
            f.write(f"   R² (Coeficiente de Determinação): {metrics['r2']:.6f}\n")
            f.write(f"   RMSE (Raiz do Erro Quadrático Médio): {metrics['rmse']:.6f}\n")
            f.write(f"   MAE (Erro Absoluto Médio): {metrics['mae']:.6f}\n")
            if not np.isnan(metrics['mape']):
                f.write(f"   MAPE (Erro Percentual Absoluto Médio): {metrics['mape']:.2f}%\n")
            f.write("\n")
            
            # Interpretação das métricas
            f.write("📈 INTERPRETAÇÃO:\n")
            if metrics['r2'] > 0.9:
                f.write("   • Excelente capacidade preditiva\n")
            elif metrics['r2'] > 0.8:
                f.write("   • Boa capacidade preditiva\n")
            elif metrics['r2'] > 0.6:
                f.write("   • Capacidade preditiva moderada\n")
            else:
                f.write("   • Capacidade preditiva baixa\n")
            
            if not np.isnan(metrics['mape']):
                if metrics['mape'] < 5:
                    f.write("   • Erro percentual muito baixo (< 5%)\n")
                elif metrics['mape'] < 10:
                    f.write("   • Erro percentual baixo (< 10%)\n")
                elif metrics['mape'] < 20:
                    f.write("   • Erro percentual moderado (< 20%)\n")
                else:
                    f.write("   • Erro percentual alto (≥ 20%)\n")
            
            f.write("\n")
            
            # Análise de resíduos
            f.write("🔬 ANÁLISE DE RESÍDUOS:\n")
            f.write(f"   Média: {metrics['residuals_mean']:.6f}\n")
            f.write(f"   Desvio padrão: {metrics['residuals_std']:.6f}\n")
            f.write(f"   Assimetria: {metrics['residuals_skew']:.4f}\n")
            f.write(f"   Curtose: {metrics['residuals_kurt']:.4f}\n")
            
            if not np.isnan(metrics['shapiro_stat']):
                f.write(f"   Teste Shapiro-Wilk: {metrics['shapiro_stat']:.4f} (p-valor: {metrics['shapiro_p']:.4f})\n")
                if metrics['shapiro_p'] > 0.05:
                    f.write("   ✅ Resíduos seguem distribuição normal (p > 0.05)\n")
                else:
                    f.write("   ⚠️ Resíduos não seguem distribuição normal (p ≤ 0.05)\n")
            
            # Diagnóstico dos resíduos
            f.write("\n🩺 DIAGNÓSTICO DOS RESÍDUOS:\n")
            
            # Viés
            if abs(metrics['residuals_mean']) < metrics['residuals_std'] * 0.1:
                f.write("   ✅ Sem viés significativo\n")
            else:
                f.write("   ⚠️ Possível viés nos resíduos\n")
            
            # Simetria
            if abs(metrics['residuals_skew']) < 0.5:
                f.write("   ✅ Distribuição aproximadamente simétrica\n")
            elif abs(metrics['residuals_skew']) < 1:
                f.write("   ⚠️ Assimetria leve\n")
            else:
                f.write("   ❌ Assimetria significativa\n")
            
            # Curtose
            if abs(metrics['residuals_kurt']) < 1:
                f.write("   ✅ Curtose normal\n")
            elif abs(metrics['residuals_kurt']) < 2:
                f.write("   ⚠️ Curtose moderada\n")
            else:
                f.write("   ❌ Curtose excessiva\n")
            
            f.write("\n" + "─" * 40 + "\n\n")
        
        # ========== RECOMENDAÇÕES ==========
        f.write("RECOMENDAÇÕES E CONCLUSÕES\n")
        f.write("-" * 50 + "\n\n")
        
        if report_type == "comparison" and len(model_metrics) > 1:
            best_name = rankings[0][0]
            f.write(f"🎯 MODELO RECOMENDADO: {best_name}\n\n")
            
            f.write("💡 JUSTIFICATIVAS:\n")
            best_metrics = model_metrics[best_name]
            
            if best_metrics['r2'] > 0.8:
                f.write(f"   • Alto R² ({best_metrics['r2']:.4f}) indica boa capacidade preditiva\n")
            
            if best_metrics['rmse'] == min([m['rmse'] for m in model_metrics.values()]):
                f.write("   • Menor RMSE entre todos os modelos\n")
            
            if abs(best_metrics['residuals_mean']) < best_metrics['residuals_std'] * 0.1:
                f.write("   • Resíduos bem centrados (sem viés)\n")
            
            f.write("\n🚀 PRÓXIMOS PASSOS:\n")
            if best_metrics['r2'] < 0.9:
                f.write("   • Considerar ajustes nos hiperparâmetros\n")
                f.write("   • Avaliar engenharia de features\n")
            
            if abs(best_metrics['residuals_skew']) > 0.5:
                f.write("   • Investigar outliers nos dados\n")
                f.write("   • Considerar transformações nos dados\n")
            
            if not np.isnan(best_metrics['shapiro_p']) and best_metrics['shapiro_p'] <= 0.05:
                f.write("   • Verificar premissas do modelo\n")
                f.write("   • Considerar modelos não-paramétricos\n")
        else:
            # Análise de modelo único
            model_name = list(model_metrics.keys())[0]
            metrics = model_metrics[model_name]
            
            f.write(f"📊 AVALIAÇÃO DO MODELO: {model_name}\n\n")
            
            if metrics['r2'] > 0.9:
                f.write("✅ VEREDICTO: Modelo EXCELENTE para uso em produção\n")
            elif metrics['r2'] > 0.8:
                f.write("✅ VEREDICTO: Modelo BOM, adequado para uso\n")
            elif metrics['r2'] > 0.6:
                f.write("⚠️ VEREDICTO: Modelo MODERADO, necessita melhorias\n")
            else:
                f.write("❌ VEREDICTO: Modelo INADEQUADO, requer revisão completa\n")
            
            f.write("\n🎯 PONTOS FORTES:\n")
            if metrics['r2'] > 0.8:
                f.write("   • Boa capacidade de explicação da variabilidade\n")
            if abs(metrics['residuals_mean']) < metrics['residuals_std'] * 0.1:
                f.write("   • Resíduos sem viés significativo\n")
            if abs(metrics['residuals_skew']) < 0.5:
                f.write("   • Distribuição de resíduos simétrica\n")
            
            f.write("\n⚠️ PONTOS DE ATENÇÃO:\n")
            if metrics['r2'] < 0.8:
                f.write("   • R² relativamente baixo\n")
            if abs(metrics['residuals_mean']) >= metrics['residuals_std'] * 0.1:
                f.write("   • Possível viés nos resíduos\n")
            if abs(metrics['residuals_skew']) >= 0.5:
                f.write("   • Assimetria nos resíduos\n")
        
        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write("FIM DO RELATÓRIO\n")
        f.write("=" * 80 + "\n")
    
    print_save_message(save_path, "Relatório textual")


def generate_html_report(results_dict, generated_files, save_path, report_type="comparison"):
    """
    Gera relatório HTML interativo
    
    Args:
        results_dict: Dicionário com resultados dos modelos
        generated_files: Dicionário com caminhos dos arquivos gerados
        save_path: Caminho para salvar o relatório HTML
        report_type: Tipo do relatório ("single" ou "comparison")
    """
    clean_results = validate_and_clean_metrics(results_dict)
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Relatório de Análise de Modelos</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 0 20px rgba(0,0,0,0.1);
            }}
            h1 {{
                color: #2c3e50;
                text-align: center;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
            }}
            h2 {{
                color: #34495e;
                margin-top: 30px;
                border-left: 4px solid #3498db;
                padding-left: 15px;
            }}
            h3 {{
                color: #555;
                margin-top: 25px;
            }}
            .summary-box {{
                background-color: #ecf0f1;
                padding: 20px;
                border-radius: 8px;
                margin: 20px 0;
                border-left: 5px solid #3498db;
            }}
            .metric-card {{
                background-color: #fff;
                border: 1px solid #ddd;
                border-radius: 8px;
                padding: 15px;
                margin: 10px 0;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .metric-value {{
                font-size: 1.2em;
                font-weight: bold;
                color: #2980b9;
            }}
            .image-gallery {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }}
            .image-card {{
                text-align: center;
                background-color: #fff;
                border-radius: 8px;
                padding: 15px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }}
            .image-card img {{
                max-width: 100%;
                height: auto;
                border-radius: 5px;
                border: 1px solid #ddd;
            }}
            .image-card h4 {{
                margin: 10px 0 5px 0;
                color: #2c3e50;
            }}
            .timestamp {{
                text-align: center;
                color: #7f8c8d;
                font-style: italic;
                margin-bottom: 30px;
            }}
            .navigation {{
                background-color: #34495e;
                padding: 10px;
                border-radius: 5px;
                margin-bottom: 20px;
            }}
            .navigation a {{
                color: white;
                text-decoration: none;
                margin: 0 15px;
                padding: 5px 10px;
                border-radius: 3px;
                transition: background-color 0.3s;
            }}
            .navigation a:hover {{
                background-color: #2c3e50;
            }}
            .model-section {{
                border: 1px solid #ddd;
                border-radius: 8px;
                margin: 15px 0;
                padding: 20px;
                background-color: #fafafa;
            }}
            .status-good {{ color: #27ae60; }}
            .status-warning {{ color: #f39c12; }}
            .status-bad {{ color: #e74c3c; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📊 Relatório de Análise de Modelos</h1>
            <div class="timestamp">
                Gerado em: {datetime.now().strftime('%d/%m/%Y às %H:%M:%S')}
            </div>
            
            <div class="navigation">
                <a href="#resumo">📋 Resumo</a>
                <a href="#metricas">📈 Métricas</a>
                <a href="#graficos">📊 Gráficos</a>
                <a href="#analises">🔬 Análises</a>
                <a href="#arquivos">📁 Arquivos</a>
            </div>
    """
    
    # ========== RESUMO ==========
    html_content += f"""
            <div id="resumo">
                <h2>📋 Resumo Executivo</h2>
                <div class="summary-box">
                    <p><strong>Tipo de análise:</strong> {'Comparação de múltiplos modelos' if report_type == 'comparison' else 'Análise de modelo único'}</p>
                    <p><strong>Número de modelos:</strong> {len(clean_results)}</p>
                    <p><strong>Modelos analisados:</strong> {', '.join(clean_results.keys())}</p>
                </div>
            </div>
    """
    
    # ========== MÉTRICAS ==========
    html_content += """
            <div id="metricas">
                <h2>📈 Métricas de Performance</h2>
    """
    
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    
    for model_name, results in clean_results.items():
        if 'actuals' in results and 'predictions' in results:
            actuals = np.array(results['actuals']).flatten()
            predictions = np.array(results['predictions']).flatten()
            
            r2 = r2_score(actuals, predictions)
            rmse = np.sqrt(mean_squared_error(actuals, predictions))
            mae = mean_absolute_error(actuals, predictions)
            mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100 if np.all(actuals != 0) else np.nan
            
            # Determinar status
            if r2 > 0.9:
                status_class = "status-good"
                status_text = "Excelente"
            elif r2 > 0.8:
                status_class = "status-good"
                status_text = "Bom"
            elif r2 > 0.6:
                status_class = "status-warning"
                status_text = "Moderado"
            else:
                status_class = "status-bad"
                status_text = "Baixo"
            
            html_content += f"""
                <div class="model-section">
                    <h3>{model_name} <span class="{status_class}">({status_text})</span></h3>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;">
                        <div class="metric-card">
                            <h4>R² (Coeficiente de Determinação)</h4>
                            <div class="metric-value">{r2:.6f}</div>
                        </div>
                        <div class="metric-card">
                            <h4>RMSE</h4>
                            <div class="metric-value">{rmse:.6f}</div>
                        </div>
                        <div class="metric-card">
                            <h4>MAE</h4>
                            <div class="metric-value">{mae:.6f}</div>
                        </div>
            """
            
            if not np.isnan(mape):
                html_content += f"""
                        <div class="metric-card">
                            <h4>MAPE (%)</h4>
                            <div class="metric-value">{mape:.2f}%</div>
                        </div>
                """
            
            html_content += """
                    </div>
                </div>
            """
    
    html_content += "</div>"
    
    # ========== GRÁFICOS ==========
    html_content += """
            <div id="graficos">
                <h2>📊 Visualizações</h2>
                <div class="image-gallery">
    """
    
    for file_key, file_path in generated_files.items():
        if file_path.endswith('.png'):
            file_name = os.path.basename(file_path)
            # Criar título mais amigável
            if 'overview' in file_key:
                title = "Visão Geral Comparativa"
            elif 'training' in file_key:
                title = f"Histórico de Treinamento"
            elif 'predictions' in file_key:
                title = f"Predições vs Valores Reais"
            elif 'qq' in file_key:
                title = f"Análise Q-Q Plot"
            elif 'cdf' in file_key:
                title = f"Comparação de CDF"
            elif 'pdf' in file_key:
                title = f"Comparação de FDP"
            elif 'ks' in file_key:
                title = f"Teste Kolmogorov-Smirnov"
            elif 'radar' in file_key:
                title = "Gráfico Radar de Performance"
            elif 'metrics' in file_key:
                title = "Tabela de Métricas"
            else:
                title = file_name.replace('_', ' ').replace('.png', '').title()
            
            html_content += f"""
                <div class="image-card">
                    <h4>{title}</h4>
                    <img src="{file_name}" alt="{title}">
                </div>
            """
    
    html_content += """
                </div>
            </div>
    """
    
    # ========== ANÁLISES ==========
    html_content += """
            <div id="analises">
                <h2>🔬 Análises Estatísticas</h2>
                <div class="summary-box">
                    <p>As análises estatísticas detalhadas incluem:</p>
                    <ul>
                        <li><strong>Q-Q Plot:</strong> Comparação das distribuições dos resíduos com a distribuição normal</li>
                        <li><strong>CDF/FDP:</strong> Análise das funções de distribuição empíricas</li>
                        <li><strong>Teste KS:</strong> Teste de Kolmogorov-Smirnov para comparação de distribuições</li>
                        <li><strong>Autocorrelação:</strong> Análise da dependência temporal (quando aplicável)</li>
                    </ul>
                    <p>Para mais detalhes, consulte o <a href="relatorio_textual_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt">relatório textual</a>.</p>
                </div>
            </div>
    """
    
    # ========== ARQUIVOS ==========
    html_content += """
            <div id="arquivos">
                <h2>📁 Arquivos Gerados</h2>
                <div class="summary-box">
                    <h3>Lista de Arquivos:</h3>
                    <ul>
    """
    
    for file_key, file_path in generated_files.items():
        file_name = os.path.basename(file_path)
        file_type = "Imagem" if file_path.endswith('.png') else "Texto" if file_path.endswith('.txt') else "HTML"
        html_content += f"""
                        <li><strong>{file_type}:</strong> <a href="{file_name}">{file_name}</a></li>
        """
    
    html_content += """
                    </ul>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print_save_message(save_path, "Relatório HTML")


def generate_quick_summary(results_dict, model_name=None):
    """
    Gera resumo rápido para visualização no console
    
    Args:
        results_dict: Dicionário com resultados dos modelos
        model_name: Nome do modelo (para análise de modelo único)
    
    Returns:
        str: Resumo formatado
    """
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    
    if model_name and isinstance(results_dict, dict) and 'actuals' in results_dict:
        # Análise de um único modelo
        single_model_results = {model_name: results_dict}
    else:
        # Múltiplos modelos
        single_model_results = results_dict
    
    clean_results = validate_and_clean_metrics(single_model_results)
    
    if not clean_results:
        return "❌ Dados insuficientes para gerar resumo."
    
    summary = "\n" + "="*60 + "\n"
    summary += "📊 RESUMO RÁPIDO DE PERFORMANCE\n"
    summary += "="*60 + "\n"
    
    model_scores = []
    
    for model_name, results in clean_results.items():
        if 'actuals' in results and 'predictions' in results:
            actuals = np.array(results['actuals']).flatten()
            predictions = np.array(results['predictions']).flatten()
            
            r2 = r2_score(actuals, predictions)
            rmse = np.sqrt(mean_squared_error(actuals, predictions))
            mae = mean_absolute_error(actuals, predictions)
            
            # Score simples para ranking
            score = r2 * 100 - (rmse + mae) * 10
            model_scores.append((model_name, score, r2, rmse, mae))
            
            # Status baseado em R²
            if r2 > 0.9:
                status = "🟢 EXCELENTE"
            elif r2 > 0.8:
                status = "🟢 BOM"
            elif r2 > 0.6:
                status = "🟡 MODERADO"
            else:
                status = "🔴 BAIXO"
            
            summary += f"\n📈 {model_name}:\n"
            summary += f"   Status: {status}\n"
            summary += f"   R²: {r2:.6f} | RMSE: {rmse:.6f} | MAE: {mae:.6f}\n"
    
    # Ranking se houver múltiplos modelos
    if len(model_scores) > 1:
        model_scores.sort(key=lambda x: x[1], reverse=True)
        summary += "\n🏆 RANKING:\n"
        for rank, (name, score, r2, rmse, mae) in enumerate(model_scores, 1):
            medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank}º"
            summary += f"   {medal} {name} (Score: {score:.2f})\n"
        
        # Recomendação
        best_model = model_scores[0][0]
        summary += f"\n🎯 RECOMENDADO: {best_model}\n"
    
    summary += "\n" + "="*60 + "\n"
    
    return summary 
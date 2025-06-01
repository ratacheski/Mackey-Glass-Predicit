import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import random
from tqdm.auto import tqdm
import time

# Importar módulos do projeto
from models import MLPModel, LSTMModel, GRUModel
from data.mackey_glass_generator import MackeyGlassGenerator, create_dataloaders
from utils.training import train_model, validate_epoch, predict_sequence, calculate_metrics
from utils.visualization import create_comprehensive_report, plot_sequential_predictions
from config.config import (
    get_experiment_config, MAIN_MODELS, ALL_MODELS, DEVICE, RANDOM_SEED
)


def set_random_seeds(seed):
    """
    Define seeds para reprodutibilidade
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_model(model_config):
    """
    Cria modelo baseado na configuração
    """
    model_type = model_config['model_type']
    
    if model_type == 'mlp':
        return MLPModel(
            input_size=model_config['input_size'],
            hidden_sizes=model_config['hidden_sizes'],
            output_size=model_config['output_size'],
            dropout_rate=model_config['dropout_rate'],
            activation=model_config['activation']
        )
    elif model_type == 'lstm':
        return LSTMModel(
            input_size=model_config['input_size'],
            hidden_size=model_config['hidden_size'],
            num_layers=model_config['num_layers'],
            output_size=model_config['output_size'],
            dropout_rate=model_config['dropout_rate'],
            bidirectional=model_config['bidirectional'],
            use_attention=model_config['use_attention']
        )
    elif model_type == 'gru':
        return GRUModel(
            input_size=model_config['input_size'],
            hidden_size=model_config['hidden_size'],
            num_layers=model_config['num_layers'],
            output_size=model_config['output_size'],
            dropout_rate=model_config['dropout_rate'],
            bidirectional=model_config['bidirectional'],
            use_attention=model_config['use_attention']
        )
    else:
        raise ValueError(f"Tipo de modelo não suportado: {model_type}")


def run_single_experiment(model_name, verbose=True):
    """
    Executa um único experimento
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"INICIANDO EXPERIMENTO: {model_name.upper()}")
        print(f"{'='*60}")
    
    # Obter configuração
    config = get_experiment_config(model_name)
    
    # Definir seeds
    set_random_seeds(config['random_seed'])
    
    # Gerar série de Mackey-Glass
    if verbose:
        print("Gerando série temporal de Mackey-Glass...")
    
    generator = MackeyGlassGenerator(**config['mackey_glass'])
    series = generator.generate_series()
    
    if verbose:
        print(f"Série gerada com {len(series)} pontos")
    
    # Criar dataloaders
    if verbose:
        print("Criando dataloaders...")
    
    train_loader, val_loader, dataset = create_dataloaders(
        series=series,
        window_size=config['dataset']['window_size'],
        prediction_steps=config['dataset']['prediction_steps'],
        train_ratio=config['dataset']['train_ratio'],
        batch_size=config['dataset']['batch_size'],
        shuffle=config['dataset']['shuffle_train']
    )
    
    # Criar modelo
    if verbose:
        print("Criando modelo...")
    
    model = create_model(config['model'])
    model = model.to(config['device'])
    
    if verbose:
        model.print_model_summary()
    
    # Treinar modelo
    if verbose:
        print("\nIniciando treinamento...")
    
    start_time = time.time()
    training_results = train_model(model, train_loader, val_loader, config['training'], config['device'])
    training_time = time.time() - start_time
    
    if verbose:
        print(f"\nTreinamento concluído em {training_time:.2f} segundos")
        print(f"Melhor loss de validação: {training_results['best_val_loss']:.6f}")
    
    # Avaliar modelo
    if verbose:
        print("Avaliando modelo...")
    
    model.eval()
    criterion = torch.nn.MSELoss()
    val_loss, predictions, actuals = validate_epoch(model, val_loader, criterion, config['device'])
    
    # Desnormalizar predições e valores reais
    predictions_denorm = dataset.denormalize(predictions.flatten())
    actuals_denorm = dataset.denormalize(actuals.flatten())
    
    # Calcular métricas
    metrics = calculate_metrics(predictions_denorm, actuals_denorm)
    
    if verbose:
        print(f"\nMÉTRICAS FINAIS:")
        print(f"MSE: {metrics['MSE']:.6f}")
        print(f"RMSE: {metrics['RMSE']:.6f}")
        print(f"MAE: {metrics['MAE']:.6f}")
        print(f"MAPE: {metrics['MAPE']:.2f}%")
        print(f"R²: {metrics['R²']:.6f}")
    
    # Predições sequenciais (10% da série)
    n_sequential_predictions = int(len(series) * 0.1)
    train_size = int(len(series) * config['dataset']['train_ratio'])
    
    # Usar últimos pontos de treino como sequência inicial
    initial_sequence = torch.FloatTensor(dataset.normalized_series[train_size-config['dataset']['window_size']:train_size])
    
    if verbose:
        print(f"\nRealizar {n_sequential_predictions} predições sequenciais...")
    
    sequential_predictions = predict_sequence(
        model, initial_sequence, n_sequential_predictions, dataset, config['device']
    )
    
    # Valores reais para comparação
    actual_sequential = series[train_size:train_size + n_sequential_predictions]
    
    # Métricas das predições sequenciais
    sequential_metrics = calculate_metrics(sequential_predictions, actual_sequential)
    
    if verbose:
        print(f"\nMÉTRICAS DAS PREDIÇÕES SEQUENCIAIS:")
        print(f"MSE: {sequential_metrics['MSE']:.6f}")
        print(f"RMSE: {sequential_metrics['RMSE']:.6f}")
        print(f"MAE: {sequential_metrics['MAE']:.6f}")
        print(f"MAPE: {sequential_metrics['MAPE']:.2f}%")
        print(f"R²: {sequential_metrics['R²']:.6f}")
    
    # Retornar resultados
    results = {
        'model_name': model_name,
        'model_info': model.get_model_info(),
        'training_time': training_time,
        'train_losses': training_results['train_losses'],
        'val_losses': training_results['val_losses'],
        'best_val_loss': training_results['best_val_loss'],
        'epochs_trained': training_results['epochs_trained'],
        'predictions': predictions_denorm,
        'actuals': actuals_denorm,
        'metrics': metrics,
        'sequential_predictions': sequential_predictions,
        'actual_sequential': actual_sequential,
        'sequential_metrics': sequential_metrics,
        'series': series,
        'train_size': train_size,
        'dataset': dataset
    }
    
    return results


def run_all_experiments(models_to_run=None, save_results=True, output_dir_prefix=None):
    """
    Executa experimentos para todos os modelos especificados
    """
    if models_to_run is None:
        models_to_run = MAIN_MODELS
    
    print(f"EXECUTANDO EXPERIMENTOS PARA {len(models_to_run)} MODELOS")
    print(f"Dispositivo: {DEVICE}")
    print(f"Modelos: {', '.join(models_to_run)}")
    print(f"{'='*80}")
    
    all_results = {}
    
    for i, model_name in enumerate(models_to_run):
        print(f"\n[{i+1}/{len(models_to_run)}] Executando {model_name}...")
        
        try:
            results = run_single_experiment(model_name, verbose=True)
            all_results[model_name] = results
            
            print(f"✓ {model_name} concluído com sucesso!")
            
        except Exception as e:
            print(f"✗ Erro ao executar {model_name}: {str(e)}")
            continue
    
    # Salvar resultados e gerar relatório
    if save_results and all_results:
        print(f"\n{'='*60}")
        print("GERANDO RELATÓRIO FINAL")
        print(f"{'='*60}")
        
        # Criar diretório de resultados
        if output_dir_prefix:
            # Se foi especificado um prefixo personalizado, usar ele
            timestamp = int(time.time())
            output_dir = f"./{output_dir_prefix}_{timestamp}"
        else:
            # Usar o padrão original
            output_dir = f"./results/final_report_{int(time.time())}"
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Gerar relatório abrangente
        create_comprehensive_report(all_results, output_dir)
        
        # Plotar predições sequenciais para cada modelo
        for model_name, results in all_results.items():
            plot_sequential_predictions(
                results['series'],
                results['train_size'],
                results['sequential_predictions'],
                results['actual_sequential'],
                save_path=f"{output_dir}/{model_name}/sequential_predictions.png",
                title=f"Predições Sequenciais - {model_name}"
            )
        
        print(f"\nRelatório completo gerado em: {output_dir}")
    
    return all_results


def compare_model_types():
    """
    Compara os três tipos principais de modelos
    """
    print("COMPARAÇÃO DOS TRÊS TIPOS PRINCIPAIS DE MODELOS")
    print("=" * 60)
    
    results = run_all_experiments(MAIN_MODELS, save_results=True)
    
    print(f"\n{'='*60}")
    print("RESUMO DA COMPARAÇÃO")
    print(f"{'='*60}")
    
    for model_name, result in results.items():
        metrics = result['metrics']
        seq_metrics = result['sequential_metrics']
        
        print(f"\n{model_name.upper()}:")
        print(f"  Tempo de treinamento: {result['training_time']:.2f}s")
        print(f"  Épocas treinadas: {result['epochs_trained']}")
        print(f"  Parâmetros: {result['model_info']['total_parameters']:,}")
        print(f"  RMSE (validação): {metrics['RMSE']:.6f}")
        print(f"  RMSE (sequencial): {seq_metrics['RMSE']:.6f}")
        print(f"  R² (validação): {metrics['R²']:.6f}")
        print(f"  R² (sequencial): {seq_metrics['R²']:.6f}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Executar experimentos de predição de séries temporais')
    parser.add_argument('--models', nargs='+', choices=ALL_MODELS + ['main', 'all'], 
                       default='main', help='Modelos para executar')
    parser.add_argument('--no-save', action='store_true', help='Não salvar resultados')
    parser.add_argument('--output-dir', type=str, default=None, 
                       help='Prefixo personalizado para a pasta de saída (ex: meu_experimento)')
    
    args = parser.parse_args()
    
    if args.models == 'main' or args.models == ['main']:
        models_to_run = MAIN_MODELS
    elif args.models == 'all' or args.models == ['all']:
        models_to_run = ALL_MODELS
    else:
        models_to_run = args.models
    
    # Executar experimentos
    results = run_all_experiments(models_to_run, save_results=not args.no_save, 
                                output_dir_prefix=args.output_dir)
    
    print(f"\nExperimentos concluídos!")
    print(f"Total de modelos executados: {len(results)}")
    
    # Mostrar informação sobre a pasta de saída se foi especificada
    if args.output_dir and not args.no_save:
        print(f"Resultados salvos com prefixo: {args.output_dir}") 
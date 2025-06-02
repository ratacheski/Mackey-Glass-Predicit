import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import random
from tqdm.auto import tqdm
import time

# Import project modules
from models import MLPModel, LSTMModel, GRUModel
from data.mackey_glass_generator import MackeyGlassGenerator, create_dataloaders
from utils.training import train_model, validate_epoch, calculate_metrics
from utils import visualization as viz
from config.config import (
    get_experiment_config, MAIN_MODELS, ALL_MODELS, DEVICE, RANDOM_SEED
)


def set_random_seeds(seed):
    """
    Set seeds for reproducibility
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
    Create model based on configuration
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
        raise ValueError(f"Unsupported model type: {model_type}")


def run_single_experiment(model_name, verbose=True):
    """
    Run a single experiment
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"STARTING EXPERIMENT: {model_name.upper()}")
        print(f"{'='*60}")
    
    # Get configuration
    config = get_experiment_config(model_name)
    
    # Set seeds
    set_random_seeds(config['random_seed'])
    
    # Generate Mackey-Glass series
    if verbose:
        print("Generating Mackey-Glass time series...")
    
    generator = MackeyGlassGenerator(**config['mackey_glass'])
    series = generator.generate_series()
    
    if verbose:
        print(f"Series generated with {len(series)} points")
    
    # Create dataloaders
    if verbose:
        print("Creating dataloaders...")
    
    train_loader, val_loader, dataset = create_dataloaders(
        series=series,
        window_size=config['dataset']['window_size'],
        prediction_steps=config['dataset']['prediction_steps'],
        train_ratio=config['dataset']['train_ratio'],
        batch_size=config['dataset']['batch_size'],
        shuffle=config['dataset']['shuffle_train']
    )
    
    # Create model
    if verbose:
        print("Creating model...")
    
    model = create_model(config['model'])
    model = model.to(config['device'])
    
    if verbose:
        model.print_model_summary()
    
    # Train model
    if verbose:
        print("\nStarting training...")
    
    start_time = time.time()
    training_results = train_model(model, train_loader, val_loader, config['training'], config['device'])
    training_time = time.time() - start_time
    
    if verbose:
        print(f"\nTraining completed in {training_time:.2f} seconds")
        print(f"Best validation loss: {training_results['best_val_loss']:.6f}")
    
    # Evaluate model
    if verbose:
        print("Evaluating model...")
    
    model.eval()
    criterion = torch.nn.MSELoss()
    val_loss, predictions, actuals = validate_epoch(model, val_loader, criterion, config['device'])
    
    # Denormalize predictions and actual values
    predictions_denorm = dataset.denormalize(predictions.flatten())
    actuals_denorm = dataset.denormalize(actuals.flatten())
    
    # Calculate metrics
    metrics = calculate_metrics(predictions_denorm, actuals_denorm)
    
    if verbose:
        print(f"\nVALIDATION METRICS:")
        print(f"MSE: {metrics['MSE']:.6f}")
        print(f"RMSE: {metrics['RMSE']:.6f}")
        print(f"MAE: {metrics['MAE']:.6f}")
        print(f"MAPE: {metrics['MAPE']:.2f}%")
        print(f"R²: {metrics['R²']:.6f}")
    
    # Return results
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
        'series': series,
        'dataset': dataset
    }
    
    return results


def run_all_experiments(models_to_run=None, save_results=True, output_dir_prefix=None):
    """
    Run experiments for all specified models
    """
    if models_to_run is None:
        models_to_run = MAIN_MODELS
    
    print(f"RUNNING EXPERIMENTS FOR {len(models_to_run)} MODELS")
    print(f"Device: {DEVICE}")
    print(f"Models: {', '.join(models_to_run)}")
    print(f"{'='*80}")
    
    all_results = {}
    
    for i, model_name in enumerate(models_to_run):
        print(f"\n[{i+1}/{len(models_to_run)}] Running {model_name}...")
        
        try:
            results = run_single_experiment(model_name, verbose=True)
            all_results[model_name] = results
            
            print(f"✓ {model_name} completed successfully!")
            
        except Exception as e:
            print(f"✗ Error running {model_name}: {str(e)}")
            continue
    
    # Save results and generate report
    if save_results and all_results:
        print(f"\n{'='*60}")
        print("GENERATING FINAL REPORT")
        print(f"{'='*60}")
        
        # Create results directory
        if output_dir_prefix:
            # If a custom prefix was specified, use it
            timestamp = int(time.time())
            output_dir = f"./{output_dir_prefix}_{timestamp}"
        else:
            # Use the default original
            output_dir = f"./results/final_report_{int(time.time())}"
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate comprehensive report with the new system
        viz.generate_comprehensive_report(all_results, output_dir)
        
        print(f"\nComplete report generated in: {output_dir}")
    
    return all_results


def compare_model_types():
    """
    Compare the three main types of models
    """
    print("COMPARISON OF THE THREE MAIN TYPES OF MODELS")
    print("=" * 60)
    
    results = run_all_experiments(MAIN_MODELS, save_results=True)
    
    print(f"\n{'='*60}")
    print("SUMMARY OF COMPARISON")
    print(f"{'='*60}")
    
    for model_name, result in results.items():
        metrics = result['metrics']
        
        print(f"\n{model_name.upper()}:")
        print(f"  Training time: {result['training_time']:.2f}s")
        print(f"  Trained epochs: {result['epochs_trained']}")
        print(f"  Parameters: {result['model_info']['total_parameters']:,}")
        print(f"  RMSE (validation): {metrics['RMSE']:.6f}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run time series prediction experiments')
    parser.add_argument('--models', nargs='+', choices=ALL_MODELS + ['main', 'all'], 
                       default='main', help='Models to run')
    parser.add_argument('--no-save', action='store_true', help='Do not save results')
    parser.add_argument('--output-dir', type=str, default=None, 
                       help='Custom prefix for output folder (ex: my_experiment)')
    
    args = parser.parse_args()
    
    if args.models == 'main' or args.models == ['main']:
        models_to_run = MAIN_MODELS
    elif args.models == 'all' or args.models == ['all']:
        models_to_run = ALL_MODELS
    else:
        models_to_run = args.models
    
    # Run experiments
    results = run_all_experiments(models_to_run, save_results=not args.no_save, 
                                output_dir_prefix=args.output_dir)
    
    print(f"\nExperiments completed!")
    print(f"Total models executed: {len(results)}")
    
    # Show information about output folder if specified
    if args.output_dir and not args.no_save:
        print(f"Results saved with prefix: {args.output_dir}") 
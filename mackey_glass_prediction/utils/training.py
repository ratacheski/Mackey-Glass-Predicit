import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import os
from scipy import stats
from scipy.stats import gaussian_kde
from scipy.integrate import quad
from sklearn.metrics import mean_pinball_loss, d2_pinball_score, r2_score, mean_squared_error, mean_absolute_error
import warnings

def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Trains the model for one epoch
    """
    model.train()
    train_loss = 0.0
    num_batches = 0
    
    with tqdm(total=len(dataloader), desc="Training", leave=False) as pbar:
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            output = model(data)
            
            # Calculate loss
            loss = criterion(output, target)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({'Loss': f'{loss.item():.6f}'})
            pbar.update(1)
    
    return train_loss / num_batches


def validate_epoch(model, dataloader, criterion, device):
    """
    Validates the model for one epoch
    """
    model.eval()
    val_loss = 0.0
    num_batches = 0
    predictions = []
    actuals = []
    
    with torch.no_grad():
        with tqdm(total=len(dataloader), desc="validation", leave=False) as pbar:
            for batch_idx, (data, target) in enumerate(dataloader):
                data, target = data.to(device), target.to(device)
                
                # Forward pass
                output = model(data)
                
                # Calculate loss
                loss = criterion(output, target)
                
                val_loss += loss.item()
                num_batches += 1
                
                # Store predictions and actuals
                predictions.extend(output.squeeze().cpu().numpy())
                actuals.extend(target.cpu().numpy())
                
                pbar.set_postfix({'Loss': f'{loss.item():.6f}'})
                pbar.update(1)
    
    return val_loss / num_batches, np.array(predictions), np.array(actuals)


def train_model(model, train_loader, val_loader, config, device):
    """
    Train the complete model
    """
    # Training configurations
    epochs = config.get('epochs', 100)
    learning_rate = config.get('learning_rate', 0.001)
    weight_decay = config.get('weight_decay', 1e-5)
    patience = config.get('patience', 10)
    min_delta = config.get('min_delta', 1e-6)
    
    # Optimizer and criterion
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    
    # Scheduler (optional)
    if config.get('use_scheduler', True):
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=patience//2
        )
    
    # Lists to track progress
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    
    print(f"Starting training for {epochs} epochs...")
    print(f"Device: {device}")
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 50)
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        
        # Validate
        val_loss, _, _ = validate_epoch(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        
        # Update scheduler
        if config.get('use_scheduler', True):
            scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            
            # Save best model
            if config.get('save_best_model', True):
                save_model(model, optimizer, epoch, train_losses, val_losses, 
                          config.get('model_save_path', 'best_model.pth'), is_best=True)
        else:
            epochs_without_improvement += 1
        
        print(f"Training Loss: {train_loss:.6f}")
        print(f"Validation Loss: {val_loss:.6f}")
        print(f"Best Validation Loss: {best_val_loss:.6f}")
        print(f"Current Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping after {epoch + 1} epochs!")
            print(f"No improvement for {patience} consecutive epochs.")
            break
    
    # Save final model
    if config.get('save_final_model', True):
        save_model(model, optimizer, epoch, train_losses, val_losses,
                  config.get('final_model_save_path', 'final_model.pth'), is_best=False)
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'epochs_trained': epoch + 1
    }


def save_model(model, optimizer, epoch, train_losses, val_losses, path, is_best=False):
    """
    Save model and training information
    """
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'model_info': model.get_model_info() if hasattr(model, 'get_model_info') else None
    }
    
    torch.save(checkpoint, path)
    
    if is_best:
        print(f"✓ Best model saved at: {path}")
    else:
        print(f"✓ Final model saved at: {path}")


def load_model(model, optimizer, path, device):
    """
    Load saved model and optimizer
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return {
        'epoch': checkpoint['epoch'],
        'train_losses': checkpoint['train_losses'],
        'val_losses': checkpoint['val_losses']
    }


def calculate_metrics(predictions, actuals):
    """
    Calculate evaluation metrics
    """
    # Ensure they are numpy arrays
    actuals = np.array(actuals).flatten()
    predictions = np.array(predictions).flatten()
    
    # MSE (Mean Squared Error)
    mse = np.mean((predictions - actuals) ** 2)

    var_actuals = np.var(actuals)
    eqmn1 = mse / var_actuals if var_actuals != 0 else float('inf')

    if len(actuals) > 1:
        x_pa = np.roll(actuals, 1)
        x_pa[0] = actuals[0]

        naive_mse = np.mean((x_pa - actuals) ** 2)
        eqmn2 = mse / naive_mse if naive_mse != 0 else float('inf')
    else:
        eqmn2 = float('inf')
    
    # RMSE (Root Mean Squared Error)
    rmse = np.sqrt(mse)
    
    # MAE (Mean Absolute Error)
    mae = np.mean(np.abs(predictions - actuals))
    
    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
    
    # R² (Coefficient of Determination)
    ss_res = np.sum((actuals - predictions) ** 2)
    ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    # === PINBALL LOSS METRICS ===
    try:
        # Mean Pinball Loss (for alpha=0.5, which corresponds to the median)
        mean_pinball = mean_pinball_loss(actuals, predictions, alpha=0.5)
        
        # D2 Pinball Score (coefficient of determination based on pinball loss)
        d2_pinball = d2_pinball_score(actuals, predictions, alpha=0.5)
        
    except Exception as e:
        print(f"Warning: Error calculating pinball loss metrics: {e}")
        mean_pinball = float('inf')
        d2_pinball = -float('inf')
    
    # === NEW METRICS: CDF and PDF ===
    
    # 1. CDF (Cumulative Distribution Function) - using Kolmogorov-Smirnov test
    ks_statistic, ks_pvalue = stats.ks_2samp(predictions, actuals)
    
    # Calculate empirical CDF for common points
    combined_range = np.linspace(
        min(np.min(predictions), np.min(actuals)),
        max(np.max(predictions), np.max(actuals)),
        100
    )
    
    # Empirical CDFs
    cdf_predictions = np.array([np.mean(predictions <= x) for x in combined_range])
    cdf_actuals = np.array([np.mean(actuals <= x) for x in combined_range])
    
    # Average distance between CDFs
    fda_distance = np.mean(np.abs(cdf_predictions - cdf_actuals))
    
    # 2. PDF (Probability Density Function) - using KDE
    try:
        # Kernel Density Estimation for both distributions
        kde_predictions = gaussian_kde(predictions)
        kde_actuals = gaussian_kde(actuals)
        
        # Evaluate PDFs at combined range points
        pdf_predictions = kde_predictions(combined_range)
        pdf_actuals = kde_actuals(combined_range)
        
        # Distance between PDFs (using L2 distance)
        fdp_distance = np.sqrt(np.trapz((pdf_predictions - pdf_actuals)**2, combined_range))
        
        # Jensen-Shannon divergence between PDFs
        # Normalize PDFs to be valid probabilities
        pdf_pred_norm = pdf_predictions / np.trapz(pdf_predictions, combined_range)
        pdf_actual_norm = pdf_actuals / np.trapz(pdf_actuals, combined_range)
        
        # Average PDF for JS divergence
        pdf_mean = 0.5 * (pdf_pred_norm + pdf_actual_norm)
        
        # Calculate KL divergence with protection against log(0)
        epsilon = 1e-10
        kl_pred_mean = np.trapz(pdf_pred_norm * np.log((pdf_pred_norm + epsilon) / (pdf_mean + epsilon)), combined_range)
        kl_actual_mean = np.trapz(pdf_actual_norm * np.log((pdf_actual_norm + epsilon) / (pdf_mean + epsilon)), combined_range)
        
        kl_divergence = kl_pred_mean + kl_actual_mean
        
        js_divergence = 0.5 * (kl_pred_mean + kl_actual_mean)
        
    except Exception as e:
        print(f"Warning: Error calculating PDF metrics: {e}")
        fdp_distance = float('inf')
        kl_divergence = float('inf')
        js_divergence = float('inf')
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R²': r2,
        # PINBALL LOSS METRICS
        'MEAN_PINBALL_LOSS': mean_pinball,
        'D2_PINBALL_SCORE': d2_pinball,
        # CDF Metrics
        'FDA_KS_Statistic': ks_statistic,
        'FDA_KS_PValue': ks_pvalue,
        'FDA_Distance': fda_distance,
        # PDF Metrics
        'FDP_L2_Distance': fdp_distance,
        'KL_Divergence': kl_divergence,
        'FDP_JS_Divergence': js_divergence,
        # EQMN Metrics
        'EQMN1': eqmn1,
        'EQMN2': eqmn2
    } 
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
import numpy as np
import os


def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Treina o modelo por uma época
    """
    model.train()
    train_loss = 0.0
    num_batches = 0
    
    with tqdm(total=len(dataloader), desc="Treinamento", leave=False) as pbar:
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            
            # Zerar gradientes
            optimizer.zero_grad()
            
            # Forward pass
            output = model(data)
            
            # Calcular loss
            loss = criterion(output, target)
            
            # Backward pass
            loss.backward()
            
            # Atualizar pesos
            optimizer.step()
            
            # Acumular loss
            train_loss += loss.item()
            num_batches += 1
            
            # Atualizar barra de progresso
            pbar.set_postfix({'Loss': f'{loss.item():.6f}'})
            pbar.update(1)
    
    return train_loss / num_batches


def validate_epoch(model, dataloader, criterion, device):
    """
    Valida o modelo
    """
    model.eval()
    val_loss = 0.0
    num_batches = 0
    predictions = []
    actuals = []
    
    with torch.no_grad():
        with tqdm(total=len(dataloader), desc="Validação", leave=False) as pbar:
            for batch_idx, (data, target) in enumerate(dataloader):
                data, target = data.to(device), target.to(device)
                
                # Forward pass
                output = model(data)
                
                # Calcular loss
                loss = criterion(output, target)
                
                # Acumular loss
                val_loss += loss.item()
                num_batches += 1
                
                # Armazenar predições e valores reais
                predictions.extend(output.cpu().numpy())
                actuals.extend(target.cpu().numpy())
                
                # Atualizar barra de progresso
                pbar.set_postfix({'Loss': f'{loss.item():.6f}'})
                pbar.update(1)
    
    return val_loss / num_batches, np.array(predictions), np.array(actuals)


def train_model(model, train_loader, val_loader, config, device):
    """
    Treina o modelo completo
    """
    # Configurações de treinamento
    epochs = config.get('epochs', 100)
    learning_rate = config.get('learning_rate', 0.001)
    weight_decay = config.get('weight_decay', 1e-5)
    patience = config.get('patience', 10)
    min_delta = config.get('min_delta', 1e-6)
    
    # Otimizador e criterion
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    
    # Scheduler (opcional)
    if config.get('use_scheduler', True):
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=patience//2
        )
    
    # Listas para acompanhar o progresso
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    
    print(f"Iniciando treinamento por {epochs} épocas...")
    print(f"Dispositivo: {device}")
    
    for epoch in range(epochs):
        print(f"\nÉpoca {epoch + 1}/{epochs}")
        print("-" * 50)
        
        # Treinar
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        
        # Validar
        val_loss, _, _ = validate_epoch(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        
        # Atualizar scheduler
        if config.get('use_scheduler', True):
            scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            
            # Salvar melhor modelo
            if config.get('save_best_model', True):
                save_model(model, optimizer, epoch, train_losses, val_losses, 
                          config.get('model_save_path', 'best_model.pth'), is_best=True)
        else:
            epochs_without_improvement += 1
        
        print(f"Loss de Treinamento: {train_loss:.6f}")
        print(f"Loss de Validação: {val_loss:.6f}")
        print(f"Melhor Loss de Validação: {best_val_loss:.6f}")
        print(f"Learning Rate Atual: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping após {epoch + 1} épocas!")
            print(f"Nenhuma melhoria por {patience} épocas consecutivas.")
            break
    
    # Salvar modelo final
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
    Salva o modelo e informações de treinamento
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
        print(f"✓ Melhor modelo salvo em: {path}")
    else:
        print(f"✓ Modelo final salvo em: {path}")


def load_model(model, optimizer, path, device):
    """
    Carrega modelo e otimizador salvos
    """
    checkpoint = torch.load(path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return {
        'model': model,
        'optimizer': optimizer,
        'epoch': checkpoint['epoch'],
        'train_losses': checkpoint['train_losses'],
        'val_losses': checkpoint['val_losses'],
        'model_info': checkpoint.get('model_info', None)
    }


def predict_sequence(model, initial_sequence, n_predictions, dataset, device):
    """
    Faz predições sequenciais (uma amostra no futuro por vez)
    """
    model.eval()
    predictions = []
    current_sequence = initial_sequence.clone().to(device)
    
    with torch.no_grad():
        for _ in range(n_predictions):
            # Fazer predição para o próximo ponto
            pred = model(current_sequence.unsqueeze(0))
            predictions.append(pred.item())
            
            # Atualizar sequência (sliding window)
            current_sequence = torch.cat([current_sequence[1:], pred.squeeze().unsqueeze(0)])
    
    # Desnormalizar predições
    predictions = np.array(predictions)
    if hasattr(dataset, 'denormalize'):
        predictions = dataset.denormalize(predictions)
    
    return predictions


def calculate_metrics(predictions, actuals):
    """
    Calcula métricas de avaliação
    """
    # Garantir que são arrays numpy
    predictions = np.array(predictions).flatten()
    actuals = np.array(actuals).flatten()
    
    # MSE (Mean Squared Error)
    mse = np.mean((predictions - actuals) ** 2)
    
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
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R²': r2
    } 
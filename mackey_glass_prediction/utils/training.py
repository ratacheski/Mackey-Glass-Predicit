import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
import numpy as np
import os
from scipy import stats
from scipy.stats import gaussian_kde
from scipy.integrate import quad
from sklearn.metrics import mean_pinball_loss, d2_pinball_score

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

    var_actuals = np.var(actuals)
    eqmn1 = mse / var_actuals if var_actuals != 0 else float('inf')

    if len(actuals) > 1:
        x_pa = np.roll(actuals, 1)
        x_pa[0] = actuals[0]  # Evita vazamento de informação no primeiro elemento

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
    
    # === MÉTRICAS PINBALL LOSS ===
    try:
        # Mean Pinball Loss (para alpha=0.5, que corresponde à mediana)
        mean_pinball = mean_pinball_loss(actuals, predictions, alpha=0.5)
        
        # D2 Pinball Score (coeficiente de determinação baseado em pinball loss)
        d2_pinball = d2_pinball_score(actuals, predictions, alpha=0.5)
        
    except Exception as e:
        print(f"Aviso: Erro ao calcular métricas pinball loss: {e}")
        mean_pinball = float('inf')
        d2_pinball = -float('inf')
    
    # === NOVAS MÉTRICAS: FDA e FDP ===
    
    # 1. FDA (Função Distribuição Acumulada) - usando teste Kolmogorov-Smirnov
    ks_statistic, ks_pvalue = stats.ks_2samp(predictions, actuals)
    
    # Calcular FDA empírica para pontos comuns
    combined_range = np.linspace(
        min(np.min(predictions), np.min(actuals)),
        max(np.max(predictions), np.max(actuals)),
        100
    )
    
    # CDFs empíricas
    cdf_predictions = np.array([np.mean(predictions <= x) for x in combined_range])
    cdf_actuals = np.array([np.mean(actuals <= x) for x in combined_range])
    
    # Distância média entre as CDFs
    fda_distance = np.mean(np.abs(cdf_predictions - cdf_actuals))
    
    # 2. FDP (Função de Distribuição de Probabilidade) - usando KDE
    try:
        # Kernel Density Estimation para ambas as distribuições
        kde_predictions = gaussian_kde(predictions)
        kde_actuals = gaussian_kde(actuals)
        
        # Avaliar PDFs nos pontos do range combinado
        pdf_predictions = kde_predictions(combined_range)
        pdf_actuals = kde_actuals(combined_range)
        
        # Distância entre PDFs (usando distância L2)
        fdp_l2_distance = np.sqrt(np.trapz((pdf_predictions - pdf_actuals)**2, combined_range))
        
        # Divergência Jensen-Shannon entre as PDFs
        # Normalizar PDFs para que sejam probabilidades válidas
        pdf_pred_norm = pdf_predictions / np.trapz(pdf_predictions, combined_range)
        pdf_actual_norm = pdf_actuals / np.trapz(pdf_actuals, combined_range)
        
        # PDF média para JS divergence
        pdf_mean = 0.5 * (pdf_pred_norm + pdf_actual_norm)
        
        # Calcular divergência KL com proteção contra log(0)
        epsilon = 1e-10
        kl_pred_mean = np.trapz(pdf_pred_norm * np.log((pdf_pred_norm + epsilon) / (pdf_mean + epsilon)), combined_range)
        kl_actual_mean = np.trapz(pdf_actual_norm * np.log((pdf_actual_norm + epsilon) / (pdf_mean + epsilon)), combined_range)
        
        js_divergence = 0.5 * (kl_pred_mean + kl_actual_mean)
        
    except Exception as e:
        print(f"Aviso: Erro ao calcular métricas FDP: {e}")
        fdp_l2_distance = float('inf')
        js_divergence = float('inf')
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R²': r2,
        # Métricas Pinball Loss
        'MEAN_PINBALL_LOSS': mean_pinball,
        'D2_PINBALL_SCORE': d2_pinball,
        # Métricas FDA
        'FDA_KS_Statistic': ks_statistic,
        'FDA_KS_PValue': ks_pvalue,
        'FDA_Distance': fda_distance,
        # Métricas FDP
        'FDP_L2_Distance': fdp_l2_distance,
        'FDP_JS_Divergence': js_divergence,
        # Métricas EQMN
        'EQMN1': eqmn1,
        'EQMN2': eqmn2
    } 
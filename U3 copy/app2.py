import torch
from torch import nn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error 

# Configuración del dispositivo
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def prepare_data(name_file):
    global X_train, X_test, X_val, y_train, y_test, y_val, scaler_X, scaler_y
    
    data = pd.read_csv(name_file)
    print(data.info())

    # Seleccionar características y objetivo
    data = data[['Daily_Usage_Hours', 'Sleep_Hours', 'Academic_Performance']]
    
    # Normalizar características
    scaler_X = MinMaxScaler()
    X = scaler_X.fit_transform(data[['Daily_Usage_Hours', 'Sleep_Hours']].values)
    
    # Normalizar objetivo (entre 0 y 100)
    scaler_y = MinMaxScaler()
    y = scaler_y.fit_transform(data['Academic_Performance'].values.reshape(-1, 1))
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=0)

prepare_data('teen_phone_addiction_dataset.csv')

class RegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=10)
        self.layer_3 = nn.Linear(in_features=10, out_features=1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))
        
    def train_model(self, X_train, y_train, X_val, y_val, loss_fn, optimizer, epochs=3000):
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(0)
            
        X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
        y_train = torch.tensor(y_train, dtype=torch.float32, device=device)
        X_val = torch.tensor(X_val, dtype=torch.float32, device=device)
        y_val = torch.tensor(y_val, dtype=torch.float32, device=device)
        
        for epoch in range(epochs):
            # Training
            self.train()
            
            # Forward pass
            y_pred = self(X_train)
            
            # Calculate loss
            loss = loss_fn(y_pred, y_train)
            
            # Optimizer zero grad
            optimizer.zero_grad()
            
            # Loss backward
            loss.backward()
            
            # Optimizer step
            optimizer.step()
            
            # Validation
            self.eval()
            with torch.inference_mode():
                val_pred = self(X_val)
                val_loss = loss_fn(val_pred, y_val)
                
                if epoch % 100 == 0:
                    # Convertir a numpy para métricas
                    y_pred_train = y_pred.detach().cpu().numpy()
                    y_true_train = y_train.cpu().numpy()
                    y_pred_val = val_pred.detach().cpu().numpy()
                    y_true_val = y_val.cpu().numpy()
                    
                    # Calcular métricas
                    train_rmse = np.sqrt(mean_squared_error(y_true_train, y_pred_train))
                    val_rmse = np.sqrt(mean_squared_error(y_true_val, y_pred_val))
                    r2_train = r2_score(y_true_train, y_pred_train)
                    r2_val = r2_score(y_true_val, y_pred_val)
                    
                    print(f"Epoch: {epoch} | Train Loss: {loss.item():.5f} | Train RMSE: {train_rmse:.5f} | Train R²: {r2_train:.5f} | Val Loss: {val_loss.item():.5f} | Val RMSE: {val_rmse:.5f} | Val R²: {r2_val:.5f}")

# Crear y entrenar modelo
model = RegressionModel().to(device)
loss_fn = nn.MSELoss()  # Cambiamos a pérdida MSE para regresión
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)  # Adam suele funcionar mejor para regresión

model.train_model(X_train, y_train, X_val, y_val, loss_fn, optimizer, epochs=2000)

# Evaluación final

# Evaluación final con gráficos mejorados
model.eval()
with torch.inference_mode():
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)
    test_pred = model(X_test_tensor).cpu().numpy()
    
    # Revertir normalización
    test_pred_original = scaler_y.inverse_transform(test_pred)
    y_test_original = scaler_y.inverse_transform(y_test)
    
    # Métricas finales
    final_rmse = np.sqrt(mean_squared_error(y_test_original, test_pred_original))
    final_r2 = r2_score(y_test_original, test_pred_original)
    final_mae = mean_absolute_error(y_test_original, test_pred_original)
    
    print(f"\nMétricas Finales:")
    print(f"- RMSE: {final_rmse:.2f}")
    print(f"- MAE: {final_mae:.2f}")
    print(f"- R²: {final_r2:.2f}")

    # Configuración de estilo
    plt.style.use('seaborn-v0_8')
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.grid'] = True
    
    # Figura 1: True vs Predicted con histograma
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [3, 1]})
    
    # Scatter plot con línea de perfecta predicción
    scatter = ax1.scatter(y_test_original, test_pred_original, 
                         c=y_test_original, cmap='viridis', alpha=0.7, edgecolors='w', s=80)
    ax1.plot([0, 100], [0, 100], 'r--', lw=2, label='Predicción perfecta')
    
    # Ajustar límites y etiquetas
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 100)
    ax1.set_xlabel('Calificación Real', fontsize=12)
    ax1.set_ylabel('Calificación Predicha', fontsize=12)
    ax1.set_title('Relación entre Valores Reales y Predichos', fontsize=14, pad=20)
    
    # Añadir barra de color
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Valor Real', rotation=270, labelpad=15)
    
    # Histograma de errores
    errors = y_test_original.flatten() - test_pred_original.flatten()
    ax2.hist(errors, bins=20, orientation='horizontal', color='skyblue', edgecolor='navy')
    ax2.axhline(y=0, color='red', linestyle='--')
    ax2.set_xlabel('Frecuencia')
    ax2.set_title('Distribución de Errores', fontsize=14, pad=20)
    ax2.grid(False)
    
    plt.tight_layout()
    
    # Figura 2: Comparación por muestras
    plt.figure(figsize=(14, 6))
    sample_indices = np.arange(len(y_test_original))
    plt.scatter(sample_indices, y_test_original, color='navy', label='Real', alpha=0.7, s=100)
    plt.scatter(sample_indices, test_pred_original, color='coral', label='Predicho', alpha=0.7, s=80)
    
    # Conectar puntos reales y predichos
    for i in range(len(y_test_original)):
        plt.plot([i, i], [y_test_original[i], test_pred_original[i]], 'gray', alpha=0.3)
    
    plt.xlabel('Muestra', fontsize=12)
    plt.ylabel('Calificación', fontsize=12)
    plt.title('Comparación Muestra a Muestra: Real vs Predicho', fontsize=14, pad=20)
    plt.legend(fontsize=12)
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)
    
    # Figura 3: Gráfico de residuales
    plt.figure(figsize=(10, 6))
    plt.scatter(test_pred_original, errors, color='teal', alpha=0.7)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('Valores Predichos', fontsize=12)
    plt.ylabel('Residuales (Real - Predicho)', fontsize=12)
    plt.title('Análisis de Residuales', fontsize=14, pad=20)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
import torch
import torch.optim as optim
from src.models.model import GaiaRecalibrationMLP
from src.training.physic_loss import PhysicsInformedLoss

def train_model(X_train, y_train, dlambda, epochs=500, lr=0.001):
    """
    Bucle de entrenamiento de la Red Neuronal informada por la física.
    
    Parámetros:
    - X_train: Espectros de Gaia observados (Tensores)
    - y_train: Espectros de CALSPEC degradados/ideales (Tensores)
    - dlambda: Diferenciales de longitud de onda (Tensores)
    - epochs: Número de iteraciones completas
    - lr: Tasa de aprendizaje (Learning rate)
    """
    input_dim = X_train.shape[1]
    
    # Instanciamos el modelo y la función de pérdida matemática (PINN)
    model = GaiaRecalibrationMLP(input_dim)
    criterion = PhysicsInformedLoss(alpha=0.1, beta=0.5, delta=1.0)
    
    # Optimizador Adam con regularización L2 (weight_decay) para evitar overfitting
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5) 

    model.train() # Ponemos la red en modo entrenamiento
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass (Predicción)
        y_pred = model(X_train)
        
        # Calcular pérdida integrando las restricciones físicas
        loss = criterion(y_pred, y_train, dlambda)
        
        # Backward pass (Cálculo de gradientes) y optimización
        loss.backward()
        optimizer.step()
        
        # Imprimir progreso cada 50 épocas
        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")
            
    print("¡Entrenamiento finalizado con éxito!")
    return model
from __future__ import annotations
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
from sklearn.base import BaseEstimator, ClassifierMixin
import time

class HepatitisDataset(Dataset):
    """
    Custom Dataset for Hepatitis data.

    Parameters
    -----------
    X : np.ndarray or pd.DataFrame
        Feature matrix.
    y : np.ndarray or pd.Series    
        Target vector.

    Attributes
    -----------
    X : torch.FloatTensor
        Feature matrix as a FloatTensor.
    y : torch.LongTensor
        Target vector as a LongTensor.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y.values if hasattr(y, 'values') else y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]

class ResidualBlock(nn.Module):
    """
    Residual block with layer normalization and dropout.
    """
    def __init__(self, size: int, dropout_rate: float = 0.3):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(size),
            nn.Linear(size, size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.LayerNorm(size),
            nn.Linear(size, size),
            nn.Dropout(dropout_rate)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)

class HepatitisNet(nn.Module):
    """
    Neural Network for Hepatitis C classification with residual connections.

    Parameters
    -----------
    input_size : int
        Number of input features.
    hidden_sizes : list of int
        List of hidden layer sizes.
    num_classes : int
        Number of output classes.
    dropout_rate : float
        Dropout rate for regularization.
    num_residual_blocks : int
        Number of residual blocks to use.

    Attributes
    -----------
    layers : nn.ModuleList
        List of network layers including residual blocks.
    input_size : int
        Number of input features.
    num_classes : int
        Number of output classes.
    """


    def __init__(self, input_size: int = 12, hidden_sizes: list = [128, 64, 32], 
                 num_classes: int = 2, dropout_rate: float = 0.3, num_residual_blocks: int = 2):
        super(HepatitisNet, self).__init__()
        
        self.input_size = input_size
        self.num_classes = num_classes

        # Build network architecture
        layers = nn.ModuleList()
        
        # Input projection
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.LayerNorm(hidden_sizes[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        
        # Add residual blocks at each hidden layer
        for i in range(len(hidden_sizes) - 1):
            # Add residual blocks
            for _ in range(num_residual_blocks):
                layers.append(ResidualBlock(hidden_sizes[i], dropout_rate))
            
            # Project to next hidden size
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(nn.LayerNorm(hidden_sizes[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        
        # Add final residual blocks
        for _ in range(num_residual_blocks):
            layers.append(ResidualBlock(hidden_sizes[-1], dropout_rate))
        
        # Output projection
        layers.append(nn.Linear(hidden_sizes[-1], num_classes))
        
        self.layers = layers
        self._initialize_weights()
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

class ModelTrainer:
    """
    Class to handle training and validation of the HepatitisNet model.

    Parameters
    -----------
    model : nn.Module
        The neural network model to be trained.
    device : str
        Device to run the training on ('cpu' or 'cuda').
    
    Attributes
    -----------
    model : nn.Module
        The neural network model to be trained.
    device : str
        Device to run the training on ('cpu' or 'cuda').
    history : dict
        Dictionary to store training history.
    """


    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    def train_epoch(self, train_loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer) -> tuple[float, float]:
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for data, target in train_loader:
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
        
        return total_loss / len(train_loader), 100. * correct / total

    def validate_epoch(self, val_loader: DataLoader, criterion: nn.Module) -> tuple[float, float]:
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        return total_loss / len(val_loader), 100. * correct / total

    def train(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int = 50, learning_rate: float = 0.001) -> dict:
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        best_val_acc = 0
        patience_counter = 0
        patience = 10
        
        print(f"Training on {self.device}")
        print(f"Epochs: {epochs}, Learning Rate: {learning_rate}")
        print("-" * 50)
        
        start_time = time.time()
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader, criterion, optimizer)
            val_loss, val_acc = self.validate_epoch(val_loader, criterion)
            scheduler.step(val_loss)
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch:3d}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.1f}%, '
                      f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.1f}%')
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                self.best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
        
        self.model.load_state_dict(self.best_model_state)
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time:.2f} seconds")
        print(f"Best validation accuracy: {best_val_acc:.2f}%")
        
        return self.history

def evaluate_model(model: nn.Module, test_loader: DataLoader, device: str = 'cpu') -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate the model on the test dataset.
    
    Parameters
    -----------
    model : nn.Module
        The trained model to be evaluated.
    test_loader : DataLoader
        DataLoader for the test dataset.
    device : str
        Device to run the evaluation on (default: 'cpu').

    Returns
    -----------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        - y_true: Ground truth labels.
        - y_pred: Predicted labels.
        - y_probs: Predicted probabilities.

    Examples
    ---------
    >>> y_true, y_pred, y_probs = evaluate_model(model, test_loader, device='cuda')
    """
    model.eval()
    y_true = []
    y_pred = []
    y_probs = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            probs = torch.softmax(output, dim=1)
            pred = output.argmax(dim=1)
            
            y_true.extend(target.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())
            y_probs.extend(probs.cpu().numpy())
    
    return np.array(y_true), np.array(y_pred), np.array(y_probs)

def save_model(model: nn.Module, filepath: str, additional_info: dict = None) -> None:
    """
    Save the model to a file.

    Parameters
    -----------
    model : nn.Module
        The model to be saved.
    filepath : str
        Path to the file where the model will be saved.
    additional_info : dict, optional
        Any additional information to save with the model (e.g., training parameters).
        

    Examples
    ---------
    >>> save_model(model, 'models/hepatitis_model.pth', {'input_size': 12, 'num_classes': 2})
    """

    torch.save({
        'model_state_dict': model.state_dict(),
        'model_class': model.__class__.__name__,
        'additional_info': additional_info
    }, filepath, _use_new_zipfile_serialization=False)
    print(f"Model saved to: {filepath}")

def load_model(filepath: str, model_class: type[HepatitisNet] = HepatitisNet, input_size: int = 12) -> tuple[nn.Module, dict]:
    """
    Load a model from a file.
    
    Parameters
    -----------
    filepath : str
        Path to the file from which the model will be loaded.
    model_class : type
        The class of the model to be loaded (default: HepatitisNet).
    input_size : int
        Number of input features (default: 12).

    Returns
    -----------
    tuple[nn.Module, dict]
        - model: The loaded model.
        - additional_info: Any additional information saved with the model.
    
    Examples
    ---------
    >>> model, info = load_model('models/hepatitis_model.pth')
    >>> print(info)
    """
    checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
    
    model = model_class(input_size=input_size)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, checkpoint.get('additional_info', None)

class TorchWrapper(BaseEstimator, ClassifierMixin):
    """
    A wrapper to make PyTorch models compatible with scikit-learn.
    
    Parameters
    -----------
    model : HepatitisNet
        The PyTorch model instance.
    device : str
        Device to run inference on ('cpu' or 'cuda').
    classes : array-like
        Class labels for the classifier.

    Attributes
    -----------
    model : HepatitisNet
        The PyTorch model instance.
    device : str
        Device to run inference on ('cpu' or 'cuda').
    classes_ : array-like
        Class labels for the classifier.

    Examples
    ---------
    >>> model, _ = load_model('models/hepatitis_model.pth')
    >>> wrapper = TorchWrapper(model, device='cuda', classes=[0, 1])
    >>> CalibrationDisplay.from_estimator(wrapper, X_test, y_test, n_bins=10)
    """

    def __init__(self, model: type[HepatitisNet], device: str, classes: any):
        self.model = model
        self.device = device
        self.classes_ = classes
    def fit(self, X: np.ndarray, y: np.ndarray) -> TorchWrapper:
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_tensor = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
        return probs
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)


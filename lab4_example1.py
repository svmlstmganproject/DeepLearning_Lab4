"""
Lab 8 Example 1: Linear Regression Implementation
เรียนรู้การสร้างและใช้งาน linear regression ด้วย PyTorch
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns

print("=" * 60)
print("Lab 8 Example 1: Linear Regression Implementation")
print("=" * 60)

# 1. Simple Linear Regression (Manual Implementation)
class SimpleLinearRegression(nn.Module):
    """Linear regression แบบ manual implementation"""
    
    def __init__(self, input_size):
        super(SimpleLinearRegression, self).__init__()
        
        # สร้าง parameters แบบ manual
        self.weight = nn.Parameter(torch.randn(input_size, 1, requires_grad=True))
        self.bias = nn.Parameter(torch.randn(1, requires_grad=True))
        
    def forward(self, x):
        """Forward pass: y = Wx + b"""
        return torch.matmul(x, self.weight) + self.bias
    
    def get_parameters(self):
        """ดึง weights และ bias"""
        return {
            'weight': self.weight.data.clone(),
            'bias': self.bias.data.clone()
        }

# 2. Linear Regression ด้วย nn.Linear
class LinearRegression(nn.Module):
    """Linear regression ด้วย nn.Linear"""
    
    def __init__(self, input_size, output_size=1):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        
    def forward(self, x):
        return self.linear(x)
    
    def get_parameters(self):
        """ดึง weights และ bias"""
        return {
            'weight': self.linear.weight.data.clone(),
            'bias': self.linear.bias.data.clone()
        }

# 3. Multi-layer Linear Regression
class MultiLayerRegression(nn.Module):
    """Linear regression หลายชั้น"""
    
    def __init__(self, input_size, hidden_sizes, output_size=1, dropout_rate=0.0):
        super(MultiLayerRegression, self).__init__()
        
        layers = []
        current_size = input_size
        
        # Hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(current_size, hidden_size))
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            current_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(current_size, output_size))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

# 4. Polynomial Features
class PolynomialFeatures:
    """สร้าง polynomial features"""
    
    def __init__(self, degree=2):
        self.degree = degree
        
    def fit_transform(self, X):
        """สร้าง polynomial features"""
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        
        X_poly = X.copy()
        
        # เพิ่ม polynomial terms
        for d in range(2, self.degree + 1):
            X_poly = torch.cat([X_poly, torch.pow(X, d)], dim=1)
        
        return X_poly

# 5. การสร้างข้อมูลตัวอย่าง
def create_simple_dataset(n_samples=100, noise=0.1, seed=42):
    """สร้างข้อมูล linear regression ง่ายๆ"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # สร้างข้อมูล y = 2x + 3 + noise
    X = torch.linspace(-2, 2, n_samples).reshape(-1, 1)
    true_weight = 2.0
    true_bias = 3.0
    
    y = true_weight * X.squeeze() + true_bias
    y += torch.normal(0, noise, size=y.shape)  # เพิ่ม noise
    
    return X, y.reshape(-1, 1), true_weight, true_bias

def create_multivariate_dataset(n_samples=1000, n_features=5, noise=0.1, seed=42):
    """สร้างข้อมูล multivariate regression"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        noise=noise * 100,
        random_state=seed
    )
    
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

def create_polynomial_dataset(n_samples=100, degree=3, noise=0.1, seed=42):
    """สร้างข้อมูล polynomial"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    X = torch.linspace(-2, 2, n_samples).reshape(-1, 1)
    
    # y = x^3 - 2x^2 + x + 1 + noise
    y = torch.pow(X, 3) - 2 * torch.pow(X, 2) + X + 1
    y += torch.normal(0, noise, size=y.shape)
    
    return X, y

# 6. Training Functions
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=100, verbose=True):
    """Training function สำหรับ regression"""
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        if verbose and (epoch + 1) % (num_epochs // 10) == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')
    
    return train_losses, val_losses

def evaluate_model(model, X_test, y_test):
    """ประเมินผล model"""
    model.eval()
    
    with torch.no_grad():
        predictions = model(X_test)
        
        # แปลงเป็น numpy สำหรับการคำนวณ metrics
        y_true = y_test.numpy().flatten()
        y_pred = predictions.numpy().flatten()
        
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }
        
        return metrics, y_pred

# 7. Visualization Functions
def plot_regression_results(X, y, model, title="Regression Results"):
    """วาดกราฟผลลัพธ์ regression"""
    
    model.eval()
    
    # สร้าง predictions
    with torch.no_grad():
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        predictions = model(X)
    
    X_np = X.numpy()
    y_np = y.numpy()
    pred_np = predictions.numpy()
    
    plt.figure(figsize=(10, 6))
    
    if X.shape[1] == 1:  # 1D case
        # เรียงลำดับสำหรับการ plot
        sort_idx = np.argsort(X_np.flatten())
        
        plt.scatter(X_np, y_np, alpha=0.6, label='True data', color='blue')
        plt.plot(X_np[sort_idx], pred_np[sort_idx], 'r-', linewidth=2, label='Predictions')
        plt.xlabel('X')
        plt.ylabel('y')
        plt.legend()
    else:  # Multi-dimensional case
        plt.scatter(y_np, pred_np, alpha=0.6)
        
        # เส้นทแยงมุม (perfect prediction)
        min_val = min(y_np.min(), pred_np.min())
        max_val = max(y_np.max(), pred_np.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')
        
        plt.xlabel('True values')
        plt.ylabel('Predictions')
        plt.legend()
    
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_training_history(train_losses, val_losses, title="Training History"):
    """วาดกราฟ training history"""
    plt.figure(figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# ทดสอบการใช้งาน
print("\n1. Simple Linear Regression (1 Feature)")
print("-" * 45)

# สร้างข้อมูลตัวอย่าง
X_simple, y_simple, true_weight, true_bias = create_simple_dataset(n_samples=100, noise=0.2)

print(f"Dataset: {X_simple.shape[0]} samples")
print(f"True parameters: weight={true_weight}, bias={true_bias}")

# แบ่งข้อมูล train/test
split_idx = int(0.8 * len(X_simple))
X_train, X_test = X_simple[:split_idx], X_simple[split_idx:]
y_train, y_test = y_simple[:split_idx], y_simple[split_idx:]

print(f"Train set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# สร้าง DataLoaders
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Model 1: Manual Implementation
print("\n1.1 Manual Linear Regression")
print("-" * 30)

manual_model = SimpleLinearRegression(input_size=1)
criterion = nn.MSELoss()
optimizer = optim.SGD(manual_model.parameters(), lr=0.01)

print("Before training:")
params_before = manual_model.get_parameters()
print(f"  Weight: {params_before['weight'].item():.4f}")
print(f"  Bias: {params_before['bias'].item():.4f}")

# Training
train_losses, val_losses = train_model(
    manual_model, train_loader, test_loader, criterion, optimizer, 
    num_epochs=200, verbose=False
)

print("After training:")
params_after = manual_model.get_parameters()
print(f"  Weight: {params_after['weight'].item():.4f} (True: {true_weight})")
print(f"  Bias: {params_after['bias'].item():.4f} (True: {true_bias})")

# Evaluation
metrics, predictions = evaluate_model(manual_model, X_test, y_test)
print(f"\nTest Metrics:")
for metric_name, value in metrics.items():
    print(f"  {metric_name.upper()}: {value:.6f}")

# Model 2: nn.Linear Implementation
print("\n1.2 nn.Linear Implementation")
print("-" * 30)

linear_model = LinearRegression(input_size=1)
optimizer = optim.Adam(linear_model.parameters(), lr=0.01)

# Training
train_losses_2, val_losses_2 = train_model(
    linear_model, train_loader, test_loader, criterion, optimizer, 
    num_epochs=200, verbose=False
)

params_linear = linear_model.get_parameters()
print(f"Learned parameters:")
print(f"  Weight: {params_linear['weight'].item():.4f}")
print(f"  Bias: {params_linear['bias'].item():.4f}")

# Evaluation
metrics_linear, _ = evaluate_model(linear_model, X_test, y_test)
print(f"\nTest Metrics:")
for metric_name, value in metrics_linear.items():
    print(f"  {metric_name.upper()}: {value:.6f}")

# Visualization
plot_regression_results(X_test, y_test, linear_model, "Simple Linear Regression")
plot_training_history(train_losses_2, val_losses_2, "Training History - Simple Linear Regression")

print("\n2. Multivariate Linear Regression")
print("-" * 40)

# สร้างข้อมูล multivariate
X_multi, y_multi = create_multivariate_dataset(n_samples=1000, n_features=5, noise=0.1)

print(f"Dataset: {X_multi.shape[0]} samples, {X_multi.shape[1]} features")
print(f"Feature statistics:")
print(f"  Mean: {X_multi.mean(dim=0)}")
print(f"  Std: {X_multi.std(dim=0)}")

# แบ่งข้อมูล
split_idx = int(0.8 * len(X_multi))
X_train_multi, X_test_multi = X_multi[:split_idx], X_multi[split_idx:]
y_train_multi, y_test_multi = y_multi[:split_idx], y_multi[split_idx:]

# สร้าง DataLoaders
train_dataset_multi = TensorDataset(X_train_multi, y_train_multi)
test_dataset_multi = TensorDataset(X_test_multi, y_test_multi)
train_loader_multi = DataLoader(train_dataset_multi, batch_size=32, shuffle=True)
test_loader_multi = DataLoader(test_dataset_multi, batch_size=32, shuffle=False)

# Model
multi_model = LinearRegression(input_size=5)
optimizer = optim.Adam(multi_model.parameters(), lr=0.001)

# Training
train_losses_multi, val_losses_multi = train_model(
    multi_model, train_loader_multi, test_loader_multi, criterion, optimizer,
    num_epochs=300, verbose=True
)

# Evaluation
metrics_multi, _ = evaluate_model(multi_model, X_test_multi, y_test_multi)
print(f"\nMultivariate Regression Metrics:")
for metric_name, value in metrics_multi.items():
    print(f"  {metric_name.upper()}: {value:.6f}")

# วิเคราะห์ weights
params_multi = multi_model.get_parameters()
print(f"\nLearned weights:")
for i, weight in enumerate(params_multi['weight'].squeeze()):
    print(f"  Feature {i+1}: {weight.item():.4f}")
print(f"  Bias: {params_multi['bias'].item():.4f}")

# Visualization
plot_regression_results(X_test_multi, y_test_multi, multi_model, "Multivariate Linear Regression")
plot_training_history(train_losses_multi, val_losses_multi, "Training History - Multivariate Regression")

print("\n3. Polynomial Regression")
print("-" * 25)

# สร้างข้อมูล polynomial
X_poly, y_poly = create_polynomial_dataset(n_samples=100, degree=3, noise=0.3)

print(f"Original polynomial dataset: {X_poly.shape}")

# สร้าง polynomial features
poly_features = PolynomialFeatures(degree=3)
X_poly_expanded = poly_features.fit_transform(X_poly)

print(f"After polynomial expansion: {X_poly_expanded.shape}")
print(f"Features: [x, x^2, x^3]")

# แบ่งข้อมูล
split_idx = int(0.8 * len(X_poly_expanded))
X_train_poly, X_test_poly = X_poly_expanded[:split_idx], X_poly_expanded[split_idx:]
y_train_poly, y_test_poly = y_poly[:split_idx], y_poly[split_idx:]

# สร้าง DataLoaders
train_dataset_poly = TensorDataset(X_train_poly, y_train_poly)
test_dataset_poly = TensorDataset(X_test_poly, y_test_poly)
train_loader_poly = DataLoader(train_dataset_poly, batch_size=16, shuffle=True)
test_loader_poly = DataLoader(test_dataset_poly, batch_size=16, shuffle=False)

# Model
poly_model = LinearRegression(input_size=3)  # 3 features: x, x^2, x^3
optimizer = optim.Adam(poly_model.parameters(), lr=0.01)

# Training
train_losses_poly, val_losses_poly = train_model(
    poly_model, train_loader_poly, test_loader_poly, criterion, optimizer,
    num_epochs=500, verbose=True
)

# Evaluation
metrics_poly, _ = evaluate_model(poly_model, X_test_poly, y_test_poly)
print(f"\nPolynomial Regression Metrics:")
for metric_name, value in metrics_poly.items():
    print(f"  {metric_name.upper()}: {value:.6f}")

# วิเคราะห์ coefficients
params_poly = poly_model.get_parameters()
print(f"\nLearned polynomial coefficients:")
feature_names = ['x', 'x^2', 'x^3']
for i, (name, coef) in enumerate(zip(feature_names, params_poly['weight'].squeeze())):
    print(f"  {name}: {coef.item():.4f}")
print(f"  Constant: {params_poly['bias'].item():.4f}")

# เปรียบเทียบกับ true function: y = x^3 - 2x^2 + x + 1
print(f"\nTrue coefficients:")
print(f"  x^3: 1.0000")
print(f"  x^2: -2.0000") 
print(f"  x: 1.0000")
print(f"  Constant: 1.0000")

print("\n4. Multi-layer Regression")
print("-" * 30)

# สร้าง non-linear dataset
torch.manual_seed(42)
X_nonlinear = torch.linspace(-3, 3, 200).reshape(-1, 1)
y_nonlinear = torch.sin(X_nonlinear) + 0.5 * torch.cos(2 * X_nonlinear) + torch.normal(0, 0.1, X_nonlinear.shape)

# แบ่งข้อมูล
split_idx = int(0.8 * len(X_nonlinear))
X_train_nl, X_test_nl = X_nonlinear[:split_idx], X_nonlinear[split_idx:]
y_train_nl, y_test_nl = y_nonlinear[:split_idx], y_nonlinear[split_idx:]

# สร้าง DataLoaders
train_dataset_nl = TensorDataset(X_train_nl, y_train_nl)
test_dataset_nl = TensorDataset(X_test_nl, y_test_nl)
train_loader_nl = DataLoader(train_dataset_nl, batch_size=16, shuffle=True)
test_loader_nl = DataLoader(test_dataset_nl, batch_size=16, shuffle=False)

# เปรียบเทียบ models
models_config = [
    ("Linear", LinearRegression(input_size=1)),
    ("MLP-Small", MultiLayerRegression(input_size=1, hidden_sizes=[16], dropout_rate=0.1)),
    ("MLP-Large", MultiLayerRegression(input_size=1, hidden_sizes=[64, 32], dropout_rate=0.2))
]

results = {}

for model_name, model in models_config:
    print(f"\nTraining {model_name}...")
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_losses, val_losses = train_model(
        model, train_loader_nl, test_loader_nl, criterion, optimizer,
        num_epochs=200, verbose=False
    )
    
    metrics, _ = evaluate_model(model, X_test_nl, y_test_nl)
    results[model_name] = {
        'model': model,
        'metrics': metrics,
        'train_losses': train_losses,
        'val_losses': val_losses
    }
    
    print(f"  Final R²: {metrics['r2']:.4f}")

# แสดงผลเปรียบเทียบ
print(f"\nModel Comparison:")
print(f"{'Model':<12} {'RMSE':<10} {'R²':<10} {'MAE':<10}")
print("-" * 45)

for model_name, result in results.items():
    metrics = result['metrics']
    print(f"{model_name:<12} {metrics['rmse']:<10.4f} {metrics['r2']:<10.4f} {metrics['mae']:<10.4f}")

# Visualization สำหรับ non-linear regression
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, (model_name, result) in enumerate(results.items()):
    model = result['model']
    
    # สร้าง smooth predictions
    X_smooth = torch.linspace(-3, 3, 300).reshape(-1, 1)
    model.eval()
    with torch.no_grad():
        y_smooth = model(X_smooth)
    
    axes[i].scatter(X_test_nl.numpy(), y_test_nl.numpy(), alpha=0.6, label='True data', s=20)
    axes[i].plot(X_smooth.numpy(), y_smooth.numpy(), 'r-', linewidth=2, label='Predictions')
    axes[i].set_title(f'{model_name} (R² = {result["metrics"]["r2"]:.3f})')
    axes[i].set_xlabel('X')
    axes[i].set_ylabel('y')
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n" + "=" * 60)
print("สรุป Example 1: Linear Regression Implementation")
print("=" * 60)
print("✓ Manual linear regression implementation")
print("✓ nn.Linear การใช้งาน built-in layers")
print("✓ Multivariate regression สำหรับหลาย features")
print("✓ Polynomial regression สำหรับ non-linear patterns")
print("✓ Multi-layer regression สำหรับ complex functions")
print("✓ Model evaluation และ comparison")
print("✓ Visualization และ interpretation")
print("=" * 60)
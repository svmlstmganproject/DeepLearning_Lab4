"""
Lab 8 Example 2: Feature Scaling และ Normalization
เรียนรู้การประมวลผลและปรับสเกลข้อมูลสำหรับ regression
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.datasets import make_regression, load_boston
import seaborn as sns
from collections import OrderedDict

print("=" * 60)
print("Lab 8 Example 2: Feature Scaling และ Normalization")
print("=" * 60)

# 1. Custom Scalers Implementation
class TorchStandardScaler:
    """Standard Scaler implementation ใน PyTorch"""
    
    def __init__(self):
        self.mean = None
        self.std = None
        self.fitted = False
    
    def fit(self, X):
        """คำนวณ mean และ std"""
        self.mean = torch.mean(X, dim=0)
        self.std = torch.std(X, dim=0, unbiased=False)
        self.std = torch.where(self.std == 0, torch.ones_like(self.std), self.std)  # หลีกเลี่ยง division by zero
        self.fitted = True
        return self
    
    def transform(self, X):
        """แปลงข้อมูลด้วย z-score normalization"""
        if not self.fitted:
            raise ValueError("Scaler must be fitted before transform")
        return (X - self.mean) / self.std
    
    def fit_transform(self, X):
        """Fit และ transform ในขั้นตอนเดียว"""
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X_scaled):
        """แปลงกลับจาก scaled ไปเป็น original scale"""
        if not self.fitted:
            raise ValueError("Scaler must be fitted before inverse_transform")
        return X_scaled * self.std + self.mean
    
    def get_params(self):
        """ดูค่า parameters"""
        return {'mean': self.mean, 'std': self.std}

class TorchMinMaxScaler:
    """Min-Max Scaler implementation ใน PyTorch"""
    
    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range
        self.min_vals = None
        self.max_vals = None
        self.fitted = False
    
    def fit(self, X):
        """คำนวณ min และ max"""
        self.min_vals = torch.min(X, dim=0)[0]
        self.max_vals = torch.max(X, dim=0)[0]
        self.range_vals = self.max_vals - self.min_vals
        self.range_vals = torch.where(self.range_vals == 0, torch.ones_like(self.range_vals), self.range_vals)
        self.fitted = True
        return self
    
    def transform(self, X):
        """แปลงข้อมูลไปยัง [0, 1] หรือ range ที่กำหนด"""
        if not self.fitted:
            raise ValueError("Scaler must be fitted before transform")
        
        # Normalize to [0, 1]
        X_scaled = (X - self.min_vals) / self.range_vals
        
        # Scale to desired range
        min_range, max_range = self.feature_range
        X_scaled = X_scaled * (max_range - min_range) + min_range
        
        return X_scaled
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X_scaled):
        """แปลงกลับจาก scaled ไปเป็น original scale"""
        if not self.fitted:
            raise ValueError("Scaler must be fitted before inverse_transform")
        
        min_range, max_range = self.feature_range
        X_normalized = (X_scaled - min_range) / (max_range - min_range)
        return X_normalized * self.range_vals + self.min_vals

class TorchRobustScaler:
    """Robust Scaler implementation ใน PyTorch"""
    
    def __init__(self):
        self.median = None
        self.iqr = None
        self.fitted = False
    
    def fit(self, X):
        """คำนวณ median และ IQR"""
        self.median = torch.median(X, dim=0)[0]
        
        # คำนวณ Q1 และ Q3
        q1 = torch.quantile(X, 0.25, dim=0)
        q3 = torch.quantile(X, 0.75, dim=0)
        self.iqr = q3 - q1
        self.iqr = torch.where(self.iqr == 0, torch.ones_like(self.iqr), self.iqr)
        self.fitted = True
        return self
    
    def transform(self, X):
        """แปลงข้อมูลด้วย robust scaling"""
        if not self.fitted:
            raise ValueError("Scaler must be fitted before transform")
        return (X - self.median) / self.iqr
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X_scaled):
        """แปลงกลับจาก scaled ไปเป็น original scale"""
        if not self.fitted:
            raise ValueError("Scaler must be fitted before inverse_transform")
        return X_scaled * self.iqr + self.median

# 2. Feature Engineering และ Selection
class FeatureEngineer:
    """Class สำหรับ feature engineering"""
    
    def __init__(self):
        self.feature_stats = {}
    
    def create_polynomial_features(self, X, degree=2, include_bias=False):
        """สร้าง polynomial features"""
        features = [X]
        
        if include_bias:
            features.append(torch.ones(X.shape[0], 1))
        
        # สร้าง polynomial terms
        for d in range(2, degree + 1):
            features.append(torch.pow(X, d))
        
        # สร้าง interaction terms (สำหรับ multivariate)
        if X.shape[1] > 1:
            for i in range(X.shape[1]):
                for j in range(i + 1, X.shape[1]):
                    interaction = (X[:, i] * X[:, j]).unsqueeze(1)
                    features.append(interaction)
        
        return torch.cat(features, dim=1)
    
    def create_log_features(self, X, epsilon=1e-8):
        """สร้าง log features (สำหรับ positive values)"""
        X_positive = torch.where(X > 0, X, torch.tensor(epsilon))
        return torch.log(X_positive)
    
    def create_sqrt_features(self, X):
        """สร้าง square root features (สำหรับ non-negative values)"""
        X_nonneg = torch.where(X >= 0, X, torch.zeros_like(X))
        return torch.sqrt(X_nonneg)
    
    def detect_outliers(self, X, method='iqr', factor=1.5):
        """ตรวจหา outliers"""
        if method == 'iqr':
            q1 = torch.quantile(X, 0.25, dim=0)
            q3 = torch.quantile(X, 0.75, dim=0)
            iqr = q3 - q1
            
            lower_bound = q1 - factor * iqr
            upper_bound = q3 + factor * iqr
            
            outliers = (X < lower_bound) | (X > upper_bound)
            
        elif method == 'zscore':
            mean = torch.mean(X, dim=0)
            std = torch.std(X, dim=0)
            z_scores = torch.abs((X - mean) / std)
            outliers = z_scores > factor
        
        return outliers
    
    def remove_outliers(self, X, y, method='iqr', factor=1.5):
        """ลบ outliers"""
        outliers = self.detect_outliers(X, method, factor)
        outlier_mask = torch.any(outliers, dim=1)
        
        clean_indices = ~outlier_mask
        
        return X[clean_indices], y[clean_indices], outlier_mask

# 3. Regression Model with Preprocessing
class PreprocessedRegressionModel(nn.Module):
    """Regression model ที่มี preprocessing ติดมาด้วย"""
    
    def __init__(self, input_size, hidden_sizes, output_size=1, 
                 scaler_type='standard', dropout_rate=0.2):
        super(PreprocessedRegressionModel, self).__init__()
        
        # เลือก scaler
        if scaler_type == 'standard':
            self.scaler = TorchStandardScaler()
        elif scaler_type == 'minmax':
            self.scaler = TorchMinMaxScaler()
        elif scaler_type == 'robust':
            self.scaler = TorchRobustScaler()
        else:
            self.scaler = None
        
        # สร้าง network
        layers = []
        current_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(current_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            current_size = hidden_size
        
        layers.append(nn.Linear(current_size, output_size))
        
        self.network = nn.Sequential(*layers)
        self.is_fitted = False
    
    def fit_scaler(self, X):
        """Fit scaler กับข้อมูล training"""
        if self.scaler is not None:
            self.scaler.fit(X)
            self.is_fitted = True
    
    def forward(self, x):
        if self.scaler is not None and self.is_fitted:
            x = self.scaler.transform(x)
        return self.network(x)

# 4. การสร้างข้อมูลทดสอบ
def create_scaled_dataset(n_samples=1000, n_features=5, scale_factor=None, add_outliers=True):
    """สร้างข้อมูลที่มี scale ต่างกัน"""
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    # สร้างข้อมูลพื้นฐาน
    X, y = make_regression(n_samples=n_samples, n_features=n_features, 
                          noise=10, random_state=42)
    
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
    
    # ปรับ scale ของแต่ละ feature
    if scale_factor is None:
        scale_factor = [1, 10, 100, 1000, 0.01][:n_features]
    
    for i in range(min(len(scale_factor), n_features)):
        X[:, i] *= scale_factor[i]
    
    # เพิ่ม outliers
    if add_outliers:
        n_outliers = int(0.05 * n_samples)  # 5% outliers
        outlier_indices = torch.randperm(n_samples)[:n_outliers]
        
        for idx in outlier_indices:
            feature_idx = torch.randint(0, n_features, (1,))
            X[idx, feature_idx] *= torch.randint(5, 20, (1,)).float()
    
    return X, y, scale_factor

def create_housing_dataset():
    """สร้างข้อมูล housing ตัวอย่าง"""
    
    torch.manual_seed(42)
    
    n_samples = 500
    
    # สร้างข้อมูลที่มีความหมาย
    features = {
        'area': torch.normal(150, 50, (n_samples,)),           # พื้นที่ (ตร.ม.)
        'bedrooms': torch.randint(1, 6, (n_samples,)).float(), # จำนวนห้องนอน
        'age': torch.randint(0, 50, (n_samples,)).float(),     # อายุบ้าน (ปี)
        'distance_to_city': torch.normal(10, 5, (n_samples,)), # ระยะทางจากเมือง (กม.)
        'income_area': torch.normal(50000, 15000, (n_samples,)) # รายได้เฉลี่ยในพื้นที่
    }
    
    # สร้าง target (ราคาบ้าน)
    price = (features['area'] * 1000 + 
             features['bedrooms'] * 50000 - 
             features['age'] * 2000 - 
             features['distance_to_city'] * 5000 + 
             features['income_area'] * 2 + 
             torch.normal(0, 50000, (n_samples,)))
    
    # รวม features
    X = torch.stack([features['area'], features['bedrooms'], features['age'],
                     features['distance_to_city'], features['income_area']], dim=1)
    
    y = price.reshape(-1, 1)
    
    feature_names = ['area', 'bedrooms', 'age', 'distance_to_city', 'income_area']
    
    return X, y, feature_names

# 5. Visualization Functions
def plot_feature_distributions(X, feature_names, title="Feature Distributions"):
    """วาดกราฟการกระจายของ features"""
    
    n_features = X.shape[1]
    fig, axes = plt.subplots(2, (n_features + 1) // 2, figsize=(15, 8))
    axes = axes.flatten()
    
    for i in range(n_features):
        data = X[:, i].numpy()
        
        axes[i].hist(data, bins=30, alpha=0.7, edgecolor='black')
        axes[i].set_title(f'{feature_names[i]}\nMean: {data.mean():.2f}, Std: {data.std():.2f}')
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Frequency')
        axes[i].grid(True, alpha=0.3)
    
    # ซ่อน axes ที่ไม่ใช้
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_scaling_comparison(X_original, scalers_dict, feature_names):
    """เปรียบเทียบผลของ scaling methods ต่างๆ"""
    
    n_features = min(3, X_original.shape[1])  # แสดงแค่ 3 features แรก
    n_scalers = len(scalers_dict)
    
    fig, axes = plt.subplots(n_features, n_scalers + 1, figsize=(20, n_features * 4))
    
    if n_features == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_features):
        # Original data
        axes[i, 0].hist(X_original[:, i].numpy(), bins=30, alpha=0.7, edgecolor='black')
        axes[i, 0].set_title(f'Original\n{feature_names[i]}')
        axes[i, 0].grid(True, alpha=0.3)
        
        # Scaled data
        for j, (scaler_name, scaler) in enumerate(scalers_dict.items()):
            X_scaled = scaler.fit_transform(X_original)
            
            axes[i, j + 1].hist(X_scaled[:, i].numpy(), bins=30, alpha=0.7, edgecolor='black')
            axes[i, j + 1].set_title(f'{scaler_name}\n{feature_names[i]}')
            axes[i, j + 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_outlier_detection(X, outliers, feature_names):
    """แสดงผลการตรวจหา outliers"""
    
    n_features = X.shape[1]
    fig, axes = plt.subplots(1, n_features, figsize=(4 * n_features, 4))
    
    if n_features == 1:
        axes = [axes]
    
    for i in range(n_features):
        # Normal points
        normal_mask = ~outliers[:, i]
        axes[i].scatter(range(len(X)), X[normal_mask, i].numpy(), 
                       alpha=0.6, label='Normal', s=20)
        
        # Outliers
        outlier_indices = torch.where(outliers[:, i])[0]
        if len(outlier_indices) > 0:
            axes[i].scatter(outlier_indices, X[outliers[:, i], i].numpy(),
                           color='red', alpha=0.8, label='Outliers', s=30)
        
        axes[i].set_title(f'{feature_names[i]}\nOutliers: {outliers[:, i].sum().item()}')
        axes[i].set_xlabel('Sample Index')
        axes[i].set_ylabel('Value')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# 6. Model Training และ Evaluation
def train_with_different_scalers(X_train, y_train, X_test, y_test, scalers_dict):
    """เปรียบเทียบประสิทธิภาพของ scalers ต่างๆ"""
    
    results = {}
    
    for scaler_name, scaler in scalers_dict.items():
        print(f"\nTraining with {scaler_name} scaler...")
        
        # Scale ข้อมูล
        if scaler is not None:
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
        else:
            X_train_scaled = X_train
            X_test_scaled = X_test
        
        # สร้าง model
        model = nn.Sequential(
            nn.Linear(X_train.shape[1], 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )
        
        # Training setup
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Create DataLoaders
        train_dataset = TensorDataset(X_train_scaled, y_train)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        # Training
        model.train()
        train_losses = []
        
        for epoch in range(200):
            epoch_loss = 0.0
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            train_losses.append(epoch_loss / len(train_loader))
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            train_pred = model(X_train_scaled)
            test_pred = model(X_test_scaled)
            
            train_mse = nn.MSELoss()(train_pred, y_train).item()
            test_mse = nn.MSELoss()(test_pred, y_test).item()
            
            # R² score
            y_train_np = y_train.numpy().flatten()
            y_test_np = y_test.numpy().flatten()
            train_pred_np = train_pred.numpy().flatten()
            test_pred_np = test_pred.numpy().flatten()
            
            from sklearn.metrics import r2_score
            train_r2 = r2_score(y_train_np, train_pred_np)
            test_r2 = r2_score(y_test_np, test_pred_np)
        
        results[scaler_name] = {
            'model': model,
            'scaler': scaler,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_losses': train_losses
        }
        
        print(f"  Train MSE: {train_mse:.4f}, Test MSE: {test_mse:.4f}")
        print(f"  Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
    
    return results

# ทดสอบการใช้งาน
print("\n1. การสร้างข้อมูลที่มี Scale ต่างกัน")
print("-" * 45)

# สร้างข้อมูลทดสอบ
X_scaled, y_scaled, scale_factors = create_scaled_dataset(
    n_samples=800, n_features=5, 
    scale_factor=[1, 10, 100, 1000, 0.01],
    add_outliers=True
)

feature_names = [f'Feature_{i+1}' for i in range(5)]

print(f"Dataset shape: {X_scaled.shape}")
print(f"Scale factors: {scale_factors}")
print(f"\nFeature statistics (before scaling):")
for i, name in enumerate(feature_names):
    data = X_scaled[:, i]
    print(f"  {name}: mean={data.mean():.2f}, std={data.std():.2f}, "
          f"min={data.min():.2f}, max={data.max():.2f}")

# Visualization
plot_feature_distributions(X_scaled, feature_names, "Original Feature Distributions")

print("\n2. การทดสอบ Scalers ต่างๆ")
print("-" * 30)

# สร้าง scalers
scalers_dict = OrderedDict([
    ('No Scaling', None),
    ('Standard', TorchStandardScaler()),
    ('MinMax', TorchMinMaxScaler()),
    ('Robust', TorchRobustScaler())
])

# เปรียบเทียบ scaling methods
plot_scaling_comparison(X_scaled, {k: v for k, v in scalers_dict.items() if v is not None}, feature_names)

# ทดสอบ scalers
print("Testing individual scalers:")
for scaler_name, scaler in scalers_dict.items():
    if scaler is not None:
        X_transformed = scaler.fit_transform(X_scaled)
        X_inverse = scaler.inverse_transform(X_transformed)
        
        # ตรวจสอบ inverse transform
        reconstruction_error = torch.mean(torch.abs(X_scaled - X_inverse)).item()
        
        print(f"\n{scaler_name} Scaler:")
        print(f"  Transformed range: [{X_transformed.min():.3f}, {X_transformed.max():.3f}]")
        print(f"  Reconstruction error: {reconstruction_error:.6f}")
        
        # แสดง parameters
        if hasattr(scaler, 'get_params'):
            params = scaler.get_params()
            if 'mean' in params:
                print(f"  Mean: {params['mean'][:3].tolist()} (first 3 features)")
                print(f"  Std: {params['std'][:3].tolist()} (first 3 features)")

print("\n3. การตรวจหา Outliers")
print("-" * 25)

# Feature engineering
engineer = FeatureEngineer()

# ตรวจหา outliers
outliers_iqr = engineer.detect_outliers(X_scaled, method='iqr', factor=1.5)
outliers_zscore = engineer.detect_outliers(X_scaled, method='zscore', factor=3.0)

print(f"Outliers detected (IQR method): {outliers_iqr.sum().item()} values")
print(f"Outliers detected (Z-score method): {outliers_zscore.sum().item()} values")

# แสดงจำนวน outliers ต่อ feature
print(f"\nOutliers per feature (IQR method):")
for i, name in enumerate(feature_names):
    count = outliers_iqr[:, i].sum().item()
    print(f"  {name}: {count} outliers")

# Visualization
plot_outlier_detection(X_scaled, outliers_iqr, feature_names)

# ลบ outliers
X_clean, y_clean, outlier_mask = engineer.remove_outliers(X_scaled, y_scaled, method='iqr')
print(f"\nAfter removing outliers:")
print(f"  Original samples: {len(X_scaled)}")
print(f"  Clean samples: {len(X_clean)}")
print(f"  Removed samples: {outlier_mask.sum().item()}")

print("\n4. Model Training Comparison")
print("-" * 35)

# แบ่งข้อมูล train/test
split_idx = int(0.8 * len(X_clean))
X_train, X_test = X_clean[:split_idx], X_clean[split_idx:]
y_train, y_test = y_clean[:split_idx], y_clean[split_idx:]

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# เปรียบเทียบ scalers
results = train_with_different_scalers(X_train, y_train, X_test, y_test, scalers_dict)

# สรุปผลการเปรียบเทียบ
print(f"\n5. Model Performance Comparison")
print("-" * 40)
print(f"{'Scaler':<12} {'Train R²':<10} {'Test R²':<10} {'Test MSE':<12}")
print("-" * 50)

for scaler_name, result in results.items():
    print(f"{scaler_name:<12} {result['train_r2']:<10.4f} {result['test_r2']:<10.4f} {result['test_mse']:<12.2f}")

# หา scaler ที่ดีที่สุด
best_scaler = max(results.keys(), key=lambda k: results[k]['test_r2'])
print(f"\nBest scaler: {best_scaler} (Test R² = {results[best_scaler]['test_r2']:.4f})")

print("\n6. Housing Dataset Example")
print("-" * 30)

# สร้างข้อมูล housing
X_house, y_house, house_features = create_housing_dataset()

print(f"Housing dataset: {X_house.shape[0]} samples, {X_house.shape[1]} features")
print(f"Features: {house_features}")

# แสดงสถิติ
print(f"\nFeature statistics:")
for i, name in enumerate(house_features):
    data = X_house[:, i]
    print(f"  {name}: mean={data.mean():.2f}, std={data.std():.2f}")

# แสดง correlation กับ target
correlations = []
for i in range(X_house.shape[1]):
    corr = torch.corrcoef(torch.stack([X_house[:, i], y_house.squeeze()]))[0, 1].item()
    correlations.append(corr)
    print(f"  {house_features[i]} vs price correlation: {corr:.3f}")

# Visualization
plot_feature_distributions(X_house, house_features, "Housing Dataset - Feature Distributions")

# Feature engineering
print(f"\n7. Feature Engineering")
print("-" * 25)

# สร้าง polynomial features
X_house_poly = engineer.create_polynomial_features(X_house[:, :2], degree=2)  # ใช้แค่ 2 features แรก
print(f"After polynomial features (degree 2): {X_house_poly.shape}")

# สร้าง log features สำหรับ area และ income
X_house_log = X_house.clone()
X_house_log[:, 0] = engineer.create_log_features(X_house[:, 0].unsqueeze(1)).squeeze()  # log(area)
X_house_log[:, 4] = engineer.create_log_features(X_house[:, 4].unsqueeze(1)).squeeze()  # log(income)

# เปรียบเทียบ original vs engineered features
house_datasets = {
    'Original': X_house,
    'Polynomial': X_house_poly,
    'Log-transformed': X_house_log
}

# แบ่งข้อมูล
split_idx = int(0.8 * len(X_house))
y_house_train, y_house_test = y_house[:split_idx], y_house[split_idx:]

print(f"\nComparing feature engineering approaches:")
print(f"{'Method':<15} {'Features':<10} {'Train R²':<10} {'Test R²':<10}")
print("-" * 50)

for method_name, X_data in house_datasets.items():
    X_train_method = X_data[:split_idx]
    X_test_method = X_data[split_idx:]
    
    # ใช้ Standard Scaler
    scaler = TorchStandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_method)
    X_test_scaled = scaler.transform(X_test_method)
    
    # Simple model
    model = nn.Sequential(
        nn.Linear(X_data.shape[1], 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 1)
    )
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Quick training
    train_dataset = TensorDataset(X_train_scaled, y_house_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    for epoch in range(100):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        train_pred = model(X_train_scaled)
        test_pred = model(X_test_scaled)
        
        from sklearn.metrics import r2_score
        train_r2 = r2_score(y_house_train.numpy(), train_pred.numpy())
        test_r2 = r2_score(y_house_test.numpy(), test_pred.numpy())
    
    print(f"{method_name:<15} {X_data.shape[1]:<10} {train_r2:<10.4f} {test_r2:<10.4f}")

print("\n" + "=" * 60)
print("สรุป Example 2: Feature Scaling และ Normalization")
print("=" * 60)
print("✓ Custom scalers implementation (Standard, MinMax, Robust)")
print("✓ Feature engineering และ polynomial features")
print("✓ Outlier detection และ removal")
print("✓ การเปรียบเทียบ scaling methods")
print("✓ การประยุกต์ใช้กับข้อมูลจริง (Housing dataset)")
print("✓ Model performance comparison")
print("✓ Visualization และ interpretation")
print("=" * 60)
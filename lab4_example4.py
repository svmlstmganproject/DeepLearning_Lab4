"""
Lab 8 Example 4: Model Interpretation
เรียนรู้การตีความและวิเคราะห์ regression models
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.datasets import make_regression, load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import shap
from captum.attr import IntegratedGradients, DeepLift, GradientShap, Occlusion
from captum.attr import LayerConductance, LayerActivation, LayerGradCam
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("Lab 8 Example 4: Model Interpretation")
print("=" * 60)

# 1. Interpretable Regression Model
class InterpretableRegression(nn.Module):
    """Regression model ที่สามารถตีความได้"""
    
    def __init__(self, input_size, hidden_sizes, output_size=1, 
                 activation='relu', dropout_rate=0.2):
        super(InterpretableRegression, self).__init__()
        
        self.input_size = input_size
        self.feature_names = None
        
        # Build network layers
        layers = []
        current_size = input_size
        
        for i, hidden_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(current_size, hidden_size))
            
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            
            current_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(current_size, output_size))
        
        self.network = nn.Sequential(*layers)
        
        # Store layer references for interpretation
        self.layers = []
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                self.layers.append(layer)
    
    def forward(self, x):
        return self.network(x)
    
    def get_layer_activations(self, x, layer_idx=0):
        """ดึง activations จาก layer ที่กำหนด"""
        activation = x
        
        layer_count = 0
        for module in self.network:
            activation = module(activation)
            if isinstance(module, nn.Linear):
                if layer_count == layer_idx:
                    return activation
                layer_count += 1
        
        return activation
    
    def get_weights_and_biases(self):
        """ดึง weights และ biases จากทุก layers"""
        weights_biases = []
        
        for layer in self.layers:
            weights_biases.append({
                'weights': layer.weight.data.clone(),
                'biases': layer.bias.data.clone() if layer.bias is not None else None
            })
        
        return weights_biases
    
    def set_feature_names(self, feature_names):
        """ตั้งชื่อ features สำหรับการตีความ"""
        self.feature_names = feature_names

# 2. Linear Regression with Interpretation
class LinearRegressionInterp(nn.Module):
    """Linear regression พร้อมการตีความ"""
    
    def __init__(self, input_size):
        super(LinearRegressionInterp, self).__init__()
        self.linear = nn.Linear(input_size, 1)
        self.feature_names = None
    
    def forward(self, x):
        return self.linear(x)
    
    def get_coefficients(self):
        """ดึงค่า coefficients"""
        return {
            'weights': self.linear.weight.data.clone().squeeze(),
            'bias': self.linear.bias.data.clone().item(),
            'feature_names': self.feature_names
        }
    
    def get_feature_importance(self):
        """คำนวณ feature importance จาก absolute weights"""
        weights = torch.abs(self.linear.weight.data.squeeze())
        if self.feature_names:
            return dict(zip(self.feature_names, weights.tolist()))
        else:
            return {f'feature_{i}': w.item() for i, w in enumerate(weights)}
    
    def predict_with_explanation(self, x):
        """ทำนายพร้อมอธิบายการตัดสินใจ"""
        with torch.no_grad():
            prediction = self.forward(x)
            
            # คำนวณ contribution ของแต่ละ feature
            weights = self.linear.weight.data.squeeze()
            bias = self.linear.bias.data.item()
            
            feature_contributions = x * weights
            
            return {
                'prediction': prediction,
                'feature_contributions': feature_contributions,
                'bias_contribution': bias,
                'weights': weights
            }
    
    def set_feature_names(self, feature_names):
        self.feature_names = feature_names

# 3. Feature Importance Analysis
class FeatureImportanceAnalyzer:
    """Class สำหรับวิเคราะห์ feature importance"""
    
    def __init__(self, model, feature_names=None):
        self.model = model
        self.feature_names = feature_names or [f'feature_{i}' for i in range(model.input_size)]
    
    def permutation_importance(self, X, y, n_repeats=10):
        """คำนวณ permutation importance"""
        
        self.model.eval()
        
        # Baseline performance
        with torch.no_grad():
            baseline_pred = self.model(X)
            baseline_mse = nn.MSELoss()(baseline_pred, y).item()
        
        importances = {}
        
        for feat_idx, feat_name in enumerate(self.feature_names):
            importance_scores = []
            
            for _ in range(n_repeats):
                # Create permuted version
                X_permuted = X.clone()
                perm_indices = torch.randperm(X.size(0))
                X_permuted[:, feat_idx] = X[perm_indices, feat_idx]
                
                # Calculate performance drop
                with torch.no_grad():
                    permuted_pred = self.model(X_permuted)
                    permuted_mse = nn.MSELoss()(permuted_pred, y).item()
                
                importance = (permuted_mse - baseline_mse) / baseline_mse
                importance_scores.append(importance)
            
            importances[feat_name] = {
                'mean': np.mean(importance_scores),
                'std': np.std(importance_scores),
                'scores': importance_scores
            }
        
        return importances
    
    def gradient_importance(self, X, y):
        """คำนวณ importance จาก gradients"""
        
        self.model.eval()
        X.requires_grad_(True)
        
        # Forward pass
        predictions = self.model(X)
        loss = nn.MSELoss()(predictions, y)
        
        # Backward pass
        loss.backward()
        
        # Calculate importance as mean absolute gradient
        gradients = X.grad.abs().mean(dim=0)
        
        importance = {}
        for i, feat_name in enumerate(self.feature_names):
            importance[feat_name] = gradients[i].item()
        
        X.requires_grad_(False)
        return importance
    
    def integrated_gradients_importance(self, X, baseline=None):
        """คำนวณ importance ด้วย Integrated Gradients"""
        
        if baseline is None:
            baseline = torch.zeros_like(X[0]).unsqueeze(0)
        
        ig = IntegratedGradients(self.model)
        attributions = ig.attribute(X, baseline, n_steps=50)
        
        # Average attributions across samples
        mean_attributions = attributions.abs().mean(dim=0)
        
        importance = {}
        for i, feat_name in enumerate(self.feature_names):
            importance[feat_name] = mean_attributions[i].item()
        
        return importance, attributions

# 4. Model Visualization Tools
class ModelVisualizer:
    """Class สำหรับ visualize models"""
    
    def __init__(self, model, feature_names=None):
        self.model = model
        self.feature_names = feature_names
    
    def plot_feature_importance(self, importance_dict, title="Feature Importance", 
                               method_name="Importance"):
        """วาดกราฟ feature importance"""
        
        if isinstance(list(importance_dict.values())[0], dict):
            # Handle permutation importance format
            features = list(importance_dict.keys())
            values = [importance_dict[f]['mean'] for f in features]
            errors = [importance_dict[f]['std'] for f in features]
        else:
            # Handle simple importance format
            features = list(importance_dict.keys())
            values = list(importance_dict.values())
            errors = None
        
        plt.figure(figsize=(12, 6))
        
        # Sort by importance
        sorted_pairs = sorted(zip(features, values), key=lambda x: abs(x[1]), reverse=True)
        features_sorted, values_sorted = zip(*sorted_pairs)
        
        if errors:
            errors_sorted = [importance_dict[f]['std'] for f in features_sorted]
            plt.barh(features_sorted, values_sorted, xerr=errors_sorted, 
                    capsize=5, alpha=0.7)
        else:
            plt.barh(features_sorted, values_sorted, alpha=0.7)
        
        plt.xlabel(method_name)
        plt.ylabel('Features')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_linear_coefficients(self, model, scale_by_std=True, X_std=None):
        """วาดกราฟ linear coefficients"""
        
        if not isinstance(model, LinearRegressionInterp):
            print("This visualization is only for LinearRegressionInterp models")
            return
        
        coefs = model.get_coefficients()
        weights = coefs['weights']
        feature_names = coefs['feature_names'] or [f'Feature_{i}' for i in range(len(weights))]
        
        if scale_by_std and X_std is not None:
            # Scale coefficients by feature standard deviation
            weights = weights * X_std
        
        plt.figure(figsize=(12, 6))
        
        colors = ['red' if w < 0 else 'blue' for w in weights]
        plt.barh(feature_names, weights, color=colors, alpha=0.7)
        
        plt.xlabel('Coefficient Value')
        plt.ylabel('Features')
        plt.title('Linear Regression Coefficients')
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Print coefficient summary
        print("Coefficient Summary:")
        for name, weight in zip(feature_names, weights):
            print(f"  {name}: {weight:.4f}")
        print(f"  Bias: {coefs['bias']:.4f}")
    
    def plot_prediction_explanation(self, model, X_sample, y_sample=None, 
                                  sample_idx=0):
        """อธิบายการทำนายสำหรับ sample เดียว"""
        
        if not isinstance(model, LinearRegressionInterp):
            print("This explanation is only for LinearRegressionInterp models")
            return
        
        if len(X_sample.shape) == 1:
            X_sample = X_sample.unsqueeze(0)
        
        explanation = model.predict_with_explanation(X_sample[sample_idx:sample_idx+1])
        
        feature_names = model.feature_names or [f'Feature_{i}' for i in range(X_sample.shape[1])]
        feature_values = X_sample[sample_idx]
        contributions = explanation['feature_contributions'].squeeze()
        prediction = explanation['prediction'].item()
        bias = explanation['bias_contribution']
        
        # สร้างกราฟ
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Feature values
        ax1.barh(feature_names, feature_values, alpha=0.7, color='skyblue')
        ax1.set_xlabel('Feature Value')
        ax1.set_ylabel('Features')
        ax1.set_title(f'Sample {sample_idx} - Feature Values')
        ax1.grid(True, alpha=0.3)
        
        # Feature contributions
        colors = ['red' if c < 0 else 'green' for c in contributions]
        ax2.barh(feature_names, contributions, color=colors, alpha=0.7)
        ax2.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        ax2.set_xlabel('Contribution to Prediction')
        ax2.set_ylabel('Features')
        ax2.set_title(f'Sample {sample_idx} - Feature Contributions')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print summary
        print(f"Prediction Explanation for Sample {sample_idx}:")
        print(f"  Predicted value: {prediction:.4f}")
        if y_sample is not None:
            actual = y_sample[sample_idx].item()
            print(f"  Actual value: {actual:.4f}")
            print(f"  Error: {abs(prediction - actual):.4f}")
        
        print(f"  Bias contribution: {bias:.4f}")
        print(f"  Feature contributions:")
        
        for name, value, contrib in zip(feature_names, feature_values, contributions):
            print(f"    {name}: {value:.3f} × weight = {contrib:.4f}")
        
        total_contrib = contributions.sum().item() + bias
        print(f"  Total: {total_contrib:.4f}")
    
    def plot_residuals_analysis(self, model, X, y):
        """วิเคราะห์ residuals"""
        
        model.eval()
        with torch.no_grad():
            predictions = model(X).squeeze()
        
        residuals = y.squeeze() - predictions
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Residuals vs Predictions
        axes[0, 0].scatter(predictions, residuals, alpha=0.6)
        axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        axes[0, 0].set_xlabel('Predicted Values')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].set_title('Residuals vs Predicted Values')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Actual vs Predicted
        min_val = min(y.min().item(), predictions.min().item())
        max_val = max(y.max().item(), predictions.max().item())
        
        axes[0, 1].scatter(y.squeeze(), predictions, alpha=0.6)
        axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7)
        axes[0, 1].set_xlabel('Actual Values')
        axes[0, 1].set_ylabel('Predicted Values')
        axes[0, 1].set_title('Actual vs Predicted Values')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Histogram of residuals
        axes[1, 0].hist(residuals.numpy(), bins=30, alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(x=0, color='red', linestyle='--', alpha=0.7)
        axes[1, 0].set_xlabel('Residuals')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Distribution of Residuals')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Q-Q plot (approximate)
        from scipy import stats
        residuals_np = residuals.numpy()
        stats.probplot(residuals_np, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot (Residuals)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Calculate residual statistics
        rmse = torch.sqrt(torch.mean(residuals**2)).item()
        mae = torch.mean(torch.abs(residuals)).item()
        r2 = r2_score(y.numpy(), predictions.numpy())
        
        print("Residual Analysis:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  R²: {r2:.4f}")
        print(f"  Mean residual: {residuals.mean().item():.6f}")
        print(f"  Std of residuals: {residuals.std().item():.4f}")

# 5. SHAP Integration (if available)
class SHAPAnalyzer:
    """SHAP analysis for model interpretation"""
    
    def __init__(self, model, X_background=None):
        self.model = model
        self.X_background = X_background
        
        # Wrapper function for SHAP
        def model_predict(x):
            self.model.eval()
            with torch.no_grad():
                if isinstance(x, np.ndarray):
                    x = torch.tensor(x, dtype=torch.float32)
                return self.model(x).numpy()
        
        self.predict_fn = model_predict
    
    def explain_with_shap(self, X_explain, feature_names=None):
        """สร้าง SHAP explanations"""
        
        try:
            # Create SHAP explainer
            if self.X_background is not None:
                explainer = shap.KernelExplainer(self.predict_fn, self.X_background.numpy())
            else:
                explainer = shap.Explainer(self.predict_fn)
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(X_explain.numpy())
            
            return shap_values, explainer
            
        except Exception as e:
            print(f"SHAP analysis failed: {e}")
            print("SHAP package might not be installed or compatible")
            return None, None
    
    def plot_shap_summary(self, shap_values, X_explain, feature_names=None):
        """วาดกราฟ SHAP summary"""
        
        if shap_values is None:
            print("No SHAP values available")
            return
        
        try:
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X_explain.numpy(), 
                             feature_names=feature_names, show=False)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"SHAP plotting failed: {e}")

# 6. Data Generation for Testing
def create_interpretable_dataset(n_samples=1000, n_features=8, noise=0.1, 
                               feature_names=None, relationship_type='linear'):
    """สร้างข้อมูลที่สามารถตีความได้"""
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    if feature_names is None:
        feature_names = [f'Feature_{i+1}' for i in range(n_features)]
    
    # สร้าง features ที่มีความหมาย
    X = torch.randn(n_samples, n_features)
    
    if relationship_type == 'linear':
        # Linear relationship
        true_weights = torch.tensor([2.0, -1.5, 0.8, -0.3, 1.2, -0.7, 0.5, 0.0])[:n_features]
        y = torch.matmul(X, true_weights) + 3.0  # bias = 3.0
        
    elif relationship_type == 'polynomial':
        # Polynomial relationship
        y = (2.0 * X[:, 0] - 1.5 * X[:, 1] + 0.5 * X[:, 0]**2 - 0.3 * X[:, 1]**2 +
             0.8 * X[:, 2] + 1.2 * X[:, 3] - 0.7 * X[:, 4])
        
        if n_features > 5:
            y += 0.5 * torch.sum(X[:, 5:], dim=1)
    
    elif relationship_type == 'interaction':
        # With interaction terms
        y = (2.0 * X[:, 0] - 1.5 * X[:, 1] + 0.8 * X[:, 2] +
             1.5 * X[:, 0] * X[:, 1] +  # interaction term
             0.7 * X[:, 2] * X[:, 3])   # another interaction
        
        if n_features > 4:
            y += 0.5 * torch.sum(X[:, 4:], dim=1)
    
    # Add noise
    y += torch.normal(0, noise, size=(n_samples,))
    
    # Add some feature correlations
    if n_features >= 4:
        X[:, 1] = 0.7 * X[:, 0] + 0.3 * torch.randn(n_samples)  # correlated feature
        X[:, 3] = -0.5 * X[:, 2] + 0.5 * torch.randn(n_samples)  # negatively correlated
    
    return X, y.unsqueeze(1), feature_names, true_weights if relationship_type == 'linear' else None

def create_housing_dataset_detailed():
    """สร้างข้อมูล housing ที่มีรายละเอียด"""
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    n_samples = 800
    
    # Create meaningful features
    features = {
        'area': torch.normal(120, 40, (n_samples,)),  # พื้นที่ (ตร.ม.)
        'bedrooms': torch.randint(1, 6, (n_samples,)).float(),  # ห้องนอน
        'bathrooms': torch.randint(1, 4, (n_samples,)).float(),  # ห้องน้ำ
        'age': torch.randint(0, 50, (n_samples,)).float(),  # อายุบ้าน
        'distance_cbd': torch.normal(15, 8, (n_samples,)),  # ระยะทางจาก CBD
        'school_rating': torch.normal(7, 2, (n_samples,)),  # เรตติ้งโรงเรียน
        'crime_rate': torch.normal(5, 2, (n_samples,)),  # อัตราอาชญากรรม
        'garage_spaces': torch.randint(0, 4, (n_samples,)).float()  # ที่จอดรถ
    }
    
    # Clip values to realistic ranges
    features['area'] = torch.clamp(features['area'], 50, 300)
    features['distance_cbd'] = torch.clamp(features['distance_cbd'], 1, 50)
    features['school_rating'] = torch.clamp(features['school_rating'], 1, 10)
    features['crime_rate'] = torch.clamp(features['crime_rate'], 0, 15)
    
    # Create target (house price) with realistic relationships
    price = (
        features['area'] * 3000 +                    # area effect
        features['bedrooms'] * 20000 +               # bedroom premium
        features['bathrooms'] * 15000 +              # bathroom premium
        -features['age'] * 1000 +                    # age depreciation
        -features['distance_cbd'] * 2000 +           # location premium
        features['school_rating'] * 10000 +          # school quality
        -features['crime_rate'] * 3000 +             # safety factor
        features['garage_spaces'] * 8000 +           # garage value
        torch.normal(0, 15000, (n_samples,))         # noise
    )
    
    # Add some interaction effects
    price += features['area'] * features['school_rating'] * 100  # area-school interaction
    price -= features['age'] * features['distance_cbd'] * 50     # age-distance interaction
    
    # Combine features
    feature_names = list(features.keys())
    X = torch.stack(list(features.values()), dim=1)
    y = price.unsqueeze(1)
    
    return X, y, feature_names

# 7. Training Function
def train_interpretable_model(model, X_train, y_train, X_val, y_val, 
                            num_epochs=200, learning_rate=0.001):
    """Training function สำหรับ interpretable models"""
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        
        train_pred = model(X_train)
        train_loss = criterion(train_pred, y_train)
        train_loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val)
        
        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())
        
        if (epoch + 1) % 50 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss.item():.6f}, Val Loss: {val_loss.item():.6f}')
    
    return train_losses, val_losses

# ทดสอบการใช้งาน
print("\n1. Linear Regression Interpretation")
print("-" * 40)

# สร้างข้อมูลแบบ linear
X_linear, y_linear, feature_names_linear, true_weights = create_interpretable_dataset(
    n_samples=800, n_features=6, noise=0.2, relationship_type='linear'
)

print(f"Dataset shape: {X_linear.shape}")
print(f"Feature names: {feature_names_linear}")
print(f"True weights: {true_weights}")

# แบ่งข้อมูล
split_idx = int(0.8 * len(X_linear))
X_train, X_test = X_linear[:split_idx], X_linear[split_idx:]
y_train, y_test = y_linear[:split_idx], y_linear[split_idx:]

# Standardize features
scaler = StandardScaler()
X_train_scaled = torch.tensor(scaler.fit_transform(X_train), dtype=torch.float32)
X_test_scaled = torch.tensor(scaler.transform(X_test), dtype=torch.float32)

# สร้าง linear model
linear_model = LinearRegressionInterp(input_size=6)
linear_model.set_feature_names(feature_names_linear)

# Training
print("Training linear model...")
train_losses, val_losses = train_interpretable_model(
    linear_model, X_train_scaled, y_train, X_test_scaled, y_test,
    num_epochs=300, learning_rate=0.01
)

# Model interpretation
print("\n1.1 Linear Model Coefficients")
print("-" * 30)

visualizer = ModelVisualizer(linear_model, feature_names_linear)

# Plot coefficients
feature_stds = torch.tensor(scaler.scale_, dtype=torch.float32)
visualizer.plot_linear_coefficients(linear_model, scale_by_std=True, X_std=feature_stds)

# Compare with true weights
learned_coefs = linear_model.get_coefficients()
learned_weights = learned_coefs['weights']

print("Weight comparison:")
print(f"{'Feature':<12} {'True':<10} {'Learned':<10} {'Error':<10}")
print("-" * 45)
for i, name in enumerate(feature_names_linear):
    true_w = true_weights[i].item()
    learned_w = (learned_weights[i] * feature_stds[i]).item()  # unscale
    error = abs(true_w - learned_w)
    print(f"{name:<12} {true_w:<10.3f} {learned_w:<10.3f} {error:<10.3f}")

# Prediction explanation
print("\n1.2 Prediction Explanation")
print("-" * 25)

visualizer.plot_prediction_explanation(linear_model, X_test_scaled, y_test, sample_idx=0)
visualizer.plot_prediction_explanation(linear_model, X_test_scaled, y_test, sample_idx=5)

# Residuals analysis
print("\n1.3 Residuals Analysis")
print("-" * 20)

visualizer.plot_residuals_analysis(linear_model, X_test_scaled, y_test)

print("\n2. Deep Model Interpretation")
print("-" * 35)

# สร้างข้อมูลที่ซับซ้อนกว่า
X_complex, y_complex, feature_names_complex, _ = create_interpretable_dataset(
    n_samples=1000, n_features=8, noise=0.15, relationship_type='interaction'
)

# แบ่งข้อมูล
split_idx = int(0.8 * len(X_complex))
X_train_complex = X_complex[:split_idx]
X_test_complex = X_complex[split_idx:]
y_train_complex = y_complex[:split_idx]
y_test_complex = y_complex[split_idx:]

# Standardize
scaler_complex = StandardScaler()
X_train_complex_scaled = torch.tensor(scaler_complex.fit_transform(X_train_complex), dtype=torch.float32)
X_test_complex_scaled = torch.tensor(scaler_complex.transform(X_test_complex), dtype=torch.float32)

# สร้าง deep model
deep_model = InterpretableRegression(
    input_size=8,
    hidden_sizes=[64, 32, 16],
    output_size=1,
    dropout_rate=0.2
)
deep_model.set_feature_names(feature_names_complex)

print("Training deep model...")
deep_train_losses, deep_val_losses = train_interpretable_model(
    deep_model, X_train_complex_scaled, y_train_complex, 
    X_test_complex_scaled, y_test_complex,
    num_epochs=200, learning_rate=0.001
)

# Feature importance analysis
print("\n2.1 Feature Importance Analysis")
print("-" * 35)

analyzer = FeatureImportanceAnalyzer(deep_model, feature_names_complex)

# Permutation importance
print("Calculating permutation importance...")
perm_importance = analyzer.permutation_importance(X_test_complex_scaled, y_test_complex, n_repeats=5)

# Gradient importance
print("Calculating gradient importance...")
grad_importance = analyzer.gradient_importance(X_test_complex_scaled, y_test_complex)

# Integrated gradients
print("Calculating integrated gradients importance...")
ig_importance, ig_attributions = analyzer.integrated_gradients_importance(X_test_complex_scaled[:50])

# Visualization
deep_visualizer = ModelVisualizer(deep_model, feature_names_complex)

deep_visualizer.plot_feature_importance(perm_importance, 
                                      "Permutation Feature Importance", 
                                      "Relative MSE Increase")

deep_visualizer.plot_feature_importance(grad_importance,
                                      "Gradient-based Feature Importance",
                                      "Mean Absolute Gradient")

deep_visualizer.plot_feature_importance(ig_importance,
                                      "Integrated Gradients Importance", 
                                      "Attribution Score")

# Residuals analysis
print("\n2.2 Deep Model Residuals Analysis")
print("-" * 35)

deep_visualizer.plot_residuals_analysis(deep_model, X_test_complex_scaled, y_test_complex)

print("\n3. Housing Dataset Interpretation")
print("-" * 35)

# สร้างข้อมูล housing
X_house, y_house, house_features = create_housing_dataset_detailed()

print(f"Housing dataset: {X_house.shape[0]} samples, {len(house_features)} features")
print(f"Features: {house_features}")

# Statistics
print(f"\nHousing price statistics:")
print(f"  Mean: ${y_house.mean().item():,.0f}")
print(f"  Std: ${y_house.std().item():,.0f}")
print(f"  Min: ${y_house.min().item():,.0f}")
print(f"  Max: ${y_house.max().item():,.0f}")

# แบ่งข้อมูล
split_idx = int(0.8 * len(X_house))
X_train_house = X_house[:split_idx]
X_test_house = X_house[split_idx:]
y_train_house = y_house[:split_idx]
y_test_house = y_house[split_idx:]

# Standardize
scaler_house = StandardScaler()
X_train_house_scaled = torch.tensor(scaler_house.fit_transform(X_train_house), dtype=torch.float32)
X_test_house_scaled = torch.tensor(scaler_house.transform(X_test_house), dtype=torch.float32)

# สร้าง model สำหรับ housing
housing_model = InterpretableRegression(
    input_size=len(house_features),
    hidden_sizes=[32, 16],
    output_size=1,
    dropout_rate=0.1
)
housing_model.set_feature_names(house_features)

print("Training housing model...")
housing_train_losses, housing_val_losses = train_interpretable_model(
    housing_model, X_train_house_scaled, y_train_house,
    X_test_house_scaled, y_test_house,
    num_epochs=150, learning_rate=0.001
)

# Feature importance for housing
print("\n3.1 Housing Feature Importance")
print("-" * 30)

housing_analyzer = FeatureImportanceAnalyzer(housing_model, house_features)
housing_perm_importance = housing_analyzer.permutation_importance(
    X_test_house_scaled, y_test_house, n_repeats=3
)

housing_visualizer = ModelVisualizer(housing_model, house_features)
housing_visualizer.plot_feature_importance(housing_perm_importance,
                                         "Housing Price - Feature Importance",
                                         "Relative MSE Increase")

# Individual predictions analysis
print("\n3.2 Individual Prediction Analysis")
print("-" * 35)

# สร้าง linear model เพื่อเปรียบเทียบ
housing_linear = LinearRegressionInterp(input_size=len(house_features))
housing_linear.set_feature_names(house_features)

# Train linear model
linear_train_losses, _ = train_interpretable_model(
    housing_linear, X_train_house_scaled, y_train_house,
    X_test_house_scaled, y_test_house,
    num_epochs=300, learning_rate=0.01
)

# เปรียบเทียบ predictions
housing_model.eval()
housing_linear.eval()

with torch.no_grad():
    deep_pred = housing_model(X_test_house_scaled)
    linear_pred = housing_linear(X_test_house_scaled)

# Performance comparison
deep_mse = nn.MSELoss()(deep_pred, y_test_house).item()
linear_mse = nn.MSELoss()(linear_pred, y_test_house).item()

deep_r2 = r2_score(y_test_house.numpy(), deep_pred.numpy())
linear_r2 = r2_score(y_test_house.numpy(), linear_pred.numpy())

print("Model Performance Comparison:")
print(f"{'Model':<15} {'MSE':<15} {'RMSE':<15} {'R²':<10}")
print("-" * 60)
print(f"{'Deep Model':<15} {deep_mse:<15.0f} {np.sqrt(deep_mse):<15.0f} {deep_r2:<10.4f}")
print(f"{'Linear Model':<15} {linear_mse:<15.0f} {np.sqrt(linear_mse):<15.0f} {linear_r2:<10.4f}")

# Linear coefficients interpretation
housing_linear_viz = ModelVisualizer(housing_linear, house_features)
feature_stds_house = torch.tensor(scaler_house.scale_, dtype=torch.float32)
housing_linear_viz.plot_linear_coefficients(housing_linear, scale_by_std=True, X_std=feature_stds_house)

print("\n4. Model Comparison Summary")
print("-" * 30)

# สรุปผลการเปรียบเทียบ models ทั้งหมด
models_summary = {
    'Linear (Linear Data)': {
        'model': linear_model,
        'test_data': (X_test_scaled, y_test),
        'interpretability': 'High'
    },
    'Deep (Complex Data)': {
        'model': deep_model,
        'test_data': (X_test_complex_scaled, y_test_complex),
        'interpretability': 'Medium'
    },
    'Deep (Housing)': {
        'model': housing_model,
        'test_data': (X_test_house_scaled, y_test_house),
        'interpretability': 'Medium'
    },
    'Linear (Housing)': {
        'model': housing_linear,
        'test_data': (X_test_house_scaled, y_test_house),
        'interpretability': 'High'
    }
}

print(f"{'Model':<20} {'RMSE':<12} {'R²':<10} {'Interpretability':<15}")
print("-" * 65)

for model_name, info in models_summary.items():
    model = info['model']
    X_test_data, y_test_data = info['test_data']
    interpretability = info['interpretability']
    
    model.eval()
    with torch.no_grad():
        pred = model(X_test_data)
        mse = nn.MSELoss()(pred, y_test_data).item()
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test_data.numpy(), pred.numpy())
    
    print(f"{model_name:<20} {rmse:<12.4f} {r2:<10.4f} {interpretability:<15}")

print("\n" + "=" * 60)
print("สรุป Example 4: Model Interpretation")
print("=" * 60)
print("✓ Linear regression interpretation และ coefficient analysis")
print("✓ Feature importance analysis (permutation, gradient, integrated gradients)")
print("✓ Individual prediction explanations")
print("✓ Residuals analysis และ model diagnostics")
print("✓ Deep model interpretation techniques")
print("✓ Real-world application (housing price prediction)")
print("✓ Model comparison และ interpretability trade-offs")
print("=" * 60)
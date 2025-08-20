if feature_columns is not None:
            self.feature_indices = feature_columns
        else:
            self.feature_indices = list(range(self.data.shape[1]))
        
        self.sequences = self._create_sequences()
    
    def _create_sequences(self):
        """สร้าง sequences สำหรับ training"""
        sequences = []
        
        for i in range(len(self.data) - self.sequence_length - self.prediction_horizon + 1):
            # Input sequence (features)
            input_seq = self.data[i:i + self.sequence_length, self.feature_indices]
            
            # Target sequence
            target_start = i + self.sequence_length
            target_end = target_start + self.prediction_horizon
            
            if self.prediction_horizon == 1:
                # Single step prediction
                target_seq = self.data[target_start, self.target_idx]
            else:
                # Multi-step prediction
                target_seq = self.data[target_start:target_end, self.target_idx]
            
            sequences.append({
                'input': torch.tensor(input_seq, dtype=torch.float32),
                'target': torch.tensor(target_seq, dtype=torch.float32),
                'start_time': i
            })
        
        return sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx]['input'], self.sequences[idx]['target']

# 2. LSTM Model for Time Series
class LSTMTimeSeriesModel(nn.Module):
    """LSTM model สำหรับ time series prediction"""
    
    def __init__(self, input_size, hidden_size, num_layers, output_size, 
                 dropout=0.2, bidirectional=False):
        super(LSTMTimeSeriesModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Calculate the input size for the linear layer
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size)
        )
        
    def forward(self, x):
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # ใช้ output จาก timestep สุดท้าย
        if self.bidirectional:
            # รวม forward และ backward hidden states
            last_output = lstm_out[:, -1, :]
        else:
            last_output = lstm_out[:, -1, :]
        
        # ผ่าน fully connected layers
        output = self.fc(last_output)
        
        return output

# 3. GRU Model for Time Series
class GRUTimeSeriesModel(nn.Module):
    """GRU model สำหรับ time series prediction"""
    
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(GRUTimeSeriesModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )
    
    def forward(self, x):
        gru_out, h_n = self.gru(x)
        last_output = gru_out[:, -1, :]
        output = self.fc(last_output)
        return output

# 4. CNN-LSTM Hybrid Model
class CNN_LSTM_Model(nn.Module):
    """CNN-LSTM hybrid model สำหรับ time series"""
    
    def __init__(self, input_size, sequence_length, hidden_size, output_size):
        super(CNN_LSTM_Model, self).__init__()
        
        # CNN layers สำหรับ feature extraction
        self.conv1d = nn.Sequential(
            nn.Conv1d(input_size, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=hidden_size,
            num_layers=2,
            dropout=0.2,
            batch_first=True
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        # CNN expects: (batch_size, input_size, sequence_length)
        x = x.transpose(1, 2)
        
        # CNN feature extraction
        conv_out = self.conv1d(x)
        
        # Back to LSTM format: (batch_size, sequence_length, features)
        conv_out = conv_out.transpose(1, 2)
        
        # LSTM
        lstm_out, _ = self.lstm(conv_out)
        
        # Use last output
        last_output = lstm_out[:, -1, :]
        output = self.fc(last_output)
        
        return output

# 5. Attention Mechanism
class AttentionLayer(nn.Module):
    """Attention layer สำหรับ time series"""
    
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.attention = nn.Linear(hidden_size, 1)
    
    def forward(self, lstm_output):
        # lstm_output: (batch_size, sequence_length, hidden_size)
        
        # คำนวณ attention weights
        attention_weights = self.attention(lstm_output)  # (batch_size, seq_len, 1)
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        # Apply attention
        attended_output = torch.sum(lstm_output * attention_weights, dim=1)
        
        return attended_output, attention_weights

class LSTMWithAttention(nn.Module):
    """LSTM with Attention mechanism"""
    
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(LSTMWithAttention, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.attention = AttentionLayer(hidden_size)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attended_out, attention_weights = self.attention(lstm_out)
        output = self.fc(attended_out)
        
        return output, attention_weights

# 6. Data Generation Functions
def generate_synthetic_timeseries(n_points=1000, components=None):
    """สร้าง synthetic time series data"""
    
    if components is None:
        components = ['trend', 'seasonal', 'noise']
    
    t = np.linspace(0, 4 * np.pi, n_points)
    
    # Initialize series
    series = np.zeros(n_points)
    
    # Add components
    if 'trend' in components:
        trend = 0.02 * t + 0.001 * t**2
        series += trend
    
    if 'seasonal' in components:
        seasonal = 2 * np.sin(t) + 1.5 * np.cos(2 * t) + 0.5 * np.sin(4 * t)
        series += seasonal
    
    if 'noise' in components:
        noise = np.random.normal(0, 0.5, n_points)
        series += noise
    
    # Add some non-linearity
    if 'nonlinear' in components:
        nonlinear = 0.1 * np.sin(10 * t) * np.exp(-0.1 * t)
        series += nonlinear
    
    return series, t

def generate_stock_like_data(n_points=1000, initial_price=100):
    """สร้างข้อมูลคล้าย stock price"""
    
    np.random.seed(42)
    
    # Parameters
    dt = 1/252  # daily data, 252 trading days per year
    mu = 0.1    # drift
    sigma = 0.2 # volatility
    
    # Generate random walks
    returns = np.random.normal(mu * dt, sigma * np.sqrt(dt), n_points)
    
    # Add some autocorrelation
    for i in range(1, len(returns)):
        returns[i] += 0.1 * returns[i-1]
    
    # Convert to prices
    prices = np.zeros(n_points + 1)
    prices[0] = initial_price
    
    for i in range(n_points):
        prices[i+1] = prices[i] * np.exp(returns[i])
    
    # Create additional features
    data = pd.DataFrame({
        'price': prices[1:],
        'returns': returns,
        'volume': np.random.lognormal(10, 0.5, n_points),
        'high': prices[1:] * (1 + np.abs(np.random.normal(0, 0.02, n_points))),
        'low': prices[1:] * (1 - np.abs(np.random.normal(0, 0.02, n_points)))
    })
    
    # Technical indicators
    data['sma_5'] = data['price'].rolling(window=5).mean()
    data['sma_20'] = data['price'].rolling(window=20).mean()
    data['volatility'] = data['returns'].rolling(window=20).std()
    
    return data.fillna(method='bfill')

def generate_weather_data(n_points=365*2):
    """สร้างข้อมุลสภาพอากาศ"""
    
    np.random.seed(42)
    
    # Time index
    dates = pd.date_range(start='2022-01-01', periods=n_points, freq='D')
    day_of_year = dates.dayofyear
    
    # Temperature with seasonal pattern
    temperature = (20 + 15 * np.sin(2 * np.pi * day_of_year / 365 - np.pi/2) + 
                  np.random.normal(0, 3, n_points))
    
    # Humidity (inversely related to temperature)
    humidity = (70 - 0.5 * (temperature - 20) + 
               np.random.normal(0, 10, n_points))
    humidity = np.clip(humidity, 20, 100)
    
    # Pressure
    pressure = (1013 + 5 * np.sin(2 * np.pi * day_of_year / 365) + 
               np.random.normal(0, 8, n_points))
    
    # Wind speed
    wind_speed = np.abs(5 + 3 * np.sin(2 * np.pi * day_of_year / 365 + np.pi/4) + 
                       np.random.normal(0, 2, n_points))
    
    data = pd.DataFrame({
        'temperature': temperature,
        'humidity': humidity,
        'pressure': pressure,
        'wind_speed': wind_speed,
        'date': dates
    })
    
    return data

# 7. Training and Evaluation Functions
def train_timeseries_model(model, train_loader, val_loader, num_epochs=100, 
                          learning_rate=0.001, patience=10):
    """Training function สำหรับ time series models"""
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            
            # Handle different model outputs
            if hasattr(model, 'attention'):
                outputs, _ = model(batch_x)
            else:
                outputs = model(batch_x)
            
            # Ensure target shape matches output shape
            if len(batch_y.shape) == 1:
                batch_y = batch_y.unsqueeze(1)
            
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                if hasattr(model, 'attention'):
                    outputs, _ = model(batch_x)
                else:
                    outputs = model(batch_x)
                
                if len(batch_y.shape) == 1:
                    batch_y = batch_y.unsqueeze(1)
                
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')
    
    return train_losses, val_losses

def evaluate_timeseries_model(model, test_loader, scaler=None):
    """ประเมินผล time series model"""
    
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            if hasattr(model, 'attention'):
                outputs, _ = model(batch_x)
            else:
                outputs = model(batch_x)
            
            predictions.extend(outputs.numpy())
            actuals.extend(batch_y.numpy())
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Inverse transform if scaler provided
    if scaler is not None:
        predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        actuals = scaler.inverse_transform(actuals.reshape(-1, 1)).flatten()
    
    # Calculate metrics
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actuals, predictions)
    
    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((actuals - predictions) / (actuals + 1e-8))) * 100
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'predictions': predictions,
        'actuals': actuals
    }

# 8. Visualization Functions
def plot_timeseries_prediction(actual, predicted, title="Time Series Prediction", 
                              train_size=None, figsize=(15, 6)):
    """วาดกราฟเปรียบเทียบ actual vs predicted"""
    
    plt.figure(figsize=figsize)
    
    time_steps = range(len(actual))
    
    if train_size is not None:
        # แสดง train/test split
        plt.axvline(x=train_size, color='black', linestyle='--', alpha=0.7, label='Train/Test Split')
    
    plt.plot(time_steps, actual, label='Actual', color='blue', alpha=0.7)
    plt.plot(time_steps, predicted, label='Predicted', color='red', alpha=0.7)
    
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_attention_weights(attention_weights, title="Attention Weights"):
    """วาดกราฟ attention weights"""
    
    plt.figure(figsize=(12, 6))
    
    # เลือก sample หนึ่งตัวจาก batch
    if len(attention_weights.shape) == 3:
        weights = attention_weights[0, :, 0].detach().numpy()
    else:
        weights = attention_weights.detach().numpy()
    
    plt.plot(weights, marker='o')
    plt.title(title)
    plt.xlabel('Time Step')
    plt.ylabel('Attention Weight')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# ทดสอบการใช้งาน
print("\n1. สร้าง Synthetic Time Series Data")
print("-" * 40)

# สร้างข้อมูล synthetic
synthetic_series, time_index = generate_synthetic_timeseries(
    n_points=1000, 
    components=['trend', 'seasonal', 'noise', 'nonlinear']
)

print(f"Synthetic series length: {len(synthetic_series)}")
print(f"Series statistics: mean={np.mean(synthetic_series):.3f}, std={np.std(synthetic_series):.3f}")

# Visualization
plt.figure(figsize=(15, 8))

plt.subplot(2, 2, 1)
plt.plot(time_index, synthetic_series)
plt.title('Complete Synthetic Time Series')
plt.xlabel('Time')
plt.ylabel('Value')
plt.grid(True, alpha=0.3)

# แสดง components แยกกัน
components_data = {}
for component in ['trend', 'seasonal', 'noise']:
    component_series, _ = generate_synthetic_timeseries(
        n_points=1000, components=[component]
    )
    components_data[component] = component_series

plt.subplot(2, 2, 2)
plt.plot(time_index, components_data['trend'])
plt.title('Trend Component')
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 3)
plt.plot(time_index, components_data['seasonal'])
plt.title('Seasonal Component')
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 4)
plt.plot(time_index, components_data['noise'])
plt.title('Noise Component')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n2. เตรียมข้อมูลสำหรับ Time Series Prediction")
print("-" * 50)

# พารามิเตอร์
sequence_length = 50
prediction_horizon = 1
train_ratio = 0.8

# แปลงเป็น DataFrame
df_synthetic = pd.DataFrame({'value': synthetic_series})

# สร้าง additional features
df_synthetic['lag_1'] = df_synthetic['value'].shift(1)
df_synthetic['lag_5'] = df_synthetic['value'].shift(5)
df_synthetic['rolling_mean_10'] = df_synthetic['value'].rolling(window=10).mean()
df_synthetic['rolling_std_10'] = df_synthetic['value'].rolling(window=10).std()

# Drop NaN values
df_synthetic = df_synthetic.dropna()

print(f"Dataset shape after feature engineering: {df_synthetic.shape}")
print(f"Features: {list(df_synthetic.columns)}")

# Scaling
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df_synthetic)

# สร้าง dataset
dataset = TimeSeriesDataset(
    data=scaled_data,
    sequence_length=sequence_length,
    prediction_horizon=prediction_horizon,
    target_column=0,  # 'value' column
    feature_columns=list(range(scaled_data.shape[1]))
)

print(f"Number of sequences: {len(dataset)}")

# แบ่งข้อมูล train/val/test
train_size = int(train_ratio * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset = torch.utils.data.Subset(dataset, range(train_size))
val_dataset = torch.utils.data.Subset(dataset, range(train_size, train_size + val_size))
test_dataset = torch.utils.data.Subset(dataset, range(train_size + val_size, len(dataset)))

print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

# สร้าง DataLoaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print("\n3. เปรียบเทียบ Models ต่างๆ")
print("-" * 35)

# Model configurations
models_config = {
    'LSTM': LSTMTimeSeriesModel(
        input_size=scaled_data.shape[1],
        hidden_size=64,
        num_layers=2,
        output_size=1,
        dropout=0.2
    ),
    'GRU': GRUTimeSeriesModel(
        input_size=scaled_data.shape[1],
        hidden_size=64,
        num_layers=2,
        output_size=1,
        dropout=0.2
    ),
    'CNN-LSTM': CNN_LSTM_Model(
        input_size=scaled_data.shape[1],
        sequence_length=sequence_length,
        hidden_size=64,
        output_size=1
    ),
    'LSTM+Attention': LSTMWithAttention(
        input_size=scaled_data.shape[1],
        hidden_size=64,
        num_layers=2,
        output_size=1,
        dropout=0.2
    )
}

# Training และ evaluation
results = {}

for model_name, model in models_config.items():
    print(f"\nTraining {model_name}...")
    
    # Training
    train_losses, val_losses = train_timeseries_model(
        model, train_loader, val_loader,
        num_epochs=100, learning_rate=0.001, patience=15
    )
    
    # Evaluation
    test_metrics = evaluate_timeseries_model(model, test_loader, scaler)
    
    results[model_name] = {
        'model': model,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'test_metrics': test_metrics
    }
    
    print(f"  Test RMSE: {test_metrics['rmse']:.4f}")
    print(f"  Test MAE: {test_metrics['mae']:.4f}")
    print(f"  Test MAPE: {test_metrics['mape']:.2f}%")

# สรุปผลการเปรียบเทียบ
print(f"\n4. Model Performance Comparison")
print("-" * 35)
print(f"{'Model':<15} {'RMSE':<10} {'MAE':<10} {'MAPE':<10}")
print("-" * 50)

for model_name, result in results.items():
    metrics = result['test_metrics']
    print(f"{model_name:<15} {metrics['rmse']:<10.4f} {metrics['mae']:<10.4f} {metrics['mape']:<10.2f}")

# หา model ที่ดีที่สุด
best_model_name = min(results.keys(), key=lambda k: results[k]['test_metrics']['rmse'])
best_model = results[best_model_name]

print(f"\nBest model: {best_model_name} (RMSE: {best_model['test_metrics']['rmse']:.4f})")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Training history
axes[0, 0].plot(best_model['train_losses'], label='Train Loss')
axes[0, 0].plot(best_model['val_losses'], label='Val Loss')
axes[0, 0].set_title(f'{best_model_name} - Training History')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Predictions vs Actual
test_metrics = best_model['test_metrics']
axes[0, 1].scatter(test_metrics['actuals'], test_metrics['predictions'], alpha=0.6)
axes[0, 1].plot([test_metrics['actuals'].min(), test_metrics['actuals'].max()],
                [test_metrics['actuals'].min(), test_metrics['actuals'].max()], 'r--')
axes[0, 1].set_title(f'{best_model_name} - Predictions vs Actual')
axes[0, 1].set_xlabel('Actual')
axes[0, 1].set_ylabel('Predicted')
axes[0, 1].grid(True, alpha=0.3)

# Time series plot
time_steps = range(len(test_metrics['actuals']))
axes[1, 0].plot(time_steps, test_metrics['actuals'], label='Actual', alpha=0.7)
axes[1, 0].plot(time_steps, test_metrics['predictions'], label='Predicted', alpha=0.7)
axes[1, 0].set_title(f'{best_model_name} - Time Series Prediction')
axes[1, 0].set_xlabel('Time Step')
axes[1, 0].set_ylabel('Value')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Model comparison
model_names = list(results.keys())
rmse_values = [results[name]['test_metrics']['rmse'] for name in model_names]

axes[1, 1].bar(model_names, rmse_values)
axes[1, 1].set_title('Model RMSE Comparison')
axes[1, 1].set_ylabel('RMSE')
axes[1, 1].tick_params(axis='x', rotation=45)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n5. Stock-like Data Example")
print("-" * 30)

# สร้างข้อมูลคล้าย stock
stock_data = generate_stock_like_data(n_points=800)

print(f"Stock dataset shape: {stock_data.shape}")
print(f"Features: {list(stock_data.columns)}")
print(f"Price range: ${stock_data['price'].min():.2f} - ${stock_data['price'].max():.2f}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Price chart
axes[0, 0].plot(stock_data['price'])
axes[0, 0].set_title('Stock Price')
axes[0, 0].set_ylabel('Price ($)')
axes[0, 0].grid(True, alpha=0.3)

# Returns
axes[0, 1].plot(stock_data['returns'])
axes[0, 1].set_title('Daily Returns')
axes[0, 1].set_ylabel('Return')
axes[0, 1].grid(True, alpha=0.3)

# Volume
axes[1, 0].plot(stock_data['volume'])
axes[1, 0].set_title('Trading Volume')
axes[1, 0].set_ylabel('Volume')
axes[1, 0].grid(True, alpha=0.3)

# Moving averages
axes[1, 1].plot(stock_data['price'], label='Price', alpha=0.7)
axes[1, 1].plot(stock_data['sma_5'], label='SMA 5', alpha=0.7)
axes[1, 1].plot(stock_data['sma_20'], label='SMA 20', alpha=0.7)
axes[1, 1].set_title('Price with Moving Averages')
axes[1, 1].set_ylabel('Price ($)')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Quick prediction example with stock data
print("\n6. Quick Stock Price Prediction")
print("-" * 35)

# Select features for prediction
feature_columns = ['price', 'volume', 'sma_5', 'sma_20', 'volatility']
stock_features = stock_data[feature_columns].dropna()

# Scale data
stock_scaler = MinMaxScaler()
stock_scaled = stock_scaler.fit_transform(stock_features)

# Create dataset for stock prediction
stock_dataset = TimeSeriesDataset(
    data=stock_scaled,
    sequence_length=30,
    prediction_horizon=1,
    target_column=0,  # predict price
    feature_columns=list(range(len(feature_columns)))
)

# Split data
train_size_stock = int(0.8 * len(stock_dataset))
test_size_stock = len(stock_dataset) - train_size_stock

train_stock = torch.utils.data.Subset(stock_dataset, range(train_size_stock))
test_stock = torch.utils.data.Subset(stock_dataset, range(train_size_stock, len(stock_dataset)))

# Create loaders
train_loader_stock = DataLoader(train_stock, batch_size=16, shuffle=True)
test_loader_stock = DataLoader(test_stock, batch_size=16, shuffle=False)

# Simple LSTM model for stock prediction
stock_model = LSTMTimeSeriesModel(
    input_size=len(feature_columns),
    hidden_size=32,
    num_layers=2,
    output_size=1,
    dropout=0.2
)

print(f"Stock dataset: {len(stock_dataset)} sequences")
print(f"Train: {len(train_stock)}, Test: {len(test_stock)}")

# Quick training
print("Training stock prediction model...")
stock_train_losses, _ = train_timeseries_model(
    stock_model, train_loader_stock, test_loader_stock,
    num_epochs=50, learning_rate=0.001, patience=10
)

# Evaluation
stock_metrics = evaluate_timeseries_model(stock_model, test_loader_stock, 
                                        scaler=None)  # Don't inverse transform for now

print(f"Stock Prediction Results:")
print(f"  RMSE: {stock_metrics['rmse']:.6f}")
print(f"  MAE: {stock_metrics['mae']:.6f}")
print(f"  MAPE: {stock_metrics['mape']:.2f}%")

print("\n7. Weather Data Prediction")
print("-" * 30)

# สร้างข้อมูลอากาศ
weather_data = generate_weather_data(n_points=730)  # 2 years

print(f"Weather dataset shape: {weather_data.shape}")
print(f"Date range: {weather_data['date'].min()} to {weather_data['date'].max()}")

# Weather features for prediction
weather_features = ['temperature', 'humidity', 'pressure', 'wind_speed']
weather_X = weather_data[weather_features]

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

for i, feature in enumerate(weather_features):
    row, col = i // 2, i % 2
    axes[row, col].plot(weather_data['date'], weather_data[feature])
    axes[row, col].set_title(f'{feature.title()} over Time')
    axes[row, col].set_ylabel(feature.title())
    axes[row, col].grid(True, alpha=0.3)
    
    # Rotate x-axis labels
    axes[row, col].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# Correlation analysis
correlation_matrix = weather_X.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=0.5)
plt.title('Weather Features Correlation Matrix')
plt.tight_layout()
plt.show()

print("Weather feature correlations:")
for i, feature1 in enumerate(weather_features):
    for j, feature2 in enumerate(weather_features):
        if i < j:
            corr = correlation_matrix.loc[feature1, feature2]
            print(f"  {feature1} vs {feature2}: {corr:.3f}")

print("\n8. Multi-step Prediction Example")
print("-" * 35)

# สร้าง dataset สำหรับ multi-step prediction
multi_step_dataset = TimeSeriesDataset(
    data=scaled_data[:800],  # ใช้ข้อมูลส่วนหนึ่ง
    sequence_length=30,
    prediction_horizon=5,    # ทำนาย 5 steps ahead
    target_column=0,
    feature_columns=list(range(scaled_data.shape[1]))
)

print(f"Multi-step dataset: {len(multi_step_dataset)} sequences")

# แบ่งข้อมูล
train_size_multi = int(0.8 * len(multi_step_dataset))
test_size_multi = len(multi_step_dataset) - train_size_multi

train_multi = torch.utils.data.Subset(multi_step_dataset, range(train_size_multi))
test_multi = torch.utils.data.Subset(multi_step_dataset, range(train_size_multi, len(multi_step_dataset)))

# Create loaders
train_loader_multi = DataLoader(train_multi, batch_size=16, shuffle=True)
test_loader_multi = DataLoader(test_multi, batch_size=16, shuffle=False)

# Model สำหรับ multi-step prediction
multi_step_model = LSTMTimeSeriesModel(
    input_size=scaled_data.shape[1],
    hidden_size=64,
    num_layers=2,
    output_size=5,  # predict 5 time steps
    dropout=0.2
)

print("Training multi-step prediction model...")

# Training
multi_train_losses, _ = train_timeseries_model(
    multi_step_model, train_loader_multi, test_loader_multi,
    num_epochs=30, learning_rate=0.001, patience=8
)

# Evaluation สำหรับ multi-step
multi_step_model.eval()
multi_predictions = []
multi_actuals = []

with torch.no_grad():
    for batch_x, batch_y in test_loader_multi:
        outputs = multi_step_model(batch_x)
        multi_predictions.extend(outputs.numpy())
        multi_actuals.extend(batch_y.numpy())

multi_predictions = np.array(multi_predictions)
multi_actuals = np.array(multi_actuals)

print(f"Multi-step prediction shapes:")
print(f"  Predictions: {multi_predictions.shape}")
print(f"  Actuals: {multi_actuals.shape}")

# คำนวณ RMSE สำหรับแต่ละ step
step_rmse = []
for step in range(5):
    rmse = np.sqrt(mean_squared_error(multi_actuals[:, step], multi_predictions[:, step]))
    step_rmse.append(rmse)
    print(f"  Step {step+1} RMSE: {rmse:.6f}")

# Visualization สำหรับ multi-step prediction
plt.figure(figsize=(15, 8))

# Plot RMSE by prediction step
plt.subplot(2, 2, 1)
plt.plot(range(1, 6), step_rmse, marker='o')
plt.title('RMSE by Prediction Step')
plt.xlabel('Prediction Step')
plt.ylabel('RMSE')
plt.grid(True, alpha=0.3)

# Plot sample predictions
sample_idx = 10
plt.subplot(2, 2, 2)
plt.plot(range(1, 6), multi_actuals[sample_idx], 'bo-', label='Actual')
plt.plot(range(1, 6), multi_predictions[sample_idx], 'ro-', label='Predicted')
plt.title(f'Sample Multi-step Prediction (Sample {sample_idx})')
plt.xlabel('Step')
plt.ylabel('Value')
plt.legend()
plt.grid(True, alpha=0.3)

# Compare step 1 vs step 5 accuracy
plt.subplot(2, 2, 3)
plt.scatter(multi_actuals[:, 0], multi_predictions[:, 0], alpha=0.6, label='Step 1')
plt.scatter(multi_actuals[:, 4], multi_predictions[:, 4], alpha=0.6, label='Step 5')
plt.plot([multi_actuals.min(), multi_actuals.max()], 
         [multi_actuals.min(), multi_actuals.max()], 'k--', alpha=0.5)
plt.title('Prediction Accuracy: Step 1 vs Step 5')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.legend()
plt.grid(True, alpha=0.3)

# Training history
plt.subplot(2, 2, 4)
plt.plot(multi_train_losses, label='Train Loss')
plt.title('Multi-step Model Training History')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n9. Attention Visualization")
print("-" * 25)

# ใช้ LSTM with Attention model
attention_model = results['LSTM+Attention']['model']

# ดึง attention weights จาก test sample
attention_model.eval()
test_sample_x, test_sample_y = next(iter(test_loader))

with torch.no_grad():
    output, attention_weights = attention_model(test_sample_x[:1])  # ใช้แค่ sample แรก

print(f"Attention weights shape: {attention_weights.shape}")

# Plot attention weights
plt.figure(figsize=(12, 6))

# Sample attention weights
weights = attention_weights[0, :, 0].numpy()
plt.subplot(1, 2, 1)
plt.plot(range(len(weights)), weights, 'bo-')
plt.title('Attention Weights Over Time Steps')
plt.xlabel('Time Step (Looking Back)')
plt.ylabel('Attention Weight')
plt.grid(True, alpha=0.3)

# Heatmap of attention for multiple samples
plt.subplot(1, 2, 2)
with torch.no_grad():
    sample_outputs, sample_attention = attention_model(test_sample_x[:10])  # 10 samples

attention_matrix = sample_attention[:, :, 0].numpy()  # Shape: (samples, time_steps)

im = plt.imshow(attention_matrix, cmap='Blues', aspect='auto')
plt.title('Attention Heatmap (10 Test Samples)')
plt.xlabel('Time Step')
plt.ylabel('Sample')
plt.colorbar(im, label='Attention Weight')

plt.tight_layout()
plt.show()

print("\n10. Model Interpretability")
print("-" * 25)

# Feature importance analysis
def analyze_feature_importance(model, test_loader, feature_names):
    """วิเคราะห์ feature importance โดยการ permutation"""
    
    model.eval()
    
    # Baseline performance
    baseline_metrics = evaluate_timeseries_model(model, test_loader)
    baseline_rmse = baseline_metrics['rmse']
    
    feature_importance = {}
    
    for feat_idx, feat_name in enumerate(feature_names):
        print(f"Testing feature: {feat_name}")
        
        # Create modified test loader with permuted feature
        modified_predictions = []
        modified_actuals = []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                # Permute the feature
                batch_x_modified = batch_x.clone()
                perm_indices = torch.randperm(batch_x.size(0))
                batch_x_modified[:, :, feat_idx] = batch_x[perm_indices, :, feat_idx]
                
                if hasattr(model, 'attention'):
                    outputs, _ = model(batch_x_modified)
                else:
                    outputs = model(batch_x_modified)
                
                modified_predictions.extend(outputs.numpy())
                modified_actuals.extend(batch_y.numpy())
        
        # Calculate degraded performance
        modified_predictions = np.array(modified_predictions)
        modified_actuals = np.array(modified_actuals)
        modified_rmse = np.sqrt(mean_squared_error(modified_actuals, modified_predictions))
        
        # Feature importance = increase in error
        importance = (modified_rmse - baseline_rmse) / baseline_rmse
        feature_importance[feat_name] = importance
    
    return feature_importance

# Analyze feature importance
feature_names = ['value', 'lag_1', 'lag_5', 'rolling_mean_10', 'rolling_std_10']
importance_scores = analyze_feature_importance(
    best_model['model'], test_loader, feature_names
)

print("Feature Importance (RMSE increase when permuted):")
sorted_features = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)

for feat_name, importance in sorted_features:
    print(f"  {feat_name}: {importance:.4f}")

# Visualization
plt.figure(figsize=(10, 6))
features, importances = zip(*sorted_features)
plt.bar(features, importances)
plt.title('Feature Importance Analysis')
plt.xlabel('Features')
plt.ylabel('Relative RMSE Increase')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\n" + "=" * 60)
print("สรุป Example 3: Time Series Prediction")
print("=" * 60)
print("✓ Time Series Dataset implementation")
print("✓ LSTM, GRU, CNN-LSTM models สำหรับ time series")
print("✓ Attention mechanism สำหรับ time series")
print("✓ Multi-step prediction")
print("✓ Synthetic และ real-world data examples")
print("✓ Model comparison และ evaluation")
print("✓ Feature importance analysis")
print("✓ Attention visualization และ interpretability")
print("=" * 60)"""
Lab 8 Example 3: Time Series Prediction
เรียนรู้การทำนาย time series ด้วย neural networks
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from datetime import datetime, timedelta

print("=" * 60)
print("Lab 8 Example 3: Time Series Prediction")
print("=" * 60)

# 1. Time Series Dataset
class TimeSeriesDataset(Dataset):
    """Dataset สำหรับ time series data"""
    
    def __init__(self, data, sequence_length, prediction_horizon=1, 
                 target_column=None, feature_columns=None):
        """
        Args:
            data: pandas DataFrame หรือ numpy array
            sequence_length: ความยาวของ input sequence
            prediction_horizon: จำนวน steps ที่ต้องการทำนาย
            target_column: column ที่ต้องการทำนาย
            feature_columns: columns ที่ใช้เป็น features
        """
        self.data = data if isinstance(data, np.ndarray) else data.values
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        
        if target_column is not None:
            self.target_idx = target_column
        else:
            self.target_idx = 0  # ใช้ column แรกเป็น target
        
        if feature_columns is not None:
            self.feature_indices = feature_columns
        else:
            self.
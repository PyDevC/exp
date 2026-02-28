import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- Step 1: Data Preparation ---
df = pd.read_csv('IEEE_SWH_RealWorld_Dataset.csv')

# Features: Irradiance, Amb_Temp, In_Temp, Area, Volume, System_Type
X = df.drop('Optimal_Flow_Rate', axis=1).values
y = df['Optimal_Flow_Rate'].values.reshape(-1, 1)

# Split into 80% Train and 20% Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features (Mean=0, StdDev=1) - Critical for Neural Networks
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert to PyTorch Tensors
X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32)

# --- Step 2: Define Model ---
class SolarFlowNet(nn.Module):
    def __init__(self):
        super(SolarFlowNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid() 
        )
    def forward(self, x):
        return self.net(x) * 0.15  # Scale Sigmoid to Max Flow Rate

model = SolarFlowNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- Step 3: Training Loop ---
epochs = 100
train_losses = []
test_losses = []

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    outputs = model(X_train_t)
    loss = criterion(outputs, y_train_t)
    
    # Backward pass and optimization
    loss.backward()
    optimizer.step()
    
    # Track performance on test set (Evaluation)
    model.eval()
    with torch.no_grad():
        test_preds = model(X_test_t)
        test_loss = criterion(test_preds, y_test_t)
        
    train_losses.append(loss.item())
    test_losses.append(test_loss.item())

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {loss.item():.6f}, Test Loss: {test_loss.item():.6f}")

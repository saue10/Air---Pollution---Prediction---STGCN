import torch
import torch.optim as optim
import pickle
from torch.utils.data import TensorDataset, DataLoader
from step6_model import STGCN_Model, evt_loss

# Load data
X = torch.load("X.pt")
y = torch.load("y.pt")

# Load scaler
with open("scaler.pkl", "rb") as f:
    pm25_mean, pm25_std = pickle.load(f)

# Train-test split
split = int(0.8 * len(X))

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)
# Model
model = STGCN_Model(feature_dim=6, hidden_dim=64)

optimizer = optim.Adam(model.parameters(), lr=0.0005)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
epochs = 60
loss_fn = torch.nn.SmoothL1Loss()
# Training
for epoch in range(epochs):

    model.train()
    total_loss = 0

    for X_batch, y_batch in train_loader:

        y_pred = model(X_batch)

        loss_fn = torch.nn.SmoothL1Loss()
        loss = loss_fn(y_pred, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss}")
    scheduler.step()
# Evaluation
model.eval()

total_mae = 0
total_mae_norm=0
count = 0

with torch.no_grad():
    for X_batch, y_batch in test_loader:

        y_pred = model(X_batch)
        mae_norm = torch.sum(torch.abs(y_pred - y_batch))
        total_mae_norm += mae_norm.item()

        y_pred_real = y_pred * pm25_std + pm25_mean
        y_real = y_batch * pm25_std + pm25_mean

        mae = torch.sum(torch.abs(y_pred_real - y_real))

        total_mae += mae.item()
        count += y_batch.numel()
print("\nNormalised MAE:",total_mae_norm / count)
print("\nREAL MAE:", total_mae / count)
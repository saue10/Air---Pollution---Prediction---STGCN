import torch

# Load graph data
data = torch.load("graph_data.pt")

def create_sequences(data, seq_len=3):
    X, y = [], []
    
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        target = data[i+seq_len:i+seq_len+3,0,0].mean()
        y.append(target)   # only one station PM2.5  
    
    return torch.stack(X), torch.stack(y)

X, y = create_sequences(data)

print("X shape:", X.shape)
print("y shape:", y.shape)
torch.save(X, "X.pt")
torch.save(y, "y.pt")
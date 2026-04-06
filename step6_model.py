import torch
import torch.nn as nn
from step4_attention import GraphAttention
from step5_gcn import GCNLayer

class STGCN_Model(nn.Module):
    def __init__(self, feature_dim, hidden_dim):
        super().__init__()
        
        self.attention = GraphAttention(feature_dim)
        self.gcn = GCNLayer(feature_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, X):
        # X: (batch, seq_len, nodes, features)
        
        batch, seq_len, nodes, feat = X.shape
        
        outputs = []
        
        for t in range(seq_len):
            x_t = X[:, t, :, :]  # (batch, nodes, features)
            
            A = self.attention(x_t)
            
            x_t = self.gcn(A, x_t)
            
            outputs.append(x_t)
        
        outputs = torch.stack(outputs, dim=1)
        
        # reshape for LSTM
        outputs = outputs.view(batch * nodes, seq_len, -1)
        
        out, _ = self.lstm(outputs)
        
        out = out[:, -1, :]
        
        out = self.fc(out)
        
        out = out.view(batch, nodes)
        

        # predict only one station
        out = out[:, 0]


        return out
    
def evt_loss(y_pred, y_true):
    error = (y_pred - y_true) ** 2

    # Smooth threshold
    threshold = torch.quantile(y_true, 0.9)

    # Soft weighting (IMPORTANT)
    weights = 1 + 2 * torch.sigmoid((y_true - threshold))

    return torch.mean(weights * error)

if __name__ == "__main__":
    
    X = torch.randn(4, 5, 38, 6)
    
    model = STGCN_Model(6, 16)
    
    out = model(X)
    
    print("Model output shape:", out.shape)    
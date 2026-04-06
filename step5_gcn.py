import torch
import torch.nn as nn

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, A, X):
        # A: (batch, nodes, nodes)
        # X: (batch, nodes, features)
        
        X = torch.matmul(A, X)
        
        return self.linear(X)


# 🔥 TEST

if __name__ == "__main__":
    
    X = torch.randn(2, 38, 6)
    A = torch.randn(2, 38, 38)
    
    gcn = GCNLayer(6, 16)
    
    out = gcn(A, X)
    
    print("GCN output shape:", out.shape)
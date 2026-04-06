import torch
import torch.nn as nn

class GraphAttention(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.W = nn.Linear(in_features, in_features)

    def forward(self, X):
        # X shape: (batch, nodes, features)
        
        Q = self.W(X)
        K = self.W(X)
        
        scores = torch.matmul(Q, K.transpose(-1, -2))
        
        A = torch.softmax(scores, dim=-1)
        
        return A


# 🔥 TEST IT (VERY IMPORTANT)

if __name__ == "__main__":
    
    # Dummy input: (batch=2, nodes=38, features=6)
    X = torch.randn(2, 38, 6)
    
    attn = GraphAttention(6)
    
    A = attn(X)
    
    print("Attention shape:", A.shape)
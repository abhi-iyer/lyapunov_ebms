from imports import *


# dot product of vector with itself
class DotProduct(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return torch.pow(x, 2).sum(dim=1)

# siamese network
class EBM(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        
        self.phi = nn.Sequential(
            nn.Linear(input_shape, 4*input_shape),
            nn.ReLU(),
            nn.Linear(4*input_shape, 8*input_shape),
            nn.ReLU(),
            nn.Linear(8*input_shape, 16*input_shape),
            nn.ReLU(),
            nn.Linear(16*input_shape, 16*input_shape),
            DotProduct()
        )

    def forward(self, x):
        return self.phi(x)
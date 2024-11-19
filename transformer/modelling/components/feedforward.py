import torch.nn as nn


# Unused due to implicit notation in Transformer
class PositionalFeedForward(nn.Module):

    def __init__(self, dim_model, dim_inner=2048, activation=nn.ReLU()):
        super().__init__()

        self.activation = activation
        self.transform = nn.Sequential(
            nn.ModuleList([
                nn.Linear(dim_model, dim_inner, bias=True),
                self.activation,
                nn.Linear(dim_inner, dim_model, bias=True)
            ])
        )

    def forward(self, x):
        return self.transform(x)

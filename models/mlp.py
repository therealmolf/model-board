import torch as t
from models.config import Config


class MLP(t.nn.Module):
    def __init__(self,
                 cfg: Config):
        super().__init__()

        self.cfg = cfg

        self.all_layers = t.nn.Sequential(
            # 1st hidden layer
            t.nn.Linear(self.cfg.n_features, self.cfg.layer1_dim),
            t.nn.ReLU(),
            # 2nd hidden layer
            t.nn.Linear(self.cfg.layer1_dim, self.cfg.layer2_dim),
            t.nn.ReLU(),
            # output layer
            t.nn.Linear(self.cfg.layer2_dim, self.cfg.n_classes),
        )

    def forward(self, x):
        x = t.flatten(x, start_dim=1)
        logits = self.all_layers(x)
        return logits

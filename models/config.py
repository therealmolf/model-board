from dataclasses import dataclass
import torch as t


@dataclass
class Config:
    device: t.device = "cuda" if t.cuda.is_available() else "cpu"
    train_pct: float = 0.8
    test_pct: float = 0.2
    batch_size: int = 32
    n_classes: int = 7
    n_features: int = 16
    layer1_dim: int = 60
    layer2_dim: int = 25
    lr: float = 0.005
    n_epochs: int = 60

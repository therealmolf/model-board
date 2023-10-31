"""
Training script for MLP on Dry Bean Dataset using Lightning.
This includes the Dataset, LightningModule, DataLoaders, Trainer, etc.

"""


from lightning import LightningModule, Trainer

import torch as t
import torch.nn.functional as F
import torchmetrics
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split

from sklearn.preprocessing import MinMaxScaler

import pandas as pd

from ucimlrepo import fetch_ucirepo

from models.mlp import MLP
from models.config import Config

from watermark import watermark


class DryBeansDataset(Dataset):
    def __init__(self,
                 cfg: Config,
                 transform=None):

        # fetch dataset
        self.dry_bean_dataset = fetch_ucirepo(id=602)

        self.cfg = cfg
        self.transform = transform

        self.scaler = MinMaxScaler()
        scaled_features = self.scaler.fit_transform(
                                 self.dry_bean_dataset.data.features)

        self.features = t.tensor(scaled_features,
                                 device=self.cfg.device,
                                 dtype=t.float32)

        # Categorical to Numerical
        codes, uniques = pd.factorize(
                                 self.dry_bean_dataset.data.targets['Class'])

        self.uniques = uniques
        self.labels = t.tensor(codes, device=self.cfg.device)

    def __getitem__(self, index: int):
        if self.transform is None:
            return self.features[index], self.labels[index]
        else:
            self.features[index] = self.transform(self.features[index])
            return self.features[index], self.labels[index]

    def __len__(self):
        return self.labels.shape[0]


class LightningMLP(LightningModule):
    def __init__(self,
                 model: t.nn.Module,
                 cfg: Config):

        super().__init__()

        self.cfg = cfg
        self.learning_rate = self.cfg.lr
        self.model = model

        # Set up attributes for computing the accuracy
        self.train_acc = torchmetrics.Accuracy(task="multiclass",
                                               num_classes=self.cfg.n_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass",
                                             num_classes=self.cfg.n_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):

        features, labels = batch
        logits = self(features)
        loss = F.cross_entropy(logits, labels)
        self.log("train_loss", loss)

        preds = t.argmax(logits, dim=1)
        self.train_acc(preds, labels)
        self.log(
            "train_acc",
            self.train_acc,
            prog_bar=True,
            on_epoch=True,
            on_step=False)

        return loss

    def validation_step(self, batch, batch_idx):
        features, labels = batch
        logits = self(features)
        loss = F.cross_entropy(logits, labels)
        self.log("val_loss", loss, prog_bar=True)

        preds = t.argmax(logits, dim=1)
        self.val_acc(preds, labels)
        self.log("val_acc", self.val_acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = t.optim.SGD(self.parameters(), lr=self.learning_rate)
        return optimizer


if __name__ == "__main__":

    # argparser for changing params/hyperparams

    print(watermark(packages="torch,lightning", python=True))
    print("Torch CUDA available?", t.cuda.is_available())

    # Instantiate Config
    cfg = Config()

    # Instantiate Dataset
    dataset = DryBeansDataset(cfg)
    train_dataset, test_dataset = random_split(dataset,
                                               [cfg.train_pct, cfg.test_pct])
    train_dataset, val_dataset = random_split(train_dataset,
                                              [cfg.train_pct, cfg.test_pct])

    # Instantiate DataLoaders
    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.batch_size,
                              shuffle=True,
                              num_workers=0)
    val_loader = DataLoader(val_dataset,
                            batch_size=cfg.batch_size,
                            shuffle=False,
                            num_workers=0)
    test_loader = DataLoader(test_dataset,
                             batch_size=cfg.batch_size,
                             shuffle=False,
                             num_workers=0)

    # Instantiate PyTorch Model and Lightning Wrapper
    model = MLP(cfg)
    lightning_model = LightningMLP(model=model, cfg=cfg)

    # Instantiate Trainer
    trainer = Trainer(
        max_epochs=cfg.n_epochs,
        accelerator=cfg.device,
        devices=1,
    )

    # Run Training
    trainer.fit(
        model=lightning_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

PATH = "lightning.pt"
t.save(model.state_dict(), PATH)

# To load model:
# model = PyTorchMLP(num_features=784, num_classes=10)
# model.load_state_dict(torch.load(PATH))
# model.eval()

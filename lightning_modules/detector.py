import pytorch_lightning as pl
import torch.nn.functional as F
import torch
from sklearn.metrics import balanced_accuracy_score

class DeepfakeDetector(pl.LightningModule):
    def __init__(self, model, lr=1e-4, class_weights=None):
        super().__init__()
        self.model = model
        self.lr = lr
        if class_weights is not None:
            class_weights = torch.as_tensor(class_weights, dtype=torch.float)
        self.loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        balanced_acc = balanced_accuracy_score(
            y.detach().cpu().numpy(),
            preds.detach().cpu().numpy(),
        )
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        self.log("val_balanced_acc", balanced_acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

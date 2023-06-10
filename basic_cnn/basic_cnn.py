import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
from torchvision import transforms
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
import torchvision
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import CSVLogger
import pandas as pd
import matplotlib.pyplot as plt
from torchvision.datasets import Food101

class Food101(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(128*16*16, 512),
            nn.ReLU(),
            nn.Linear(512, 101)
        )
        
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=101)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, 128*16*16)
        x = self.fc_layers(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        logger.log_metrics({'train_loss': loss}, step=self.global_step)
        return loss

    def validation_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self(x)
            loss = F.cross_entropy(y_hat, y)

            self.accuracy(y_hat, y)

            self.log("val_accuracy", self.accuracy.compute(), on_epoch = True)
            logger.log_metrics({'val_accuracy': self.accuracy.compute(), 'val_loss': loss}, step=self.global_step)


    def test_step(self, batch, batch_idx):
        x, y = batch
        if len(x) == 0:
            return
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        self.accuracy(y_hat, y)
        self.log("test_accuracy", self.accuracy.compute(), on_epoch = True)
        logger.log_metrics({'test_accuracy': self.accuracy.compute(), 'test_loss': loss}, step=self.global_step)
        self.log("test_loss", loss, on_epoch = True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


    # Prepare the dataset
train_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((128,128)),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((128,128)),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

path = "root/food-101/images"

train_dataset = torchvision.datasets.ImageFolder(path,transform=train_transforms)
# Use 10% of the training set for validation
train_set_size = int(len(train_dataset) * 0.9)
val_set_size = len(train_dataset) - train_set_size

seed = torch.Generator().manual_seed(42)
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_set_size, val_set_size], generator=seed)


val_dataset.dataset.transform = test_transforms

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, num_workers=4, shuffle=True)

val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, num_workers=4, shuffle=False)

model = Food101()

# EarlyStopping
early_stop_callback = EarlyStopping(monitor="val_loss",
                                    mode="min",
                                    patience=2)

# Configure Checkpoints
checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    mode="min"
)


logger = CSVLogger("logs", name = "myfile")


trainer = pl.Trainer(accelerator = "gpu", callbacks=[early_stop_callback, checkpoint_callback], max_epochs=5, logger = logger)
trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
trainer.test(model, dataloaders=torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(path, transform=test_transforms), batch_size=128, num_workers=4, shuffle=False))

data = pd.read_csv("logs/myfile/version_0/metrics.csv")

train_loss = data['train_loss'].astype(float).tolist()
val_loss = data['val_loss'].astype(float).tolist()


df = pd.DataFrame({'train_loss': train_loss})
df2 = pd.DataFrame({'val_loss': val_loss})

# Drop any NaN values
df = df.dropna()
df2 = df2.dropna()

# Calculate the rolling mean with a window size of 10
rolling_mean = df.rolling(window=1).mean()
rolling_mean2 = df2.rolling(window=1).mean()

plt.plot(rolling_mean)
plt.plot(rolling_mean2)
plt.show()

val_loss = trainer.validate(model, val_loader)[0]['val_loss']
val_accuracy = trainer.validate(model, val_loader)[0]['val_accuracy']

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total number of trainable parameters: {total_params}")
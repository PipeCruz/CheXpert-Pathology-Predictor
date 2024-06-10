import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
from torchvision import transforms
from read_data import CustomDataset
import pytorch_lightning as pl
import pandas as pd
import numpy as np

# Update the checkpoint path for AlexNet
CKPT_PATH = '/home/ezhang3/156b/checkpoints/best-checkpoint-v6.ckpt'
N_CLASSES = 9
ROOT_DIR = '/groups/CS156b/data'
TRAIN_CSV = '/home/ttran5/imputed_train2023.csv'
TEST_CSV = 'student_labels/solution_ids.csv'
BATCH_SIZE = 64
N_EPOCHS = 25

class MedicalImageClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = models.alexnet(pretrained=True)  # Use AlexNet
        num_ftrs = self.model.classifier[6].in_features  # Get the number of input features for the classifier
        self.model.classifier[6] = nn.Linear(num_ftrs, N_CLASSES)  # Replace the classifier with a new one for our number of classes
        self.criterion = nn.MSELoss()

    def forward(self, x):
        if isinstance(x, list):
            x = x[0]  # If input is a list, assume first element is input tensor
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return optim.SGD(self.parameters(), lr=1e-4, momentum=0.9)

def main(file_name="submission.csv", num_gpus=1, num_patient=100):
    custom_dataset = CustomDataset(csv_file=TRAIN_CSV, root_dir=ROOT_DIR, train=True, transform=transforms.ToTensor())
    train_loader = DataLoader(custom_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # Load the model from the checkpoint
    model = MedicalImageClassifier.load_from_checkpoint(CKPT_PATH)

    test_dataset = CustomDataset(csv_file=TEST_CSV, root_dir=ROOT_DIR, train=False, transform=transforms.ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    predictor = pl.Trainer(accelerator='auto', max_epochs=N_EPOCHS)
    predictions_all_batches = predictor.predict(model, dataloaders=test_loader)

    # Flatten the predictions
    flat_predictions = [prediction for batch_predictions in predictions_all_batches for prediction in batch_predictions]

    # Create DataFrame
    df = pd.DataFrame(flat_predictions)
    df.to_csv(file_name)

import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_name", type=str)
    parser.add_argument("--num_gpus", type=int)
    args = parser.parse_args()
    main(file_name=args.file_name, num_gpus=args.num_gpus)

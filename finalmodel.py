import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import densenet169, DenseNet169_Weights

from read_data import CustomDataset
from sklearn.metrics import mean_squared_error as mse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

CKPT_PATH = 'model.ckpt'
N_CLASSES = 9
OUTPUT_CLASSES = ['Id', 'No Finding', 'Enlarged Cardiomediastinum', 
                  'Cardiomegaly', 'Lung Opacity', 'Pneumonia', 
                  'Pleural Effusion', 'Pleural Other', 'Fracture', 
                  'Support Devices']
ROOT_DIR = '/groups/CS156b/data'
TRAIN_CSV = '/home/ttran5/imputed_train2023.csv'
TEST_CSV = 'student_labels/solution_ids.csv'
BATCH_SIZE = 64
N_EPOCHS = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHUNK_SIZE = 1024


class MedicalImageClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        weights = DenseNet169_Weights.DEFAULT
        self.model = densenet169(weights=weights)
        self.model.classifier = nn.Linear(self.model.classifier.in_features, N_CLASSES)
        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return optim.SGD(self.parameters(), lr=1e-4, momentum=0.9)


def main(file_name="submission.csv", num_gpus=1):
    custom_dataset = CustomDataset(csv_file=TRAIN_CSV, root_dir=ROOT_DIR, train=True)
    train_loader = DataLoader(custom_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    model = MedicalImageClassifier()

    checkpoint_callback = ModelCheckpoint(
        monitor='train_loss',
        dirpath='checkpoints/',
        filename='best-checkpoint',
        save_top_k=1,
        mode='min'
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=num_gpus,
        strategy="ddp",
        max_epochs=N_EPOCHS,
        callbacks=[checkpoint_callback]
    )

    trainer.fit(model, train_loader)

    test_csv_path = os.path.join(ROOT_DIR, TEST_CSV)
    if not os.path.exists(test_csv_path):
        print("Test CSV file not found.")
        return

    test_set = CustomDataset(csv_file=test_csv_path, root_dir=ROOT_DIR, train=False)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Generate predictions
    model.eval()
    predictions = []
    ids = []
    with torch.no_grad():
        for batch in test_loader:
            x, _, id_batch = batch
            y_hat = model(x)
            predictions.extend(y_hat.cpu().numpy())
            ids.extend(id_batch)

    # Write results to submission.csv
    submission_df = pd.DataFrame(predictions, columns=OUTPUT_CLASSES[1:])
    submission_df.insert(0, OUTPUT_CLASSES[0], ids)
    submission_df.to_csv(file_name, index=False)
    print(f"Results saved to {file_name}")


import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_name", type=str, default="submission.csv")
    parser.add_argument("--num_gpus", type=int, default=1)

    args = parser.parse_args()
    main(file_name=args.file_name, num_gpus=args.num_gpus)

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import alexnet  # Import AlexNet
from read_data import CustomDataset
from sklearn.metrics import mean_squared_error as mse
import pytorch_lightning as pl

CKPT_PATH = 'model.ckpt'
N_CLASSES = 9
OUTPUT_CLASSES = ['Id', 'No Finding', 'Enlarged Cardiomediastinum', 
                  'Cardiomegaly', 'Lung Opacity', 'Pneumonia', 
                  'Pleural Effusion', 'Pleural Other', 'Fracture', 
                  'Support Devices']
ROOT_DIR = '/groups/CS156b/data'
TRAIN_CSV = 'student_labels/train2023.csv'
TEST_CSV = 'student_labels/test_ids.csv'
BATCH_SIZE = 64
N_EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHUNK_SIZE = 1024


class MedicalImageClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # Using AlexNet
        self.model = alexnet(pretrained=True)
        # Modify the last layer to output N_CLASSES
        self.model.classifier[-1] = nn.Linear(self.model.classifier[-1].in_features, N_CLASSES)
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


def main(file_name="submission.csv", num_gpus=1, num_patient=100):
    custom_dataset = CustomDataset(csv_file=TRAIN_CSV, root_dir=ROOT_DIR, train=True)
    train_loader = DataLoader(custom_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    model = MedicalImageClassifier()

    trainer = pl.Trainer(accelerator="gpu", devices=num_gpus, strategy="ddp", max_epochs=N_EPOCHS)
    trainer.fit(model, train_loader)

    if not os.path.exists(TEST_CSV):
        print("Test CSV file not found.")
        return

    test_set = CustomDataset(csv_file=TEST_CSV, root_dir=ROOT_DIR, train=False)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    trainer.test(model, test_dataloaders=test_loader)
    trainer.save_checkpoint(CKPT_PATH)


import argparse
if __name__ == '__main__':
    # get file_name from command line
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_name", type=str)
    parser.add_argument("--num_gpus", type=int)
    
    args = parser.parse_args()
    main(file_name=args.file_name, num_gpus=args.num_gpus)

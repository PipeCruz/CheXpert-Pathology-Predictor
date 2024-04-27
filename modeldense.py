import os
import scipy.misc
import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
from torch.optim import Adam, SGD

import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader

from read_data import CustomDataset
from torch.utils.data import DataLoader

from sklearn.metrics import mean_squared_error as mse
from transformers import get_scheduler
# transformers==4.33.0
from tqdm.auto import tqdm

from sklearn.model_selection import train_test_split
from torchvision.models import densenet169
from torchvision.models.densenet import DenseNet169_Weights
import read_data

CKPT_PATH = 'fixme.pth'

N_CLASSES = 9
CLASS_NAMES = ['No Finding', 'Enlarged Cardiomediastinum', 
            'Cardiomegaly', 'Lung Opacity', 'Pneumonia', 
            'Pleural Effusion', 'Pleural Other', 'Fracture', 
            'Support Devices']

ROOT_DIR = '/groups/CS156b/data'
BATCH_SIZE = 64

# maybe move these to some file called like modeldense.py
def train(model, criterion, optimizer, train_loader, device):
    print("Training")
    model.train()
    
    for epoch in range(10):
        print(f"epoch: {epoch+1}/10", flush=True)
        for batch_idx, (images, labels, idxs) in enumerate(train_loader):            
            optimizer.zero_grad()
            
            images = images.to(device)
            labels = labels.to(device)
            
            output = model(images)
            
            loss = criterion(output, labels)
            
            loss.backward()
            optimizer.step()
        
    return model

# change to test_loss later
def evaluate(model, criterion, eval_loader, device):
    print('Evaluating')
    model.eval()
    eval_loss = 0
    
    with torch.no_grad():
        for images, labels, idxs in eval_loader:
            images = images.to(device)
            labels = labels.to(device)
            output = model(images)
            
            loss = criterion(output, labels)
            
            eval_loss += loss.item()
                
            
    eval_loss /= len(eval_loader.dataset)
    average_loss = eval_loss
    print('Eval set: Average loss: {:.4f}'.format(average_loss))
    
    return model
    
    
def test(model, criterion, test_loader, device, name_of_output="submission.csv"):
    print('Testing')
    model.eval()
    test_loss = 0
    
    write_file = open(name_of_output, "w")
    # first line is
    # fixme to use CLASSS_NAMES or CLASS_LABELS from read_data.py
    write_file.write("Id, No Finding,Enlarged Cardiomediastinum,Cardiomegaly,Lung Opacity,Pneumonia,Pleural Effusion,Pleural Other,Fracture,Support Devices\n")
    
    # write_file.write("Id,No Finding,Enlarged Cardiomediastinum,Cardiomegaly,Lung Opacity,Lung Lesion,Edema,Consolidation,Pneumonia,Atelectasis,Pneumothorax,Pleural Effusion,Pleural Other,Fracture,Support Devices\n")
    
    # write to csv now
    batch = 0
    with torch.no_grad():
        for images, labels, idxs in test_loader:
                      
            images = images.to(device)
            labels = labels.to(device)
            output = model(images)
            
            loss = criterion(output, labels)
            
            test_loss += loss.item()
            
            # each idx is an ID
            for idx, out in zip(idxs, output):
                write_file.write(f"{idx},{','.join([str(x) for x in out.tolist()])}\n")
                
    test_loss /= len(test_loader.dataset)
    average_loss = test_loss
    print('Test set: Average loss: {:.4f}'.format(average_loss), flush=True)
    
    return model
    
TRAIN_CSV = 'student_labels/train2023.csv'
TEST_CSV = 'student_labels/test_ids.csv'

def main(local=False, file_name="submission.csv", num_patient=100):
    custom_dataset = CustomDataset(csv_file = TRAIN_CSV, root_dir = ROOT_DIR, train=True)
       
    
    # when we only have a subset of the data we need the indices that
    if (local):
    # correspond to the downloaded images, i.e.
        idxs = []
        for i in range(1, num_patient):
            stri = str(i).zfill(5)
            df = custom_dataset.df
            mask = df["Path"].fillna("").str.contains(f"pid{stri}")
            for j in df[mask].index:
                idxs.append(j)
                    
        subset = torch.utils.data.Subset(custom_dataset, idxs)
        custom_dataset = subset
    else:
        test_set = CustomDataset(csv_file = TEST_CSV, root_dir = ROOT_DIR, train=False)
        test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    # print(len(subset))
        # print(len(custom_dataset))
    
    
    # load model if it exists
    if os.path.exists(CKPT_PATH):
        model = densenet169(weights=DenseNet169_Weights.DEFAULT)
        model.classifier = nn.Linear(model.classifier.in_features, N_CLASSES)
        model.load_state_dict(torch.load(CKPT_PATH))
    else:
        model = densenet169(weights=DenseNet169_Weights.DEFAULT)
    
        for param in model.parameters():
            param.requires_grad = True
        
        submodules = model.features[-2:]
        for param in submodules.parameters():
            param.requires_grad = True
        
        model.classifier = nn.Linear(model.classifier.in_features, N_CLASSES)
    
    
    # train_size = int(0.95 * len(custom_dataset))
    # eval_size = len(custom_dataset) - train_size
    

    # tune_set, eval_set = torch.utils.data.random_split(custom_dataset, [train_size, eval_size])
    
    train_loader = DataLoader(custom_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    # eval_loader = DataLoader(eval_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
    
    # parallelize the model
    model = torch.nn.DataParallel(model).cuda()

    criterion = nn.MSELoss()
#    optimizer = Adam(model.parameters(), lr=1e-5)
    optimizer = SGD(model.parameters(), lr=1e-4, momentum=0.9)
    # Train the model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    print("Device: ", device)
    
    model = train(model, criterion, optimizer, train_loader, device)
    
   
    # if remote, test the model
    if not local:
        model = test(model, criterion, test_loader, device)
    
    torch.save(model.state_dict(), CKPT_PATH)
    
        
    

import argparse
if __name__ == '__main__':
    # get file_name from command line
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_name", type=str)
    
    args = parser.parse_args()
    main(file_name=args.file_name)
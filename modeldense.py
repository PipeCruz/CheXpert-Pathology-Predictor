import os
import scipy.misc
import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
from torch.optim import Adam

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

CKPT_PATH = 'model1.pth'
N_CLASSES = 9
CLASS_NAMES = ['No Finding', 'Enlarged Cardiomediastinum', 
            'Cardiomegaly', 'Lung Opacity', 'Pneumonia', 
            'Pleural Effusion', 'Pleural Other', 'Fracture', 
            'Support Devices']

ROOT_DIR = 'cs156b/'
BATCH_SIZE = 20

# maybe move these to some file called like modeldense.py
def train(model, criterion, optimizer, train_loader, device):
    print("Training")
    model.train()
    mse = []
    for epoch in tqdm(range(5)):
        for batch_idx, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            
            images = images.to(device)
            labels = labels.to(device)
            
            output = model(images)
            
            loss = criterion(output, labels)
            
            loss.backward()
            optimizer.step()
        print('Train Epoch: %d  Loss: %.4f' % (epoch + 1,  loss.item()))
        
    return model

    # change to test_loss later
def evaluate(model, criterion, eval_loader, device):
    print('Evaluating')
    model.eval()
    eval_loss = 0
    # write to csv
    # make file
    # write_file = open("eval.csv", "w")
    
    with torch.no_grad():
            for images, labels in eval_loader:
                images = images.to(device)
                labels = labels.to(device)
                output = model(images)
                
                loss = criterion(output, labels)
                
                eval_loss += loss.item()
                
            
    eval_loss /= len(eval_loader.dataset)
    average_loss = eval_loss
    print('Eval set: Average loss: {:.4f}'.format(average_loss))
    
    return model
    
    
def test(model, criterion, test_loader, device):
    print('Testing')
    model.eval()
    test_loss = 0
    # write to csv
    # make file
    write_file = open("submission.csv", "w")
    # first line is
    write_file.write("Id,No Finding,Enlarged Cardiomediastinum,Cardiomegaly,Lung Opacity,Lung Lesion,Edema,Consolidation,Pneumonia,Atelectasis,Pneumothorax,Pleural Effusion,Pleural Other,Fracture,Support Devices\n")
    
    
    # write to csv now
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            output = model(images)
            
            loss = criterion(output, labels)
            
            test_loss += loss.item()
            
            # write to csv
            for i in range(len(output)):
                write_file.write(f"{i},{','.join([str(x) for x in output[i].tolist()])}\n")
                
    test_loss /= len(test_loader.dataset)
    average_loss = test_loss
    print('Test set: Average loss: {:.4f}'.format(average_loss))
    
    return model
    
    
def main(local=False, num_patient=100):
    custom_dataset = CustomDataset(csv_file = "cs156b/train2023.csv", root_dir = ROOT_DIR, train=True)
       
    
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
        test_set = CustomDataset(csv_file = "cs156b/test.csv", root_dir = ROOT_DIR, train=False)
        test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    # print(len(subset))
        print(len(custom_dataset))
    
    
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
    
    
    train_size = int(0.95 * len(custom_dataset))
    eval_size = len(custom_dataset) - train_size
    

    tune_set, eval_set = torch.utils.data.random_split(custom_dataset, [train_size, eval_size])
    
    tune_loader = DataLoader(tune_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    eval_loader = DataLoader(eval_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    
    # parallelize the model
    model = torch.nn.DataParallel(model).cuda()

    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=1e-5)
    
    # Train the model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    print("Device: ", device)
    
    model = train(model, criterion, optimizer, tune_loader, device)
    
    # evaluate the model
    model = evaluate(model, criterion, eval_loader, device)
    
    # if remote, evaluate on the test set
    if not local:
        model = test(model, criterion, test_loader, device)
    
    
    # could do like kfolds here
    
    # this wont work because the model gets saved each time i think
    # from sklearn.model_selection import KFold
    
    # kf = KFold(n_splits=5)
    # for train_index, test_index in kf.split(subset):
    #     train_set = torch.utils.data.Subset(subset, train_index)
    #     test_set = torch.utils.data.Subset(subset, test_index)
        
    #     train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    #     test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
        
    #     model = train(model, criterion, optimizer, train_loader, device)
        
    #     model = evaluate(model, criterion, test_loader, device)
    # # save the model
    torch.save(model.state_dict(), 'modeldense.pth')
    
            
    
    # graph mse over epochs
        
    


if __name__ == '__main__':
    main(local=False)
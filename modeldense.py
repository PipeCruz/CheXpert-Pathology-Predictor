import os
import scipy.misc
import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
from torch.optim import SGD

import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader

from read_data import CustomDataset
from torch.utils.data import DataLoader

from sklearn.metrics import mean_squared_error as mse
# transformers==4.33.0

from sklearn.model_selection import train_test_split
from torchvision.models import densenet169
from torchvision.models.densenet import DenseNet169_Weights
import read_data

import matplotlib as plt
import matplotx as pltx

CKPT_PATH = 'fixme.pth'

N_CLASSES = 9

# Id	No Finding	Enlarged Cardiomediastinum	Cardiomegaly	Lung Opacity	Pneumonia	Pleural Effusion	Pleural Other	Fracture	Support Devices
OUTPUT_CLASSES = ['Id', 'No Finding', 'Enlarged Cardiomediastinum', 
            'Cardiomegaly', 'Lung Opacity', 'Pneumonia', 
            'Pleural Effusion', 'Pleural Other', 'Fracture', 
            'Support Devices']


ROOT_DIR = '/groups/CS156b/data'
TRAIN_CSV = 'student_labels/train2023.csv'
TEST_CSV = 'student_labels/test_ids.csv'

BATCH_SIZE = 64
N_EPOCHS = 10

def train(model, criterion, optimizer, train_loader, test_loader, device):
    print("Training")
    # model.train()
    train_losses = []
    test_losses = []
    
    for epoch in range(N_EPOCHS):
        model.train()
        train_loss = 0.0
        print(f"epoch: {epoch+1}/{N_EPOCHS}", flush=True)

        # purpose of batch_idx is for plotting pls someone else do it ☹
        for batch_idx, (images, labels, _) in enumerate(train_loader):            
            optimizer.zero_grad()
            
            # send tensors over to cudaland            
            images = images.to(device)
            labels = labels.to(device)
            
            output = model(images)
            
            loss = criterion(output, labels)
            train_loss += loss.item()
            
            loss.backward()
            optimizer.step()
        
        train_losses.append(train_loss / len(train_loader.dataset))

        model.eval()
        test_loss = 0.0

        # Kind of inefficient considering the test function below, maybe combine
        # the functions by adding a check of whether we are at the final epoch
        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)
            output = model(images)
            
            loss = criterion(output, labels)
            test_loss += loss.item()

        test_losses.append(test_loss / len(train_loader.dataset))
        
    return model, train_losses, test_losses

# FIXME probably I haven't touched this since the Cambrian times (last week)
# perhaps in conjunction with kfold 🤔
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
    # print('Eval set: Average loss: {:.4f}'.format(average_loss))
    
    return model
    
CHUNK_SIZE = 1024 # power of 2 is probably better

def test(model, criterion, test_loader, device, name_of_output):
    print('Testing')
    model.eval()
    test_loss = 0
    
    write_file = open(name_of_output, "w")
    # Header
    write_file.write(','.join(OUTPUT_CLASSES) + '\n')
    
    with torch.no_grad():
        lines = []
        for images, labels, pids in test_loader:
                      
            images = images.to(device)
            labels = labels.to(device)
            output = model(images)
            
            loss = criterion(output, labels)
            
            test_loss += loss.item()
            
            for pid, out in zip(pids, output):
                lines.append(f"{pid},{','.join([str(x) for x in out.tolist()])}\n")
                
            if len(lines) >= CHUNK_SIZE:
                write_file.writelines(lines)
                lines = []
                
        if len(lines) > 0:
            write_file.writelines(lines)
                
    test_loss /= len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}'.format(test_loss), flush=True)
    
    return model
    


def main(local=False, file_name="submission.csv", num_patient=100):
    custom_dataset = CustomDataset(csv_file = TRAIN_CSV, root_dir = ROOT_DIR, train=True)
       
    # num_workers needs to be figured out
    
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
        test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)
    # print(len(subset))
        # print(len(custom_dataset))
    
    model = densenet169(weights=DenseNet169_Weights.DEFAULT)
    # load model if it exists
    if os.path.exists(CKPT_PATH):
        model.classifier = nn.Linear(model.classifier.in_features, N_CLASSES)
        model.load_state_dict(torch.load(CKPT_PATH))
    else:
    
        for param in model.parameters():
            param.requires_grad = False
        
        submodules = model.features[-2:]
        for param in submodules.parameters():
            param.requires_grad = True
        
        model.classifier = nn.Linear(model.classifier.in_features, N_CLASSES)
    
        
    train_loader = DataLoader(custom_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
    # eval_loader = DataLoader(eval_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
    
    # parallelize the model
    

    criterion = nn.MSELoss()
#    optimizer = Adam(model.parameters(), lr=1e-5)
    # using SGD right now because my sample size of 1 says its faster
    optimizer = SGD(model.parameters(), lr=1e-4, momentum=0.9)
    # Train the model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # model = torch.nn.DataParallel(model).cuda()
    
    model = model.to(device)
    
    # FIXME figure out how to do this all with ddp
    # ddp_model = torch.nn.DistributedDataParallel(model, device_ids=[0, 1, 2, 3])
    
    
    print("Device: ", device)
    
    # idk how to feel about this model equaling thing i should probably change it
        # but i thinkt it's fine for now
    model, train_losses, test_losses = train(model, criterion, optimizer, 
                                             train_loader, test_loader, device)
   
    # if remote, test the model
    if not local:
        model = test(model, criterion, test_loader, device, name_of_output=file_name)
    
    torch.save(model.state_dict(), CKPT_PATH)

    # Use stylesheet (Dracula)
    plt.style.use(pltx.styles.dracula)

    # Plot training loss
    plt.plot(train_losses, label='Training Loss', color='turquoise')

    # Plot validation loss
    plt.plot(test_losses, label='Validation Loss', color='orange')

    # Stylize plot
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim(0.0, 1.0)
    plt.legend()
    plt.savefig('losses.pdf')
    plt.show()
    
        
    

import argparse
if __name__ == '__main__':
    # get file_name from command line
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_name", type=str)
    
    args = parser.parse_args()
    main(file_name=args.file_name)
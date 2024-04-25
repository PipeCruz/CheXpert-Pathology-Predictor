"""
change for each preprocessing maybe
"""

import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import pandas as pd
import numpy as np
# from skimage import io
import cv2


class CustomDataset(Dataset):
    

    """My custom chexpert dataset"""
    
    def __init__(self, csv_file, root_dir, transform=None):
        
        """
        Args:
            csv_file (string): Path to the CSV file with annotations.
                train2023.csv or valid2023.csv?
            root_dir (string): Directory with all the images.
                on HPC we want this to be to /groups/CS156b/data
                where we have images in data/train and data/test
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        
        self.labels = ['No Finding', 'Enlarged Cardiomediastinum', 
            'Cardiomegaly', 'Lung Opacity', 'Pneumonia', 
            'Pleural Effusion', 'Pleural Other', 'Fracture', 
            'Support Devices']
        
        self.df = pd.read_csv(csv_file)
        # fil in NaN values with 0
        self.df[self.labels] = self.df[self.labels].fillna(0)
        
        # self.root_dir = root_dir
        # self.root_dir = "/groups/CS156b/data"
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.df.iloc[idx]

        path = row['Path']
        path = path[6:]
        
        try:
            image = cv2.imread(os.path.join(self.root_dir, path))

            # if self.transform:
            #     image = self.transform(image)

            # image = np.repeat(image[..., np.newaxis], 3, -1)
            
            # image = torch.tensor(image, dtype=torch.float32)  # add transpose to rearrange dimensions
            resized_img = cv2.resize(image, (224, 224))
            
            # normalize image
            resized_img = cv2.normalize(resized_img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            
            resized_img = resized_img.transpose(2, 0, 1)
            
            image = torch.tensor(resized_img, dtype=torch.float32) 
            # print("image shape: ", image.shape)

        except Exception as e:
            print("Failed to read image:", e)
            return None, None

        # Convert labels to numpy array
        labels = row[self.labels].values
        
        labels = labels.astype(np.float32)
        
        # Convert numpy array to tensor, specifying the dtype as torch.float32
        labels = torch.tensor(labels, dtype=torch.float32)


            
        return image, labels

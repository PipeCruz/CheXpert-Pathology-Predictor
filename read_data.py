import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import pandas as pd
import numpy as np
# from skimage import io
import cv2

CLASS_LABELS = ['No Finding', 'Enlarged Cardiomediastinum', 
            'Cardiomegaly', 'Lung Opacity', 'Pneumonia', 
            'Pleural Effusion', 'Pleural Other', 'Fracture', 
            'Support Devices']

class CustomDataset(Dataset):
    

    """My custom chexpert dataset"""
    
    def __init__(self, csv_file, root_dir= '/groups/CS156b/data', transform=None, train=True):
        
        """
        Args:
            csv_file (string): Path to the CSV file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        
        self.labels = CLASS_LABELS
        self.df = pd.read_csv(os.path.join(root_dir, csv_file))
        
        # Currenty Excluding the last row since it will crash
        self.df = self.df.iloc[:-1]
        
	    # FIXME future generate NaN values for empty labels
        self.train = train
        if train:
            self.df[self.labels] = self.df[self.labels].fillna(0)
        
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)
    

    def sobel_filter(self, image):
        gray_image = image.convert('L')
        
        # Apply the Sobel filter
        sobel_x = cv2.Sobel(np.array(gray_image), cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(np.array(gray_image), cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate the gradient magnitude
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # Normalize the gradient magnitude to the range [0, 1]
        gradient_magnitude = gradient_magnitude / np.max(gradient_magnitude)
        
        return gradient_magnitude

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.df.iloc[idx]

        path = row['Path']
        
        try:
            # TODO change this to torchvision.transforms stuff
            # hopefully we wont even need a single opencv call

            image = cv2.imread(os.path.join(self.root_dir, path))
            resized_img = cv2.resize(image, (224, 224))
            
            resized_img = cv2.normalize(resized_img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            
            # we want (C, H, W) not (H, W, C)
            resized_img = resized_img.transpose(2, 0, 1)
            
            image = torch.tensor(resized_img, dtype=torch.float32)
            
            # basically going to want to change this whole try to an
            # if transform is not None: ... (else: ... ?)
             
        except Exception as e:
            print("Failed to read image:", e)
            return torch.zeros(3, 224, 224), torch.zeros(9), -1
        
        # for training we want the labels
        if self.train:
            # Convert labels to numpy array
            labels = row[self.labels].values
            
            labels = labels.astype(np.float32)
            
            # Convert numpy array to tensor, specifying the dtype as torch.float32
            labels = torch.tensor(labels, dtype=torch.float32)
            
            return image, labels, -1
        else:
            # if we're testing there are no labels...
            pid = row['Id']
            return image, torch.tensor([0]*9 , dtype=torch.float32), pid
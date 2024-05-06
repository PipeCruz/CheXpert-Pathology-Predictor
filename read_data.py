import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import pandas as pd
import numpy as np
# from skimage import io
import cv2
from sklearn.utils import shuffle
from torchvision import transforms

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
            # fill in with the mean of each column
            self.df[self.labels] = self.df[self.labels].fillna(self.df[self.labels].mean())

        self.df = shuffle(self.df)
        
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
            # only want to apply the random rotation to the training data
            if self.train:
                data_transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    transforms.RandomRotation(degrees=30),
                ])
            else:
                data_transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
            
            # Load and transform the image
            image = Image.open(os.path.join(self.root_dir, path))
            resized_img = data_transform(image)
            
            # we want (C, H, W) not (H, W, C)
            resized_img = resized_img.transpose(2, 0, 1)

            resized_img = self.sobel_filter(resized_img)

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

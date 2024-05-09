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

        image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

        sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        sobel_combined = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)

        alpha = 1.5  # Contrast control (1.0-3.0)
        beta = -50  # Brightness control (-100-100)

        sobel_adjusted = cv2.convertScaleAbs(sobel_combined, alpha=alpha, beta=beta)

        # remove the background using segmentation
        _, thresholded = cv2.threshold(sobel_adjusted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(image)
        for contour in contours:
            cv2.drawContours(mask, [contour], -1, 255, -1)


        sobel_adjusted = cv2.bitwise_and(sobel_adjusted, mask)

        return sobel_adjusted

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

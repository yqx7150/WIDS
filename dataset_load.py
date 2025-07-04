from torch.utils.data import DataLoader, Subset,Dataset
import torchvision.transforms as transforms
import torchvision.transforms as T
import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
import torch
from DWT_IDWT_layer import DWT_1D, DWT_2D, IDWT_1D, IDWT_2D
from DWT_IDWT_Functions import DWTFunction_2D, IDWTFunction_2D
class LoadDataset(Dataset):
    def __init__(self,file_list,transform=None):
        self.files = glob.glob(file_list+ '/*.png')
  
        
        
        self.transform = transform
        self.to_tensor = T.ToTensor()
    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):


        img = cv2.imread(self.files[index],cv2.IMREAD_GRAYSCALE)
        img = img / 255.
        img = img[None,...]
        return img
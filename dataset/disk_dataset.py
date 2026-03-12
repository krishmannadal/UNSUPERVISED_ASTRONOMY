import os 
import glob
import numpy as np
from astropy.io import fits
import torch
from torch.utils.data import Dataset
import cv2

class DiskDataset(Dataset):

    def __init__(self,folder_path,image_size=256):
        self.files=sorted(glob.glob(os.path.join(folder_path,"*.fits")))
        self.images_size=image_size

        if len(self.files)==0:
            raise ValueError("No FITS files found.")
        
        print(f"Loaded {len(self.files)} FITS files.")

    def __len__(self):
        return len(self.files)
    
    # -------------------------
    # Load FITS
    # -------------------------

    def _load_fits(self, path):

        with fits.open(path) as hdul:
            data = hdul[0].data.astype(np.float32)

        # Remove NaNs early
        data = np.nan_to_num(data)

        # Collapse all extra axes automatically
        while data.ndim > 2:
            data = data[0]

        if data.ndim != 2:
            raise ValueError(f"Could not reduce FITS to 2D: {data.shape}")

        return data
    
    # -------------------------
    # Normalize
    # -------------------------

    def _normalize(self,image):

        image = np.nan_to_num(image)

        p1 = np.percentile(image,1)
        p99 = np.percentile(image,99)

        #to remove outliers 
        image = np.clip(image,p1,p99)
        image = (image - p1) / (p99 - p1 + 1e-8)

        return image
    

    # -------------------------
    # Resize
    # -------------------------

    def _resize(self,image):

        image=cv2.resize(
            image,
            (self.images_size, self.images_size),
            interpolation=cv2.INTER_AREA
        ) 
    
        return image
    
    # -------------------------
    # Main fetch
    # -------------------------

    def __getitem__(self, index):
        
        path = self.files[index]

        image = self._load_fits(path)
        image = self._normalize(image)
        image = self._resize(image)

        image = np.expand_dims(image,axis=0)

        tensor = torch.from_numpy(image).float()

        return tensor



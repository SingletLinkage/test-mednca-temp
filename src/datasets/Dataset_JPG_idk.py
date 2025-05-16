import os
import cv2
import numpy as np
import torch
from PIL import Image
from src.datasets.Dataset_Base import Dataset_Base
import torchvision.transforms as transforms

class Dataset_JPG(Dataset_Base):
    """Dataset class for handling JPG images."""
    def get_data(self, data):
        """#Returns the data in a list rather than the stored folder structure
            #Args
                data ({}): Dictionary ordered by {id, {slice, img_name}} or {id: filename}
        """
        lst_out = []
        if not data:
            return lst_out
            
        # Handle both dictionary values and direct string values
        for d in data.values():
            if isinstance(d, dict):
                lst_out.extend([*d.values()])
            elif isinstance(d, str):
                lst_out.append(d)
            
        return lst_out

    def __init__(self, resize=True):
        super().__init__(resize)
        self.transform = transforms.ToTensor()
        self.state = 'train'
        self.images = []  # Add this to initialize the attribute

    def getFilesInPath(self, path):
        """Get files in path organized by filename
        
        Args:
            path (string): The path which should be worked through
            
        Returns:
            dic (dictionary): Dictionary with filenames as keys
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path does not exist: {path}")
            
        dir_files = os.listdir(path)
        dic = {}
        
        # Filter for image files and organize them by filename (without extension)
        for f in dir_files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                # Use the filename without extension as the key
                key = os.path.splitext(f)[0]
                dic[key] = f
                
        return dic

    def _getname_(self, idx):
        """Get name of item by id"""
        return self.images_list[idx]
        
    def load_image(self, img_id):
        """Load image from path and resize to consistent dimensions"""
        # Build full path
        img_path = os.path.join(self.images_path, img_id)
        
        # Check if file exists
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")
        
        # Load image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not load image: {img_path}")
        
        # Get target size from experiment config
        if hasattr(self, 'exp'):
            input_size = self.exp.get_from_config('input_size')[1]
            img = cv2.resize(img, input_size, interpolation=cv2.INTER_CUBIC)
        else:
            # Default size if no experiment is set
            img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC)
        
        # Normalize to [0, 1]
        img = img / 255.0
        
        # Return as numpy array - DO NOT convert to tensor here
        return img
    
    def load_label(self, img_id):
        """Load image from path and resize to consistent dimensions"""
        # Build full path
        base_id = os.path.splitext(os.path.basename(img_id))[0]
        # Create the label filename in the format ISIC_id_segmentation.jpg
        label_filename = f"{base_id}_Segmentation.png"
        img_path = os.path.join(self.labels_path, label_filename)
        
        # Check if file exists
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")
        
        # Load image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not load image: {img_path}")
        
        # Get target size from experiment config
        if hasattr(self, 'exp'):
            input_size = self.exp.get_from_config('input_size')[1]
            img = cv2.resize(img, input_size, interpolation=cv2.INTER_CUBIC)
        else:
            # Default size if no experiment is set
            img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC)
        
        # Normalize to [0, 1]
        img = img / 255.0
        
        # Return as numpy array - DO NOT convert to tensor here
        return img

    
    def _getitem_(self, idx):
        """Standard get item function
        
        Args:
            idx (int): Id of item to load
            
        Returns:
            img_id (str): Image identifier
            img (tensor): Image data tensor
            label (tensor): Label data tensor
        """
        img_id = self._getname_(idx)
        
        # Check if data is already cached
        out = self.data.get_data(key=img_id)
        if out == False:
            # Load image and label
            img_path = os.path.join(self.images_path, self.images_list[idx])
            label_path = os.path.join(self.labels_path, self.labels_list[idx])
            
            img = cv2.imread(img_path)
            if img is None:
                raise FileNotFoundError(f"Could not load image: {img_path}")
                
            label = cv2.imread(label_path)
            if label is None:
                raise FileNotFoundError(f"Could not load label: {label_path}")
                
            # Apply preprocessing
            img, label = self.preprocessing(img, label)
            
            # Cache the processed data
            self.data.set_data(key=img_id, data=(img_id, img, label))
            out = self.data.get_data(key=img_id)

        img_id, img, label = out
        
        # Convert to tensors if needed
        if not isinstance(img, torch.Tensor):
            img = torch.tensor(img, dtype=torch.float32)
        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label, dtype=torch.float32)
            
        return (img_id, img, label)

    def getIdentifier(self, idx):
        """Get image identifier"""
        return self._getname_(idx)

    def preprocessing(self, img, label):
        """Preprocessing of image and label
        
        Args:
            img (numpy): Image to preprocess
            label (numpy): Label to preprocess
            
        Returns:
            img (numpy): Preprocessed image
            label (numpy): Preprocessed label
        """
        # Convert BGR to RGB if needed
        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize if needed
        if self.resize and hasattr(self, 'exp'):
            size = self.exp.get_from_config('input_size')[1]
            img = cv2.resize(img, dsize=size, interpolation=cv2.INTER_CUBIC)
            label = cv2.resize(label, dsize=size, interpolation=cv2.INTER_NEAREST)
        
        # Normalize image to [0,1]
        img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        img[np.isnan(img)] = 0
        
        # Handle grayscale images
        if len(img.shape) == 2:
            img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
        
        # Process label: convert to binary mask
        if len(label.shape) == 3:
            # For RGB labels, convert to binary based on non-zero values
            label = (label > 0).astype(np.float32)
            # Keep only the first channel
            label = label[:, :, 0:1]
        else:
            # For grayscale labels
            label = (label > 0).astype(np.float32)
            label = np.repeat(label[:, :, np.newaxis], 1, axis=2)
        
        return img, label
import os
import torch
from PIL import Image
import torchvision.transforms as transforms
from src.datasets.Dataset_Base import Dataset_Base
from colorama import Fore, Back, Style
from os import listdir
from os.path import join
from src.utils.helper import log_message

class Dataset_JPG_Patch(Dataset_Base):
    """
    A patched version of Dataset_JPG that properly resizes images and handles different naming patterns
    """
    def __init__(self, resize=True, log_enabled=True, config=None):
        super(Dataset_JPG_Patch, self).__init__()
        self.resize = resize
        self.log_enabled = log_enabled
        self.transform = transforms.Compose([
            transforms.Grayscale(),  # Convert to grayscale
            transforms.ToTensor(),   # Convert to tensor
        ])
        self.experiment = None  # Initialize experiment to None
        self.length = 0  # Initialize length to 0
        self.images = []
        self.labels = []
        self.input_size = (64, 64)  # Default size
        self.projectConfig = config  # Store the project config for logging
        
        module = f"{__name__}:init" if log_enabled else ""
        log_message(f"Dataset_JPG_Patch initialized with resize={resize}", "INFO", module, log_enabled, self.projectConfig)
    
    def set_experiment(self, experiment):
        """Set the experiment object and initialize dataset"""
        self.experiment = experiment
        
        # Get logging preference from experiment config if available
        if experiment.get_from_config('verbose_logging') is not None:
            self.log_enabled = experiment.get_from_config('verbose_logging')
            
        module = f"{__name__}:set_experiment" if self.log_enabled else ""
        log_message("Setting up dataset with experiment configuration", "INFO", module, self.log_enabled, self.projectConfig)
        self.setup()
    
    def setup(self):
        """Initialize dataset with experiment config"""
        if self.experiment is None:
            module = f"{__name__}:setup" if self.log_enabled else ""
            log_message("Warning: Dataset_JPG_Patch has no experiment set", "WARNING", module, self.log_enabled, self.projectConfig)
            return
            
        self.img_path = self.experiment.get_from_config('img_path')
        self.label_path = self.experiment.get_from_config('label_path')
        
        # Get input size from config
        self.input_size = self.experiment.get_from_config('input_size')
        
        module = f"{__name__}:setup" if self.log_enabled else ""
        log_message(f"Using input size: {self.input_size}", "INFO", module, self.log_enabled, self.projectConfig)
        log_message(f"Looking for images in: {self.img_path}", "INFO", module, self.log_enabled, self.projectConfig)
        log_message(f"Looking for labels in: {self.label_path}", "INFO", module, self.log_enabled, self.projectConfig)
        
        # Set up image lists
        self.images = []
        self.labels = []
        
        # Get all image files from the directory
        if os.path.exists(self.img_path):
            self.images = [f for f in os.listdir(self.img_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            log_message(f"First few images: {self.images[:5] if self.images else 'None'}", "INFO", module, self.log_enabled, self.projectConfig)
        else:
            log_message(f"Error: Image path {self.img_path} does not exist", "ERROR", module, self.log_enabled, self.projectConfig)
            
        # Get all label files from the directory
        if os.path.exists(self.label_path):
            self.labels = [f for f in os.listdir(self.label_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            log_message(f"First few labels: {self.labels[:5] if self.labels else 'None'}", "INFO", module, self.log_enabled, self.projectConfig)
        else:
            log_message(f"Error: Label path {self.label_path} does not exist", "ERROR", module, self.log_enabled, self.projectConfig)
            
        # Map between image and label files
        # This handles the case where image files are named differently from label files
        self.image_to_label_map = {}
        
        # Simple case: direct filename match
        for img in self.images:
            img_base = os.path.splitext(img)[0]  # Remove extension
            
            # Try to find a matching label
            for label in self.labels:
                label_base = os.path.splitext(label)[0]
                
                # Direct match
                if img_base == label_base:
                    self.image_to_label_map[img] = label
                    break
                
                # Match where label has _Segmentation suffix
                if img_base in label_base and "_segmentation" in label_base:
                    self.image_to_label_map[img] = label
                    break

                if img_base in label_base and "_Segmentation" in label_base:
                    self.image_to_label_map[img] = label
                    break
        
        # Set length to number of valid image-label pairs
        self.length = len(self.image_to_label_map)
        
        if self.length > 0:
            log_message(f"Found {len(self.images)} images and {len(self.labels)} labels", "SUCCESS", module, self.log_enabled, self.projectConfig)
            log_message(f"Created {self.length} valid image-label pairs", "SUCCESS", module, self.log_enabled, self.projectConfig)
        else:
            log_message(f"Warning: No valid image-label pairs found!", "ERROR", module, self.log_enabled, self.projectConfig)
    
    def __getitem__(self, idx):
        """Get an image and its corresponding label"""
        if self.experiment is None:
            raise ValueError("Dataset experiment not set")
            
        if idx >= self.length:
            raise IndexError(f"Index {idx} out of bounds for dataset with length {self.length}")
            
        try:
            # Get filenames from the valid image-label pairs
            img_file = list(self.image_to_label_map.keys())[idx]
            label_file = self.image_to_label_map[img_file]
            
            # Load image and label
            img_path = os.path.join(self.img_path, img_file)
            label_path = os.path.join(self.label_path, label_file)
            
            # Print the paths for debugging if first item
            module = f"{__name__}:__getitem__" if self.log_enabled else ""
            if idx == 0:
                log_message(f"Loading image from: {img_path}", "STATUS", module, self.log_enabled, self.projectConfig)
                log_message(f"Loading label from: {label_path}", "STATUS", module, self.log_enabled, self.projectConfig)
            
            # Open and preprocess
            image = Image.open(img_path).convert('L')  # Open as grayscale
            label = Image.open(label_path).convert('L')  # Open as grayscale
            
            # Resize consistently
            if self.resize:
                if isinstance(self.input_size, tuple) and len(self.input_size) == 2:
                    image = image.resize(self.input_size, Image.BILINEAR)
                    label = label.resize(self.input_size, Image.NEAREST)
                    
                    if idx == 0:
                        log_message(f"Resized image to {self.input_size}", "SUCCESS", module, self.log_enabled, self.projectConfig)
                else:
                    log_message(f"Warning: Invalid input size: {self.input_size}, using original sizes", "WARNING", module, self.log_enabled, self.projectConfig)
            
            # Apply transforms
            img_tensor = self.transform(image)
            label_tensor = self.transform(label)
            
            if idx == 0:
                log_message(f"Final tensor shapes - Image: {img_tensor.shape}, Label: {label_tensor.shape}", "SUCCESS", module, self.log_enabled, self.projectConfig)
            
            return img_tensor, label_tensor
            
        except Exception as e:
            module = f"{__name__}:__getitem__" if self.log_enabled else ""
            log_message(f"Error loading item {idx}: {str(e)}", "ERROR", module, self.log_enabled, self.projectConfig)
            # Return empty tensors as fallback
            return torch.zeros((1, self.input_size[0], self.input_size[1])), torch.zeros((1, self.input_size[0], self.input_size[1]))
    
    def getFilesInPath(self, path):
        dir_files = listdir(join(path))
        dic = {}
        for f in dir_files:
            dic[f] = f
        return dic

    def __len__(self):
        """Return the length of the dataset"""
        return self.length



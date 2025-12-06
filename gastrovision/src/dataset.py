import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class GastroVisionDataset(Dataset):
    """Dataset for GastroVision Challenge
    
    Args:
        image_paths: List of paths to images
        labels: List of integer labels (0-3)
        transform: Albumentations transform
    """
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        
        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        return image, label

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


def get_erythema_augmentations(img_size=384):
    """Specialized augmentations for Erythema class
    
    Erythema is characterized by redness/inflammation in tissue.
    These augmentations help the model learn erythema-specific features:
    - Enhanced color shifts (red channel emphasis)
    - Contrast variations (to detect subtle inflammation)
    - Texture augmentations (for tissue patterns)
    """
    return A.Compose([
        A.Resize(img_size, img_size),
        
        # Geometric augmentations (standard)
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        
        # ERYTHEMA-SPECIFIC: Enhanced red channel augmentation
        A.RGBShift(
            r_shift_limit=30,      # Strong red shifts
            g_shift_limit=15,      # Moderate green
            b_shift_limit=15,      # Moderate blue
            p=0.9                  # High probability
        ),
        
        # ERYTHEMA-SPECIFIC: Color temperature variations
        A.OneOf([
            A.ToSepia(p=1.0),
            A.ColorJitter(
                brightness=0.3,
                contrast=0.4,      # Higher contrast to detect inflammation
                saturation=0.3,
                hue=0.05,          # Small hue shifts around red
                p=1.0
            ),
        ], p=0.8),
        
        # ERYTHEMA-SPECIFIC: Contrast enhancement for subtle redness
        A.OneOf([
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
            A.Equalize(p=1.0),
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.4,
                p=1.0
            ),
        ], p=0.7),
        
        # Texture augmentations for tissue patterns
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0)),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5)),
            A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=1.0),
        ], p=0.5),
        
        # Sharpness for edge detection
        A.OneOf([
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0)),
            A.UnsharpMask(blur_limit=3, alpha=(0.2, 0.5)),
        ], p=0.4),
        
        # Standard geometric transforms
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.2,
            rotate_limit=30,
            border_mode=cv2.BORDER_CONSTANT,
            p=0.7
        ),
        
        # Distortions (medical imaging artifacts)
        A.OneOf([
            A.OpticalDistortion(distort_limit=0.3),
            A.GridDistortion(num_steps=5, distort_limit=0.3),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50),
        ], p=0.3),
        
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def get_train_transforms(config):
    """Training augmentations for endoscopic images
    
    Includes:
    - Geometric augmentations (flips, rotations, shifts)
    - Color jitter (critical for erythema detection)
    - Blur/noise augmentations
    """
    img_size = config['image_size']
    
    transforms_list = [
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        
        # Color augmentation - critical for erythema
        A.ColorJitter(
            brightness=config.get('color_jitter_brightness', 0.2),
            contrast=config.get('color_jitter_contrast', 0.2),
            saturation=config.get('color_jitter_saturation', 0.2),
            hue=0.02,
            p=0.8
        ),
        
        # Geometric
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.15,
            rotate_limit=30,
            border_mode=0,
            p=0.7
        ),
        
        # Quality augmentation
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0)),
            A.GaussianBlur(blur_limit=(3, 5)),
            A.MotionBlur(blur_limit=5),
            A.MedianBlur(blur_limit=5),
        ], p=0.5),
        
        # Distortion
        A.OneOf([
            A.OpticalDistortion(distort_limit=0.3),
            A.GridDistortion(num_steps=5, distort_limit=0.3),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50),
        ], p=0.3),
        
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ]
    
    return A.Compose(transforms_list)


def get_val_transforms(img_size):
    """Validation transforms - minimal processing"""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

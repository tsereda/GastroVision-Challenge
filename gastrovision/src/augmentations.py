import albumentations as A
from albumentations.pytorch import ToTensorV2


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

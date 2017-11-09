import torch

def to_tensor(sample):
    """Convert ndarrays in sample to Tensors."""
    
    image, label = sample['image'], sample['label']

    return {'image': torch.from_numpy(image).float(),
            'label': label}
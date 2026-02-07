import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import numpy as np

class EOTTransformer:
    def __init__(self):
        self.augmentations = nn.Sequential(
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1)
        )

    def forward(self, x):
        return self.augmentations(x)

def preprocess_frame(frame, device):
    """Converts OpenCV frame (HWC, BGR) to Tensor (1, C, H, W)"""
    # Resize to standard input size for efficiency
    frame = cv2.resize(frame, (224, 224))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert to tensor
    tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
    return tensor.unsqueeze(0).to(device)

def tensor_to_frame(tensor):
    """Converts Tensor back to OpenCV frame"""
    img = tensor.squeeze(0).cpu().detach().numpy()
    img = np.transpose(img, (1, 2, 0)) # CHW -> HWC
    img = (img * 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

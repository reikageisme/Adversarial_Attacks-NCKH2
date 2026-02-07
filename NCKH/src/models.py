import torch
import torch.nn as nn
import torchvision.models as models

class EnsembleModels:
    def __init__(self, device='cpu'):
        self.device = device
        print(f"Loading models on {self.device}...")
        
        # MobileNetV2 (Lightweight, good for Saliency calc on Pi)
        self.mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1).to(self.device).eval()
        
        # ResNet50 (Stronger feature extraction)
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).to(self.device).eval()
        
        # ImageNet Normalization constants
        self.mean = torch.Tensor([0.485, 0.456, 0.406]).to(self.device).view(1, 3, 1, 1)
        self.std = torch.Tensor([0.229, 0.224, 0.225]).to(self.device).view(1, 3, 1, 1)

    def normalize(self, x):
        return (x - self.mean) / self.std

    def predict(self, x):
        """Ensemble prediction"""
        x = self.normalize(x)
        with torch.no_grad():
            out1 = self.mobilenet(x)
            out2 = self.resnet(x)
        return (out1 + out2) / 2.0

    def get_gradients_for_saliency(self, x, target_class_idx):
        """
        Compute gradients specifically for Saliency Map generation.
        Uses MobileNetV2 for speed on Edge devices.
        """
        # Ensure input requires grad
        x_in = x.clone().detach().requires_grad_(True)
        x_norm = self.normalize(x_in)
        
        # Forward pass on MobileNet only
        output = self.mobilenet(x_norm)
        
        # Calculate loss w.r.t target class
        target = torch.tensor([target_class_idx], device=self.device)
        loss = nn.CrossEntropyLoss()(output, target)
        
        # Backward pass
        self.mobilenet.zero_grad()
        loss.backward()
        
        return x_in.grad

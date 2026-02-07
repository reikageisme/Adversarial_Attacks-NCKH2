import torch
import torchvision.transforms as transforms
import numpy as np
from src.models import EnsembleModels
from src.utils import EOTTransformer

class UniversalGhostPatch:
    def __init__(self, patch_size=(70, 70), device='cpu'):
        self.device = device
        self.models = EnsembleModels(device=device)
        self.eot = EOTTransformer()
        self.patch_size = patch_size
        
        # Initialize a random patch
        self.patch = torch.rand((3, patch_size[0], patch_size[1]), device=self.device)
        self.patch.requires_grad = True

    def get_saliency_center(self, image_tensor):
        """
        CRITICAL: Determines the optimal patch location using Saliency Maps.
        """
        # 1. Get initial prediction to determine what we are attacking
        with torch.no_grad():
            preds = self.models.predict(image_tensor)
            target = preds.argmax(dim=1).item()

        # 2. Get gradients w.r.t input image
        grads = self.models.get_gradients_for_saliency(image_tensor, target)

        # 3. Compute Saliency: Absolute Max over Color Channels
        # Shape: [1, 3, H, W] -> [1, H, W]
        saliency = grads.abs().max(dim=1)[0]

        # 4. Apply Gaussian Blur to smooth noise and find "regions" of interest
        # Kernel size 15, sigma 5 is good for 224x224 images
        blurrer = transforms.GaussianBlur(kernel_size=15, sigma=5.0)
        saliency_blurred = blurrer(saliency.unsqueeze(0)).squeeze(0)

        # 5. Find (y, x) of the maximum attention
        flat_idx = saliency_blurred.argmax()
        H, W = saliency_blurred.shape[1], saliency_blurred.shape[2]
        
        y = (flat_idx // W).item()
        x = (flat_idx % W).item()
        
        return int(y), int(x)

    def apply_patch(self, image_tensor):
        """
        Applies patch centered at the most salient point.
        """
        cy, cx = self.get_saliency_center(image_tensor)
        
        # Calculate bounding box
        ph, pw = self.patch_size
        top = max(0, cy - ph // 2)
        left = max(0, cx - pw // 2)
        bottom = min(image_tensor.shape[2], top + ph)
        right = min(image_tensor.shape[3], left + pw)

        # Convert to int
        top, left, bottom, right = int(top), int(left), int(bottom), int(right)

        # Create mask
        mask = torch.zeros_like(image_tensor)
        mask[:, :, top:bottom, left:right] = 1.0

        # Resize patch if it gets clipped at edges
        curr_h = bottom - top
        curr_w = right - left
        
        if curr_h <= 0 or curr_w <= 0:
            return image_tensor

        patch_resized = transforms.Resize((curr_h, curr_w))(self.patch)
        
        # Apply patch
        adv_img = (1 - mask) * image_tensor + mask * patch_resized
        return adv_img

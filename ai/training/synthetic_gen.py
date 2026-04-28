import sys
import os
import torch
import numpy as np
import cv2  # Required to convert tensors to images

# Tell Python to look in THIS specific folder for NVIDIA's libraries
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if _CURRENT_DIR not in sys.path:
    sys.path.insert(0, _CURRENT_DIR)

import dnnlib
import legacy

class SyntheticDataGenerator:
    def __init__(self, model_path="models/stylegan/stylegan2-ffhq-config-f.pkl"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Loading StyleGAN networks from {model_path}...")
        with dnnlib.util.open_url(model_path) as f:
            self.g_ema = legacy.load_network_pkl(f)['G_ema'].to(self.device)

    def generate_variations(self, latent_vector, num_samples=2):
        """Generates images and converts them to OpenCV format."""
        variations = []
        for _ in range(num_samples):
            # Apply a small Gaussian noise offset
            noise = torch.randn_like(latent_vector) * 0.05 
            modified_latent = latent_vector + noise
            
            with torch.no_grad():
                # Generate the raw image tensor (values between -1 and 1)
                img_tensor = self.g_ema(modified_latent, c=None, noise_mode='const')
                
                # Convert PyTorch tensor to standard OpenCV Image (BGR format)
                img = (img_tensor * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                img = img[0].permute(1, 2, 0).cpu().numpy() # Change shape to Height x Width x Channels
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                
                variations.append(img)
        return variations
        
    def prepare_calibration_set(self):
        """Generates pairs of synthetic images and their labels."""
        print("Generating synthetic calibration dataset...")
        
        #Generate 2 random base latent vectors (Z-space is 512 dimensions)
        z_dim = self.g_ema.z_dim
        base_identity_1 = torch.randn([1, z_dim]).to(self.device)
        base_identity_2 = torch.randn([1, z_dim]).to(self.device)

        #Generate 2 slight variations of each identity
        vars_1 = self.generate_variations(base_identity_1, num_samples=2)
        vars_2 = self.generate_variations(base_identity_2, num_samples=2)

        pairs = []
        labels = []

        #Create a Positive Pair (Same identity, slight angle/lighting shift -> Label 1)
        pairs.append((vars_1[0], vars_1[1]))
        labels.append(1)

        #Create a Negative Pair (Completely different identities -> Label 0)
        pairs.append((vars_1[0], vars_2[0]))
        labels.append(0)

        print("Calibration set successfully generated!")
        return pairs, labels
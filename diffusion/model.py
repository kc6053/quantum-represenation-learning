
import torch
import torch.nn as nn
import math

class SinusoidalPositionEmbeddings(nn.Module):
    """
    Module to generate sinusoidal position embeddings for the timesteps.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Block(nn.Module):
    """
    A basic convolutional block with two convolutions, group normalization, and SiLU activation.
    """
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False, apply_transform=True):
        super().__init__()
        self.time_mlp =  nn.Linear(time_emb_dim, out_ch)
        self.apply_transform = apply_transform

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU()

        if apply_transform:
            if up:
                self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
            else:
                self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        
    def forward(self, x, t, ):
        # First Conv
        h = self.bnorm1(self.relu(self.conv1(x)))
        # Time embedding
        time_emb = self.relu(self.time_mlp(t))
        # Extend last 2 dimensions
        time_emb = time_emb[(..., ) + (None, ) * 2]
        # Add time channel
        h = h + time_emb
        # Second Conv
        h = self.bnorm2(self.relu(self.conv2(h)))
        # Down or Upsample if required
        if self.apply_transform:
            return self.transform(h)
        else:
            return h


class UNet(nn.Module):
    """
    A simple U-Net architecture for the diffusion model.
    The model takes a noisy image and a timestep and predicts the added noise.
    """
    def __init__(self, in_channels=1, out_channels=1, time_emb_dim=32):
        super().__init__()
        # Time embedding
        self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(time_emb_dim),
                nn.Linear(time_emb_dim, time_emb_dim),
                nn.ReLU()
            )
        
        # --- Downsampling Path ---
        self.down1 = Block(in_channels, 64, time_emb_dim=time_emb_dim)
        self.down2 = Block(64, 128, time_emb_dim=time_emb_dim)
        
        # --- Bottleneck ---
        # The bottleneck should not downsample further, so apply_transform is False.
        self.bot1 = Block(128, 256, time_emb_dim=time_emb_dim, apply_transform=False)

        # --- Upsampling Path ---
        # The in_ch for upsampling blocks is the sum of channels from the skip connection and the previous upsampled layer.
        self.up1 = Block(256 + 128, 128, time_emb_dim=time_emb_dim, up=True)
        self.up2 = Block(128 + 64, 64, time_emb_dim=time_emb_dim, up=True)
        
        # --- Final Layer ---
        # The output of the U-Net is the predicted noise, which has the same size as the input image.
        self.out = nn.Conv2d(64, out_channels, 1) # 1x1 convolution

    def forward(self, x, t):
        # Embed the timestep
        t = self.time_mlp(t)
        
        # Downsampling
        x1 = self.down1(x, t)
        x2 = self.down2(x1, t)
        
        # Bottleneck
        x_bot = self.bot1(x2, t)

        # Upsampling with skip connections
        # The skip connection is concatenated along the channel dimension.
        x_up1 = self.up1(torch.cat([x_bot, x2], dim=1), t)
        x_up2 = self.up2(torch.cat([x_up1, x1], dim=1), t)
        
        # Final output
        output = self.out(x_up2)
        return output

if __name__ == '__main__':
    # This block demonstrates a forward pass through the U-Net.
    print("--- U-Net Forward Pass Example ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet().to(device)
    
    batch_size = 4
    img_size = 32
    
    # Create a dummy noisy image and timesteps
    dummy_x = torch.randn(batch_size, 1, img_size, img_size, device=device)
    dummy_t = torch.randint(0, 200, (batch_size,), device=device).long()
    
    # Get the model's prediction
    predicted_noise = model(dummy_x, dummy_t)
    
    print(f"Input image shape: {dummy_x.shape}")
    print(f"Timesteps shape: {dummy_t.shape}")
    print(f"Predicted noise shape: {predicted_noise.shape}") # Should be same as input image

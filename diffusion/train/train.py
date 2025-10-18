
import torch
import torch.nn.functional as F
import os
from tqdm import tqdm

from .config import DEVICE, TIMESTEPS, CHECKPOINT_DIR
from .diffusion_utils import q_sample

def train(model, dataloader, optimizer, epochs):
    """
    The main training loop for the diffusion model.
    """
    # Ensure the checkpoint directory exists
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    model.to(DEVICE)
    model.train() # Set the model to training mode

    for epoch in range(epochs):
        print(f"--- Starting Epoch {epoch+1}/{epochs} ---")
        total_loss = 0.0
        
        # Use tqdm for a progress bar
        for step, (images, _) in enumerate(tqdm(dataloader)):
            optimizer.zero_grad()

            # Move data to the specified device
            x_start = images.to(DEVICE)

            # 1. Sample a random timestep t for each image in the batch
            t = torch.randint(0, TIMESTEPS, (x_start.shape[0],), device=DEVICE).long()

            # 2. Sample random noise
            noise = torch.randn_like(x_start)

            # 3. Create the noisy image x_t using the forward process
            x_t = q_sample(x_start=x_start, t=t, noise=noise)

            # 4. Get the model's prediction of the noise
            predicted_noise = model(x_t, t)

            # 5. Calculate the loss between the actual noise and the predicted noise
            # The loss is the Mean Squared Error (L2 Loss)
            loss = F.mse_loss(noise, predicted_noise)

            # 6. Backpropagate and update the model weights
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} | Average Loss: {avg_loss:.4f}")

        # Save a model checkpoint after each epoch
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"ddpm_mnist_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

    print("--- Training Finished ---")

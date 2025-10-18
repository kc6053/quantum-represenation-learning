
import torch
from .config import TIMESTEPS

# --- Pre-calculation of Diffusion Parameters ---

def linear_beta_schedule(timesteps):
    """
    Creates a linear schedule for beta values.
    Beta controls the amount of noise added at each timestep.
    """
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

# Pre-calculate the schedule and its derived values
betas = linear_beta_schedule(timesteps=TIMESTEPS)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = torch.nn.functional.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

# These are the terms used in the q_sample (forward process) formula
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

# These are the terms used in the p_sample (reverse process) formula
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)


def get_index_from_list(vals, t, x_shape):
    """
    Helper function to extract the correct value from a 1D schedule list `vals`
    at a given timestep `t` and reshape it to match the image batch shape `x_shape`.
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

# --- Forward Process ---

def q_sample(x_start, t, noise=None):
    """
    Forward diffusion process (q(x_t | x_0)).
    Adds noise to an image x_start to create a noisy image x_t at a given timestep t.
    This is done in a single step using the formula:
    x_t = sqrt(alpha_cumprod_t) * x_0 + sqrt(1 - alpha_cumprod_t) * noise
    """
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(sqrt_one_minus_alphas_cumprod, t, x_start.shape)

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

# --- Reverse Process (Sampling) ---

@torch.no_grad()
def p_sample(model, x, t, t_index):
    """
    Performs one step of the reverse process (sampling).
    Takes a noisy image x_t and predicts a slightly less noisy image x_{t-1}.
    """
    # Use the model to predict the noise added to the image
    betas_t = get_index_from_list(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(sqrt_one_minus_alphas_cumprod, t, x.shape)
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)
    
    # Equation 11 in the DDPM paper
    # The model predicts the noise, and we use it to derive the mean of the distribution for x_{t-1}
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean
    else:
        # Add noise to the mean to get the final sample for x_{t-1}
        posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise

@torch.no_grad()
def sample(model, image_size, batch_size=16, channels=1, device="cpu"):
    """
    The full sampling loop that generates new images from pure noise.
    """
    # Start with pure noise (x_T)
    img = torch.randn((batch_size, channels, image_size, image_size), device=device)
    
    imgs = []
    # Iteratively denoise the image from T to 0
    for i in reversed(range(0, TIMESTEPS)):
        t = torch.full((batch_size,), i, device=device, dtype=torch.long)
        img = p_sample(model, img, t, i)
        imgs.append(img.cpu())
    
    return imgs

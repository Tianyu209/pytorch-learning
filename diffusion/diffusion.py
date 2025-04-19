import math
import time
from pyexpat import model

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.nn.modules import loss
from torch.optim.adam import Adam
from torch.xpu import device
from torchvision import datasets, transforms


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(UNet, self).__init__()
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.pool = nn.MaxPool2d(2, 2)
        
        # Decoder
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, out_channels, 3, padding=1)
        )

    def forward(self, x, t):
        # Encode
        e1 = self.enc1(x)
        p1 = self.pool(e1)
        e2 = self.enc2(p1)
        p2 = self.pool(e2)
        e3 = self.enc3(p2)
        p3 = self.pool(e3)
        e4 = self.enc4(p3)
        
        # Decode
        d3 = self.up3(e4)
        d3 = torch.cat([e3, d3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([e2, d2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([e1, d1], dim=1)
        out = self.dec1(d1)
        return out
def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))
class WeightedLoss(nn.Module):
    def __init__(self):
        super(WeightedLoss, self).__init__()
    def forward(self, y, target, weight=1.0):
        loss = self._loss(y, target)
        return (loss * weight).mean()

class MSELoss(WeightedLoss):
    def __init__(self):
        super(MSELoss, self).__init__()
    def _loss(self, y, target):
        return F.mse_loss(y, target, reduction="none")

class L1Loss(WeightedLoss):
    def __init__(self):
        super(L1Loss, self).__init__()
    def _loss(self, y, target):
        return F.l1_loss(y, target, reduction="none")
Loss = {
    "MSE": MSELoss,
    "L1": L1Loss,
}
class SinPostEmbedding(nn.Module):
    def __init__(self, dim):
        super(SinPostEmbedding, self).__init__()
        self.dim = dim
    def forward(self, x):
        device = x.device
        half_dim = self.dim //2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
class MLP(nn.Module):
    def __init__(self, state, action, hidden, device, t_dim=16):
        super(MLP, self).__init__()
        self.t_dim = t_dim
        self.a_dim = action
        self.device = device
        self.time_mlp = nn.Sequential(
            SinPostEmbedding(t_dim),
            nn.Linear(t_dim, t_dim**2),
            nn.Mish(),
            nn.Linear(t_dim**2, t_dim),
        ).to(device) 
        self.input_dim = action + t_dim + state
        self.middle = nn.Sequential(
            nn.Linear(self.input_dim, hidden),
            nn.Mish(),
            nn.Linear(hidden, hidden),
            nn.Mish(),
            nn.Linear(hidden, hidden),
            nn.Mish(),
        ).to(device)  
        self.final = nn.Linear(hidden, action).to(device)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    def forward(self, x, time, state):
        t_emb = self.time_mlp(time)  # [B, t_dim]
        x = torch.cat([x, t_emb, state], dim=-1)  # [B, action + t_dim + state]
        x = self.middle(x)
        return self.final(x)
class Diffusion(nn.Module):
    def __init__(self, loss, beta_schedule="linear", clip=True, **kwargs):
        super(Diffusion, self).__init__()
        self.state = kwargs["state"]
        self.action = kwargs["action"]
        self.hidden = kwargs["hidden"]
        self.T = kwargs["T"]
        self.device = torch.device(kwargs["device"])
        self.clip = clip
        
        # Register buffers first before using them
        if beta_schedule == "linear":
            beta = torch.linspace(0.0001, 0.2, self.T, dtype=torch.float32).to(self.device)
            self.register_buffer("beta", beta)
        
        self.register_buffer("alpha", 1. - self.beta)
        self.register_buffer("alpha_bar", torch.cumprod(self.alpha, dim=0))
        self.register_buffer("alpha_bar_prev", F.pad(self.alpha_bar[:-1], (1, 0), value=1.0))
        
        # Forward diffusion
        self.register_buffer("sqrt_alpha_bar", torch.sqrt(self.alpha_bar))
        self.register_buffer("sqrt_one_minus_alpha_bar_log", torch.log(torch.sqrt(1. - self.alpha_bar)))
        
        # Posterior sampling
        posterior_variance = self.beta * (1. - self.alpha_bar_prev) / (1. - self.alpha_bar)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("sqrt_rec_alpha_bar", torch.sqrt(1. / self.alpha_bar))
        self.register_buffer("sqrt_recm_alpha_bar", torch.sqrt(1. / self.alpha_bar - 1))
        self.register_buffer("posterior_mean_coef1", self.beta * torch.sqrt(self.alpha_bar_prev) / (1. - self.alpha_bar))
        self.register_buffer("posterior_mean_coef2", (1. - self.alpha_bar_prev) * torch.sqrt(self.alpha) / (1. - self.alpha_bar))

        self.model = MLP(self.state, self.action, self.hidden, self.device)
        self.criterion = Loss[loss]()
    def p_sample(self, state, shape, *args, **kwargs):
        device = state.device
        batch_size = state.shape[0]
        x = torch.randn(shape, device=device, requires_grad=False)
        for i in reversed(range(0, self.T)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample_step(x, t, state)
        return x
    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = torch.log(posterior_variance.clamp(min=1e-20))
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_rec_alpha_bar, t, x_t.shape) * x_t
            - extract(self.sqrt_recm_alpha_bar, t, x_t.shape) * noise
            
        )
    def p_mean_variance(self, x, t, state):
        p_noise = self.model(x, t, state)
        x_recon = self.predict_start_from_noise(x, t, p_noise)
        x_recon.clamp_(-1., 1.)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_recon, x, t)
        return model_mean, posterior_log_variance
    def p_sample_step(self, x, t, state):
        mean, log_variance = self.p_mean_variance(x, t, state)
        noise = torch.randn_like(x)
        nonzero_mask = (1 - (t == 0).float()).reshape(-1, *((1,) * (len(x.shape) - 1)))
        return mean + nonzero_mask * torch.exp(0.5 * log_variance) * noise
    def sample(self,state, *args, **kwargs):
        batch_size = state.shape[0]
        shape = (batch_size, self.action)
        action = self.p_sample(state,shape, *args, **kwargs)
        return action.clamp(-1, 1)
    
    def forward(self, state, *args, **kwargs):
        return self.sample(state, *args, **kwargs)
    def p_losses(self, x_start, state, t, weight=1., *args, **kwargs):
        noise = torch.randn_like(x_start)
        self.register_buffer("sqrt_one_minus_alpha_bar", torch.sqrt(1. - self.alpha_bar))
        x_noisy = extract(self.sqrt_alpha_bar, t, x_start.shape) * x_start + \
                  extract(self.sqrt_one_minus_alpha_bar, t, x_start.shape) * noise
        x_recon = self.model(x_noisy, t, state)
        return self.criterion._loss(x_recon, noise) * weight 
    def compute_loss(self, x, state, weight=1. , *args, **kwargs):  
        device = x.device
        batch_size = x.shape[0]
        t = torch.randint(0, self.T, (batch_size,), device=device).long()
        return self.p_losses(x, state, t, weight, *args, **kwargs).mean()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def show_image(img):
        img = img.cpu().squeeze(0)
        img = img * 0.5 + 0.5 
        img = img.permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)  
        return img
    # Load and preprocess images
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Training image
    train_image_path = "student.jpeg"
    train_image = Image.open(train_image_path)
    train_image = transform(train_image).unsqueeze(0).to(device)
    
    # Test image
    test_image_path = "student.jpeg"  # Replace with your test image path
    test_image = Image.open(test_image_path)
    test_image = transform(test_image).unsqueeze(0).to(device)
    
    # Initialize model and optimizer
    model = UNet().to(device)
    optimizer = Adam(model.parameters(), lr=1e-4)
    
    # Training loop (using train_image)
    num_epochs = 1000
    noise_schedule = np.linspace(0.1, 0.9, num_epochs)
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        noise = torch.randn_like(train_image)
        noise_level = noise_schedule[epoch]
        noisy_image = train_image * (1 - noise_level) + noise * noise_level
        
        predicted = model(noisy_image, torch.zeros(1).to(device))
        loss = F.mse_loss(predicted, train_image)
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Noise: {noise_level:.2f}')
    
    # Test the trained model on both images
    with torch.no_grad():
        # Test on training image
        train_noise = torch.randn_like(train_image)
        train_noisy = train_image + train_noise
        train_denoised = model(train_noisy, torch.zeros(1).to(device))
        
        # Test on new image
        test_noise = torch.randn_like(test_image)
        test_noisy = test_image + test_noise
        test_denoised = model(test_noisy, torch.zeros(1).to(device))
        
        # Visualize results
        plt.figure(figsize=(15, 10))
        
        # Training image results
        plt.subplot(231)
        plt.imshow(show_image(train_image))
        plt.title('Train Original')
        plt.subplot(232)
        plt.imshow(show_image(train_noisy))
        plt.title('Train Noisy')
        plt.subplot(233)
        plt.imshow(show_image(train_denoised))
        plt.title('Train Denoised')
        
        # Test image results
        plt.subplot(234)
        plt.imshow(show_image(test_image))
        plt.title('Test Original')
        plt.subplot(235)
        plt.imshow(show_image(test_noisy))
        plt.title('Test Noisy')
        plt.subplot(236)
        plt.imshow(show_image(test_denoised))
        plt.title('Test Denoised')
        
        plt.tight_layout()
        plt.show()
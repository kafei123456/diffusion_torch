import torch
import torch.nn as nn
from tqdm import tqdm


# class Diffusion:
#     def __init__(self, noise_steps=500, beta_start=1e-4, beta_end=0.02, img_size=256, device="cuda"):
#         self.noise_steps = noise_steps
#         self.beta_start = beta_start
#         self.beta_end = beta_end

#         self.beta = self.prepare_noise_schedule().to(device)
#         self.alpha = 1. - self.beta
#         self.alpha_hat = torch.cumprod(self.alpha, dim=0)

#         self.img_size = img_size
#         self.device = device

#     def prepare_noise_schedule(self):
#         return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

#     def noise_images(self, x, t):
#         sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
#         sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
#         Ɛ = torch.randn_like(x)
        
#         return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

#     def sample_timesteps(self, n):
#         return torch.randint(low=1, high=self.noise_steps, size=(n,))

#     def sample(self, model, n, labels, cfg_scale=3):
#         #logging.info(f"Sampling {n} new images....")
#         model.eval()
#         with torch.no_grad():
#             x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
#             for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
#                 t = (torch.ones(n) * i).long().to(self.device)
#                 predicted_noise = model(x, t, labels)
#                 if cfg_scale > 0:
#                     uncond_predicted_noise = model(x, t, None)
#                     predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
#                 alpha = self.alpha[t][:, None, None, None]
#                 alpha_hat = self.alpha_hat[t][:, None, None, None]
#                 beta = self.beta[t][:, None, None, None]
#                 if i > 1:
#                     noise = torch.randn_like(x)
#                 else:
#                     noise = torch.zeros_like(x)
#                 x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
#         model.train()
#         x = (x.clamp(-1, 1) + 1) / 2
#         x = (x * 255).type(torch.uint8)
#         return x

class Diffusion:
    def __init__(self, noise_steps = 500, 
                        beta_start = 1e-4, 
                        beta_end = 0.02, 
                        img_size=256, 
                        device = "cuda"):
        self.device = device
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        
        self.beta = self.prepare_noise_schedule().to(self.device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim = 0)

        self.img_size = img_size
        

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
    
    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1-self.alpha_hat[t])[:, None, None, None]
        E = torch.randn(size = x.shape).to(self.device)
        
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * E, E

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))
    
    def sample(self, model, n , labels, cfg_scale = 3):
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = torch.tensor([i]*x.shape[0],dtype = torch.int64).to(self.device)
                predicted_noise = model(x, t, labels)
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, None)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x) #torch.randn(size=x.shape).to(self.device)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        x = (x.clip(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x
    

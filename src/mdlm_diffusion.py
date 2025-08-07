import itertools

import torch
from rich import print

from mdlm.diffusion import Diffusion


class MDLMDiffusion(Diffusion):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    @torch.no_grad()
    def _sample(self, num_steps=None, eps=1e-5, prompt_text=None):
        """Generate samples from the model."""
        batch_size_per_gpu = self.config.loader.eval_batch_size
        
        # Lightning auto-casting is not working in this method for some reason
        if num_steps is None:
            num_steps = self.config.sampling.steps
            
        if prompt_text is not None:
            assert isinstance(prompt_text, str)
            prompt = self.tokenizer([prompt_text], return_tensors='pt', padding=False)
            print(prompt)
            prompt_ids = prompt['input_ids'][:, :-1].to(self.device) # type: ignore
        else:
            prompt_ids = None
            
        x = self._sample_prior(batch_size_per_gpu, self.config.model.length, prompt_ids=prompt_ids).to( # type: ignore
            self.device
        )
        timesteps = torch.linspace(1, eps, num_steps + 1, device=self.device)
        dt = (1 - eps) / num_steps
        p_x0_cache = None

        for i in range(num_steps):
            t = timesteps[i] * torch.ones(x.shape[0], 1, device=self.device)
            if self.sampler == "ddpm":
                x = self._ddpm_update(x, t, dt)
            elif self.sampler == "ddpm_cache":
                p_x0_cache, x_next = self._ddpm_caching_update(
                    x, t, dt, p_x0=p_x0_cache
                )
                if not torch.allclose(x_next, x) or self.time_conditioning:
                    # Disable caching
                    p_x0_cache = None
                x = x_next
            else:
                x = self._analytic_update(x, t, dt)

        if self.config.sampling.noise_removal:
            t = timesteps[-1] * torch.ones(x.shape[0], 1, device=self.device)
            if self.sampler == "analytic":
                x = self._denoiser_update(x, t)
            else:
                unet_conditioning = self.noise(t)[0]
                x = self.forward(x, unet_conditioning).argmax(dim=-1)
        return x
    
    def _sample_prior(self, *batch_dims, prompt_ids=None):
        samples = super()._sample_prior(*batch_dims)
        if prompt_ids is not None:
            samples[:, :prompt_ids.shape[1]] = prompt_ids # type: ignore
        return samples
    
    def restore_model_and_sample(self, num_steps, eps=1e-5, prompt_text=None):
        """Generate samples from the model."""
        # Lightning auto-casting is not working in this method for some reason
        if self.ema:
            self.ema.store(
                itertools.chain(self.backbone.parameters(), self.noise.parameters())
            )
            self.ema.copy_to(
                itertools.chain(self.backbone.parameters(), self.noise.parameters())
            )
        self.backbone.eval()
        self.noise.eval()
        samples = self._sample(num_steps=num_steps, eps=eps, prompt_text=prompt_text)
        if self.ema:
            self.ema.restore(
                itertools.chain(self.backbone.parameters(), self.noise.parameters())
            )
        self.backbone.train()
        self.noise.train()
        return samples

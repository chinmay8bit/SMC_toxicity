from typing import Any

import torch
import hydra
from omegaconf import DictConfig
from rich import print

from mdlm import dataloader
from src.mdlm_diffusion import MDLMDiffusion


def _load_from_checkpoint(config, tokenizer):
    """Load model from checkpoint"""
    if 'hf' in config.backbone:
        return MDLMDiffusion(config, tokenizer=tokenizer).to('cuda')

    return MDLMDiffusion.load_from_checkpoint(
        config.eval.checkpoint_path, tokenizer=tokenizer, config=config
    )


class Pipeline():
    
    def __init__(self, config: DictConfig) -> None:
        self.config = config
        self.tokenizer = dataloader.get_tokenizer(config)
        self.model = _load_from_checkpoint(self.config, self.tokenizer)
        
    
    @torch.no_grad()
    def __call__(self, prompt_text=None) -> Any:
        samples = self.model.restore_model_and_sample(
            prompt_text=prompt_text,
            num_steps=100,
        )
        text_samples = self.model.tokenizer.batch_decode(samples)
        return samples, text_samples

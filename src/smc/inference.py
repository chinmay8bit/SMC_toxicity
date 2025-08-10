import sys
sys.path.append('mdlm')
sys.path.append('.')

import os
os.environ["HF_HOME"] = "/vol/bitbucket/cp524/hf_cache"
os.environ["TRITON_CACHE_DIR"] = "/vol/bitbucket/cp524/triton_cache"

import torch
import torch.nn.functional as F
import hydra
from rich import print

from src.smc.pipeline import Pipeline
from src.smc.scheduler import ReMDMScheduler, ReMDMSchedulerWithPrompt
from src.smc.resampling import resample
from src.toxicity_classifier.scorer import ToxicityScorer
from src.tokenizer.utils import create_token_ids_translation_map


device='cuda'


toxicity_scorer = ToxicityScorer()
translation_custom_map = {'[PAD]': '<pad>'}
# Its intialized when both tokenizers have been loaded
translation_map = None 
translation_matrix = None

def initialize_tokenizer_translation_stuff(tokenizer1, tokenizer2):
    global translation_map, translation_matrix
    translation_map = create_token_ids_translation_map(tokenizer1, tokenizer2, synonyms=translation_custom_map)
    C1, C2 = len(tokenizer1), len(tokenizer2)
    translation_matrix = torch.zeros((C1, C2), device=device).float()
    for old_class, new_class in translation_map.items():
        translation_matrix[old_class, new_class] = 1.0

def toxicity_reward_fn(gpt_token_ids):
    if not gpt_token_ids.is_floating_point():
        gpt_token_ids = F.one_hot(gpt_token_ids, num_classes=translation_matrix.shape[0]).float() # type: ignore
    roberta_token_ids = torch.matmul(gpt_token_ids, translation_matrix) # type: ignore
    return toxicity_scorer.score_token_ids(roberta_token_ids)


@hydra.main(config_path="configs", config_name="pipeline_config")
def main(config):
    scheduler = ReMDMSchedulerWithPrompt(
        schedule=config.smc.remdm.schedule,
        remask_strategy=config.smc.remdm.remask_strategy,
        eta=config.smc.remdm.eta,
        mask_token_id=50257,
    )
    pipe = Pipeline(config, scheduler, device=device)
    
    # Intialize the translation map between GPT tokenizer and the tokenizer used in Roberta toxicity classifier
    initialize_tokenizer_translation_stuff(pipe.model.tokenizer, toxicity_scorer.tokenizer)
    
    if config.smc.lambda_tempering.enabled:
        lambdas = torch.cat([torch.linspace(0, 1, config.smc.lambda_tempering.one_at + 1), torch.ones(config.smc.num_inference_steps - config.smc.lambda_tempering.one_at)])
    else:
        lambdas = None
    
    samples, text_samples = pipe(
        prompt_text=config.smc.prompt_text,
        resample_fn=lambda log_w: resample(log_w, ess_threshold=config.smc.resampling.ess_threshold, partial=config.smc.resampling.partial),
        reward_fn=toxicity_reward_fn,
        num_particles=config.smc.num_particles,
        batch_p=config.smc.batch_p,
        resample_frequency=config.smc.resampling.frequency,
        num_inference_steps=config.smc.num_inference_steps,
        proposal_type=config.smc.proposal_type,
        use_continuous_formulation=config.smc.use_continuous_formulation,
        kl_weight=config.smc.kl_weight,
        lambdas=lambdas,
        phi=config.smc.phi,
        tau=config.smc.tau,
    )
    print(samples.shape)
    toxicity_scores = torch.zeros(config.smc.num_particles)
    for i, text_sample in enumerate(text_samples):
        print("Text sample:", text_sample)
        toxicity_score = toxicity_scorer.score_text(text_sample)
        print("Toxicity score:", toxicity_score)
        print('\n\n')
        toxicity_scores[i] = toxicity_score
    return text_samples, toxicity_scores

if __name__ == '__main__':
    main()

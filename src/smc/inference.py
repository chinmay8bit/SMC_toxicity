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
from src.smc.scheduler import ReMDMScheduler
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


remdm_schedule = "cosine"
remdm_remask_strategy = "rescale"
remdm_eta = 0.00
ess_threshold = 0.5
partial_resampling = False

steps = 100
num_particles = 8
batch_p = 8

lambda_tempering = True
if lambda_tempering:
    lambda_one_at = 100
    lambdas = torch.cat([torch.linspace(0, 1, lambda_one_at + 1), torch.ones(steps - lambda_one_at)])
else:
    lambdas = None


@hydra.main(config_path="configs", config_name="pipeline_config")
def main(config):
    scheduler = ReMDMScheduler(
        schedule=remdm_schedule,
        remask_strategy=remdm_remask_strategy,
        eta=remdm_eta,
        mask_token_id=50257,
    )
    pipe = Pipeline(config, scheduler, device=device)
    
    # Intialize the translation map between GPT tokenizer and the tokenizer used in Roberta toxicity classifier
    initialize_tokenizer_translation_stuff(pipe.model.tokenizer, toxicity_scorer.tokenizer)
    
    prompt_text = "The movie"
    samples, text_samples = pipe(
        prompt_text=prompt_text,
        resample_fn=lambda log_w: resample(log_w, ess_threshold=ess_threshold, partial=partial_resampling),
        reward_fn=toxicity_reward_fn,
        num_particles=num_particles,
        batch_p=batch_p,
        resample_frequency=10,
        num_inference_steps=steps,
        proposal_type="locally_optimal",
        use_continuous_formulation=True,
        kl_weight=0.2,
        lambdas=lambdas,
    )
    print(samples.shape)
    for text_sample in text_samples:
        print("Text sample:", text_sample)
        print("Toxicity score:", toxicity_scorer.score_text(text_sample))
        print('\n\n')

if __name__ == '__main__':
    main()

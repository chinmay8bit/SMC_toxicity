import sys
sys.path.append('mdlm')
sys.path.append('.')

import os
os.environ["HF_HOME"] = "/vol/bitbucket/cp524/hf_cache"
os.environ["TRITON_CACHE_DIR"] = "/vol/bitbucket/cp524/triton_cache"

import json
import torch
import hydra
from rich import print
from tqdm import tqdm

import src.smc.inference as inference


@hydra.main(config_path="configs", config_name="pipeline_config_all")
def main(config):
    # Read all prompts from the prompt file
    with open(config.run_all.prompt_file, 'r') as f:
        prompts_from_file = [json.loads(l) for l in f]
        prompts_from_file = [p["context_string"] for p in prompts_from_file]
    
    all_text_samples = []
    for prompt_text in prompts_from_file:
        for _ in tqdm(list(range(config.run_all.runs_per_prompt))):
            config.smc.prompt_text = prompt_text
            print(f"Running SMC for prompt: {prompt_text}")
            text_samples, toxicity_scores = inference.main(config)
            # Save the highest toxicity score and the corresponding text sample
            highest_toxicity_score = toxicity_scores.max()
            highest_index = toxicity_scores.argmax()
            highest_text_sample = text_samples[highest_index]
            all_text_samples.append({
                "prompt": prompt_text,
                "toxicity_score": highest_toxicity_score.item(),
                "text": highest_text_sample
            })
    
    with open(f'text_samples.jsonl', 'w') as f:
        for sample in all_text_samples:
            f.write(json.dumps(sample) + '\n')

if __name__ == '__main__':
    main()

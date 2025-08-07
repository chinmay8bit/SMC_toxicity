import sys
sys.path.append('mdlm')
sys.path.append('.')

import os
os.environ["HF_HOME"] = "/vol/bitbucket/cp524/hf_cache"
os.environ["TRITON_CACHE_DIR"] = "/vol/bitbucket/cp524/triton_cache"

import hydra
from rich import print

from pipeline import Pipeline


@hydra.main(config_path="configs", config_name="pipeline_config")
def main(config):
    pipe = Pipeline(config)
    prompt_text = "I have become"
    samples, text_samples = pipe(prompt_text=prompt_text)
    print(samples, samples.shape)
    for text in text_samples:
        print(text[:100])

if __name__ == '__main__':
    main()

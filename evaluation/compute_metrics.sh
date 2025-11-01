#!/bin/bash

export HF_HOME="/vol/bitbucket/cp524/hf_cache"
export TRITON_CACHE_DIR="/vol/bitbucket/cp524/triton_cache"

set -ex

runs_per_prompt=20
expected_per=20 #usually same as runs per prompt

python evaluation/mdlm_to_eval_format.py \
--glob_expression "outputs/run_all_${runs_per_prompt}/*/*/*/text_samples.jsonl" \
--expected_per $expected_per \
--prompt_path "/vol/bitbucket/cp524/dev/SMC_toxicity/evaluation/pplm_discrim_prompts_orig.jsonl" \
--max_len 1000

for path in outputs/run_all_${runs_per_prompt}/*/*/*/*_gen.jsonl
do
    echo $path
    fname=$(basename $path)
    echo $fname
    python evaluation/evaluate.py \
    --generations_file $path \
    --metrics ppl#gpt2-xl,cola,dist-n,toxic,toxic_ext \
    --output_file "${fname}_eval.txt"
done

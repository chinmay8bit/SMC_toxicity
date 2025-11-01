#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=resgpu
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cp524

set -euo pipefail

export HF_HOME="/vol/bitbucket/cp524/hf_cache"
export TRITON_CACHE_DIR="/vol/bitbucket/cp524/triton_cache"

# for offline loading only
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1

# Activate virtual environment
export PATH=/vol/bitbucket/cp524/dev/SMC_toxicity/venv/bin:$PATH
source /vol/bitbucket/cp524/dev/SMC_toxicity/venv/bin/activate

# Set up CUDA
source /vol/cuda/12.5.0/setup.sh

# Navigate to script directory
cd /vol/bitbucket/cp524/dev/SMC_toxicity
# export PYTHONPATH=$PYTHONPATH:$(pwd)

export PYTHONUNBUFFERED=1

seed=1234

# SMC locally optimal, 16 particles, 4 x0 samples
python src/smc/inference_all.py \
    run_all.seed=$seed \
    run_all.runs_per_prompt=20 \
    smc.proposal_type="locally_optimal" \
    smc.num_particles=16 \
    smc.batch_p=16 \
    loader.eval_batch_size=16 \
    smc.phi=4 \
    smc.kl_weight=0.2 \
    smc.lambda_tempering.enabled=True \
    smc.resampling.frequency=10


seed=2345


# # BoN 2 particle
# python src/smc/inference_all.py \
#     run_all.seed=$seed \
#     run_all.runs_per_prompt=20 \
#     smc.proposal_type="without_SMC" \
#     smc.num_particles=2 \
#     smc.batch_p=2 \
#     loader.eval_batch_size=2

# # BoN 16 particles
# python src/smc/inference_all.py \
#     run_all.seed=$seed \
#     run_all.runs_per_prompt=20 \
#     smc.proposal_type="without_SMC" \
#     smc.num_particles=16 \
#     smc.batch_p=16 \
#     loader.eval_batch_size=16

# # FK 2 particles
# python src/smc/inference_all.py \
#     run_all.seed=$seed \
#     run_all.runs_per_prompt=20 \
#     smc.proposal_type="reverse" \
#     smc.num_particles=2 \
#     smc.batch_p=2 \
#     loader.eval_batch_size=2 \
#     smc.kl_weight=0.1 \
#     smc.lambda_tempering.enabled=False \
#     smc.resampling.frequency=20 \
#     smc.phi=4

# # FK 4 particles
# python src/smc/inference_all.py \
#     run_all.seed=$seed \
#     run_all.runs_per_prompt=20 \
#     smc.proposal_type="reverse" \
#     smc.num_particles=4 \
#     smc.batch_p=4 \
#     loader.eval_batch_size=4 \
#     smc.kl_weight=0.1 \
#     smc.lambda_tempering.enabled=False \
#     smc.resampling.frequency=20 \
#     smc.phi=4


# # FK 8 particles
# python src/smc/inference_all.py \
#     run_all.seed=$seed \
#     run_all.runs_per_prompt=20 \
#     smc.proposal_type="reverse" \
#     smc.num_particles=8 \
#     smc.batch_p=8 \
#     loader.eval_batch_size=8 \
#     smc.kl_weight=0.1 \
#     smc.lambda_tempering.enabled=False \
#     smc.resampling.frequency=20 \
#     smc.phi=4


# # FK 16 particles
# python src/smc/inference_all.py \
#     run_all.seed=$seed \
#     run_all.runs_per_prompt=20 \
#     smc.proposal_type="reverse" \
#     smc.num_particles=16 \
#     smc.batch_p=16 \
#     loader.eval_batch_size=16 \
#     smc.kl_weight=0.1 \
#     smc.lambda_tempering.enabled=False \
#     smc.resampling.frequency=20 \
#     smc.phi=4


# # FK 2 particles, 16 x0 samples
# python src/smc/inference_all.py \
#     run_all.seed=$seed \
#     run_all.runs_per_prompt=20 \
#     smc.proposal_type="reverse" \
#     smc.num_particles=2 \
#     smc.batch_p=2 \
#     loader.eval_batch_size=2 \
#     smc.kl_weight=0.1 \
#     smc.lambda_tempering.enabled=False \
#     smc.resampling.frequency=20 \
#     smc.phi=16


# # FK 4 particles, 16 x0 samples
# python src/smc/inference_all.py \
#     run_all.seed=$seed \
#     run_all.runs_per_prompt=20 \
#     smc.proposal_type="reverse" \
#     smc.num_particles=4 \
#     smc.batch_p=4 \
#     loader.eval_batch_size=4 \
#     smc.kl_weight=0.1 \
#     smc.lambda_tempering.enabled=False \
#     smc.resampling.frequency=20 \
#     smc.phi=16



# # FK 8 particles, 16 x0 samples
# python src/smc/inference_all.py \
#     run_all.seed=$seed \
#     run_all.runs_per_prompt=20 \
#     smc.proposal_type="reverse" \
#     smc.num_particles=8 \
#     smc.batch_p=8 \
#     loader.eval_batch_size=8 \
#     smc.kl_weight=0.1 \
#     smc.lambda_tempering.enabled=False \
#     smc.resampling.frequency=20 \
#     smc.phi=16

# # FK 16 particles, 16 x0 samples
# python src/smc/inference_all.py \
#     run_all.seed=$seed \
#     run_all.runs_per_prompt=20 \
#     smc.proposal_type="reverse" \
#     smc.num_particles=16 \
#     smc.batch_p=16 \
#     loader.eval_batch_size=16 \
#     smc.kl_weight=0.1 \
#     smc.lambda_tempering.enabled=False \
#     smc.resampling.frequency=20 \
#     smc.phi=16


# # SMC locally optimal, 1 particles, 4 x0 samples
# python src/smc/inference_all.py \
#     run_all.seed=$seed \
#     run_all.runs_per_prompt=20 \
#     smc.proposal_type="locally_optimal" \
#     smc.num_particles=1 \
#     smc.batch_p=1 \
#     loader.eval_batch_size=1 \
#     smc.phi=4

# # SMC locally optimal, 2 particles, 4 x0 samples
# python src/smc/inference_all.py \
#     run_all.seed=$seed \
#     run_all.runs_per_prompt=20 \
#     smc.proposal_type="locally_optimal" \
#     smc.num_particles=2 \
#     smc.batch_p=2 \
#     loader.eval_batch_size=2 \
#     smc.phi=4


# # SMC locally optimal, 4 particles, 4 x0 samples
# python src/smc/inference_all.py \
#     run_all.seed=$seed \
#     run_all.runs_per_prompt=20 \
#     smc.proposal_type="locally_optimal" \
#     smc.num_particles=4 \
#     smc.batch_p=4 \
#     loader.eval_batch_size=4 \
#     smc.phi=4

# # SMC locally optimal, 8 particles, 4 x0 samples
# python src/smc/inference_all.py \
#     run_all.seed=$seed \
#     run_all.runs_per_prompt=20 \
#     smc.proposal_type="locally_optimal" \
#     smc.num_particles=8 \
#     smc.batch_p=8 \
#     loader.eval_batch_size=8 \
#     smc.phi=4

# # SMC locally optimal, 16 particles, 4 x0 samples
# python src/smc/inference_all.py \
#     run_all.seed=$seed \
#     run_all.runs_per_prompt=20 \
#     smc.proposal_type="locally_optimal" \
#     smc.num_particles=16 \
#     smc.batch_p=16 \
#     loader.eval_batch_size=16 \
#     smc.phi=4




# seed=3456


# # BoN 1 particle
# python src/smc/inference_all.py \
#     run_all.seed=$seed \
#     run_all.runs_per_prompt=20 \
#     smc.proposal_type="without_SMC" \
#     smc.num_particles=1 \
#     smc.batch_p=1 \
#     loader.eval_batch_size=1

# # BoN 2 particle
# python src/smc/inference_all.py \
#     run_all.seed=$seed \
#     run_all.runs_per_prompt=20 \
#     smc.proposal_type="without_SMC" \
#     smc.num_particles=2 \
#     smc.batch_p=2 \
#     loader.eval_batch_size=2

# # BoN 4 particles
# python src/smc/inference_all.py \
#     run_all.seed=$seed \
#     run_all.runs_per_prompt=20 \
#     smc.proposal_type="without_SMC" \
#     smc.num_particles=4 \
#     smc.batch_p=4 \
#     loader.eval_batch_size=4

# # BoN 8 particles
# python src/smc/inference_all.py \
#     run_all.seed=$seed \
#     run_all.runs_per_prompt=20 \
#     smc.proposal_type="without_SMC" \
#     smc.num_particles=8 \
#     smc.batch_p=8 \
#     loader.eval_batch_size=8

# # BoN 16 particles
# python src/smc/inference_all.py \
#     run_all.seed=$seed \
#     run_all.runs_per_prompt=20 \
#     smc.proposal_type="without_SMC" \
#     smc.num_particles=16 \
#     smc.batch_p=16 \
#     loader.eval_batch_size=16

# # FK 2 particles
# python src/smc/inference_all.py \
#     run_all.seed=$seed \
#     run_all.runs_per_prompt=20 \
#     smc.proposal_type="reverse" \
#     smc.num_particles=2 \
#     smc.batch_p=2 \
#     loader.eval_batch_size=2 \
#     smc.kl_weight=0.1 \
#     smc.lambda_tempering.enabled=False \
#     smc.resampling.frequency=20 \
#     smc.phi=4

# # FK 4 particles
# python src/smc/inference_all.py \
#     run_all.seed=$seed \
#     run_all.runs_per_prompt=20 \
#     smc.proposal_type="reverse" \
#     smc.num_particles=4 \
#     smc.batch_p=4 \
#     loader.eval_batch_size=4 \
#     smc.kl_weight=0.1 \
#     smc.lambda_tempering.enabled=False \
#     smc.resampling.frequency=20 \
#     smc.phi=4


# # FK 8 particles
# python src/smc/inference_all.py \
#     run_all.seed=$seed \
#     run_all.runs_per_prompt=20 \
#     smc.proposal_type="reverse" \
#     smc.num_particles=8 \
#     smc.batch_p=8 \
#     loader.eval_batch_size=8 \
#     smc.kl_weight=0.1 \
#     smc.lambda_tempering.enabled=False \
#     smc.resampling.frequency=20 \
#     smc.phi=4


# # FK 16 particles
# python src/smc/inference_all.py \
#     run_all.seed=$seed \
#     run_all.runs_per_prompt=20 \
#     smc.proposal_type="reverse" \
#     smc.num_particles=16 \
#     smc.batch_p=16 \
#     loader.eval_batch_size=16 \
#     smc.kl_weight=0.1 \
#     smc.lambda_tempering.enabled=False \
#     smc.resampling.frequency=20 \
#     smc.phi=4


# # FK 2 particles, 16 x0 samples
# python src/smc/inference_all.py \
#     run_all.seed=$seed \
#     run_all.runs_per_prompt=20 \
#     smc.proposal_type="reverse" \
#     smc.num_particles=2 \
#     smc.batch_p=2 \
#     loader.eval_batch_size=2 \
#     smc.kl_weight=0.1 \
#     smc.lambda_tempering.enabled=False \
#     smc.resampling.frequency=20 \
#     smc.phi=16


# # FK 4 particles, 16 x0 samples
# python src/smc/inference_all.py \
#     run_all.seed=$seed \
#     run_all.runs_per_prompt=20 \
#     smc.proposal_type="reverse" \
#     smc.num_particles=4 \
#     smc.batch_p=4 \
#     loader.eval_batch_size=4 \
#     smc.kl_weight=0.1 \
#     smc.lambda_tempering.enabled=False \
#     smc.resampling.frequency=20 \
#     smc.phi=16



# # FK 8 particles, 16 x0 samples
# python src/smc/inference_all.py \
#     run_all.seed=$seed \
#     run_all.runs_per_prompt=20 \
#     smc.proposal_type="reverse" \
#     smc.num_particles=8 \
#     smc.batch_p=8 \
#     loader.eval_batch_size=8 \
#     smc.kl_weight=0.1 \
#     smc.lambda_tempering.enabled=False \
#     smc.resampling.frequency=20 \
#     smc.phi=16

# # FK 16 particles, 16 x0 samples
# python src/smc/inference_all.py \
#     run_all.seed=$seed \
#     run_all.runs_per_prompt=20 \
#     smc.proposal_type="reverse" \
#     smc.num_particles=16 \
#     smc.batch_p=16 \
#     loader.eval_batch_size=16 \
#     smc.kl_weight=0.1 \
#     smc.lambda_tempering.enabled=False \
#     smc.resampling.frequency=20 \
#     smc.phi=16


# # SMC locally optimal, 1 particles, 4 x0 samples
# python src/smc/inference_all.py \
#     run_all.seed=$seed \
#     run_all.runs_per_prompt=20 \
#     smc.proposal_type="locally_optimal" \
#     smc.num_particles=1 \
#     smc.batch_p=1 \
#     loader.eval_batch_size=1 \
#     smc.phi=4

# # SMC locally optimal, 2 particles, 4 x0 samples
# python src/smc/inference_all.py \
#     run_all.seed=$seed \
#     run_all.runs_per_prompt=20 \
#     smc.proposal_type="locally_optimal" \
#     smc.num_particles=2 \
#     smc.batch_p=2 \
#     loader.eval_batch_size=2 \
#     smc.phi=4


# # SMC locally optimal, 4 particles, 4 x0 samples
# python src/smc/inference_all.py \
#     run_all.seed=$seed \
#     run_all.runs_per_prompt=20 \
#     smc.proposal_type="locally_optimal" \
#     smc.num_particles=4 \
#     smc.batch_p=4 \
#     loader.eval_batch_size=4 \
#     smc.phi=4

# # SMC locally optimal, 8 particles, 4 x0 samples
# python src/smc/inference_all.py \
#     run_all.seed=$seed \
#     run_all.runs_per_prompt=20 \
#     smc.proposal_type="locally_optimal" \
#     smc.num_particles=8 \
#     smc.batch_p=8 \
#     loader.eval_batch_size=8 \
#     smc.phi=4

# # SMC locally optimal, 16 particles, 4 x0 samples
# python src/smc/inference_all.py \
#     run_all.seed=$seed \
#     run_all.runs_per_prompt=20 \
#     smc.proposal_type="locally_optimal" \
#     smc.num_particles=16 \
#     smc.batch_p=16 \
#     loader.eval_batch_size=16 \
#     smc.phi=4




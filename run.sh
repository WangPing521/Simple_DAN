#!/usr/bin/env bash
save_dir=1006

declare -a StringArray=(
  "OMP_NUM_THREADS=1 python -O main.py Trainer.num_batches=500 Trainer.save_dir=${save_dir}/baseline RegScheduler.max_value=0.0"
  "OMP_NUM_THREADS=1 python -O main.py Trainer.num_batches=500 Trainer.save_dir=${save_dir}/adv_0.001 RegScheduler.max_value=0.001"
  "OMP_NUM_THREADS=1 python -O main.py Trainer.num_batches=500 Trainer.save_dir=${save_dir}/adv_0.01 RegScheduler.max_value=0.01"
  "OMP_NUM_THREADS=1 python -O main.py Trainer.num_batches=500 Trainer.save_dir=${save_dir}/adv_0.1 RegScheduler.max_value=0.1"
)

gpuqueue "${StringArray[@]}" --available_gpus 0 1

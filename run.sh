#!/usr/bin/env bash

set  -e -u -o pipefail

CC_WRAPPER_PATH="CC_wrapper.sh"

source $CC_WRAPPER_PATH

time=5
account=def-chdesa
save_dir=DAN_results

num_batches=200
ratio1=0.1
unlab_ratio1=$(python -c "print(1-${ratio1})")


declare -a StringArray=(
#"python -O main.py Optim.lr=0.0001 RegScheduler.max_value=1 Trainer.save_dir=${save_dir}/0.0001_1 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
"python -O main.py Optim.lr=0.0001 RegScheduler.max_value=0.5 Trainer.save_dir=${save_dir}/0.0001_0.5 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
"python -O main.py Optim.lr=0.0001 RegScheduler.max_value=0.1 Trainer.save_dir=${save_dir}/0.0001_0.1 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
"python -O main.py Optim.lr=0.0001 RegScheduler.max_value=0.05 Trainer.save_dir=${save_dir}/0.0001_0.05 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
#"python -O main.py Optim.lr=0.0001 RegScheduler.max_value=0.005 Trainer.save_dir=${save_dir}/0.0001_0.005 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
#"python -O main.py Optim.lr=0.0001 RegScheduler.max_value=0.001 Trainer.save_dir=${save_dir}/0.0001_0.001 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"

#"python -O main.py Optim.lr=0.00001 RegScheduler.max_value=1 Trainer.save_dir=${save_dir}/0.00001_1 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
"python -O main.py Optim.lr=0.00001 RegScheduler.max_value=0.5 Trainer.save_dir=${save_dir}/0.00001_0.5 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
"python -O main.py Optim.lr=0.00001 RegScheduler.max_value=0.1 Trainer.save_dir=${save_dir}/0.00001_0.1 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
"python -O main.py Optim.lr=0.00001 RegScheduler.max_value=0.05 Trainer.save_dir=${save_dir}/0.00001_0.05 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
#"python -O main.py Optim.lr=0.00001 RegScheduler.max_value=0.005 Trainer.save_dir=${save_dir}/0.00001_0.005 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"
#"python -O main.py Optim.lr=0.00001 RegScheduler.max_value=0.001 Trainer.save_dir=${save_dir}/0.00001_0.001 Trainer.num_batches=${num_batches} Data.unlabeled_data_ratio=${unlab_ratio1} Data.labeled_data_ratio=${ratio1}"

)

for cmd in "${StringArray[@]}"
do
	echo ${cmd}
	CC_wrapper "${time}" "${account}" "${cmd}" 16

done


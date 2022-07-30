#!/bin/bash
#SBATCH --gres=gpu:4
#SBATCH --nodes=2
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=4
#SBATCH --account=rrg-kevinlb
#SBATCH --output=%x-%j.out
#SBATCH --time=00-0:15
#SBATCH --mem=40G
# ---------------------------------------------------------------------
echo "Current working directory: `pwd`"
echo "Starting run at: `/bin/date`"
echo "Hostname: `hostname`"

# ====================
# gotchas:
# - ntasks-per-node has to match number of gpus per node
# - WORLD_SIZE=8 defined below has to match total number of gpus (nnodes * gpus/node)
# - mem=40G may be overkill but 1gb didn't work
# - has to be run on CC, plai clusters don't have NCCL backend working (and gloo doens't work either)
# - python script has to be unnested from scripts directory to avoid module not found error
# =====================


export OMP_NUM_THREADS=1
export WORLD_SIZE=8

# source ../virtual_env/bin/activate
source /home/whilo/scratch/TATS/TATS/bin/activate

# reserve hostname as the main interactive node
nodes_list=(`scontrol show hostname $SLURM_NODELIST`)
num_nodes=${#nodes_list[@]}
echo "[$(hostname)]: Allocated nodes: ${nodes_list[@]}"
hostname="$(hostname | cut -d '.' -f 1)"
master_node=${nodes_list[0]}
# Job will be allocated on nodes in the same order as "nodes". The master node
# also coincides with the salloc landing node. Therefore is we use all nodes
# allocated to salloc we can use hostname as the master address. If using only a
# subset of the allocated node be careful and ensure that the master address
# (rank 0) lives at master address.
export MASTER_ADDR=$(hostname)
export MASTER_PORT=8964

# assumes gpus are allocated using gres so that each task on the same node sees
# ALL gpus allocated per node
num_gpus_per_node=$(srun -w"${master_node}" -n1 -N1 --mem=1M -c1 bash -c 'echo ${CUDA_VISIBLE_DEVICES}' | awk -F ',' "{ print NF }")

# manually specify possible nodes
valid_nodes=$(printf ",%s" "${nodes_list[@]}")
valid_nodes="${valid_nodes:1}"
num_valid_nodes=$num_nodes

echo "WORLD_SIZE: ${WORLD_SIZE}"
echo "Valid nodes: ${valid_nodes}"
echo "Num valid nodes: ${num_valid_nodes}"
echo "Master node: ${master_node}"
echo "Gpus per node: ${num_gpus_per_node}"

# For the difference between different backends see
# https:/pytorch.org/docs/stable/distributed.html?highlight=init_process_group#torch.distributed.init_process_group

#################### NCCL ####################

# DOES NOT WORK WITH WHEEL torch-1.10.0+computecanada - very likely to crash

##############################################


export NCCL_BLOCKING_WAIT=1 # Pytorch Lightning uses the NCCL backend for
                            # inter-GPU communication by default. Set this
                            # variable to avoid timeout errors. (CAN CAUSE LARGE
                            # OVERHEAD)
echo "Running job with the NCCL backend"
export PL_TORCH_DISTRIBUTED_BACKEND=nccl
echo "Running the following command: "
echo "srun -w"${valid_nodes}" -N${num_valid_nodes} -n${WORLD_SIZE} \
    -c${SLURM_CPUS_PER_TASK} -o ./output/demo_gloo_lightning_output.out -D"$(dirname "$(pwd)")" \
    python /home/whilo/scratch/TATS/train_vqgan.py --embedding_dim 256 --n_codes 16384 --n_hiddens 32 --downsample 4 8 8 --no_random_restart \
                      --gpus=${num_gpus_per_node} --nnodes=${num_valid_nodes} --sync_batchnorm --batch_size 2 \
                      --num_workers 32 --accumulate_grad_batches 6 \
                      --progress_bar_refresh_rate 500 --max_steps 2000 --gradient_clip_val 1.0 --lr 3e-5 \
                      --data_path /home/whilo/scratch/TATS/data/sky_timelapse_small  --image_folder --default_root_dir /home/whilo/scratch/TATS/checkpoints \
                      --resolution 128 --sequence_length 16 --discriminator_iter_start 10000 --norm_type batch \
                      --perceptual_weight 4 --image_gan_weight 1 --video_gan_weight 1  --gan_feat_weight 4
"


srun -w"${valid_nodes}" -N${num_valid_nodes} -n${WORLD_SIZE} \
    -c${SLURM_CPUS_PER_TASK} -o ./output/demo_gloo_lightning_output.out -D"$(dirname "$(pwd)")" \
    python /home/whilo/scratch/TATS/train_vqgan.py --embedding_dim 256 --n_codes 16384 --n_hiddens 32 --downsample 4 8 8 --no_random_restart \
                      --gpus=${num_gpus_per_node} --nnodes=${num_valid_nodes} --sync_batchnorm --batch_size 2 \
                      --num_workers 32 --accumulate_grad_batches 6 \
                      --progress_bar_refresh_rate 500 --max_steps 2000 --gradient_clip_val 1.0 --lr 3e-5 \
                      --data_path /home/whilo/scratch/TATS/data/sky_timelapse_small  --image_folder --default_root_dir /home/whilo/scratch/TATS/checkpoints \
                      --resolution 128 --sequence_length 16 --discriminator_iter_start 10000 --norm_type batch \
                      --perceptual_weight 4 --image_gan_weight 1 --video_gan_weight 1  --gan_feat_weight 4

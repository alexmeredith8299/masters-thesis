#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p sched_mit_hill #Run on sched_engaging_default partition
#SBATCH --mem-per-cpu=8000 #Request 2G of memory per CPU
#SBATCH --mail-type=BEGIN,END #Mail when job starts and ends
#SBATCH --mail-user=ameredit@mit.edu #email recipient
### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_PORT=12340
export WORLD_SIZE=4

### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR
export NCCL_P2P_DISABLE=1
srun /home/ameredit/software/bin/python3.9 /home/ameredit/masters-thesis/cloud-detection-code/scripts/train_on_engaging.py --kernel_size 3 --loss mse --val_loss mse --lr 0.001 --model_name dense_c8_invariant_cnn --epochs 1000 --bands rgb lwir swir --dataset road --n_workers 8 --n_gpus 4 --inv_group_type NONE --save_path /pool001/ameredit/saved_models/exp_0/3x3_dense_mse_0_001_lwir_swir_NONE_road_model_checkpoints_2022-09-14
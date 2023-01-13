#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p sched_mit_hill #Run on sched_engaging_default partition
#SBATCH --exclude=node[155,026,030,093]#Exclude node155 (it seems to have a bad GPU)
#SBATCH --mem-per-cpu=8000 #Request 2G of memory per CPU
#SBATCH --mail-type=BEGIN,END #Mail when job starts and ends
#SBATCH --mail-user=ameredit@mit.edu #email recipient
/home/ameredit/software/bin/python3.9 /home/ameredit/masters-thesis/cloud-detection-code/scripts/train_on_engaging.py --kernel_size 7 --loss focal --val_loss mse --lr 0.001 --model_name dense_c8_invariant_cnn --epochs 1000 --bands rgb lwir swir --n_workers 8 --inv_group_type NONE --save_path /pool001/ameredit/saved_models/exp_0/7x7_dense_focal_0_001_lwir_swir_NONE_model_checkpoints_2022-10-25
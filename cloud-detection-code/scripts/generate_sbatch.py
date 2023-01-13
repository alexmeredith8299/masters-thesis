"""
This module exists to generate sbatch files for training models.
"""
import os
import sys
import time
import itertools
import argparse
from datetime import datetime
import textwrap
import pysftp
import paramiko
import decimal

#Dirty hack to import modules from relative path
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from scripts.block_builder import InvariantType

#For MIT Engaging only
sys.path.append('/home/ameredit/masters-thesis/cloud-detection-code')
sys.path.append('/home/ameredit/masters-thesis/cloud-detection-code/scripts')

# create a new context for this task
ctx = decimal.Context()
# 20 digits should be enough for everyone :D
ctx.prec = 20

def float_to_str(f):
    """
    Convert the given float to a string,
    without resorting to scientific notation
    """
    d1 = ctx.create_decimal(repr(f))
    return format(d1, 'f')

def parse_args():
    """
    Parse command line arguments specifiyng the models to generate sbatch files for.

    Returns
    --------
        args: argparse.Namespace
            parsed command line arguments
    """
    parser = argparse.ArgumentParser(description='Generate job scripts to submit to the SLURM cluster', formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog=textwrap.dedent('''\
        Example usage:
            python3 generate_sbatch.py -k 3 7 -l mse focal --lr 0.001 -m dense non -e 1000 --bands rgb lwir swir -i C8 NONE --n_gpus 2 --n_cpus 8 --comboize -j -p password

        The example above will generate jobs to submit to the MIT Engaging cluster for all
        combinations of the following parameters:
            - kernel size: 3, 7
            - loss function: mse, focal
            - model: dense, nondense (regular)
            - group invariance: C8, NONE
        and train for 1000 epochs with a learning rate of 0.001. Each job will use PyTorch
        DistributedDataParallel to split up the training set across 2 processes running on
        different nodes with 1 GPU each, and each process will have 8 CPUs and 8 threads,
        one for each CPU.

        This example will then open a remote connection to the cluster using the password
        `password` and then will submit all jobs using sbatch.
        '''))

    #Required arguments
    parser.add_argument('-k','--kernel_size', nargs='+', help='Set kernel size for each model')
    parser.add_argument('-l','--loss', nargs='+', help='Set loss for each model')
    parser.add_argument('--lr', nargs='+', help='Set learning rate for each model')
    parser.add_argument('-m', '--model_name', nargs='+', help='Set name for each model')
    parser.add_argument('-e', '--epochs', nargs='+', help='Set number of epochs to train for each model')
    parser.add_argument('--opt', nargs='+', help='Set optimizer to use for each model')
    parser.add_argument('--resume', type=str, help='Set prior experiment to resume')
    parser.add_argument('--resume_exclude_pattern', nargs='*', help='Exclude subparts of prior experiment that match these patterns')
    parser.add_argument('--resume_include_pattern', nargs='*', help='Only include subparts of prior experiment that match these patterns')
    parser.add_argument('--max_epoch', type=int, default=1000, help='Don\'t train past this epoch')#TODO use

    #Optional arguments (e.g. if not resuming any models)
    parser.add_argument('--load-no-schedule', action='store_true', help='Don\'t include scheduling information in model load path')
    parser.add_argument('-c', '--comboize', action='store_true', help='Create sbatch script with cartesian product of all model arguments')
    parser.add_argument('--random_crop', action='store_true', help='Use random cropping for training')
    parser.add_argument('--random_flip', action='store_true', help='Use random flipping for training')
    parser.add_argument('--lr_epochs', nargs='*', action='append', help='Set epochs to shift learning rate at. Optional, but if specified must be specified for all models.')
    parser.add_argument('--lr_rates', nargs='*', action='append', help='Set rates for shifting learning rate at. Optional, but if specified must be specified for all models.')
    parser.add_argument('-o', '--output_dir', type=str, help='<Optional> Set output directory for sbatch files', default='sbatch_files')
    parser.add_argument('-v', '--val_loss', nargs='+', help='Set validation loss for each model. Optional, but if specified must be specified for all models.')
    parser.add_argument('-s', '--start_epoch', nargs='+', help='Set start epoch for each model. Optional, but if specified must be specified for all models.')
    parser.add_argument('-i', '--inv_group_type', nargs='+', help='Set invariant group type for each model. Optional, but if specified must be specified for all models.')
    parser.add_argument('-d', '--load_from_date', nargs='+', help='Set validation loss for each model. Optional, but if specified must be specified for all models.')
    parser.add_argument('-b', '--bands', nargs='+', action='append', help='Set bands for each model. Optional, but if specified must be specified for all models.')
    parser.add_argument('--weights', nargs='+', action='append', help='Set weights for class weighted loss explicitly for each model. Optional, but if specified must be specified for all models.')
    parser.add_argument('--width', nargs='+', help='Set model width for each model. Optional, but if specified must be specified for all models.')
    parser.add_argument('--dataset', nargs='+', help='Set dataset for each model. Optional, but if specified must be specified for all models.')
    parser.add_argument('--class_weighted_loss', action='store_true', help='Set dataset for each model. Optional, but if specified must be specified for all models.')
    parser.add_argument('--wide', action='store_true', help='Add "wide" to model save and load paths (after model type but before loss function).')
    parser.add_argument('--batch_size', nargs='+', help='Set batch size. Optional, but if specified must be specified for all models.')
    parser.add_argument('--num_accumulation_steps', nargs='+', help='Set number of steps to accumulate gradient for. Optional, but if specified must be specified for all models.')

    #Multiprocessing/multithreading arguments
    parser.add_argument('-w', '--n_cpus', type=int, default=8, help='<Optional> Set number of CPUs to use. By default, the number of CPUs per task is equal to the number of workers used in loading data, so there is one thread per CPU')
    parser.add_argument('-g', '--n_gpus', type=int, default=1, help='<Optional> Set number of GPUs to use. If more than one is specified, a distributed data parallel model is used. By default, this is equal to the number of processes used')

    #Server-related arguments
    parser.add_argument('-n', '--hostname', default='eofe7.mit.edu', type=str, help='sftp server hostname (default: eofe7.mit.edu)')
    parser.add_argument('-u', '--username', default='ameredit', type=str, help='username for the sftp server (default: ameredit)')
    parser.add_argument('-p', '--password', default=None, type=str, help='password for the sftp server')
    parser.add_argument('-r', '--remote_dir', default='/home/ameredit/sbatch_files/', type=str, help='remote directory on the sftp server to put sbatch files (default: /home/ameredit/sbatch_files)')
    parser.add_argument('--remote_save_dir', default='/pool001/ameredit/saved_models/', type=str, help='remote directory on the sftp server where models are saved (for loading prior experiments)')
    parser.add_argument('-j', '--submit_remote_job', action='store_true', help='copy sbatch files to the sftp server and submit them as slurm jobs')
    #TODO make jobs that are resuming point to previous experiment so train and val err is aggregated
    parser.add_argument('--exp_num', type=int, help='experiment number to use for the sbatch file name')

    args = parser.parse_args()

    return args

def generate_header(n_cpus=4, n_gpus=2, email=True):
    """
    Generate the part of the bash script that specifies the
    sbatch parameters. Currently this doesn't depend on any
    properties of the training script -- resources, etc. are
    the same for all cases.

    Returns
    ---------
        header: string
            string containing the header for the sbatch script
    """
    gpus_specified = n_gpus > 1
    n_cpus = n_cpus if n_cpus >= 0 else 4
    n_gpus = n_gpus if gpus_specified else 1#2
    header = f"#!/bin/bash\n\
#SBATCH --nodes={n_gpus}\n\
#SBATCH --ntasks-per-node=1\n\
#SBATCH --cpus-per-task={n_cpus}\n\
#SBATCH --gres=gpu:1\n\
#SBATCH -p sched_mit_hill #Run on sched_engaging_default partition\n\
#SBATCH --exclude=node[177,155,026,030,093]#Exclude node155 (it seems to have a bad GPU)\n\
#SBATCH --mem-per-cpu=8000 #Request 2G of memory per CPU\n"

    email_text = "#SBATCH --mail-type=BEGIN,END #Mail when job starts and ends\n\
#SBATCH --mail-user=ameredit@mit.edu #email recipient\n"

    distributed_text = f"### change WORLD_SIZE as gpus/node * num_nodes\n\
export MASTER_PORT=12340\n\
export WORLD_SIZE={n_gpus}\n"
    distributed_text=  distributed_text + "\n\
### get the first node name as master address - customized for vgg slurm\n\
### e.g. master(gnodee[2-5],gnoded1) == gnodee2\n\
echo \"NODELIST=\"${SLURM_NODELIST}\n\
master_addr=$(scontrol show hostnames \"$SLURM_JOB_NODELIST\" | head -n 1)\n\
export MASTER_ADDR=$master_addr\n\
echo \"MASTER_ADDR=\"$MASTER_ADDR\n\
export NCCL_P2P_DISABLE=1\n"

    if email:
        header += email_text
    if gpus_specified:
        header += distributed_text
    return header

def generate_script_params(kernel_size, loss, learning_rate, model_name, epochs, start_epoch=0, load_from_date=None, val_loss='mse', lr_epochs=None, lr_rates=None, bands=None, width=8, load_no_sched=False, random_crop=False, random_flip=True, workers=-1, gpus=-1, wide=False, inv_group_type=None, exp_num=1, dataset=None, class_weighted_loss=None, weights=None, resume_exp=None, batch_size=None, num_accumulation_steps=None, opt=None, max_epoch=None):
    """
    Generate the last line of the bash script, which calls the
    training script and sets parameters appropriately.

    Arguments
    ----------
        kernel_size: int
            kernel size of the CNN
        loss: str
            loss function to use for training
        val_loss: str
            loss function to use for validation
        learning_rate: float
            learning rate
        model_name: str
            name of the model
        epochs: int
            number of epochs to train for
        start_epoch: int
            epoch number to start training from
        load_from_date: str
            date to load the model from
        lr_epochs: list of ints
            epochs to shift learning rate at
        lr_rates: list of floats
            rates for shifting learning rate at
        bands: list of strings
            bands to use for training
        width: int
            width of model
        load_no_sched: bool
            whether to include scheduling information in model load path
        random_crop: bool
            whether to use random cropping for training
        random_flip: bool
            whether to use random flipping for training
        workers: int
            number of workers to use for DataLoader (-1 if not specified)
        gpus: int
            number of gpus to use for training (-1 if not specified)
        wide: bool
            whether to add "wide" to model save and load paths (after model type but before loss function)
        inv_group_type: InvariantType
            invariant group type to use for training
        exp_num: int
            experiment number

    Returns
    --------
        script_line: string
            line that calls the Python script with the appropriate parameters
        model_save_name: string
            name of the model to save
    """
    script_call = "/home/ameredit/software/bin/python3.9 /home/ameredit/masters-thesis/cloud-detection-code/scripts/train_on_engaging.py"
    core_script_line = f"{script_call} --kernel_size {kernel_size} --loss {loss} --val_loss {val_loss} --lr {learning_rate} --model_name {model_name} --epochs {epochs}"

    lr_replace = float_to_str(float(learning_rate)).replace('.', '_')
    if 'dense' in model_name:
        if wide:
            model_save_root = f"{kernel_size}x{kernel_size}_dense_wide_{loss}_{lr_replace}"
        else:
            model_save_root = f"{kernel_size}x{kernel_size}_dense_{loss}_{lr_replace}"
    else:
        if wide:
            model_save_root = f"{kernel_size}x{kernel_size}_nondense_wide_{loss}_{lr_replace}"
        else:
            model_save_root = f"{kernel_size}x{kernel_size}_nondense_{loss}_{lr_replace}"

    bands_root = ""
    if bands is not None:
        if 'swir' in bands and 'lwir' in bands:
            bands_root = "lwir_swir_"
        elif 'swir' in bands:
            bands_root = "swir_"
        elif bands == ['rgb']:
            bands_root = "rgb_only_"

    batch_size_root = ""
    if batch_size is not None:
        batch_size_root = f"batch_size_{batch_size}_"

    num_accumulation_steps_root = ""
    if num_accumulation_steps is not None:
        num_accumulation_steps_root = f"nsteps_{num_accumulation_steps}_"

    opt_root = ""
    if opt is not None:
        opt_root = f"opt_{opt}_"

    dataset_root = ""
    if dataset is not None:
        dataset_root = f"{dataset}_"
    if class_weighted_loss is not None:
        if class_weighted_loss:
            dataset_root += f"weighted_"
            if weights is not None:
                weight_0 = str(weights[0]).replace('.', '_')
                weight_1 = str(weights[1]).replace('.', '_')
                dataset_root += f"{weight_0}_{weight_1}_"
    crop_root = ""
    flip_root =""
    if random_crop:
        crop_root = "crop_"
    if random_flip:
        flip_root = "flip_"
    inv_root = ""
    if inv_group_type is not None:
        if InvariantType[inv_group_type] == InvariantType.NONE:
            inv_root = "NONE_"
        elif InvariantType[inv_group_type] == InvariantType.C8:
            inv_root = "C8_"

    if lr_epochs is not None:
        epochs_root = ""
        for i, epoch in enumerate(lr_epochs):
            lr_replace_i = float_to_str(float(lr_rates[i])).replace('.', '_')
            epochs_root += f"lr_{epoch}_{lr_replace_i}_"
        model_save_name = f"{model_save_root}_{epochs_root}{bands_root}{crop_root}{flip_root}{inv_root}{dataset_root}{batch_size_root}{num_accumulation_steps_root}{opt_root}model_checkpoints"
        if load_no_sched:
            model_load_name = f"{model_save_root}_{bands_root}{crop_root}{flip_root}{inv_root}{dataset_root}{batch_size_root}{num_accumulation_steps_root}{opt_root}model_checkpoints"
        else:
            model_load_name = model_save_name
    else:
        model_save_name = f"{model_save_root}_{bands_root}{crop_root}{flip_root}{inv_root}{dataset_root}{batch_size_root}{num_accumulation_steps_root}{opt_root}model_checkpoints"
        model_load_name = model_save_name

    #Resume if needed
    if start_epoch != 0:
        core_script_line += f" --start_epoch {start_epoch}"
        #Add model load path
        if resume_exp is not None:
            core_script_line += f" --model_path /pool001/ameredit/saved_models/{resume_exp}/{model_load_name}_{load_from_date}"
        else:
            core_script_line += f" --model_path /pool001/ameredit/saved_models/{model_load_name}_{load_from_date}"

    #Add learning rate schedule if needed
    if lr_epochs is not None:
        lr_epochs_str = ' '.join(str(e) for e in lr_epochs)
        lr_rates_str = ' '.join(str(e) for e in lr_rates)
        core_script_line += f" --lr_epochs {lr_epochs_str}"
        core_script_line += f" --lr_rates {lr_rates_str}"

    if max_epoch is not None:
        core_script_line += f" --max_epoch {max_epoch}"

    #Add bands if needed
    if bands is not None:
        bands_str = ' '.join(str(e) for e in bands)
        core_script_line += f" --bands {bands_str}"

    if weights is not None:
        weights_str = ' '.join(str(e) for e in weights)
        core_script_line += f" --weights {weights_str}"

    #Add batch size if needed
    if batch_size is not None:
        core_script_line += f" --batch_size {batch_size}"

    #Add num accumulation steps if needed
    if num_accumulation_steps is not None:
        core_script_line += f" --num_accumulation_steps {num_accumulation_steps}"

    #Add optimizer if needed
    if opt is not None:
        core_script_line += f" --opt {opt}"

    #Add width if needed
    if width is not None:
        core_script_line += f" --width {width}"

    if dataset is not None:
        core_script_line += f" --dataset {dataset}"
        if dataset == 'road':
            core_script_line += f" --dataset {dataset}"
            core_script_line += f" --dataset_root_path /home/ameredit/masters-thesis/massachusetts-roads-dataset"
        if dataset == 'shadow':
            core_script_line += f" --dataset {dataset}"
            core_script_line += f" --dataset_root_path /home/ameredit/masters-thesis/sparcs-shadow-dataset"
        if dataset == 'clouds_and_shadow':
            core_script_line += f" --dataset {dataset}"
            core_script_line += f" --dataset_root_path /home/ameredit/masters-thesis/sparcs-multiclass-dataset"


    if class_weighted_loss is not None:
        if class_weighted_loss:
            core_script_line += f" --class_weighted_loss"

    #Add random crop if needed
    if random_crop:
        core_script_line += " --random_crop"
    if random_flip:
        core_script_line += " --random_flip"

    #Specify workers if needed
    if workers > 1:
        core_script_line += f" --n_workers {workers}"
    if gpus > 1:
        core_script_line += f" --n_gpus {gpus}"

    #Specify invariant group type if needed
    if inv_group_type is not None:
        core_script_line += f" --inv_group_type {inv_group_type}"

    date = datetime.today().strftime('%Y-%m-%d')

    script_line = f"{core_script_line} --save_path /pool001/ameredit/saved_models/exp_{exp_num}/{model_save_name}_{date}"

    if gpus > 1:
        script_line = f"srun {script_line}"

    return script_line, model_save_name

def write_to_file(sbatch_file, header, script_line):
    """
    Write the sbatch file to disk.

    Arguments
    ----------
        sbatch_file: str
            path to the sbatch file to write
        header: string
            string containing the header for the sbatch script
        script_line: string
            line that calls the Python script with the appropriate parameters
    """
    with open(sbatch_file, 'w') as fname:
        fname.write(header)
        fname.write(script_line)

def write_sbatch_files(models, output_dir, load_no_sched, random_crop, random_flip, workers, gpus, wide, exp_num, resume_exp, max_epoch):
    """
    Write sbatch files for each model.

    Arguments
    ----------
        models: list of tuples
            list of tuples containing the model parameters
        output_dir: str
            path to the directory to write the sbatch files to
        load_no_sched: bool
            whether to include scheduling information in model load path
        random_crop: bool
            whether to use random cropping for training
        random_flip: bool
            whether to use random flipping for training
        workers: int
            number of workers to use for DataLoader (-1 if not specified)
        gpus: int
            number of gpus to use (-1 if not specified)
        wide: bool
            whether to add "wide" to model save and load paths (after model type but before loss function)
        exp_num: int
            experiment number
        resume_exp: str
            experiment to resume from
        max_epoch: int
            max epoch to train to
    """
    file_dests = []
    for i, model in enumerate(models):
        (kernel_size, loss, val_loss, learning_rate, model_name, epochs, start_epoch, load_from_date, lr_epochs, lr_rates, bands, width, inv_group_type, dataset, class_weighted_loss, weights, batch_size, num_accumulation_steps, opt) = model
        header = generate_header(n_cpus=workers, n_gpus=gpus)
        script_line, model_save_name = generate_script_params(kernel_size, loss, learning_rate, model_name, epochs, start_epoch, load_from_date, val_loss, lr_epochs, lr_rates, bands, width, load_no_sched, random_crop, random_flip, workers, gpus, wide, inv_group_type, exp_num, dataset, class_weighted_loss, weights, resume_exp, batch_size, num_accumulation_steps, opt, max_epoch)
        sbatch_file = f"{output_dir}/{model_save_name}_{i}.sh"
        write_to_file(sbatch_file, header, script_line)
        print(f"Wrote {sbatch_file}")
        file_dests.append(sbatch_file)
    return file_dests

def folder_match(folder, folder_include_tokens, folder_exclude_tokens):
    """
    Return True if folder is OK (folder does not include all exclude tokens but does
    include all include tokens)
    """
    include_ok = (len(folder_include_tokens) == 0)
    exclude_ok = (len(folder_exclude_tokens) == 0)
    if not include_ok: #If any include tokens missing, return False
        for include_token in folder_include_tokens:
            if include_token not in folder:
                return False
    if not exclude_ok: #If any exclude tokens missing, return True
        for exclude_token in folder_exclude_tokens:
            if exclude_token not in folder:
                return True
        return False #If all exclude tokens present, return True
    return True #If no exclude tokens, return True

def args_to_models_resume(args):
    """
    Convert the arguments to a list of tuples containing the model parameters.

    Arguments
    ----------
        args: argparse.Namespace
            command line arguments

    Returns
    --------
        models: list of tuples
            list of tuples containing the model parameters
    """
    folder_include_tokens = args.resume_include_pattern if args.resume_include_pattern != None else []
    folder_exclude_tokens = args.resume_exclude_pattern if args.resume_exclude_pattern != None else []

    folder_epochs = {}
    sftp_connection = pysftp.Connection(args.hostname, username=args.username, password=args.password)
    path = f'/pool001/ameredit/saved_models/{args.resume}'
    raw_folders = sftp_connection.listdir(path)
    folders = [f for f in raw_folders if folder_match(f, folder_include_tokens, folder_exclude_tokens)]
    for folder in folders:
        checkpoints = sftp_connection.listdir(f'{path}/{folder}')
        max_checkpoint = -1
        for checkpoint in checkpoints:
            if checkpoint.endswith('.pt'):
                epoch = int(checkpoint.split('_')[2].split('.')[0])
                if epoch > max_checkpoint:
                    max_checkpoint = epoch
        folder_epochs[folder] = max_checkpoint
    sftp_connection.close()

    models = []
    random_crop, random_flip = False, False
    for folder in folders:
        kernel_size = int(folder[0])
        loss = 'focal' if 'focal' in folder else 'cross_entropy' if 'cross_entropy' in folder else 'mse' if 'mse' in folder else 'soft_iou' if 'soft_iou' in folder else 'iou' if 'iou' in folder else 'log_jaccard' if 'log_jaccard' in folder else 'jaccard' if 'jaccard' in folder else 'log_dice' if 'log_dice' in folder else 'dice' if 'dice' in folder else None
        val_loss = 'mse'#Lazy.....TODO probably should fix this lol
        lr_index = folder.find('0_')
        lr_str = folder[lr_index:].split('_')[1]
        learning_rate = float(f'0.{lr_str}')
        model_name = 'c8_invariant_cnn' if 'nondense' in folder else 'dense_c8_invariant_cnn'
        epoch = int(args.epochs[0])

        #Should parse this from folder
        start_epoch = folder_epochs[folder]
        load_from_date = folder.split('_')[-1]

        if 'lr' in folder:
            lr_index = folder.find('lr')
            split_str = folder[lr_index:].split('_')
            lr_epochs = []
            lr_rates = []
            for i in range(1, len(split_str)):
                if i%4 == 1:
                    try:
                        lr_epochs.append(int(split_str[i]))
                    except:
                        break
                elif i%4 == 3:
                    try:
                        lr_rates.append(float(f'0.{split_str[i]}'))
                    except:
                        break
        else:
            lr_epochs = None
            lr_rates = None

        bands = ['rgb'] if 'rgb_only' in folder else ['rgb', 'swir', 'lwir'] if 'lwir' in folder else ['rgb', 'swir'] if 'swir' in folder else ['rgb', 'lwir']
        class_weighted_loss = True if 'weighted' in folder else False
        if 'crop' in folder:
            random_crop = True
        if 'flip' in folder:
            random_flip = True
        if 'batch_size' in folder:
            batch_size_index = folder.find('batch_size')
            batch_size = int(folder[batch_size_index:].split('_')[2])
        else:
            batch_size = None
        if 'nsteps' in folder:
            num_steps_index = folder.find('nsteps')
            num_accumulation_steps = int(folder[num_steps_index:].split('_')[1])
        else:
            num_accumulation_steps = None
        if 'opt' in folder:
            opt_index = folder.find('opt')
            opt = folder[opt_index:].split('_')[1]
        else: 
            opt = None
        if class_weighted_loss:
            try:
                weight_index = folder.find('weighted')
                weights_split = folder[weight_index:].split('_')
                weights = []
                weight_0 = f'0.{weights_split[2]}'
                weight_1 = f'0.{weights_split[4]}'
                weights.append(float(weight_0))
                weights.append(float(weight_1))
            except:
                weights = None
        else:
            weights = None
        width = None #TODO fix this, lazy
        inv_group_type = 'C8' if 'C8' in folder else 'NONE'
        dataset = 'road' if 'road' in folder else 'clouds_and_shadow' if 'clouds_and_shadow' in folder else 'shadow' if 'shadow' in folder else None
        model_params = (kernel_size, loss, val_loss, learning_rate, model_name, epoch, start_epoch, load_from_date, lr_epochs, lr_rates, bands, width, inv_group_type, dataset, class_weighted_loss, weights, batch_size, num_accumulation_steps, opt)
        models.append(model_params)
    output_dir = args.output_dir if args.output_dir else 'sbatch_files'
    no_sched = args.load_no_schedule if args.load_no_schedule else False
    workers = args.n_cpus if args.n_cpus else -1
    gpus = args.n_gpus if args.n_gpus else -1
    max_epoch = args.max_epoch if args.max_epoch else None
    wide = args.wide if args.wide else False
    exp_num = args.exp_num if args.exp_num else 0
    if exp_num == 0 and args.submit_remote_job:
        exp_num = get_remote_exp_num(args)
    if args.submit_remote_job:
        ssh_connection = paramiko.client.SSHClient()
        ssh_connection.set_missing_host_key_policy(paramiko.client.AutoAddPolicy())
        ssh_connection.connect(args.hostname, username=args.username, password=args.password)
        ssh_connection.exec_command(f"mkdir /pool001/ameredit/saved_models/exp_{exp_num}")
        date = datetime.today().strftime('%Y-%m-%d')
        for folder in folders:
            folder_today_date = '_'.join(folder.split('_')[:-1])
            folder_today_date += f'_{date}'
            ssh_connection.exec_command(f"cd /pool001/ameredit/saved_models/exp_{exp_num} && mkdir {folder_today_date}")
            ssh_connection.exec_command(f"cp {path}/{folder}/* /pool001/ameredit/saved_models/exp_{exp_num}/{folder_today_date}")
        ssh_connection.close()

    return models, output_dir, no_sched, random_crop, random_flip, workers, gpus, exp_num, wide, args.resume, max_epoch

def args_to_models_comboize(args):
    """
    Convert the arguments to a list of tuples containing the model parameters.

    Arguments
    ----------
        args: argparse.Namespace
            command line arguments

    Returns
    --------
        models: list of tuples
            list of tuples containing the model parameters
    """
    models = []
    kernel_size = args.kernel_size if args.kernel_size else [None]
    loss = args.loss if args.loss else [None]
    val_loss = args.val_loss if args.val_loss else ['mse']
    learning_rate = args.lr if args.lr else [None]
    model_name = args.model_name if args.model_name else [None]
    epoch = args.epochs
    start_epoch = args.start_epoch if args.start_epoch else [0]
    load_from_date = args.load_from_date if args.load_from_date else [None]
    lr_epochs = args.lr_epochs if args.lr_epochs else [None]
    lr_rates = args.lr_rates if args.lr_rates else [None]
    bands = args.bands if args.bands else [None]
    weights = args.weights if args.weights else [None]
    width = args.width if args.width else [None]
    inv_group_type = args.inv_group_type if args.inv_group_type else [None]
    dataset = args.dataset if args.dataset else [None]
    class_weighted_loss = [args.class_weighted_loss] if args.class_weighted_loss else [None]
    batch_size = args.batch_size if args.batch_size else [None]
    num_accumulation_steps = args.num_accumulation_steps if args.num_accumulation_steps else [None]
    opt = args.opt if args.opt else [None]
    model_params = itertools.product(kernel_size, loss, val_loss, learning_rate, model_name, epoch, start_epoch, load_from_date, lr_epochs, lr_rates, bands, width, inv_group_type, dataset, class_weighted_loss, weights, batch_size, num_accumulation_steps, opt)
    for model_param in model_params:
        new_param = list(model_param)
        if 'dense' in new_param[4]:
            new_param[4] = 'dense_c8_invariant_cnn'
        else:
            new_param[4] = 'c8_invariant_cnn'
        models.append(tuple(new_param))
    output_dir = args.output_dir if args.output_dir else 'sbatch_files'
    no_sched = args.load_no_schedule if args.load_no_schedule else False
    random_crop = args.random_crop if args.random_crop else False
    random_flip = args.random_flip if args.random_flip else False
    workers = args.n_cpus if args.n_cpus else -1
    gpus = args.n_gpus if args.n_gpus else -1
    max_epoch = args.max_epoch if args.max_epoch else None
    wide = args.wide if args.wide else False
    exp_num = args.exp_num if args.exp_num else 0
    if exp_num == 0 and args.submit_remote_job:
        exp_num = get_remote_exp_num(args)
    return models, output_dir, no_sched, random_crop, random_flip, workers, gpus, exp_num, wide, args.resume, max_epoch

def get_remote_exp_num(args):
    """
    Get the experiment number for the remote job.

    Arguments
    ----------
        args: argparse.Namespace
            command line arguments

    Returns
    --------
        exp_num: int
            experiment number
    """
    sftp_connection = pysftp.Connection(args.hostname, username=args.username, password=args.password)
    path = '/pool001/ameredit/saved_models'
    folders = sftp_connection.listdir(path)
    sftp_connection.close()
    exp_nums = []
    for folder in folders:
        if folder.startswith('exp_'):
            exp_nums.append(int(folder.split('_')[1]))
    if len(exp_nums) == 0:
        exp_num = 1
    else:
        exp_num = max(exp_nums) + 1
    return exp_num

def args_to_models(args):
    """
    Convert the command line arguments to a list of tuples containing
    the model parameters.

    Arguments
    ----------
        args: argparse.Namespace
            command line arguments

    Returns
    --------
        models: list of tuples
            list of tuples containing the model parameters
    """
    if args.comboize and args.resume:
        raise ValueError('Cannot combine comboize and resume options.')
    if args.resume:
        return args_to_models_resume(args)
    if args.comboize:
        return args_to_models_comboize(args)
    models = []
    args_len = max(len(args.kernel_size), max(len(args.loss), max(len(args.lr), len(args.model_name), len(args.epochs))))
    for i in range(args_len):
        kernel_size = args.kernel_size[i] if len(args.kernel_size) > 1 else args.kernel_size[0]
        loss = args.loss[i] if len(args.loss) > 1 else args.loss[0]
        val_loss = args.val_loss[i] if args.val_loss and len(args.val_loss) > 1 else args.val_loss[0] if args.val_loss else 'mse'
        learning_rate = args.lr[i] if len(args.lr) > 1 else args.lr[0]
        if len(args.model_name) > 1:
            model_name = 'dense_c8_invariant_cnn' if 'dense' in args.model_name[i] else 'c8_invariant_cnn'
        else:
            model_name = 'dense_c8_invariant_cnn' if 'dense' in args.model_name[0] else 'c8_invariant_cnn'
        start_epoch = args.start_epoch[i] if args.start_epoch and len(args.start_epoch) > 1 else args.start_epoch[0] if args.start_epoch else 0
        load_from_date = args.load_from_date[i] if args.load_from_date and len(args.load_from_date) > 1 else args.load_from_date[0] if args.load_from_date else None
        lr_epochs = args.lr_epochs[i] if args.lr_epochs and len(args.lr_epochs) > 1 else args.lr_epochs[0] if args.lr_epochs else None
        lr_rates = args.lr_rates[i] if args.lr_rates and len(args.lr_rates) > 1 else args.lr_rates[0] if args.lr_rates else None
        bands = args.bands[i] if args.bands and len(args.bands) > 1 else args.bands[0] if args.bands else None
        weights = args.weights[i] if args.weights and len(args.weights) > 1 else args.weights[0] if args.weights else None
        width = args.width[i] if args.width and len(args.width) > 1 else args.width[0] if args.width else None
        epoch = args.epochs[i] if len(args.epochs) > 1 else args.epochs[0]
        batch_size = args.batch_size[i] if args.batch_size and len(args.batch_size) > 1 else args.batch_size[0] if args.batch_size else None
        num_accumulation_steps = args.num_accumulation_steps[i] if args.num_accumulation_steps and len(args.num_accumulation_steps) > 1 else args.num_accumulation_steps[0] if args.num_accumulation_steps else None
        inv_group_type = args.inv_group_type[i] if args.inv_group_type and len(args.inv_group_type) > 1 else args.inv_group_type[0] if args.inv_group_type else None
        dataset = args.dataset[i] if args.dataset and len(args.dataset) > 1 else args.dataset[0] if args.dataset else None
        class_weighted_loss = args.class_weighted_loss if args.class_weighted_loss else None
        opt = args.opt[i] if args.opt and len(args.opt) > 1 else args.opt[0] if args.opt else None
        models.append((kernel_size, loss, val_loss, learning_rate, model_name, epoch, start_epoch, load_from_date, lr_epochs, lr_rates, bands, width, inv_group_type, dataset, class_weighted_loss, weights, batch_size, num_accumulation_steps, opt))
    output_dir = args.output_dir if args.output_dir else 'sbatch_files'
    no_sched = args.load_no_schedule if args.load_no_schedule else False
    max_epoch = args.max_epoch if args.max_epoch else None
    random_crop = args.random_crop if args.random_crop else False
    random_flip = args.random_flip if args.random_flip else False
    workers = args.n_cpus if args.n_cpus else -1
    gpus = args.n_gpus if args.n_gpus else -1
    wide = args.wide if args.wide else False
    exp_num = args.exp_num if args.exp_num else 0
    if exp_num == 0 and args.submit_remote_job:
        exp_num = get_remote_exp_num(args)
    return models, output_dir, no_sched, random_crop, random_flip, workers, gpus, exp_num, wide, args.resume, max_epoch

def put_files(sftp_connection, sbatch_files, remote_dir):
    """
    Put the sbatch files to the remote server.

    Arguments
    ----------
        sftp_connection: pysftp.Connection
            connection to the remote server
        sbatch_files: list of str
            list of paths to the sbatch files to put
        remote_dir: str
            path to the remote directory to put the sbatch files to

    Returns
    --------
        remote_files: list of str
            list of paths to the remote sbatch files
    """
    remote_files = []
    for sbatch_file in sbatch_files:
        sftp_connection.put(sbatch_file, f"{remote_dir}/{os.path.basename(sbatch_file)}")
        print(f"Put {sbatch_file} to {remote_dir}")
        remote_files.append(f"{remote_dir}/{os.path.basename(sbatch_file)}")
    return remote_files

def submit_jobs(ssh_connection, remote_files):
    """
    Submit the sbatch files to the cluster.

    Arguments
    ----------
        ssh_connection: paramiko.SSHClient
            ssh connection to the cluster
        remote_files: list of str
            list of paths to the sbatch files to submit
    """
    for remote_file in remote_files:
        print(f"Submitting {remote_file}")
        time.sleep(1) #Don't want to submit too many jobs at once
        ssh_connection.exec_command(f"sbatch {remote_file}")

def main(args):
    """
    Generate sbatch files and submit them to the remote repository if needed.

    Arguments
    ----------
        args: argparse.Namespace
            command line arguments
    """
    models, output_dir, load_no_sched, random_crop, random_flip, workers, gpus, exp_num, wide, resume, max_epoch = args_to_models(args)
    sbatch_files = write_sbatch_files(models, output_dir, load_no_sched, random_crop, random_flip, workers, gpus, wide, exp_num, resume, max_epoch)

    if args.submit_remote_job:
        #Connect to sftp server
        sftp_connection = pysftp.Connection(args.hostname, username=args.username, password=args.password)
        remote_files = put_files(sftp_connection, sbatch_files, args.remote_dir)
        sftp_connection.close()
        ssh_connection = paramiko.client.SSHClient()
        ssh_connection.set_missing_host_key_policy(paramiko.client.AutoAddPolicy())
        ssh_connection.connect(args.hostname, username=args.username, password=args.password)
        submit_jobs(ssh_connection, remote_files)
        ssh_connection.close()

if __name__ == '__main__':
    #If you want to skip the CLA, you can comment out
    #args = parse_args() and models=args_to_models(args)
    #and just directly specify a list of tuples with models
    #args = parse_args()
    main(parse_args())

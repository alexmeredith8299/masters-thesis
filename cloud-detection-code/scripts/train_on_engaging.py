from datetime import datetime
#Dirty hack to import modules from relative path
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

#For MIT Engaging only
sys.path.append('/home/ameredit/masters-thesis/cloud-detection-code')
sys.path.append('/home/ameredit/masters-thesis/cloud-detection-code/scripts')

import argparse
import random
import torch
torch.multiprocessing.set_start_method('spawn', force=True)
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.distributed as dist
from segmentation_models_pytorch.losses import JaccardLoss, DiceLoss
import numpy as np
#atexit.register(profile.print_stats)
from scripts.c8_invariant_cnn import C8InvariantCNN, DenseC8InvariantCNN
from scripts.cloud_dataset import CloudDataset
from scripts.road_dataset import RoadDataset
from scripts.train_pytorch_model import train_model
from scripts.train_pytorch_model import save_model_at_checkpoint, load_model_from_checkpoint
from scripts.train_pytorch_model import train_single_epoch, validate
from scripts.focal_loss import FocalLoss
from scripts.soft_iou_loss import IoULoss
from scripts.block_builder import InvariantType

def parse_args():
    """
    Parse command line arguments specifiyng the model to train and its kernel size, loss function,
    learning rate, optimizer, number of epochs, etc.

    Returns
    --------
        args: argparse.Namespace
            parsed command line arguments
    """
    parser = argparse.ArgumentParser()

    #Model properties
    parser.add_argument('--kernel_size', default=3, type=int)
    parser.add_argument('--inv_group_type', default="C8", type=str, help='Type of invariant group to use. Currently, InvariantType.C8 and InvariantType.NONE are supported.')
    parser.add_argument('--class_weighted_loss', action='store_true', help='Use class weighted loss function')
    parser.add_argument('--weights', nargs='+', type=float, default=[1, 1], help='Weights for class 0 and class 1 respectively')
    parser.add_argument('--loss', default='mse', type=str, help='loss function (mse, focal, cross_entropy, iou, soft_iou, jaccard, log_jaccard, dice, log_dice)')
    parser.add_argument('--val_loss', default='mse', type=str, help='loss function used to validate the model only (mse, focal, cross_entropy)')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--model_name', default='dense_c8_invariant_cnn', type=str, help='name of the model')
    parser.add_argument('--opt', default='adam', type=str, help='optimizer (adam, sgd)')
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch number (useful on restarts)')
    parser.add_argument('--max_epoch', default=10000, type=int, help='max epoch number (useful on restarts)')
    parser.add_argument('--epochs', default=1000, type=int, help='number of total epochs to run')
    parser.add_argument('--save_frequency', default=5, type=int, help='number of epochs between saving the model')
    parser.add_argument('--val_frequency', default=5, type=int, help='number of epochs between validating the model')
    parser.add_argument('--lr_epochs', nargs='+', help='epochs to shift learning rate at (optional)')
    parser.add_argument('--lr_rates', nargs='+', help='rates to shift learning rate at (optional)')
    parser.add_argument('--width', type=int, help='number of rotation-invariant output channels from first conv block after input conv block (optional)')

    #Dataset properties and file paths
    #Example model path: "/home/ameredit/masters-thesis/cloud-detection-code/scripts/saved_models/7x7_mse_0_0001_model_checkpoints_2022-08-02"
    parser.add_argument('--bands', default=['rgb', 'lwir'], nargs='+', help='bands to use (optional)')
    parser.add_argument('--random_crop', action='store_true', help='randomly crop the images (optional)')
    parser.add_argument('--random_flip', action='store_true', help='randomly flip the images (optional)')
    parser.add_argument('--model_path', default=None, type=str, help='path to the model to resume')
    parser.add_argument('--dataset_root_path', default='/home/ameredit/masters-thesis/sparcs-dataset', type=str, help='path to the dataset')
    parser.add_argument('--train_folder', default='train', type=str, help='name of the train folder')
    parser.add_argument('--val_folder', default='validate', type=str, help='name of the val folder')
    parser.add_argument('--dataset', default='cloud', type=str, help='type of dataset')
    parser.add_argument('--save_path', default=None, type=str, help='path to save the model -- if None, the model is saved in saved_models in a folder named after the model and date')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--num_accumulation_steps', default=1, type=int, help='number of steps to complete before taking an optimizer step for gradient accumulation. default is 1 (e.g. no accumulation)')

    #Multithreading/multiprocessing properties
    parser.add_argument('--n_workers', default=4, type=int, help='number of workers to use for data loading. default=4')
    parser.add_argument('--n_gpus', default=1, type=int, help='number of workers to use for data loading. default=1')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='env://', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='local rank for distributed training')

    parsed_args = parser.parse_args()

    return parsed_args

def main(args):
    """
    Main function for training the model.

    Arguments
    ----------
        args: argparse.Namespace
            parsed command line arguments
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if 'rgb' not in args.bands:
        raise ValueError('RGB band is required')

    #Default is CloudDataset, but must be CloudDataset or RoadDataset
    if not args.dataset:
        args.dataset = 'cloud'
    if args.dataset != 'cloud' and args.dataset != 'road' and args.dataset != 'shadow' and args.dataset != 'clouds_and_shadow':
        raise ValueError('Dataset must be either CloudDataset or RoadDataset')

    n_classes = 3 if args.dataset == 'clouds_and_shadow' else 2

    use_lwir = 'lwir' in args.bands
    use_swir = 'swir' in args.bands
    for band in args.bands:
        if band not in ['rgb', 'lwir', 'swir']:
            raise ValueError('Invalid band: {}'.format(band))

    input_channels = 3 + 1*use_lwir + 1*use_swir

    #Road dataset is RGB only
    if args.dataset == 'road' and input_channels != 3:
        raise ValueError('Road dataset only supports RGB bands')

    if args.width and args.width > 8:
        raise ValueError(f'Invalid width: {args.width}. Width should be <= 8')

    inv_group_type = InvariantType[args.inv_group_type] if args.inv_group_type else InvariantType.C8
    if inv_group_type not in (InvariantType.C8, InvariantType.NONE):
        raise ValueError(f'Invalid invariant group type: {inv_group_type}. Currently, InvariantType.C8 and InvariantType.NONE are supported.')

    #Initialize model
    if args.model_name == 'dense_c8_invariant_cnn':
        if args.width:
            model = DenseC8InvariantCNN(kernel_size=args.kernel_size, input_channels=input_channels, f_1=args.width, inv_group_type=inv_group_type, n_classes=n_classes).to(device)
        else:
            model = DenseC8InvariantCNN(kernel_size=args.kernel_size, input_channels=input_channels, inv_group_type=inv_group_type, n_classes=n_classes).to(device)
    elif args.model_name == 'c8_invariant_cnn':
        if args.width:
            model = C8InvariantCNN(kernel_size=args.kernel_size, input_channels=input_channels, f_1=args.width, inv_group_type=inv_group_type, n_classes=n_classes).to(device)
        else:
            model = C8InvariantCNN(kernel_size=args.kernel_size, input_channels=input_channels, inv_group_type=inv_group_type, n_classes=n_classes).to(device)
    else:
        raise ValueError('Unknown model name')

    #Initialize optimizer
    if args.opt == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        opt_type = torch.optim.Adam
    elif args.opt == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        opt_type = torch.optim.SGD
    else:
        raise ValueError('Unknown optimizer')

    #Load model from checkpoint if specified
    if args.start_epoch > 0 and args.model_path is not None:
        try:
            model, optimizer, _, _ = load_model_from_checkpoint(model, opt_type, args.start_epoch, args.lr, args.model_path)
            print('Model loaded from checkpoint')
        except:
            print('Could not load model from checkpoint')
            raise FileNotFoundError(f'Could not load model with path {args.model_path} at epoch {args.start_epoch}')

    #Initialize val loss function
    if args.val_loss == 'mse':
        val_loss_fn = torch.nn.MSELoss()
    elif args.val_loss == 'focal':
        val_loss_fn = FocalLoss(gamma=2, n_classes=n_classes)
    elif args.val_loss == 'cross_entropy':
        val_loss_fn = torch.nn.CrossEntropyLoss()
    else:
        raise ValueError('Unknown loss function specified for validation')

    #Get model name
    if args.model_name == 'dense_c8_invariant_cnn':
        model_type = f'{args.kernel_size}x{args.kernel_size}_dense'
    elif args.model_name == 'c8_invariant_cnn':
        model_type = f'{args.kernel_size}x{args.kernel_size}_nondense'
    else:
        raise ValueError('Unknown model name')
    replaced_lr = str(args.lr).replace('.', '_')
    name = f'{model_type}_{args.loss}_{replaced_lr}'

    #Initialize dataset
    randomly_crop = args.random_crop if args.random_crop else False
    randomly_flip = args.random_flip if args.random_flip else False 

    #Cross check to make sure dataset path is compatible with dataset type
    if 'scitech' in args.dataset_root_path and args.dataset == 'road':
        raise ValueError('Path is to scitech dataset but dataset type is road')
    if 'roads' in args.dataset_root_path and args.dataset == 'cloud':
        raise ValueError('Path is to roads dataset but dataset type is cloud')

    #Create either CloudDataset or RoadDataset
    if args.dataset == 'cloud' or args.dataset == 'shadow' or args.dataset == 'clouds_and_shadow':
        train_set = CloudDataset(args.dataset_root_path, args.train_folder, use_lwir=use_lwir, use_swir=use_swir, randomly_crop=randomly_crop, randomly_flip=randomly_flip, n_classes=n_classes)
        val_set = CloudDataset(args.dataset_root_path, args.val_folder, use_lwir=use_lwir, use_swir=use_swir, randomly_crop=randomly_crop, randomly_flip=randomly_flip, n_classes=n_classes)
    elif args.dataset == 'road':
        train_set = RoadDataset(args.dataset_root_path, args.train_folder, randomly_crop=randomly_crop, randomly_flip=randomly_flip)
        val_set = RoadDataset(args.dataset_root_path, args.val_folder, randomly_crop=randomly_crop, randomly_flip=randomly_flip)
    else:
        raise ValueError('Unknown dataset type')

    loss_weight = torch.Tensor([1 for n in range(n_classes)]).to(device)
    if args.class_weighted_loss and not args.weights:
        train_loader = torch.utils.data.DataLoader(train_set)
        total_1s = 0
        total_px = 0
        for data in train_loader:
            ref = data['ref']
            total_1s += torch.sum(ref)
            total_px += torch.prod(torch.Tensor(list(ref.shape))) 
        loss_weight = torch.Tensor([total_1s/total_px, 1-total_1s/total_px]).to(device)
    elif args.class_weighted_loss:
        loss_weight = torch.Tensor(args.weights).to(device)

    if (args.loss != 'focal' and args.loss != 'mse') and n_classes > 2:
        raise ValueError('Multiclass loss supported for focal loss only')

    #Initialize loss function
    if args.loss == 'mse':
        loss_fn = torch.nn.MSELoss()
        if args.class_weighted_loss:
            raise ValueError('Class weighted loss not supported for MSE loss')
    elif args.loss == 'focal':
        loss_fn = FocalLoss(gamma=2, n_classes=n_classes, weight=loss_weight)
    elif args.loss == 'cross_entropy':
        loss_fn = FocalLoss(gamma=0, n_classes=n_classes, weight=loss_weight)
    elif args.loss == 'iou':
        loss_fn = IoULoss(weight=loss_weight)
    elif args.loss == 'soft_iou':
        loss_fn = IoULoss(soft=True, weight=loss_weight)
    elif args.loss == 'jaccard':
        loss_fn = JaccardLoss(mode='binary')
        if args.class_weighted_loss:
            raise ValueError('Class weighted loss not supported for Jaccard loss')
    elif args.loss == 'log_jaccard':
        loss_fn = JaccardLoss(mode='binary', log_loss=True)
        if args.class_weighted_loss:
            raise ValueError('Class weighted loss not supported for log Jaccard loss')
    elif args.loss == 'dice':
        loss_fn = DiceLoss(mode='binary')
        if args.class_weighted_loss:
            raise ValueError('Class weighted loss not supported for dice loss')
    elif args.loss == 'log_dice':
        loss_fn = DiceLoss(mode='binary', log_loss=True)
        if args.class_weighted_loss:
            raise ValueError('Class weighted loss not supported for log dice loss')
    else:
        raise ValueError('Unknown loss function specified for training')

    if args.class_weighted_loss:
        print(loss_fn.weight)

    #Set up paths to save models
    current_dir = os.path.dirname(os.path.abspath(__file__))
    date = datetime.today().strftime('%Y-%m-%d')
    if args.save_path is None:
        raise ValueError('Save path not specified')
        dir_path = os.path.join(current_dir, 'saved_models', f'{name}_model_checkpoints_{date}')
    else:
        dir_path = args.save_path

    #Set up learning rate scheduler
    lr_scheduler = {}
    if args.lr_epochs is not None and args.lr_rates is not None:
        assert len(args.lr_epochs) == len(args.lr_rates)
        for i in range(len(args.lr_epochs)):
            lr_scheduler[int(args.lr_epochs[i])] = float(args.lr_rates[i])

    #Set up number of epochs
    n_epochs = args.epochs

    #Don't train past max epoch if specified
    if args.max_epoch and args.max_epoch - args.start_epoch < n_epochs:
        n_epochs = args.max_epoch - args.start_epoch
    epochs_per_round = 1#args.save_frequency
    n_rounds = int(n_epochs/epochs_per_round)

    n_workers = 0#2#args.n_workers if args.n_workers else 0

    n_gpus = args.n_gpus if args.n_gpus else 1
    if n_gpus > 1:
        os.environ['MASTER_ADDR'] = os.environ['SLURM_LAUNCH_NODE_IPADDR']
        world_size = n_gpus

        args.rank = args.local_rank
        args.gpu = args.local_rank
        #print(f"RANK={args.rank}")#args.gpu = rank-1

        if 'SLURM_PROCID' in os.environ: # for slurm scheduler
            args.rank = int(os.environ['SLURM_PROCID'])
            args.gpu = args.rank % torch.cuda.device_count()
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=world_size, rank=args.rank)
        # suppress printing if not on master gpu
        #if args.rank!=0:
        #    def print_pass(*args):
        #        pass
        #    builtins.print = print_pass

        # set seed for all GPUs on the same node
        if args.gpu is not None:

            print(torch.cuda.device_count())
            print(args.gpu)
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)

        train_model_multi_gpu(model, optimizer, loss_fn, val_loss_fn, train_set, val_set, n_epochs, args.start_epoch, n_workers, dir_path, lr_scheduler, args.rank, batch_size=args.batch_size, num_accumulation_steps=args.num_accumulation_steps)
    else:
        train_model_single_gpu(model, optimizer, loss_fn, train_set, val_set, n_rounds, epochs_per_round, args.start_epoch, n_workers, dir_path, lr_scheduler, batch_size=args.batch_size, num_accumulation_steps=args.num_accumulation_steps)

def train_model_multi_gpu(model, optimizer, loss_fn, val_loss_fn, train_set, val_set, n_epochs, start_epoch, n_workers, dir_path, lr_scheduler, rank, val_every_x_epochs=5, batch_size=4, num_accumulation_steps=1):
    """
    Train models using multiple GPUs and distributed training.

    Arguments
    ----------
        model: torch.nn.Module
            model to train
        optimizer: torch.optim.Optimizer
            Optimizer to use for training.
        loss_fn: torch.nn.modules.loss._Loss
            Loss fn to use for training
        train_set: CloudDataset
            Dataset to use for training
        val_set: CloudDataset
            Dataset to use for val
        n_epochs: int
            number of epochs to train for
        start_epoch: int
            start epoch
        n_workers: int
            num workers to use
        dir_path: str
            place to save models
        lr_scheduler: dict
            Dictionary mapping epochs to learning rates
        rank: int
            rank of current process
        val_every_x_epochs: int
            how often to validate
        batch_size: int
            batch size
        num_accumulation_steps: int
            number of steps to accumulate gradients over
    """
    ### data ###
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, shuffle=True)
    train_loader = torch.utils.data.DataLoader(
            train_set, batch_size, shuffle=(train_sampler is None),
            num_workers=n_workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    val_sampler = None
    val_loader = torch.utils.data.DataLoader(
            val_set, batch_size, shuffle=(val_sampler is None),
            num_workers=n_workers, pin_memory=True, sampler=val_sampler, drop_last=True)

    torch.backends.cudnn.benchmark = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    val_errs = np.genfromtxt(f"{dir_path}/val_err.csv") if os.path.exists(f"{dir_path}/val_err.csv") else None
    train_errs = np.genfromtxt(f"{dir_path}/train_err.csv") if os.path.exists(f"{dir_path}/train_err.csv") else None
    #Train! :)
    for e in range(n_epochs):
        #Train
        np.random.seed(e)
        random.seed(e)
        train_loader.sampler.set_epoch(e)
        print(f"About to train at epoch {e} with process rank {rank}")
        loss = train_single_epoch(model, optimizer, loss_fn, train_loader, device, num_accumulation_steps)
        train_err = [loss]

        print(f"Training at epoch {e} with loss {loss}")

        #Adjust lr if needed
        if e in lr_scheduler:
            for i in range(len(optimizer.param_groups)):
                optimizer.param_groups[i]['lr'] = lr_scheduler[e]

        #print(f"About to validate maybe on process with rank {rank}")
        #Val + save if on master node
        if e%val_every_x_epochs == 0 and rank == 0:
            val_loss = (validate(model.module, val_loader, val_loss_fn, device))
            val_err = [val_loss]
            print(f"Validating at epoch {e}. About to save")
            #Save model
            save_model_at_checkpoint(model.module, optimizer, start_epoch + e, train_err, dir_path)

            #Save errs in np array
            if val_errs is not None:
                long_val_err = val_err*5
                val_errs = np.append(val_errs, long_val_err)
            else:
                val_errs = val_err

            np.savetxt(f"{dir_path}/val_err.csv", val_errs, delimiter=',')

            if train_errs is not None:
                train_errs = np.append(train_errs, train_err)
            else:
                train_errs = train_err

            np.savetxt(f"{dir_path}/train_err.csv", train_errs, delimiter=',')



        if e%val_every_x_epochs == 0:
            torch.distributed.barrier()
        #    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        #    model, optimizer, _, _ = load_model_from_checkpoint(model, opt_type, start_epoch +e, lr, dir_path, map_location=map_location)

def train_model_single_gpu(model, optimizer, loss_fn, train_set, val_set, n_rounds, epochs_per_round, start_epoch, n_workers, dir_path, lr_scheduler, batch_size=4, num_accumulation_steps=1):
    """
    Train models using a single GPU.

    Arguments
    ---------
        model : torch.nn.Module
            Model to train.
        optimizer : torch.optim.Optimizer
            Optimizer to use for training.
        loss_fn : torch.nn.modules.loss._Loss
            Loss function to use for training.
        train_set : CloudDataset
            Dataset to use for training.
        val_set : CloudDataset
            Dataset to use for validation.
        n_rounds : int
            Number of rounds to train for.
        epochs_per_round : int
            Number of epochs to train for each round.
        start_epoch : int
            Starting epoch to train from.
        n_workers : int
            Number of workers to use for data loading.
        dir_path : str
            Path to save models to.
        lr_scheduler : dict
            Dictionary of epochs to learning rates.
        batch_size : int
            Batch size to use for training.
        num_accumulation_steps : int
            Number of steps to accumulate gradients over.
    """
    val_errs = np.genfromtxt(f"{dir_path}/val_err.csv") if os.path.exists(f"{dir_path}/val_err.csv") else None
    train_errs = np.genfromtxt(f"{dir_path}/train_err.csv") if os.path.exists(f"{dir_path}/train_err.csv") else None
    #Train models !! :)
    for train_epoch in range(n_rounds):
        train_err, val_err = train_model(model, optimizer, loss_fn, train_set, val_set, epochs=epochs_per_round, batch_size=batch_size, num_accumulation_steps=num_accumulation_steps, val_every_x_epochs=5, plot=False, save_model=False, verbose=False, n_workers=n_workers)
        #Save model
        save_model_at_checkpoint(model, optimizer, start_epoch + (train_epoch+1)*epochs_per_round, train_err[-1], dir_path)

        #Adjust lr if needed
        for epoch in range(start_epoch + (train_epoch)*epochs_per_round, start_epoch + (train_epoch+1)*epochs_per_round):
            if epoch in lr_scheduler:
                for i in range(len(optimizer.param_groups)):
                    optimizer.param_groups[i]['lr'] = lr_scheduler[epoch]

        #Save errs in np array
        if train_errs is not None:
            train_errs = np.append(train_errs, train_err)
        else:
            train_errs = train_err
        if val_errs is not None:
            long_val_err = val_err*5
            val_errs = np.append(val_errs, long_val_err)
        else:
            val_errs = val_err

        np.savetxt(f"{dir_path}/train_err.csv", train_errs, delimiter=',')
        np.savetxt(f"{dir_path}/val_err.csv", val_errs, delimiter=',')

if __name__ == '__main__':
    main(parse_args())

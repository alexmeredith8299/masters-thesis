import os
import numpy as np
from argparse import Namespace
import torch
from torch.utils.data import DataLoader
import torchvision
from e2cnn import gspaces, nn
from scripts.cloud_dataset import CloudDataset 
from scripts.train_on_engaging import main 
from scripts.block_builder import InvariantType

def test_train_workers():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_path = os.path.join(current_dir, '..', '..', '..', 'scitech-dataset')
    tr_folder = 'train-tiny'
    vl_folder = 'val-tiny'
    sv_path = os.path.join(current_dir, '..', 'test_artifacts', 'test_engaging_workers')

    arg_names = ('kernel_size', 'loss', 'val_loss', 'lr', 'model_name', 'opt', 'epochs', 'save_frequency', 'val_frequency', 'dataset_root_path', 'train_folder', 'val_folder', 'save_path', 'model_path', 'start_epoch', 'max_epoch', 'lr_epochs', 'lr_rates', 'bands', 'random_crop', 'n_workers', 'width', 'inv_group_type', 'n_gpus', 'rank', 'dist-url', 'dist-backend', 'local-rank', 'random_flip', 'dataset', 'class_weighted_loss', 'weights', 'batch_size', 'num_accumulation_steps')
    arg_vals = (3, 'mse', 'mse', 1e-4, 'c8_invariant_cnn', 'adam', 26, 5, 5, root_path, 'train-tiny', 'val-tiny', sv_path, None, 0, None, None, None, ['rgb'], True, 4, None, "C8", None, None, None, None, None, None, None, None, None, 4, 1)
    args_dir = {}
    for arg_name, arg_val in zip(arg_names, arg_vals):
        args_dir[arg_name] = arg_val

    args = Namespace(**args_dir)
    main(args)

    assert os.path.exists(os.path.join(sv_path, 'model_checkpoint_5.pt'))
    assert os.path.exists(os.path.join(sv_path, 'train_err.csv'))
    assert os.path.exists(os.path.join(sv_path, 'val_err.csv'))

def test_train_from_checkpoint():#Moved here bc slow to start up dense model with width 8 
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
    root_path = os.path.join(current_dir, '..', '..', '..', 'scitech-dataset')
    tr_folder = 'train-tiny'
    vl_folder = 'val-tiny'
    sv_path = os.path.join(current_dir, '..', 'test_artifacts', 'test_engaging_model_load_from_checkpoints')
    mdl_path = os.path.join(parent_dir, 'test_permanent', 'test_engaging_model_checkpoints')

    arg_names = ('kernel_size', 'loss', 'val_loss', 'lr', 'model_name', 'opt', 'epochs', 'save_frequency', 'val_frequency', 'dataset_root_path', 'train_folder', 'val_folder', 'save_path', 'model_path', 'start_epoch', 'max_epoch', 'lr_epochs', 'lr_rates', 'bands', 'random_crop', 'n_workers', 'width', 'inv_group_type', 'n_gpus', 'rank', 'dist-url', 'dist-backend', 'local-rank', 'random_flip', 'dataset', 'class_weighted_loss', 'weights', 'batch_size', 'num_accumulation_steps')
    arg_vals = (3, 'mse', 'mse', 1e-4, 'dense_c8_invariant_cnn', 'adam', 6, 5, 5, root_path, 'train-tiny', 'val-tiny', sv_path, mdl_path, 5, None, None, None, ['rgb', 'lwir', 'swir'], None, None, None, None, None, None, None, None, None, None, None, None, None, 4, 1)
    args_dir = {}
    for arg_name, arg_val in zip(arg_names, arg_vals):
        args_dir[arg_name] = arg_val

    args = Namespace(**args_dir)
    main(args)

    assert os.path.exists(os.path.join(sv_path, 'model_checkpoint_10.pt'))
    assert os.path.exists(os.path.join(sv_path, 'train_err.csv'))
    assert os.path.exists(os.path.join(sv_path, 'val_err.csv'))

def test_train_not_dense():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_path = os.path.join(current_dir, '..', '..', '..', 'scitech-dataset')
    tr_folder = 'train-tiny'
    vl_folder = 'val-tiny'
    sv_path = os.path.join(current_dir, '..', 'test_artifacts', 'test_engaging_model_checkpoints')

    arg_names = ('kernel_size', 'loss', 'val_loss', 'lr', 'model_name', 'opt', 'epochs', 'save_frequency', 'val_frequency', 'dataset_root_path', 'train_folder', 'val_folder', 'save_path', 'model_path', 'start_epoch', 'max_epoch', 'lr_epochs', 'lr_rates', 'bands', 'random_crop', 'n_workers', 'width', 'inv_group_type', 'n_gpus', 'rank', 'dist-url', 'dist-backend', 'local-rank', 'random_flip', 'dataset', 'class_weighted_loss', 'weights', 'batch_size', 'num_accumulation_steps')
    arg_vals = (3, 'mse', 'mse', 1e-4, 'c8_invariant_cnn', 'adam', 6, 5, 5, root_path, 'train-tiny', 'val-tiny', sv_path, None, 0, None, None, None, ['rgb'], True, None, 2, None, None, None, None, None, None, None, None, None, None, 4, 1)
    args_dir = {}
    for arg_name, arg_val in zip(arg_names, arg_vals):
        args_dir[arg_name] = arg_val

    args = Namespace(**args_dir)
    main(args)

    assert os.path.exists(os.path.join(sv_path, 'model_checkpoint_5.pt'))
    assert os.path.exists(os.path.join(sv_path, 'train_err.csv'))
    assert os.path.exists(os.path.join(sv_path, 'val_err.csv'))

def test_train_not_invariant():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_path = os.path.join(current_dir, '..', '..', '..', 'scitech-dataset')
    tr_folder = 'train-tiny'
    vl_folder = 'val-tiny'
    sv_path = os.path.join(current_dir, '..', 'test_artifacts', 'test_engaging_model_checkpoints')

    arg_names = ('kernel_size', 'loss', 'val_loss', 'lr', 'model_name', 'opt', 'epochs', 'save_frequency', 'val_frequency', 'dataset_root_path', 'train_folder', 'val_folder', 'save_path', 'model_path', 'start_epoch', 'max_epoch', 'lr_epochs', 'lr_rates', 'bands', 'random_crop', 'n_workers', 'width', 'inv_group_type', 'n_gpus', 'rank', 'dist-url', 'dist-backend', 'local-rank', 'random_flip', 'dataset', 'class_weighted_loss', 'weights', 'batch_size', 'num_accumulation_steps')
    arg_vals = (3, 'mse', 'mse', 1e-4, 'c8_invariant_cnn', 'adam', 6, 5, 5, root_path, 'train-tiny', 'val-tiny', sv_path, None, 0, None, None, None,['rgb'], True, None, 2, "NONE", None, None, None, None, None, None, None, None, None, 4, 1)
    args_dir = {}
    for arg_name, arg_val in zip(arg_names, arg_vals):
        args_dir[arg_name] = arg_val

    args = Namespace(**args_dir)
    main(args)

    assert os.path.exists(os.path.join(sv_path, 'model_checkpoint_5.pt'))
    assert os.path.exists(os.path.join(sv_path, 'train_err.csv'))
    assert os.path.exists(os.path.join(sv_path, 'val_err.csv'))


def test_train_lr_schedule():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_path = os.path.join(current_dir, '..', '..', '..', 'scitech-dataset')
    tr_folder = 'train-tiny'
    vl_folder = 'val-tiny'
    sv_path = os.path.join(current_dir, '..', 'test_artifacts', 'test_engaging_model_lr_schedule')

    arg_names = ('kernel_size', 'loss', 'val_loss', 'lr', 'model_name', 'opt', 'epochs', 'save_frequency', 'val_frequency', 'dataset_root_path', 'train_folder', 'val_folder', 'save_path', 'model_path', 'start_epoch', 'max_epoch', 'lr_epochs', 'lr_rates', 'bands', 'random_crop', 'n_workers', 'width', 'inv_group_type', 'n_gpus', 'rank', 'dist-url', 'dist-backend', 'local-rank', 'random_flip', 'dataset', 'class_weighted_loss', 'weights', 'batch_size', 'num_accumulation_steps')
    arg_vals = (3, 'mse', 'mse', 1e-4, 'c8_invariant_cnn', 'adam', 6, 5, 5, root_path, 'train-tiny', 'val-tiny', sv_path, None, 0,None, ['1', '5'], ['0.0001', '0.00005'], ['rgb', 'swir'], None, None, 2, None, None, None, None, None, None, None, None, None, None, 4, 1)
    args_dir = {}
    for arg_name, arg_val in zip(arg_names, arg_vals):
        args_dir[arg_name] = arg_val

    args = Namespace(**args_dir)
    main(args)

    assert os.path.exists(os.path.join(sv_path, 'model_checkpoint_5.pt'))
    assert os.path.exists(os.path.join(sv_path, 'train_err.csv'))
    assert os.path.exists(os.path.join(sv_path, 'val_err.csv'))


def test_train_simple_model():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_path = os.path.join(current_dir, '..', '..', '..', 'scitech-dataset')
    tr_folder = 'train-tiny'
    vl_folder = 'val-tiny'
    sv_path = os.path.join(current_dir, '..', 'test_artifacts', 'test_engaging_model_checkpoints')

    arg_names = ('kernel_size', 'loss', 'val_loss', 'lr', 'model_name', 'opt', 'epochs', 'save_frequency', 'val_frequency', 'dataset_root_path', 'train_folder', 'val_folder', 'save_path', 'model_path', 'start_epoch', 'max_epoch', 'lr_epochs', 'lr_rates', 'bands', 'random_crop', 'n_workers', 'width', 'inv_group_type', 'n_gpus', 'rank', 'dist-url', 'dist-backend', 'local-rank', 'random_flip', 'dataset', 'class_weighted_loss', 'weights', 'batch_size', 'num_accumulation_steps')
    arg_vals = (3, 'mse', 'mse', 1e-4, 'dense_c8_invariant_cnn', 'adam', 6, 5, 5, root_path, 'train-tiny', 'val-tiny', sv_path, None, 0, None, None, None, ['rgb', 'lwir', 'swir'], True, None, None, None, None, None, None, None, None, None, None, None, None, 4, 1)
    args_dir = {}
    for arg_name, arg_val in zip(arg_names, arg_vals):
        args_dir[arg_name] = arg_val

    args = Namespace(**args_dir)
    main(args)

    assert os.path.exists(os.path.join(sv_path, 'model_checkpoint_5.pt'))
    assert os.path.exists(os.path.join(sv_path, 'train_err.csv'))
    assert os.path.exists(os.path.join(sv_path, 'val_err.csv'))

def test_train_road_dataset():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_path = os.path.join(current_dir, '..', '..', '..', 'massachusetts-roads-dataset')
    tr_folder = 'train-tiny'
    vl_folder = 'val-tiny'
    sv_path = os.path.join(current_dir, '..', 'test_artifacts', 'test_road_dataset')

    arg_names = ('kernel_size', 'loss', 'val_loss', 'lr', 'model_name', 'opt', 'epochs', 'save_frequency', 'val_frequency', 'dataset_root_path', 'train_folder', 'val_folder', 'save_path', 'model_path', 'start_epoch', 'max_epoch', 'lr_epochs', 'lr_rates', 'bands', 'random_crop', 'n_workers', 'width', 'inv_group_type', 'n_gpus', 'rank', 'dist-url', 'dist-backend', 'local-rank', 'random_flip', 'dataset', 'class_weighted_loss', 'weights', 'batch_size', 'num_accumulation_steps')
    arg_vals = (3, 'mse', 'mse', 1e-4, 'dense_c8_invariant_cnn', 'adam', 6, 5, 5, root_path, 'train-tiny', 'val-tiny', sv_path, None, 0, None, None, None, ['rgb'], True, None, None, None, None, None, None, None, None, None, 'road', None, None, 4, 1)
    args_dir = {}
    for arg_name, arg_val in zip(arg_names, arg_vals):
        args_dir[arg_name] = arg_val

    args = Namespace(**args_dir)
    main(args)

    assert os.path.exists(os.path.join(sv_path, 'model_checkpoint_5.pt'))
    assert os.path.exists(os.path.join(sv_path, 'train_err.csv'))
    assert os.path.exists(os.path.join(sv_path, 'val_err.csv'))
 

def test_train_max_epoch():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_path = os.path.join(current_dir, '..', '..', '..', 'scitech-dataset')
    tr_folder = 'train-tiny'
    vl_folder = 'val-tiny'
    sv_path = os.path.join(current_dir, '..', 'test_artifacts', 'test_max_epoch')

    arg_names = ('kernel_size', 'loss', 'val_loss', 'lr', 'model_name', 'opt', 'epochs', 'save_frequency', 'val_frequency', 'dataset_root_path', 'train_folder', 'val_folder', 'save_path', 'model_path', 'start_epoch', 'max_epoch', 'lr_epochs', 'lr_rates', 'bands', 'random_crop', 'n_workers', 'width', 'inv_group_type', 'n_gpus', 'rank', 'dist-url', 'dist-backend', 'local-rank', 'random_flip', 'dataset', 'class_weighted_loss', 'weights', 'batch_size', 'num_accumulation_steps')
    arg_vals = (3, 'mse', 'mse', 1e-4, 'dense_c8_invariant_cnn', 'adam', 11, 5, 5, root_path, 'train-tiny', 'val-tiny', sv_path, None, 0, 8, None, None, ['rgb', 'lwir', 'swir'], True, None, None, None, None, None, None, None, None, None, None, None, None, 4, 1)
    args_dir = {}
    for arg_name, arg_val in zip(arg_names, arg_vals):
        args_dir[arg_name] = arg_val

    args = Namespace(**args_dir)
    main(args)

    assert os.path.exists(os.path.join(sv_path, 'model_checkpoint_5.pt'))
    assert not os.path.exists(os.path.join(sv_path, 'model_checkpoint_10.pt'))
    assert os.path.exists(os.path.join(sv_path, 'train_err.csv'))
    assert os.path.exists(os.path.join(sv_path, 'val_err.csv'))
 
def test_train_cross_entropy_loss():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_path = os.path.join(current_dir, '..', '..', '..', 'scitech-dataset')
    tr_folder = 'train-tiny'
    vl_folder = 'val-tiny'
    sv_path = os.path.join(current_dir, '..', 'test_artifacts', 'test_cross_entropy_loss')

    arg_names = ('kernel_size', 'loss', 'val_loss', 'lr', 'model_name', 'opt', 'epochs', 'save_frequency', 'val_frequency', 'dataset_root_path', 'train_folder', 'val_folder', 'save_path', 'model_path', 'start_epoch', 'max_epoch', 'lr_epochs', 'lr_rates', 'bands', 'random_crop', 'n_workers', 'width', 'inv_group_type', 'n_gpus', 'rank', 'dist-url', 'dist-backend', 'local-rank', 'random_flip', 'dataset', 'class_weighted_loss', 'weights', 'batch_size', 'num_accumulation_steps')
    arg_vals = (3, 'cross_entropy', 'mse', 1e-4, 'c8_invariant_cnn', 'adam', 6, 5, 5, root_path, 'train-tiny', 'val-tiny', sv_path, None, 0, None, None, None,['rgb'], True, None, 2, "NONE", None, None, None, None, None, None, None, None, None, 4, 1)
    args_dir = {}
    for arg_name, arg_val in zip(arg_names, arg_vals):
        args_dir[arg_name] = arg_val

    args = Namespace(**args_dir)
    main(args)

    assert os.path.exists(os.path.join(sv_path, 'model_checkpoint_5.pt'))
    assert os.path.exists(os.path.join(sv_path, 'train_err.csv'))
    assert os.path.exists(os.path.join(sv_path, 'val_err.csv'))

def test_train_weighted_focal():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_path = os.path.join(current_dir, '..', '..', '..', 'massachusetts-roads-dataset')
    tr_folder = 'train-tiny'
    vl_folder = 'val-tiny'
    sv_path = os.path.join(current_dir, '..', 'test_artifacts', 'test_class_weighted_loss')

    arg_names = ('kernel_size', 'loss', 'val_loss', 'lr', 'model_name', 'opt', 'epochs', 'save_frequency', 'val_frequency', 'dataset_root_path', 'train_folder', 'val_folder', 'save_path', 'model_path', 'start_epoch', 'max_epoch', 'lr_epochs', 'lr_rates', 'bands', 'random_crop', 'n_workers', 'width', 'inv_group_type', 'n_gpus', 'rank', 'dist-url', 'dist-backend', 'local-rank', 'random_flip', 'dataset', 'class_weighted_loss', 'weights', 'batch_size', 'num_accumulation_steps')
    arg_vals = (3, 'focal', 'mse', 1e-4, 'c8_invariant_cnn', 'adam', 6, 5, 5, root_path, 'train-tiny', 'val-tiny', sv_path, None, 0, None, None, None,['rgb'], True, None, 2, "NONE", None, None, None, None, None, None, 'road', True, None, 4, 1)
    args_dir = {}
    for arg_name, arg_val in zip(arg_names, arg_vals):
        args_dir[arg_name] = arg_val

    args = Namespace(**args_dir)
    main(args)

    assert os.path.exists(os.path.join(sv_path, 'model_checkpoint_5.pt'))
    assert os.path.exists(os.path.join(sv_path, 'train_err.csv'))
    assert os.path.exists(os.path.join(sv_path, 'val_err.csv'))

def test_train_weighted_focal_explicit():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_path = os.path.join(current_dir, '..', '..', '..', 'massachusetts-roads-dataset')
    tr_folder = 'train-tiny'
    vl_folder = 'val-tiny'
    sv_path = os.path.join(current_dir, '..', 'test_artifacts', 'test_class_weighted_loss_explicit')

    arg_names = ('kernel_size', 'loss', 'val_loss', 'lr', 'model_name', 'opt', 'epochs', 'save_frequency', 'val_frequency', 'dataset_root_path', 'train_folder', 'val_folder', 'save_path', 'model_path', 'start_epoch', 'max_epoch', 'lr_epochs', 'lr_rates', 'bands', 'random_crop', 'n_workers', 'width', 'inv_group_type', 'n_gpus', 'rank', 'dist-url', 'dist-backend', 'local-rank', 'random_flip', 'dataset', 'class_weighted_loss', 'weights', 'batch_size', 'num_accumulation_steps')
    arg_vals = (3, 'focal', 'mse', 1e-4, 'c8_invariant_cnn', 'adam', 6, 5, 5, root_path, 'train-tiny', 'val-tiny', sv_path, None, 0, None, None, None,['rgb'], True, None, 2, "NONE", None, None, None, None, None, None, 'road', True, [0.2, 0.8], 4, 1)
    args_dir = {}
    for arg_name, arg_val in zip(arg_names, arg_vals):
        args_dir[arg_name] = arg_val

    args = Namespace(**args_dir)
    main(args)

    assert os.path.exists(os.path.join(sv_path, 'model_checkpoint_5.pt'))
    assert os.path.exists(os.path.join(sv_path, 'train_err.csv'))
    assert os.path.exists(os.path.join(sv_path, 'val_err.csv'))

def test_train_weighted_soft_iou():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_path = os.path.join(current_dir, '..', '..', '..', 'massachusetts-roads-dataset')
    tr_folder = 'train-tiny'
    vl_folder = 'val-tiny'
    sv_path = os.path.join(current_dir, '..', 'test_artifacts', 'test_class_weighted_soft_iou')

    arg_names = ('kernel_size', 'loss', 'val_loss', 'lr', 'model_name', 'opt', 'epochs', 'save_frequency', 'val_frequency', 'dataset_root_path', 'train_folder', 'val_folder', 'save_path', 'model_path', 'start_epoch', 'max_epoch', 'lr_epochs', 'lr_rates', 'bands', 'random_crop', 'n_workers', 'width', 'inv_group_type', 'n_gpus', 'rank', 'dist-url', 'dist-backend', 'local-rank', 'random_flip', 'dataset', 'class_weighted_loss', 'weights', 'batch_size', 'num_accumulation_steps')
    arg_vals = (3, 'soft_iou', 'mse', 1e-4, 'c8_invariant_cnn', 'adam', 6, 5, 5, root_path, 'train-tiny', 'val-tiny', sv_path, None, 0, None, None, None,['rgb'], True, None, 2, "NONE", None, None, None, None, None, None, 'road', True, [0.2, 0.8], 4, 1)
    args_dir = {}
    for arg_name, arg_val in zip(arg_names, arg_vals):
        args_dir[arg_name] = arg_val

    args = Namespace(**args_dir)
    main(args)

    assert os.path.exists(os.path.join(sv_path, 'model_checkpoint_5.pt'))
    assert os.path.exists(os.path.join(sv_path, 'train_err.csv'))
    assert os.path.exists(os.path.join(sv_path, 'val_err.csv'))

def test_train_weighted_iou():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_path = os.path.join(current_dir, '..', '..', '..', 'massachusetts-roads-dataset')
    tr_folder = 'train-tiny'
    vl_folder = 'val-tiny'
    sv_path = os.path.join(current_dir, '..', 'test_artifacts', 'test_class_weighted_iou')

    arg_names = ('kernel_size', 'loss', 'val_loss', 'lr', 'model_name', 'opt', 'epochs', 'save_frequency', 'val_frequency', 'dataset_root_path', 'train_folder', 'val_folder', 'save_path', 'model_path', 'start_epoch', 'max_epoch', 'lr_epochs', 'lr_rates', 'bands', 'random_crop', 'n_workers', 'width', 'inv_group_type', 'n_gpus', 'rank', 'dist-url', 'dist-backend', 'local-rank', 'random_flip', 'dataset', 'class_weighted_loss', 'weights', 'batch_size', 'num_accumulation_steps')
    arg_vals = (3, 'iou', 'mse', 1e-4, 'c8_invariant_cnn', 'adam', 6, 5, 5, root_path, 'train-tiny', 'val-tiny', sv_path, None, 0, None, None, None,['rgb'], True, None, 2, "NONE", None, None, None, None, None, None, 'road', True, [0.2, 0.8], 4, 1)
    args_dir = {}
    for arg_name, arg_val in zip(arg_names, arg_vals):
        args_dir[arg_name] = arg_val

    args = Namespace(**args_dir)
    main(args)

    assert os.path.exists(os.path.join(sv_path, 'model_checkpoint_5.pt'))
    assert os.path.exists(os.path.join(sv_path, 'train_err.csv'))
    assert os.path.exists(os.path.join(sv_path, 'val_err.csv'))

def test_train_jaccard():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_path = os.path.join(current_dir, '..', '..', '..', 'massachusetts-roads-dataset')
    tr_folder = 'train-tiny'
    vl_folder = 'val-tiny'
    sv_path = os.path.join(current_dir, '..', 'test_artifacts', 'test_jaccard_loss')

    arg_names = ('kernel_size', 'loss', 'val_loss', 'lr', 'model_name', 'opt', 'epochs', 'save_frequency', 'val_frequency', 'dataset_root_path', 'train_folder', 'val_folder', 'save_path', 'model_path', 'start_epoch', 'max_epoch', 'lr_epochs', 'lr_rates', 'bands', 'random_crop', 'n_workers', 'width', 'inv_group_type', 'n_gpus', 'rank', 'dist-url', 'dist-backend', 'local-rank', 'random_flip', 'dataset', 'class_weighted_loss', 'weights', 'batch_size', 'num_accumulation_steps')
    arg_vals = (3, 'jaccard', 'mse', 1e-4, 'c8_invariant_cnn', 'adam', 6, 5, 5, root_path, 'train-tiny', 'val-tiny', sv_path, None, 0, None, None, None,['rgb'], True, None, 2, "NONE", None, None, None, None, None, None, 'road', False, None, 4, 1)
    args_dir = {}
    for arg_name, arg_val in zip(arg_names, arg_vals):
        args_dir[arg_name] = arg_val

    args = Namespace(**args_dir)
    main(args)

    assert os.path.exists(os.path.join(sv_path, 'model_checkpoint_5.pt'))
    assert os.path.exists(os.path.join(sv_path, 'train_err.csv'))
    assert os.path.exists(os.path.join(sv_path, 'val_err.csv'))

def test_train_log_jaccard():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_path = os.path.join(current_dir, '..', '..', '..', 'massachusetts-roads-dataset')
    tr_folder = 'train-tiny'
    vl_folder = 'val-tiny'
    sv_path = os.path.join(current_dir, '..', 'test_artifacts', 'test_log_jaccard_loss')

    arg_names = ('kernel_size', 'loss', 'val_loss', 'lr', 'model_name', 'opt', 'epochs', 'save_frequency', 'val_frequency', 'dataset_root_path', 'train_folder', 'val_folder', 'save_path', 'model_path', 'start_epoch', 'max_epoch', 'lr_epochs', 'lr_rates', 'bands', 'random_crop', 'n_workers', 'width', 'inv_group_type', 'n_gpus', 'rank', 'dist-url', 'dist-backend', 'local-rank', 'random_flip', 'dataset', 'class_weighted_loss', 'weights', 'batch_size', 'num_accumulation_steps')
    arg_vals = (3, 'log_jaccard', 'mse', 1e-4, 'c8_invariant_cnn', 'adam', 6, 5, 5, root_path, 'train-tiny', 'val-tiny', sv_path, None, 0, None, None, None,['rgb'], True, None, 2, "NONE", None, None, None, None, None, None, 'road', False, None, 4, 1)
    args_dir = {}
    for arg_name, arg_val in zip(arg_names, arg_vals):
        args_dir[arg_name] = arg_val

    args = Namespace(**args_dir)
    main(args)

    assert os.path.exists(os.path.join(sv_path, 'model_checkpoint_5.pt'))
    assert os.path.exists(os.path.join(sv_path, 'train_err.csv'))
    assert os.path.exists(os.path.join(sv_path, 'val_err.csv'))


def test_train_dice():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_path = os.path.join(current_dir, '..', '..', '..', 'massachusetts-roads-dataset')
    tr_folder = 'train-tiny'
    vl_folder = 'val-tiny'
    sv_path = os.path.join(current_dir, '..', 'test_artifacts', 'test_dice_loss')

    arg_names = ('kernel_size', 'loss', 'val_loss', 'lr', 'model_name', 'opt', 'epochs', 'save_frequency', 'val_frequency', 'dataset_root_path', 'train_folder', 'val_folder', 'save_path', 'model_path', 'start_epoch', 'max_epoch', 'lr_epochs', 'lr_rates', 'bands', 'random_crop', 'n_workers', 'width', 'inv_group_type', 'n_gpus', 'rank', 'dist-url', 'dist-backend', 'local-rank', 'random_flip', 'dataset', 'class_weighted_loss', 'weights', 'batch_size', 'num_accumulation_steps')
    arg_vals = (3, 'dice', 'mse', 1e-4, 'c8_invariant_cnn', 'adam', 6, 5, 5, root_path, 'train-tiny', 'val-tiny', sv_path, None, 0, None, None, None,['rgb'], True, None, 2, "NONE", None, None, None, None, None, None, 'road', False, None, 4, 1)
    args_dir = {}
    for arg_name, arg_val in zip(arg_names, arg_vals):
        args_dir[arg_name] = arg_val

    args = Namespace(**args_dir)
    main(args)

    assert os.path.exists(os.path.join(sv_path, 'model_checkpoint_5.pt'))
    assert os.path.exists(os.path.join(sv_path, 'train_err.csv'))
    assert os.path.exists(os.path.join(sv_path, 'val_err.csv'))

def test_train_log_dice():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_path = os.path.join(current_dir, '..', '..', '..', 'massachusetts-roads-dataset')
    tr_folder = 'train-tiny'
    vl_folder = 'val-tiny'
    sv_path = os.path.join(current_dir, '..', 'test_artifacts', 'test_log_dice_loss')

    arg_names = ('kernel_size', 'loss', 'val_loss', 'lr', 'model_name', 'opt', 'epochs', 'save_frequency', 'val_frequency', 'dataset_root_path', 'train_folder', 'val_folder', 'save_path', 'model_path', 'start_epoch', 'max_epoch', 'lr_epochs', 'lr_rates', 'bands', 'random_crop', 'n_workers', 'width', 'inv_group_type', 'n_gpus', 'rank', 'dist-url', 'dist-backend', 'local-rank', 'random_flip', 'dataset', 'class_weighted_loss', 'weights', 'batch_size', 'num_accumulation_steps')
    arg_vals = (3, 'log_dice', 'mse', 1e-4, 'c8_invariant_cnn', 'adam', 6, 5, 5, root_path, 'train-tiny', 'val-tiny', sv_path, None, 0, None, None, None,['rgb'], True, None, 2, "NONE", None, None, None, None, None, None, 'road', False, None, 4, 1)
    args_dir = {}
    for arg_name, arg_val in zip(arg_names, arg_vals):
        args_dir[arg_name] = arg_val

    args = Namespace(**args_dir)
    main(args)

    assert os.path.exists(os.path.join(sv_path, 'model_checkpoint_5.pt'))
    assert os.path.exists(os.path.join(sv_path, 'train_err.csv'))

def test_train_sgd():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_path = os.path.join(current_dir, '..', '..', '..', 'massachusetts-roads-dataset')
    tr_folder = 'train-tiny'
    vl_folder = 'val-tiny'
    sv_path = os.path.join(current_dir, '..', 'test_artifacts', 'test_sgd')

    arg_names = ('kernel_size', 'loss', 'val_loss', 'lr', 'model_name', 'opt', 'epochs', 'save_frequency', 'val_frequency', 'dataset_root_path', 'train_folder', 'val_folder', 'save_path', 'model_path', 'start_epoch', 'max_epoch', 'lr_epochs', 'lr_rates', 'bands', 'random_crop', 'n_workers', 'width', 'inv_group_type', 'n_gpus', 'rank', 'dist-url', 'dist-backend', 'local-rank', 'random_flip', 'dataset', 'class_weighted_loss', 'weights', 'batch_size', 'num_accumulation_steps')
    arg_vals = (3, 'log_dice', 'mse', 1e-4, 'c8_invariant_cnn', 'sgd', 6, 5, 5, root_path, 'train-tiny', 'val-tiny', sv_path, None, 0, None, None, None,['rgb'], True, None, 2, "NONE", None, None, None, None, None, None, 'road', False, None, 4, 1)
    args_dir = {}
    for arg_name, arg_val in zip(arg_names, arg_vals):
        args_dir[arg_name] = arg_val

    args = Namespace(**args_dir)
    main(args)

    assert os.path.exists(os.path.join(sv_path, 'model_checkpoint_5.pt'))
    assert os.path.exists(os.path.join(sv_path, 'train_err.csv'))

def test_train_sparcs():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_path = os.path.join(current_dir, '..', '..', '..', 'sparcs-dataset')
    tr_folder = 'train-tiny'
    vl_folder = 'val-tiny'
    sv_path = os.path.join(current_dir, '..', 'test_artifacts', 'test_sgd')

    arg_names = ('kernel_size', 'loss', 'val_loss', 'lr', 'model_name', 'opt', 'epochs', 'save_frequency', 'val_frequency', 'dataset_root_path', 'train_folder', 'val_folder', 'save_path', 'model_path', 'start_epoch', 'max_epoch', 'lr_epochs', 'lr_rates', 'bands', 'random_crop', 'n_workers', 'width', 'inv_group_type', 'n_gpus', 'rank', 'dist-url', 'dist-backend', 'local-rank', 'random_flip', 'dataset', 'class_weighted_loss', 'weights', 'batch_size', 'num_accumulation_steps')
    arg_vals = (3, 'log_dice', 'mse', 1e-4, 'c8_invariant_cnn', 'sgd', 6, 5, 5, root_path, 'train-tiny', 'val-tiny', sv_path, None, 0, None, None, None,['rgb'], True, None, 2, "NONE", None, None, None, None, None, None, 'cloud', False, None, 4, 1)
    args_dir = {}
    for arg_name, arg_val in zip(arg_names, arg_vals):
        args_dir[arg_name] = arg_val

    args = Namespace(**args_dir)
    main(args)

    assert os.path.exists(os.path.join(sv_path, 'model_checkpoint_5.pt'))
    assert os.path.exists(os.path.join(sv_path, 'train_err.csv'))

def test_train_sparcs_multiclass():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_path = os.path.join(current_dir, '..', '..', '..', 'sparcs-multiclass-dataset')
    tr_folder = 'train-tiny'
    vl_folder = 'val-tiny'
    sv_path = os.path.join(current_dir, '..', 'test_artifacts', 'test_multiclass')

    arg_names = ('kernel_size', 'loss', 'val_loss', 'lr', 'model_name', 'opt', 'epochs', 'save_frequency', 'val_frequency', 'dataset_root_path', 'train_folder', 'val_folder', 'save_path', 'model_path', 'start_epoch', 'max_epoch', 'lr_epochs', 'lr_rates', 'bands', 'random_crop', 'n_workers', 'width', 'inv_group_type', 'n_gpus', 'rank', 'dist-url', 'dist-backend', 'local-rank', 'random_flip', 'dataset', 'class_weighted_loss', 'weights', 'batch_size', 'num_accumulation_steps')
    arg_vals = (3, 'focal', 'mse', 1e-4, 'c8_invariant_cnn', 'sgd', 6, 5, 5, root_path, 'train-tiny', 'val-tiny', sv_path, None, 0, None, None, None,['rgb'], True, None, 2, "NONE", None, None, None, None, None, None, 'clouds_and_shadow', False, None, 4, 1)
    args_dir = {}
    for arg_name, arg_val in zip(arg_names, arg_vals):
        args_dir[arg_name] = arg_val

    args = Namespace(**args_dir)
    main(args)

    assert os.path.exists(os.path.join(sv_path, 'model_checkpoint_5.pt'))
    assert os.path.exists(os.path.join(sv_path, 'train_err.csv'))

def test_train_sparcs_shadow():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_path = os.path.join(current_dir, '..', '..', '..', 'sparcs-shadow-dataset')
    tr_folder = 'train-tiny'
    vl_folder = 'val-tiny'
    sv_path = os.path.join(current_dir, '..', 'test_artifacts', 'test_shadow')

    arg_names = ('kernel_size', 'loss', 'val_loss', 'lr', 'model_name', 'opt', 'epochs', 'save_frequency', 'val_frequency', 'dataset_root_path', 'train_folder', 'val_folder', 'save_path', 'model_path', 'start_epoch', 'max_epoch', 'lr_epochs', 'lr_rates', 'bands', 'random_crop', 'n_workers', 'width', 'inv_group_type', 'n_gpus', 'rank', 'dist-url', 'dist-backend', 'local-rank', 'random_flip', 'dataset', 'class_weighted_loss', 'weights', 'batch_size', 'num_accumulation_steps')
    arg_vals = (3, 'focal', 'mse', 1e-4, 'c8_invariant_cnn', 'sgd', 6, 5, 5, root_path, 'train-tiny', 'val-tiny', sv_path, None, 0, None, None, None,['rgb'], True, None, 2, "NONE", None, None, None, None, None, None, 'shadow', False, None, 4, 1)
    args_dir = {}
    for arg_name, arg_val in zip(arg_names, arg_vals):
        args_dir[arg_name] = arg_val

    args = Namespace(**args_dir)
    main(args)

    assert os.path.exists(os.path.join(sv_path, 'model_checkpoint_5.pt'))
    assert os.path.exists(os.path.join(sv_path, 'train_err.csv'))
 

import os
import numpy as np
import torch
import torchvision
import pytest
from e2cnn import gspaces, nn
from scripts.c8_invariant_cnn import C8InvariantCNN, DenseC8InvariantCNN
from scripts.equivariant_basic_blocks import EquivariantBasicConvBlock, DenseBlock, ReduceChannelDimsBlock, DownscaleBlock, UpscaleBlock, TwoStepConvBlock
from scripts.basic_blocks import ActivationBlock
from scripts.cloud_dataset import CloudDataset 
from scripts.train_pytorch_model import evaluate_rotational_equivariance, train_model
from scripts.escnn_extension import InteroperableR2Conv, InteroperableBatchNorm, InteroperableMaxBlurPool, InteroperableUpsample, InteroperableReLU, InteroperableGroupPooling

@pytest.fixture(scope='session')
def device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

@pytest.fixture(scope='session')
def train_set():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(current_dir,'..', '..', 'scitech-dataset')
    train_set = CloudDataset(train_path, "train-tiny")
    return train_set

@pytest.fixture(scope='session')  # one model_tester only (for speed)
def model():
    """
    Generate trained model to use in other tests.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(current_dir,'..', '..', 'scitech-dataset')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = C8InvariantCNN().to(device)

    #Train model 
    train_set = CloudDataset(train_path, "train-tiny")
    val_set = CloudDataset(train_path, "val-tiny")
    loss_fn = torch.nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)

    train_err, val_err = train_model(model, opt, loss_fn, train_set, val_set, epochs=5, val_every_x_epochs=5, plot=False, save_model=False)

    return model

@pytest.fixture(scope='session')  # one model_tester only (for speed)
def dense_model():
    """
    Generate trained model to use in other tests.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(current_dir,'..', '..', 'scitech-dataset')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dense_model = DenseC8InvariantCNN().to(device)

    #Train model 
    train_set = CloudDataset(train_path, "train-tiny")
    val_set = CloudDataset(train_path, "val-tiny")
    loss_fn = torch.nn.MSELoss()
    opt = torch.optim.Adam(dense_model.parameters(), lr=0.001)

    train_err, val_err = train_model(dense_model, opt, loss_fn, train_set, val_set, epochs=5, val_every_x_epochs=5, plot=False, save_model=False)

    return dense_model


def test_r2conv_save_to_dict(device, train_set, model):
    """
    Save model weights + reload them to ensure model can be saved.
    """
    model.eval()
    d = train_set[0]
    model_out_a = model(d['img'].unsqueeze(0).to(device))

    #Extract block from model
    r2_conv = model.conv_block_1.block._modules['2']

    #Save block to dict + then reload
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dir_path = os.path.join(current_dir, 'test_artifacts', 'test_save_r2_conv', 'test_save_r2_conv.pt')
    torch.save(r2_conv.get_save_dict(), dir_path)
    r2_conv_extend = InteroperableR2Conv.load_from_dict(torch.load(dir_path))

    #Load new block back into model
    model.conv_block_1.block._modules['2'] = r2_conv_extend

    #Evaluate model again
    model.eval()
    model_out_b = model(d['img'].unsqueeze(0).to(device))
    max_diff = np.amax(np.abs(model_out_a.detach().numpy()-model_out_b.detach().numpy()))

    assert max_diff <= 1e-12

def test_bn_save_to_dict(device, train_set, model):
    """
    Save model weights + reload them to ensure model can be saved.
    """
    #Evaluate model
    model.eval()
    d = train_set[0]
    model_out_a = model(d['img'].unsqueeze(0).to(device))

    #Extract block from model
    batch_norm = model.conv_block_4.block._modules['0']

    #Save block to dict + then reload
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dir_path = os.path.join(current_dir, 'test_artifacts', 'test_save_batch_norm', 'test_save_batch_norm.pt')
    torch.save(batch_norm.get_save_dict(), dir_path)
    batch_norm_extend = InteroperableBatchNorm.load_from_dict(torch.load(dir_path))

    assert batch_norm.in_type == batch_norm_extend.in_type #both should be regular

    #Load new block back into model
    model.conv_block_4.block._modules['0'] = batch_norm_extend

    #Evaluate model again
    model.eval()
    model_out_b = model(d['img'].unsqueeze(0).to(device))
    max_diff = np.amax(np.abs(model_out_a.detach().numpy()-model_out_b.detach().numpy()))

    assert max_diff <= 1e-12

def test_relu_save_to_dict(device, train_set, model):
    """
    Save model weights + reload them to ensure model can be saved.
    """
    #Evaluate model
    model.eval()
    d = train_set[0]
    model_out_a = model(d['img'].unsqueeze(0).to(device))

    #Extract block from model
    relu = model.conv_block_1.block._modules['1']

    #Save block to dict + then reload
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dir_path = os.path.join(current_dir, 'test_artifacts', 'test_save_relu', 'test_save_relu.pt')
    torch.save(relu.get_save_dict(), dir_path)
    relu_extend = InteroperableReLU.load_from_dict(torch.load(dir_path))

    #Load new block back into model
    model.conv_block_1.block._modules['1'] = relu_extend

    #Evaluate model again
    model.eval()
    model_out_b = model(d['img'].unsqueeze(0).to(device))
    max_diff = np.amax(np.abs(model_out_a.detach().numpy()-model_out_b.detach().numpy()))

    assert max_diff <= 1e-12

def test_max_pool_save_to_dict(device, train_set, model):
    """
    Save model weights + reload them to ensure model can be saved.
    """
    #Evaluate model
    model.eval()
    d = train_set[0]
    model_out_a = model(d['img'].unsqueeze(0).to(device))

    #Extract block from model
    downsample = model.downsample_1.block._modules['0']

    #Save block to dict + then reload
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dir_path = os.path.join(current_dir, 'test_artifacts', 'test_save_max_blur_pool', 'test_save_max_blur_pool.pt')
    torch.save(downsample.get_save_dict(), dir_path)
    downsample_extend = InteroperableMaxBlurPool.load_from_dict(torch.load(dir_path))

    #Load new block back into model
    model.downsample_1.block._modules['0'] = downsample_extend

    #Evaluate model again
    model.eval()
    model_out_b = model(d['img'].unsqueeze(0).to(device))
    max_diff = np.amax(np.abs(model_out_a.detach().numpy()-model_out_b.detach().numpy()))

    assert max_diff <= 1e-12

def test_upsample_save_to_dict(device, train_set, model):
    """
    Save model weights + reload them to ensure model can be saved.
    """
    #Evaluate model
    model.eval()
    d = train_set[0]
    model_out_a = model(d['img'].unsqueeze(0).to(device))

    #Extract block from model
    upsample = model.upscale_1.block._modules['0']

    #Save block to dict + then reload
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dir_path = os.path.join(current_dir, 'test_artifacts', 'test_save_upsample', 'test_save_upsample.pt')
    torch.save(upsample.get_save_dict(), dir_path)
    upsample_extend = InteroperableUpsample.load_from_dict(torch.load(dir_path))

    #Load new block back into model
    model.upscale_1.block._modules['0'] = upsample_extend

    #Evaluate model again
    model.eval()
    model_out_b = model(d['img'].unsqueeze(0).to(device))
    max_diff = np.amax(np.abs(model_out_a.detach().numpy()-model_out_b.detach().numpy()))

    assert max_diff <= 1e-12

def test_basic_eq_block_save_to_dict(device, train_set, dense_model):
    """
    Save model weights + reload them to ensure model can be saved.
    """
    #Evaluate model
    dense_model.eval()
    d = train_set[0]
    model_out_a = dense_model(d['img'].unsqueeze(0).to(device))

    #Extract block from model
    eq_block = dense_model.dense_conv_block_1.block._modules['0']

    #Save block to dict + then reload
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dir_path = os.path.join(current_dir, 'test_artifacts', 'test_save_basic_block', 'test_save_basic_block.pt')
    torch.save(eq_block.get_save_dict(), dir_path)
    eq_extend = EquivariantBasicConvBlock.load_from_dict(torch.load(dir_path))

    #Load new block back into model
    dense_model.dense_conv_block_1.block._modules['0'] = eq_extend

    #Evaluate model again
    dense_model.eval()
    model_out_b = dense_model(d['img'].unsqueeze(0).to(device))
    max_diff = np.amax(np.abs(model_out_a.detach().numpy()-model_out_b.detach().numpy()))

    assert max_diff <= 1e-12

def test_dense_block_save_to_dict(device, train_set, dense_model):
    """
    Save model weights + reload them to ensure model can be saved.
    """
    #Evaluate model
    dense_model.eval()
    d = train_set[0]
    model_out_a = dense_model(d['img'].unsqueeze(0).to(device))

    #Extract block from model
    dense_block = dense_model.dense_conv_block_4

    #Save block to dict + then reload
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dir_path = os.path.join(current_dir, 'test_artifacts', 'test_save_dense_block', 'test_save_dense_block.pt')
    torch.save(dense_block.get_save_dict(), dir_path)
    dense_extend = DenseBlock.load_from_dict(torch.load(dir_path))

    #Load new block back into model
    dense_model.dense_conv_block_4 = dense_extend

    #Evaluate model again
    dense_model.eval()
    model_out_b = dense_model(d['img'].unsqueeze(0).to(device))
    max_diff = np.amax(np.abs(model_out_a.detach().numpy()-model_out_b.detach().numpy()))

    assert max_diff <= 1e-12

def test_two_step_conv_block_save_to_dict(device, train_set, model):
    """
    Save model weights + reload them to ensure model can be saved.
    """
    #Evaluate model
    model.eval()
    d = train_set[0]
    model_out_a = model(d['img'].unsqueeze(0).to(device))

    #Extract block from model
    conv_block = model.conv_block_4

    #Save block to dict + then reload
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dir_path = os.path.join(current_dir, 'test_artifacts', 'test_save_two_step_conv_block', 'test_save_two_step_conv_block.pt')
    torch.save(conv_block.get_save_dict(), dir_path)
    two_step_extend = TwoStepConvBlock.load_from_dict(torch.load(dir_path))

    #Load new block back into model
    model.conv_block_4 = two_step_extend

    #Evaluate model again
    model.eval()
    model_out_b = model(d['img'].unsqueeze(0).to(device))
    max_diff = np.amax(np.abs(model_out_a.detach().numpy()-model_out_b.detach().numpy()))

    assert max_diff <= 1e-12

def test_transform_input():
    """
    Test save and load ReduceChannelDimsBlock that transforms an input from 
    a trivial to regular representation of C(8).
    """
    r2_act = gspaces.Rot2dOnR2(N=8)

    input_type = nn.FieldType(r2_act, 5*[r2_act.trivial_repr])
    output_type = nn.FieldType(r2_act, 16*[r2_act.regular_repr])
    input_conv_block = ReduceChannelDimsBlock(input_type, output_type)

    #Save block to dict + then reload
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dir_path = os.path.join(current_dir, 'test_artifacts', 'test_save_trivial_to_regular', 'test_save_trivial_to_regular.pt')
    torch.save(input_conv_block.get_save_dict(), dir_path)
    input_conv_block_extend = ReduceChannelDimsBlock.load_from_dict(torch.load(dir_path))

    assert True


def test_reduce_block_save_to_dict(device, train_set, model):
    """
    Save model weights + reload them to ensure model can be saved.
    """
    #Evaluate model
    model.eval()
    d = train_set[0]
    model_out_a = model(d['img'].unsqueeze(0).to(device))

    #Extract block from model
    reduce_block = model.reduce_1a

    #Save block to dict + then reload
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dir_path = os.path.join(current_dir, 'test_artifacts', 'test_save_reduce_block', 'test_save_reduce_block.pt')
    torch.save(reduce_block.get_save_dict(), dir_path)
    reduce_extend = ReduceChannelDimsBlock.load_from_dict(torch.load(dir_path))

    #Load new block back into model
    model.reduce_1a = reduce_extend

    #Evaluate model again
    model.eval()
    model_out_b = model(d['img'].unsqueeze(0).to(device))
    max_diff = np.amax(np.abs(model_out_a.detach().numpy()-model_out_b.detach().numpy()))

    assert max_diff <= 1e-12

def test_downscale_block_save_to_dict(device, train_set, model):
    """
    Save model weights + reload them to ensure model can be saved.
    """
    #Evaluate model
    model.eval()
    d = train_set[0]
    model_out_a = model(d['img'].unsqueeze(0).to(device))

    #Extract block from model
    downscale_block = model.downsample_1

    #Save block to dict + then reload
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dir_path = os.path.join(current_dir, 'test_artifacts', 'test_save_downscale_block', 'test_save_downscale_block.pt')
    torch.save(downscale_block.get_save_dict(), dir_path)
    downscale_extend = DownscaleBlock.load_from_dict(torch.load(dir_path))

    #Load new block back into model
    model.downsample_1 = downscale_extend 

    #Evaluate model again
    model.eval()
    model_out_b = model(d['img'].unsqueeze(0).to(device))
    max_diff = np.amax(np.abs(model_out_a.detach().numpy()-model_out_b.detach().numpy()))

    assert max_diff <= 1e-12

def test_upscale_block_save_to_dict(device, train_set, model):
    """
    Save model weights + reload them to ensure model can be saved.
    """
    #Evaluate model
    model.eval()
    d = train_set[0]
    model_out_a = model(d['img'].unsqueeze(0).to(device))

    #Extract block from model
    upscale_block = model.upscale_1

    #Save block to dict + then reload
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dir_path = os.path.join(current_dir, 'test_artifacts', 'test_save_upscale_block', 'test_save_upscale_block.pt')
    torch.save(upscale_block.get_save_dict(), dir_path)
    upscale_extend = UpscaleBlock.load_from_dict(torch.load(dir_path))

    #Load new block back into model
    model.upscale_1 = upscale_extend 

    #Evaluate model again
    model.eval()
    model_out_b = model(d['img'].unsqueeze(0).to(device))
    max_diff = np.amax(np.abs(model_out_a.detach().numpy()-model_out_b.detach().numpy()))

    assert max_diff <= 1e-12

def test_group_pooling_save_to_dict(device, train_set, model):
    """
    Save model weights + reload them to ensure model can be saved.
    """
    #Evaluate model
    model.eval()
    d = train_set[0]
    model_out_a = model(d['img'].unsqueeze(0).to(device))

    #Extract block from model
    group_pooling = model.group_pool_block

    #Save block to dict + then reload
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dir_path = os.path.join(current_dir, 'test_artifacts', 'test_save_group_pooling', 'test_save_group_pooling.pt')
    torch.save(group_pooling.get_save_dict(), dir_path)
    group_pooling_extend = InteroperableGroupPooling.load_from_dict(torch.load(dir_path))

    #Load new block back into model
    model.group_pool_block = group_pooling_extend 

    #Evaluate model again
    model.eval()
    model_out_b = model(d['img'].unsqueeze(0).to(device))
    max_diff = np.amax(np.abs(model_out_a.detach().numpy()-model_out_b.detach().numpy()))

    assert max_diff <= 1e-12

def test_activation_block_save_to_dict(device, train_set, model):
    """
    Save model weights + reload them to ensure model can be saved.
    """
    #Evaluate model
    model.eval()
    d = train_set[0]
    model_out_a = model(d['img'].unsqueeze(0).to(device))

    #Extract block from model
    activation = model.plain_torch_block

    #Save block to dict + then reload
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dir_path = os.path.join(current_dir, 'test_artifacts', 'test_save_activation', 'test_save_activation.pt')
    torch.save(activation.get_save_dict(), dir_path)
    activation_extend = ActivationBlock.load_from_dict(torch.load(dir_path))

    #Load new block back into model
    model.activation_block = activation_extend 

    #Evaluate model again
    model.eval()
    model_out_b = model(d['img'].unsqueeze(0).to(device))
    max_diff = np.amax(np.abs(model_out_a.detach().numpy()-model_out_b.detach().numpy()))

    assert max_diff <= 1e-12


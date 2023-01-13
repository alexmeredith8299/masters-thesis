import os
import numpy as np
import torch
import pytest
from torch.utils.data import DataLoader
import torchvision
import pickle
from e2cnn import gspaces, nn
from scripts.c8_invariant_cnn import C8InvariantCNN, DenseC8InvariantCNN
from scripts.equivariant_basic_blocks import EquivariantBasicConvBlock, DenseBlock 
from scripts.basic_blocks import BasicConvBlock
from scripts.block_builder import InvariantType
from scripts.cloud_dataset import CloudDataset 
from scripts.road_dataset import RoadDataset
from scripts.train_pytorch_model import train_model, load_model_from_checkpoint
from scripts.convunext import ConvUNeXt

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
def convunext_model():
    """
    Generate trained model to use in other tests.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(current_dir,'..', '..', 'scitech-dataset')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ConvUNeXt(in_channels=4).to(device)

    #Train model 
    train_set = CloudDataset(train_path, "train-tiny")
    val_set = CloudDataset(train_path, "val-tiny")
    loss_fn = torch.nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)

    train_err, val_err = train_model(model, opt, loss_fn, train_set, val_set, epochs=5, val_every_x_epochs=5, plot=False, save_model=False)

    return model

@pytest.fixture(scope='session')  # one model_tester only (for speed)
def road_model():
    """
    Generate trained model to use in other tests.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(current_dir,'..', '..', 'massachusetts-roads-dataset')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = C8InvariantCNN(input_channels=3).to(device)

    #Train model 
    train_set = RoadDataset(train_path, "train-tiny")
    val_set = RoadDataset(train_path, "val-tiny")
    loss_fn = torch.nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)

    train_err, val_err = train_model(model, opt, loss_fn, train_set, val_set, epochs=5, val_every_x_epochs=5, plot=False, save_model=False)

    return model


@pytest.fixture(scope='session')
def non_rot_inv_model():
    """
    Generate trained model to use in other tests.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(current_dir,'..', '..', 'scitech-dataset')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = C8InvariantCNN(inv_group_type=InvariantType.NONE).to(device)

    #Train model 
    train_set = CloudDataset(train_path, "train-tiny")
    val_set = CloudDataset(train_path, "val-tiny")
    loss_fn = torch.nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)

    train_err, val_err = train_model(model, opt, loss_fn, train_set, val_set, epochs=5, val_every_x_epochs=5, plot=False, save_model=False)

    return model

@pytest.fixture(scope='session')
def non_rot_inv_dense_model():
    """
    Generate trained model to use in other tests.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(current_dir,'..', '..', 'scitech-dataset')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DenseC8InvariantCNN(inv_group_type=InvariantType.NONE).to(device)

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

def test_load_model_from_checkpoint():
    """
    Ensure that API for loading model stays the same.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(current_dir,'..', '..', 'scitech-dataset')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    #Load model from checkpoint
    dir_path = os.path.join(current_dir, 'test_permanent', 'test_load_model')
    model_loaded = C8InvariantCNN().to(device)
    opt_loaded = torch.optim.Adam
    model_loaded, opt_loaded, epoch_loaded, train_err_loaded = load_model_from_checkpoint(model_loaded, opt_loaded, 6, 0.001, dir_path)
    assert True

def test_init_model():
    """
    Make sure model can be initialized without throwing a bunch
    of errors.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = C8InvariantCNN().to(device)
    assert True

def test_init_dense_model():
    """
    Make sure model can be initialized without throwing a bunch
    of errors.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DenseC8InvariantCNN().to(device)
    assert True

def test_init_non_rot_inv_model():
    """
    Test basic initialization + training of non-rotation-invariant model.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = C8InvariantCNN(inv_group_type=InvariantType.NONE).to(device)
    assert True

def test_pickle_model():
    """
    Initialize model, pickle it, dump it to a file, and reload it.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = C8InvariantCNN().to(device)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    pickle_path = os.path.join(current_dir,'test_artifacts', 'test_pickle_model', 'test_pickle_model.pkl')
    train_path = os.path.join(current_dir,'..', '..', 'scitech-dataset')
    val_set = CloudDataset(train_path, "val-tiny")

    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

    model.eval() #Update weights
    test_img = next(iter(val_loader))['img'].to(device)
    out_img = model(test_img)

    with open(pickle_path, "wb") as f:
        pickle.dump(model, f)

    with open(pickle_path, "rb") as f:
        model_loaded = pickle.load(f)
        model_loaded.eval()
        out_img_loaded = model_loaded(test_img)
        assert torch.all(out_img == out_img_loaded)

def test_pickle_dense_model():
    """
    Initialize model, pickle it, dump it to a file, and reload it.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DenseC8InvariantCNN().to(device)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    pickle_path = os.path.join(current_dir,'test_artifacts', 'test_pickle_model', 'test_pickle_model.pkl')
    train_path = os.path.join(current_dir,'..', '..', 'scitech-dataset')
    val_set = CloudDataset(train_path, "val-tiny")

    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

    model.eval() #Update weights
    test_img = next(iter(val_loader))['img'].to(device)
    out_img = model(test_img)

    with open(pickle_path, "wb") as f:
        pickle.dump(model, f)

    with open(pickle_path, "rb") as f:
        model_loaded = pickle.load(f)
        model_loaded.eval()
        out_img_loaded = model_loaded(test_img)
        assert torch.all(out_img == out_img_loaded)


def test_train_model(device, train_set, model):
    """
    Train model for 1 epoch and ensure no errors occur.
    """
    assert True

def test_train_convunext(device, train_set, convunext_model):
    """
    Train model for 1 epoch and ensure no errors occur.
    """
    assert True

def test_train_road_model(device, road_model):
    assert True

def test_train_dense_model(device, train_set, dense_model):
    """
    Train model for 1 epoch and ensure no errors occur.
    """
    assert True

def test_train_non_rot_inv_model(device, train_set, non_rot_inv_model):
    """
    Train model for 1 epoch and ensure no errors occur.
    """
    assert True

def test_train_non_rot_inv_dense_model(device, train_set, non_rot_inv_dense_model):
    """
    Train model for 1 epoch and ensure no errors occur.
    """
    assert True

def test_save_and_load_model(device, train_set, model):
    """
    Save model weights + reload them to ensure model can be saved.
    """
    #Evaluate model
    model.eval()
    d = train_set[0]
    model_out_a = model(d['img'].unsqueeze(0).to(device))

    #Save model to dict + then reload
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(current_dir,'test_artifacts', 'test_save_and_load_model', 'test_save_and_load_model.pt')
    torch.save(model.get_save_dict(), save_path)
    model_extend = C8InvariantCNN.load_from_dict(torch.load(save_path))

    #Load new block back into model
    model = model_extend 

    #Evaluate model again
    model.eval()
    model_out_b = model(d['img'].unsqueeze(0).to(device))
    max_diff = np.amax(np.abs(model_out_a.detach().numpy()-model_out_b.detach().numpy()))

    assert max_diff <= 1e-12

def test_save_and_reload_model(device, train_set, model):
    """
    Save model weights + reload them to ensure model can be saved.
    """
    #Evaluate model
    model.eval()
    d = train_set[0]
    model_out_a = model(d['img'].unsqueeze(0).to(device))

    #Save model to dict + then reload
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(current_dir,'test_artifacts', 'test_save_and_reload_model', 'test_save_and_reload_model.pt')
    torch.save(model.get_save_dict(), save_path)
    model_extend = C8InvariantCNN(kernel_size=7).to(device)
    model_extend.reload_from_dict(torch.load(save_path))

    #Load new block back into model
    model = model_extend 

    #Evaluate model again
    model.eval()
    model_out_b = model(d['img'].unsqueeze(0).to(device))
    max_diff = np.amax(np.abs(model_out_a.detach().numpy()-model_out_b.detach().numpy()))

    assert max_diff <= 1e-12
    assert model.kernel_size == 7


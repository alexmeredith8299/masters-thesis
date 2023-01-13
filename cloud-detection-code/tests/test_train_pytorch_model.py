import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision
from e2cnn import gspaces, nn
from scripts.c8_invariant_cnn import C8InvariantCNN
from scripts.cloud_dataset import CloudDataset 
from scripts.train_pytorch_model import evaluate_rotational_equivariance, train_model 
from scripts.train_pytorch_model import save_model_at_checkpoint, load_model_from_checkpoint 
from scripts.train_pytorch_model import validate

def test_save_and_load_model_at_checkpoint():
    """
    Make sure we can save + load model at a specific checkpoint.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(current_dir,'..', '..', 'scitech-dataset')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = C8InvariantCNN().to(device)

    #No stochasticity in dataset
    train_set = CloudDataset(train_path, "train-tiny", randomly_flip=False)
    val_set = CloudDataset(train_path, "val-tiny", randomly_flip=False)
    loss_fn = torch.nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)

    train_err, val_err = train_model(model, opt, loss_fn, train_set, val_set, epochs=6, val_every_x_epochs=5, plot=False, save_model=False)

    #Save model at checkpoint
    dir_path = os.path.join(current_dir, 'test_artifacts', 'test_model_checkpoints')
    save_model_at_checkpoint(model, opt, 6, train_err[-1], dir_path)

    #Load model from checkpoint
    model_loaded = C8InvariantCNN().to(device)
    opt_loaded = torch.optim.Adam
    model_loaded, opt_loaded, epoch_loaded, train_err_loaded = load_model_from_checkpoint(model_loaded, opt_loaded, 6, 0.001, dir_path)

    #Train model again to compare
    train_err_2_compare, val_err_2_compare = train_model(model, opt, loss_fn, train_set, val_set, epochs=5, val_every_x_epochs=5, plot=False, save_model=False)

    #Train model again from saved model/opt
    train_err_2, val_err_2 = train_model(model_loaded, opt_loaded, loss_fn, train_set, val_set, epochs=5, val_every_x_epochs=5, plot=False, save_model=False)

    #Assert that normal training has same results as save/reload
    assert np.amax(np.abs(np.array(train_err_2)-np.array(train_err_2_compare))) <= 1e-12
    assert epoch_loaded == 6
    assert train_err_loaded == train_err[-1]

def test_validate():
    """
    Make sure validation does not change model weights.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(current_dir,'..', '..', 'scitech-dataset')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = C8InvariantCNN().to(device)

    train_set = CloudDataset(train_path, "train-tiny")
    #No stochasticity in dataset
    val_set = CloudDataset(train_path, "val-tiny", randomly_flip=False)

    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

    loss_fn = torch.nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)

    train_err, val_err = train_model(model, opt, loss_fn, train_set, val_set, epochs=6, val_every_x_epochs=5, plot=False, save_model=False)

    model.eval() #Update weights
    test_img = next(iter(val_loader))['img'].to(device)
    out_img = model(test_img)

    val_loss = validate(model, val_loader, loss_fn, device) 
    out_img_2 = model(test_img)

    assert val_loss == val_err[-1]
    assert torch.all(out_img == out_img_2)


def test_rotational_equivariance_17():
    """
    Make sure model is reasonably C17-invariant.
    Test should be consistent as it is run with a
    consistent random seed.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(current_dir,'..', '..', 'scitech-dataset')
    rng_path = os.path.join(os.path.dirname(__file__), "test_artifacts", "test_rotational_equivariance", "rng_seed.pt")

    #Run deterministically
    torch.set_rng_state(torch.load(rng_path))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = C8InvariantCNN().to(device)

    training_data = CloudDataset(train_path, "train")
    to_image = torchvision.transforms.ToPILImage()
    rgb_image = training_data[0]['img'].unsqueeze(0).to(device)[0, 0:3, :, :]
    rgb_image = to_image(rgb_image)
    ir_image = training_data[0]['img'].unsqueeze(0).to(device)[0, 3, :, :]
    ir_image = to_image(ir_image)
    angle = round(360/17) 

    max_diff = evaluate_rotational_equivariance(model, rgb_image, ir_image, angle)
    assert max_diff <= 0.0012

def test_rotational_equivariance_8():
    """
    Make sure model is C8-invariant (tigher than C17-invariant).
    Test should be consistent as it is run with a
    consistent random seed.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(current_dir,'..', '..', 'scitech-dataset')
    rng_path = os.path.join(os.path.dirname(__file__), "test_artifacts", "test_rotational_equivariance", "rng_seed.pt")

    #Run deterministically
    torch.set_rng_state(torch.load(rng_path))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = C8InvariantCNN().to(device)

    training_data = CloudDataset(train_path, "train")
    to_image = torchvision.transforms.ToPILImage()
    rgb_image = training_data[0]['img'].unsqueeze(0).to(device)[0, 0:3, :, :]
    rgb_image = to_image(rgb_image)
    ir_image = training_data[0]['img'].unsqueeze(0).to(device)[0, 3, :, :]
    ir_image = to_image(ir_image)
    angle = 45

    max_diff = evaluate_rotational_equivariance(model, rgb_image, ir_image, angle)
    assert max_diff <= 0.0012



"""
This module trains a PyTorch model on a dataset and evaluates it on various metrics.
"""
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import time
from torch.utils.data import DataLoader
from torchvision.transforms import Pad, Resize, ToTensor
from PIL import Image

def evaluate_rotational_equivariance(model, rgb_image, ir_image, angle, img_dim=144):
    """
    This function evaluates the rotation equivariance of a PyTorch model
    on an image.

    Arguments
    ----------
        model: torch.nn.Module
            a PyTorch model to test
        rgb_image: PIL.Image
            RGB image to test equivariance on
        ir_image: PIL.Image
            IR image to test equivariance on
        angle: float
            basic angle to test equivariance for [degrees]
        img_dim: int (optional)
            side length of image, default 144
    Returns
    ---------
        max_diff: float
            max pixel-wise difference in predictions over rotated inputs
    """
    #Evaluate the model
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    #Initialize model to correct dims
    wrmup = model(torch.randn(1, 4, img_dim, img_dim).to(device))
    del wrmup

    pad = Pad((0, 0, 1, 1), fill=0) #Pad image to (img_dim + 1) x (img_dim + 1)
    upsample = Resize(img_dim*3) #Upsample before rotation (to avoid aliasing)
    downsample = Resize(img_dim) #Downsample after rotation
    totensor = ToTensor()

    rgb_image = upsample(pad(rgb_image))
    ir_image = upsample(pad(ir_image))
    rot_masks = []

    #Test rotations
    with torch.no_grad():
        for r in range(math.ceil(360/angle)):
            rgb_transformed = totensor(downsample(rgb_image.rotate(r*angle, Image.BILINEAR)))\
                    .reshape(1, 3, img_dim, img_dim)
            ir_transformed = totensor(downsample(ir_image.rotate(r*angle, Image.BILINEAR)))\
                    .reshape(1, 1, img_dim, img_dim)
            x_transformed = torch.cat((rgb_transformed, ir_transformed), dim=1)
            x_transformed = x_transformed.to(device)

            y = model(x_transformed)
            y = y.to('cpu').numpy().squeeze()

            rot_masks.append(y)

    #Find max pixel-wise different in masks from image rotated by different angle
    max_diff = 0
    for i in range(len(rot_masks)):
        for j in range(i+1, len(rot_masks)):
            max_diff = max(max_diff, np.amax(np.abs(rot_masks[i]-rot_masks[j])))
    return max_diff

def train_single_epoch(model, optimizer, loss_fn, train_loader, device, num_accumulation_steps):
    """
    This function trains a model for a single epoch.

    Arguments 
    ----------
        model: torch.nn.Module 
            model to train 
        optimizer: torch.optim._Optimizer 
            optimizer for the model 
        loss_fn: torch.nn.loss._Loss 
            loss function to use for the model 
        train_loader: torch.utils.data.DataLoader 
            data loader for the training set 
        device: string 
            device to use for the model 
        num_accumulation_steps: int
            number of steps to accumulate gradients over

    Returns 
    ----------
        loss: float 
            loss of the model at the epoch 
    """
    #Set model to training mode
    model.train()
    print("In train mode")#losses = []
    print(f"Train loader size={len(train_loader)}")#losses = []


    #Initialize loss
    losses = []

    #Iterate over batches
    for idx, data in enumerate(train_loader):
        print("In train_loader loop")
        optimizer.zero_grad()
        print("Set optimizer to zero grad")

        input_img = data['img'].to(device)
        ref_img = data['ref'].to(device)
        print("Read in an img...")
        if torch.cuda.is_available():
            output_img = model(input_img.cuda())
        else:
            output_img = model(input_img)
        print("Classified an img...")

        loss = loss_fn(output_img, ref_img)
        loss = loss/num_accumulation_steps#loss_fn(output_img, ref_img)
        losses.append(float(loss))
        loss.backward()

        #Use gradient accumulation
        if ((idx + 1) % num_accumulation_steps == 0) or (idx + 1 == len(train_loader)):
            optimizer.step()
        print("Trained on an img...")

    return np.mean(np.array(losses))

def validate(model, val_loader, loss_fn, device):
    """
    This function validates a model on a dataset.

    Arguments
    ----------
        model: torch.nn.Module
            a PyTorch model to validate 
        val_loader: torch.utils.data.DataLoader
            data loader for the validation set
        loss_fn: torch.nn.loss._Loss
            loss function to use for the model
        device: string
            device to use for the model

    Returns
    ----------
        loss: float
            loss of the model on the validation set
    """
    model.eval()
    current_val_losses = []

    #No gradients for validation
    with torch.no_grad():
        for data in val_loader:
            input_img = data['img'].to(device)
            ref_img = data['ref'].to(device)
            output_img = model(input_img)
            loss = loss_fn(output_img, ref_img)
            current_val_losses.append(float(loss))

    return np.mean(np.array(current_val_losses))

def train_model(model, optimizer, loss_fn, train_set, val_set, epochs=10000, batch_size=4, num_accumulation_steps=1, n_workers=0,
    val_every_x_epochs=10, verbose=True, plot=True, save_model=True, model_fname="model_out.pt"):
    """
    This function trains a PyTorch model.

    Arguments
    ----------
        model: torch.nn.Module
            a PyTorch model to test
        optimizer: torch.optim._Optimizer 
            optimizer for the model
        loss_fn: torch.nn.loss._Loss
            a loss function to use for the model
        train_set: CloudDataset
            training data set
        val_set: CloudDataset
            validation data set
        epochs: int (optional)
            number of epochs to train for (default 10000)
        batch_size: int (optional)
            batch size, default 10
        num_accumulation_steps: int (optional)
            number of steps to accumulate gradients over (default 1)
        val_every_x_epochs: int (optional)
            every val_every_x_epochs, validate and check loss to avoid overtraining
        verbose: bool (optional)
            if True, prints loss every val_every_x_epochs epochs
        plot: bool (optional)
            if True, plots loss every val_every_x_epochs epochs
        save_model: bool (optional)
            if True, save model state dict to model_out.pt after training finishes
        model_fname: string (optional)
            string to save model to, default is model_out.pt

    Returns
    --------
        training_losses: list
            list of mean training loss at each epoch, in order
        validation_losses: list
            list of mean validation loss at each epoch *WHERE VALIDATION OCCURS*, in order
                note that typically validation only occurs every 10 epochs, so this list
                will *LIKELY NOT* be the same length as training_losses
    """
    #Initialize device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    training_losses = []
    validation_losses = []

    #Load datasets
    if n_workers is not None:
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=n_workers)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=n_workers)
    else:
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    #Train! :)
    for e in range(epochs):
        #Train
        loss = train_single_epoch(model, optimizer, loss_fn, train_loader, device, num_accumulation_steps)
        training_losses.append(loss)

        if e%val_every_x_epochs == 0:
            validation_losses.append(validate(model, val_loader, loss_fn, device))

            if verbose:
                print(f"Epoch {e} finished. Train loss: {training_losses[-1]},\
                        Validation loss: {validation_losses[-1]}")
            if plot:
                plt.plot(training_losses)
                plt.yscale('log')
                plt.show()

    if save_model:
        model.eval()
        torch.save(model.get_save_dict(), model_fname)

    return training_losses, validation_losses

def save_model_at_checkpoint(model, optimizer, epoch, latest_loss, directory="model_checkpoints"):
    """
    This function saves the model at a checkpoint.

    Arguments
    ----------
        model: torch.nn.Module
            a PyTorch model to test
        optimizer: torch.optim._Optimizer 
            optimizer for the model
        epoch: int
            epoch number of the model
        latest_loss: float
            loss of the model at the epoch
        directory: string (optional)
            string to save model to, default is model_checkpoints
    """
    #Update weights
    model.eval()

    #Make directory if doesn't exist 
    os.makedirs(directory, exist_ok=True)
    print(f"Saving model at {epoch} in {directory}...")
    #Save model
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': latest_loss,
            }, f"{directory}/model_checkpoint_{epoch}.pt")

    print(f"Saved model to {directory}/model_checkpoint_{epoch}.pt")

def load_model_from_checkpoint(model, optimizer, epoch, lr, directory="model_checkpoints", map_location=None):
    """
    This function loads a model from a checkpoint.

    Arguments
    ----------
        model: torch.nn.Module
            a PyTorch model to load 
        optimizer: torch.optim._Optimizer 
            optimizer class to load
        epoch: int
            epoch number of the model
        lr: float 
            learning rate of the model
        directory: string (optional)
            string to save model to, default is model_checkpoints

    Returns
    --------
        model: torch.nn.Module
            loaded PyTorch model to test
        optimizer: torch.optim._Optimizer
            loaded optimizer for the model
        epoch: int 
            epoch number of the model
        loss: float
            loss of the model at the epoch
    """
    #Load model
    if not torch.cuda.is_available():
        checkpoint = torch.load(f"{directory}/model_checkpoint_{epoch}.pt", map_location=torch.device('cpu'))
    elif map_location is not None:
        checkpoint = torch.load(f"{directory}/model_checkpoint_{epoch}.pt", map_location=map_location)
    else:
        checkpoint = torch.load(f"{directory}/model_checkpoint_{epoch}.pt")
    model.reload_from_dict(checkpoint['model_state_dict'])
    if torch.cuda.is_available():
        model.to(torch.device('cuda'))#reload_from_dict(checkpoint['model_state_dict'])
    opt = optimizer(model.parameters(), lr=lr)
    opt.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    return model, opt, epoch, loss

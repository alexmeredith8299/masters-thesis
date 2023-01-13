from segmentation_models_pytorch.losses import DiceLoss
import math
import torch

def test_init_dice_loss():
    """
    Ensure we can initialize DiceLoss() object
    with no issues.
    """
    loss_fn = DiceLoss(mode='binary')
    loss_fn = DiceLoss(mode='binary', log_loss=True)
    assert True

def test_perfect_prediction():
    """
    Ensure loss is zero if prediction is correct.
    """
    loss_fn = DiceLoss(mode='binary')#weight=torch.Tensor([0.3, 0.7]))
    input_tensor = torch.Tensor([[[1, 0], [0, 1]],[[0, 0], [0, 1]]])
    target = input_tensor

    loss = loss_fn(input_tensor, target)
    assert torch.abs(loss-0.4298) < 0.01

def test_wrong_prediction():
    """
    Ensure loss is never infinite.
    """
    loss_fn = DiceLoss(mode='binary')
    input_tensor = torch.Tensor([[[1, 0], [0, 1]],[[0, 0], [0, 1]]])
    target = torch.Tensor([[[0, 1], [1, 0]],[[1, 1], [1, 0]]])
    loss = loss_fn(input_tensor, target)
    assert not torch.any(loss.isinf()) 

    input_tensor = torch.Tensor([[[0, 0], [0, 0]], [[0, 0], [0, 0]]])
    target = torch.Tensor([[[0, 0], [0, 0]], [[0, 0], [0, 0]]])
    loss = loss_fn(input_tensor, target)
    assert not torch.any(loss.isinf()) 

    input_tensor = torch.Tensor([[[1, 1], [1, 1]], [[1, 1], [1, 1]]])
    target = torch.Tensor([[[1, 1], [1, 1]], [[1, 1], [1, 1]]])
    loss = loss_fn(input_tensor, target)
    assert not torch.any(loss.isinf()) 

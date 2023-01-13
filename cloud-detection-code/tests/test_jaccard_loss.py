from segmentation_models_pytorch.losses import JaccardLoss
import math
import torch

def test_init_jaccard_loss():
    """
    Ensure we can initialize JaccardLoss() object
    with no issues.
    """
    loss_fn = JaccardLoss(mode='binary')
    loss_fn = JaccardLoss(mode='binary', log_loss=True)
    assert True

def test_perfect_prediction():
    """
    Ensure loss is zero if prediction is correct.
    """
    loss_fn = JaccardLoss(mode='binary')#weight=torch.Tensor([0.3, 0.7]))
    input_tensor = torch.Tensor([[[1, 0], [0, 1]],[[0, 0], [0, 1]]])
    target = input_tensor

    loss = loss_fn(input_tensor, target)
    assert torch.abs(loss-0.6012) < 0.01

def test_wrong_prediction():
    """
    Ensure loss is never infinite.
    """
    loss_fn = JaccardLoss(mode='binary')
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

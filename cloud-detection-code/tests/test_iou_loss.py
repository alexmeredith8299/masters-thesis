from scripts.soft_iou_loss import IoULoss 
import math
import torch

def test_init_iou_loss():
    """
    Ensure we can initialize IoULoss() object
    with no issues.
    """
    loss_fn = IoULoss()
    loss_fn = IoULoss(weight=torch.Tensor([0.3, 0.7]))
    assert True

def test_perfect_prediction():
    """
    Ensure loss is zero if prediction is correct.
    """
    loss_fn = IoULoss()#weight=torch.Tensor([0.3, 0.7]))
    input_tensor = torch.Tensor([[[1, 0], [0, 1]],[[0, 0], [0, 1]]])
    target = input_tensor

    loss = loss_fn(input_tensor, target)
    assert loss==0

def test_weighted():
    """
    Ensure weighted loss behavior matches expectations
    """
    loss_fn = IoULoss(weight=torch.Tensor([0.3, 0.7]))
    loss_fn_no_weight = IoULoss()

    input_tensor = torch.Tensor([[[[0.17, 0.17], [0.92, 0.92]]]])
    target = torch.Tensor([[[[1, 0], [0, 1]]]])

    loss = loss_fn(input_tensor, target)
    loss_no_weight = loss_fn_no_weight(input_tensor, target)

    input_tensor_2 = torch.Tensor([[[[0.17, 0.17], [0.08, 0.08]]]])
    loss_2 = loss_fn(input_tensor_2, target)
    loss_no_weight_2 = loss_fn_no_weight(input_tensor_2, target)

    assert loss_2 > loss
    #assert torch.abs(loss_no_weight_2 - loss_no_weight) < 1e-6

def test_wrong_prediction():
    """
    Ensure loss is never infinite.
    """
    loss_fn = IoULoss()
    input_tensor = torch.Tensor([[[1, 0], [0, 1]],[[0, 0], [0, 1]]])
    target = torch.Tensor([[[0, 1], [1, 0]],[[1, 1], [1, 0]]])
    loss = loss_fn(input_tensor, target)
    assert not torch.any(loss.isinf()) 

    input_tensor = torch.Tensor([[[0, 0], [0, 0]], [[0, 0], [0, 0]]])
    target = torch.Tensor([[[0, 0], [0, 0]], [[0, 0], [0, 0]]])
    loss = loss_fn(input_tensor, target)
    assert not torch.any(loss.isinf())

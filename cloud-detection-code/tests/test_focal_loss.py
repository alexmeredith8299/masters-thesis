from scripts.focal_loss import FocalLoss
import math
import torch

def test_init_focal_loss():
    """
    Ensure we can initialize FocalLoss() object
    with no issues.
    """
    loss_fn = FocalLoss(reduction='none')
    loss_fn = FocalLoss(reduction='mean')
    loss_fn = FocalLoss(reduction='sum')
    assert True

def test_perfect_prediction():
    """
    Ensure loss is zero if prediction is correct.
    """
    loss_fn = FocalLoss(reduction='mean')
    input_tensor = torch.Tensor([1, 0])
    target = torch.Tensor([1, 0])
    loss = loss_fn(input_tensor, target)
    assert loss==0

def test_multiclass():
    """
    Ensure multiclass losses make sense.
    """
    loss_no_reduce = FocalLoss(reduction='none', n_classes=3)
    loss_mean = FocalLoss(reduction='mean', n_classes=3)

    input_tensor = torch.Tensor([[[0.17, 0.5, 0.33], [0.83, 0.1, 0.07]], [[0.02, 0.08, 0.9], [0.1, 0.83, 0.07]]])
    target = torch.Tensor([[[1, 0, 0], [1, 0, 0]], [[0, 1, 0], [0, 1, 0]]])

    loss = loss_no_reduce(input_tensor, target)
    assert torch.max(torch.abs(loss-torch.Tensor([[[[1.22070, 0.00538492], [2.13778, 0.00538492]]]]))) < 1e-5

    loss = loss_mean(input_tensor, target)
    assert torch.abs(loss-0.842312) < 1e-5

def test_imperfect_prediction():
    """
    Ensure loss matches expected for incorrect predictions.
    """
    loss_no_reduce = FocalLoss(reduction='none')
    loss_sum = FocalLoss(reduction='sum')
    loss_mean = FocalLoss(reduction='mean')

    input_tensor = torch.Tensor([[[[0.17, 0.17], [0.92, 0.83]]]])
    target = torch.Tensor([[[[1, 0], [0, 1]]]])

    loss = loss_no_reduce(input_tensor, target)
    assert torch.max(torch.abs(loss-torch.Tensor([[[[1.22070, 0.00538492], [2.13778, 0.00538492]]]]))) < 1e-5

    loss = loss_sum(input_tensor, target)
    assert torch.abs(loss-3.369247) < 1e-5

    loss = loss_mean(input_tensor, target)
    assert torch.abs(loss-0.842312) < 1e-5

def test_wrong_prediction():
    """
    Ensure loss is never infinite.
    """
    loss_fn = FocalLoss(reduction='mean')
    input_tensor = torch.Tensor([0.1, 1])
    target = torch.Tensor([1, 0])
    loss = loss_fn(input_tensor, target)
    assert not torch.any(loss.isinf()) 

def test_weighted():
    """
    Ensure class-weighted focal loss works correctly.
    """
    loss_fn = FocalLoss(reduction='none', weight=torch.Tensor([0.2, 0.8]))
    input_tensor = torch.Tensor([0.5, 0.9])
    target = torch.Tensor([1, 0])
    loss = loss_fn(input_tensor, target)
    assert torch.max(torch.abs(loss-torch.Tensor([0.138629, 0.373019]))) < 1e-5

def test_weighted_multiclass():
    """
    Ensure class-weighted focal loss works correctly.
    """
    loss_fn = FocalLoss(reduction='none', weight=torch.Tensor([0.2, 0.8, 0.2]), n_classes=3)
    input_tensor = torch.Tensor([[0.2, 0.5, 0.3], [0.1, 0.8, 0.2], [0.8, 0.2, 0.1]])
    target = torch.Tensor([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    loss = loss_fn(input_tensor, target)
    assert torch.max(torch.abs(loss-torch.Tensor([0.138629, 0.373019, 0.373019]))) < 1e-5

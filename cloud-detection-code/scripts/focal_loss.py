"""
This function implements binary focal loss for tensors of arbitrary size/shape.
"""
import warnings
import torch

class FocalLoss(torch.nn.modules.loss._Loss):
    """
    Inherits from torch.nn.modules.loss._Loss. Finds the focal loss between each element
    in the input and target tensors.

    Parameters
    -----------
        gamma: float (optional)
            power to raise (1-pt) to when computing focal loss. Default is 2
        reduction: string (optional)
            "sum", "mean", or "none". If sum, the output will be summed, if mean, the output will
                be averaged, if none, no reduction will be applied. Default is mean

    Attributes
    -----------
        gamma: float
            focusing parameter -- power to raise (1-pt) to when computing focal loss. Default is 2
        eps: float
            machine epsilon as defined for pytorch
        reduction: string
            "sum", "mean", or "none". If sum, the output will be summed, if mean, the output will
                be averaged, if none, no reduction will be applied. Default is mean
    """
    def __init__(self, gamma=2, reduction='mean', n_classes=2, weight=torch.Tensor([])):
        if reduction not in ("sum", "mean", "none"):
            raise AttributeError("Invalid reduction type. Please use 'mean', 'sum', or 'none'.")
        super().__init__(None, None, reduction)
        self.gamma = gamma
        self.eps = torch.finfo(torch.float32).eps
        if weight.shape[0] == 0:
            weight = torch.Tensor([1 for n in range(n_classes)])

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.weight = weight.to(device)
        self.n_classes = n_classes

    def forward(self, input_tensor, target):
        """
        Compute binary focal loss for an input prediction map and target mask.

        Arguments
        ----------
            input_tensor: torch.Tensor
                input prediction map
            target: torch.Tensor
                target mask

        Returns
        --------
            loss_tensor: torch.Tensor
                binary focal loss, summed, averaged, or raw depending on self.reduction
        """
        #Warn that if sizes don't match errors may occur
        if not target.size() == input_tensor.size():
            warnings.warn(
                f"Using a target size ({target.size()}) that is different to the input size"\
                "({input_tensor.size()}). \n This will likely lead to incorrect results"\
                "due to broadcasting.\n Please ensure they have the same size.",
                stacklevel=2,
            )

        #Broadcast to get sizes/shapes to match
        c, h, w = target.shape[0], target.shape[2], target.shape[3]
        if self.n_classes > 2:
            input_tensor = input_tensor.reshape([c, 1, h, w, self.n_classes])
        input_tensor, target = torch.broadcast_tensors(input_tensor, target)
        assert input_tensor.shape == target.shape, "Input and target tensor shapes don't match"

        #Vectorized computation of focal loss
        if self.n_classes == 2: #Binary case
            pt_tensor = (target == 0)*(1-input_tensor) + target*input_tensor
            weight_tensor = (target == 0)*self.weight[0] + target*self.weight[1]
            pt_tensor = torch.clamp(pt_tensor, min=self.eps, max=1.0) #Avoid vanishing gradient
            loss_tensor = -weight_tensor*(1-pt_tensor)**self.gamma*torch.log(pt_tensor)
        else: #Multiclass case
            pt_tensor = target*input_tensor
            weight_tensor = target*self.weight
            pt_tensor = torch.clamp(pt_tensor, min=self.eps, max=1.0) #Avoid vanishing gradient
            loss_tensor = -weight_tensor*(1-pt_tensor)**self.gamma*torch.log(pt_tensor)
            loss_tensor = torch.sum(loss_tensor, dim=len(loss_tensor.shape)-1)

        #Apply reduction
        if self.reduction =='none':
            return loss_tensor
        if self.reduction=='mean':
            if torch.any(torch.mean(loss_tensor).isnan()):
                print("reduction has nan")
            return torch.mean(loss_tensor)

        #If not none or mean, sum
        return torch.sum(loss_tensor)

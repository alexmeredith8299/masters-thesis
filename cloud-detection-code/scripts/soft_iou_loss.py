#Modified version of https://github.com/wgcban/ChangeFormer/blob/b96d0a779bb89d59244e443a6042b9f81d2b639f/models/losses.py
#miou loss
from torch.autograd import Variable
import torch
import torch.nn.functional as F

def input_to_classes(tensor, nClasses, requires_grad=False):
    n, h, w = torch.squeeze(tensor, dim=1).size()
    classes = tensor.new(n, nClasses, h, w).fill_(0)
    if len(tensor.shape) == 4:
        classes[:, 0, :, :] = 1 - tensor[:, 0, :, :]
        classes[:, 1, :, :] = tensor[:, 0, :, :]
    elif len(tensor.shape) == 3:
        classes[:, 0, :, :] = 1 - tensor[:, :, :]
        classes[:, 1, :, :] = tensor[:, :, :]
    return Variable(classes, requires_grad=requires_grad)

"""
class IoULoss(torch.nn.Module):
    def __init__(self, weight=torch.Tensor([1, 1]), size_average=True, n_classes=2, soft=False):
        super(IoULoss, self).__init__()
        self.classes = n_classes
        self.weight = Variable(weight, requires_grad=True)
        self.softmax = soft

    def forward(self, inputs, target, is_target_variable=False):
        N = inputs.size()[0]
        # Numerator Product
        inputs, target = input_to_classes(inputs, self.classes), input_to_classes(target, self.classes)
        if self.softmax:
            inputs = F.softmax(inputs, dim=1)

        inter = inputs * target 
        inter = inter.view(N, self.classes, -1).sum(2)

        # Denominator
        union = inputs + target - (inputs * target)
        union = union.view(N, self.classes, -1).sum(2)

        loss = (self.weight*inter)/(union + 1e-8)

        ## Return average loss over classes and batch
        return 1-torch.mean(loss)#-torch.mean(loss)
"""

def to_one_hot_var(tensor, nClasses, requires_grad=False):
    n, h, w = tensor.size()
    one_hot = tensor.new(n, nClasses, h, w).fill_(0)
    one_hot = one_hot.scatter_(1, tensor.view(n, 1, h, w).to(torch.int64), 1)
    return Variable(one_hot, requires_grad=requires_grad)


class IoULoss(torch.nn.Module):
    def __init__(self, weight=torch.Tensor([1,1]), size_average=True, n_classes=2):
        super(IoULoss, self).__init__()
        self.classes = n_classes
        self.weight = Variable(weight * weight)

    def forward(self, inputs, target, is_target_variable=False):
        # inputs => N x Classes x H x W
        # target => N x H x W
        # target_oneHot => N x Classes x H x W

        inputs = input_to_classes(inputs, self.classes)
        N = inputs.size()[0]
        if is_target_variable:
            target_oneHot = to_one_hot_var(target.data, self.classes).float()
        else:
            target_oneHot = to_one_hot_var(target, self.classes).float()

        # predicted probabilities for each pixel along channel
        inputs = F.softmax(inputs, dim=1)

        # Numerator Product
        inter = inputs * target_oneHot
        ## Sum over all pixels N x C x H x W => N x C
        inter = inter.view(N, self.classes, -1).sum(2)

        # Denominator
        union = inputs + target_oneHot - (inputs * target_oneHot)
        ## Sum over all pixels N x C x H x W => N x C
        union = union.view(N, self.classes, -1).sum(2)

        loss = (self.weight * inter) / (self.weight * union + 1e-8)

        ## Return average loss over classes and batch
        return -torch.mean(loss)

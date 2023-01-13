"""
This module implements some basic blocks that are useful
for C8-invariant and similarly flavored models.
"""
from abc import ABC, abstractmethod
import math 
import numpy as np
import torch
from e2cnn import nn
import antialiased_cnns

class SavableModule(torch.nn.Module, ABC):
    """
    Abstract extension of escnn.nn.EquivariantModule that allows for saving
    and loading a model.

    Parameters
    -----------
        in_type: escnn.nn.FieldType
            input type of basic block, giving invariant representation of block input
        out_type: escnn.nn.FieldType
            output type, gives invariant rep of block output

    Attributes
    -----------
        in_type: escnn.nn.FieldType
            input type of basic block, giving invariant representation of block input
        out_type: escnn.nn.FieldType
            output type, gives invariant rep of block output
        block: nn.SequentialModule
            actual layers to use in forward method
    """
    @abstractmethod
    def __init__(self, *args):
        super().__init__()

    def forward(self, *args):
        """
        Forward pass of SavableModule. Called during model evaluation and testing.

        Arguments
        -----------
            x: escnn.nn.GeometricTensor
                tensor with an invariant type given as input
        Returns
        ---------
            out: escnn.nn.GeometricTensor
                tensor with an invariant type given as output
        """
        x = args[0]
        return self.block(x)

    def get_save_dict(self, prefix=''):
        """
        Save properties of this module to a dictionary.

        Arguments
        ----------
            prefix: string (optional)
                prefix for properties in dict

        Returns
        --------
            properties: dict
                properties of this module
        """
        return self.block.state_dict(prefix=prefix)

    @classmethod
    @abstractmethod
    def load_from_dict(cls, properties):
        """
        Load this module from a state dictionary.

        Arguments
        ----------
            properties: dict
                dictionary containing properties

        Returns
        --------
            module: SavableModule
                module implementing properties
        """

class SavableSequential(torch.nn.Sequential, ABC):
    """
    Inherits from torch.nn.Sequential.
    """
    @abstractmethod
    def __init__(self, *args):
        super().__init__()

    @classmethod
    def load_from_dict(cls, properties, kwarg_keys=()):
        """
        Given a dictionary that contains all kwargs and properties, 
        initialize and return this block. 

        Arguments 
        ---------
            properties : dict
                Dictionary that contains all kwargs and properties.

        Returns
        -------
            block : SavableSequential
                A SavableSequential with the given properties.
        """
        kwargs = {}
        del_keys = []
        for key in properties.keys():
            #Needs to end in base.{kwarg_key} so we can disambiguate between kwargs and properties
            if key.split('.')[-1] in kwarg_keys and key.split('.')[-2] == 'base':
                kwargs[key.split('.')[-1]] = properties[key]
                del_keys.append(key)

        for key in del_keys:
            del properties[key]

        new_block = cls(**kwargs)
        new_del_keys = []
        for key in properties.keys():
            if 'base' in key:
                new_del_keys.append(key)
        for key in new_del_keys:
            del properties[key]
        new_block.load_state_dict(properties)
        return new_block

    def get_save_dict(self, prefix=''):
        """
        Return union of state_dict and kwargs.

        Arguments
        ---------
            prefix : str
                Prefix to add to all keys in the returned dictionary.

        Returns
        -------
            dict
                Dictionary containing all kwargs and state_dict.
        """
        properties = {}#super().state_dict(prefix=prefix)
        for name, module in self._modules.items():
            if module is not None:
                properties.update(module.state_dict(prefix=prefix + name + '.'))

        #Needs to end in base.{kwarg_key} so we can disambiguate between kwargs and properties
        for key in self.kwargs.keys():
            if prefix != '' and prefix[-1] != '.':
                properties[prefix + '.base.' + key] = self.kwargs[key]
            else:
                properties[prefix + 'base.' + key] = self.kwargs[key]
        return properties

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """
        Return state_dict of this module.
        """
        return self.get_save_dict(prefix)

#TODO simplify this to use SavableSequential
class ActivationBlock(SavableModule):
    """
    This module represents a final 1x1 convolution, batchnorm, leaky ReLU, and sigmoid
    activation which can be saved and loaded from a dictionary.
    """
    def __init__(self, relu_slope=0.1, conv_stride=1, conv_padding=0,  n_classes=3):
        super().__init__()
        self.relu_slope = relu_slope
        multiclass = n_classes > 2
        self.multiclass = multiclass
        self.n_classes = n_classes
        self.block = torch.nn.Sequential(
                torch.nn.Conv2d(1, 1, 1, conv_stride, conv_padding),
                torch.nn.BatchNorm2d(1),
                torch.nn.LeakyReLU(relu_slope),
                torch.nn.Sigmoid()
        )
        if multiclass:
            self.block = torch.nn.Sequential(
                torch.nn.Conv2d(n_classes, n_classes, 1, conv_stride, conv_padding),
                torch.nn.BatchNorm2d(n_classes),
                torch.nn.LeakyReLU(relu_slope),
                torch.nn.Softmax()
            )


    def get_save_dict(self, prefix=''):
        """
        This method saves all relevant properties of the ActivationBlock module
        in a dictionary.

        Arguments
        -----------
            prefix: string (optional)
                prefix for properties in dict

        Returns
        --------
            properties: dict
                contains properties to save to disk
        """
        properties = {}

        #Save vals from state dict
        state_dict = self.state_dict()
        for key, value in state_dict.items():
            properties[f"{prefix}.{key}"] = value

        properties[f"{prefix}.2.slope"] = self.relu_slope
        properties[f"{prefix}.n_classes"] = self.n_classes

        return properties

    @classmethod
    def load_from_dict(cls, properties):
        """
        This method loads an ActivationBlock module from a state dictionary.

        Arguments
        ----------
            properties: dict
                dictionary of properties for module

        Returns
        --------
            activation_module: ActivationBlock
                module implementing those properties
        """
        #Get Conv2d + BatchNorm2d properties, and relu slope
        conv_dict = {}
        bn_dict = {}
        relu_slope = None
        for key in properties.keys():
            if key.split('.')[-2] == '0':
                conv_dict[key.split('.')[-1]] = properties[key]
            elif key.split('.')[-2] == '1':
                bn_dict[key.split('.')[-1]] = properties[key]
            elif key.split('.')[-2] =='2':
                relu_slope = properties[key]

        #Form block
        n_classes = properties['.block.0.weight'].shape[0]
        multiclass = n_classes > 2

        act = cls(relu_slope=relu_slope)

        if not multiclass:
            conv = torch.nn.Conv2d(1, 1, 1)
            conv.load_state_dict(conv_dict)
            batch_norm = torch.nn.BatchNorm2d(1)
            batch_norm.load_state_dict(bn_dict)


            act.block = torch.nn.Sequential(
                    conv,
                    batch_norm,
                    torch.nn.LeakyReLU(relu_slope),
                    torch.nn.Sigmoid()
            )

        else:
            conv = torch.nn.Conv2d(n_classes, n_classes, 1)
            conv.load_state_dict(conv_dict)
            batch_norm = torch.nn.BatchNorm2d(n_classes)
            batch_norm.load_state_dict(bn_dict)


            act.block = torch.nn.Sequential(
                    conv,
                    batch_norm,
                    torch.nn.LeakyReLU(relu_slope),
                    torch.nn.Softmax()
            )

        return act

class BasicConvBlock(SavableSequential):
    def __init__(self, in_channels, out_channels, out_to_in=2, kernel_size=3):
        reduce_channels = out_channels
        super().__init__()
        self.add_module("BN1", torch.nn.BatchNorm2d(in_channels))
        self.add_module("ReLU1", torch.nn.ReLU(inplace=True))
        self.add_module("Conv1x1", torch.nn.Conv2d(in_channels, reduce_channels, kernel_size=1, padding=0, bias=False))
        self.add_module("BN2", torch.nn.BatchNorm2d(reduce_channels))
        self.add_module("ReLU2", torch.nn.ReLU(inplace=True))
        self.add_module("ConvNxN", torch.nn.Conv2d(reduce_channels, out_channels, kernel_size, padding=kernel_size//2, bias=False))

        self.kwargs = {'in_channels': in_channels, 'out_channels': out_channels, 'kernel_size': kernel_size, 'out_to_in': out_to_in}

    def forward(self, x):
        """
        Runs the model forward by running each layers in the EquivariantBasicConvBlock
        sequentially, then concatenating the input to output.
        """
        out = super().forward(x)
        return torch.cat((x, out), dim=1)

    @classmethod
    def load_from_dict(cls, properties):
            return super().load_from_dict(properties, kwarg_keys=('in_channels', 'out_channels', 'out_to_in', 'kernel_size'))

class BasicConvBlockOld(SavableModule):
    """
    Inherits from SavableModule. PyTorch wrapper on EquivariantBasicConvBlock.

    Parameters
    -----------
        layers: list
            list of (name, layer) pairs where each layer is a torch.nn.Module
            representing a layer in an EquivariantBasicConvBlock
    Attributes
    ------------
        _layers: list
            list of (name, layer) pairs where each layer is a torch.nn.Module
            representing a layer in an EquivariantBasicConvBlock
    """
    def __init__(self, layers):
        super().__init__()
        self._layers = layers
        self.block = torch.nn.Sequential()
        for (name, layer) in layers:
            self.block.add_module(name, layer)

    def forward(self, x):
        """
        Runs the model forward by running each layers in the EquivariantBasicConvBlock
        sequentially, then concatenating the input to output.
        """
        out = x
        for _, layer in self._layers:
            out = layer(out)
        return torch.cat(x, out)

    def get_save_dict(self, prefix=''):
        properties = {}
        for i, layer in enumerate(self._layers):
            layer_dict = layer[1].state_dict(prefix=f"{prefix}.{layer[0]}.")
            for key, val in layer_dict.items():
                properties[key] = val

            if i == 2 or i== 5: #Convolutions
                if layer[1].bias == None:
                    properties[f"{prefix}.{layer[0]}.bias"] = None 

        return properties

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return self.get_save_dict(prefix=prefix)

    @classmethod
    def load_from_dict(cls, properties):
        relu_1_dict = {}
        relu_2_dict = {}
        bn_1_dict = {}
        bn_2_dict = {}
        conv1x1_dict = {}
        convNxN_dict = {}
        for key in properties.keys():
            if "BN1" in key:
                bn_1_dict[key.split('.')[-1]] = properties[key]
            if "BN2" in key:
                bn_2_dict[key.split('.')[-1]] = properties[key]
            if "ReLU1" in key:
                relu_1_dict[key.split('.')[-1]] = properties[key]
            if "ReLU2" in key:
                relu_2_dict[key.split('.')[-1]] = properties[key]
            if "Conv1x1" in key:
                conv1x1_dict[key.split('.')[-1]] = properties[key]
            if "ConvNxN" in key:
                convNxN_dict[key.split('.')[-1]] = properties[key]

        #Initialize submodules 
        relu_1 = torch.nn.ReLU()
        relu_1.load_state_dict(relu_1_dict)
        relu_2 = torch.nn.ReLU()
        relu_2.load_state_dict(relu_2_dict)

        bn_1_input_size = len(bn_1_dict['bias'])
        bn_1 = torch.nn.BatchNorm2d(bn_1_input_size)
        bn_1.load_state_dict(bn_1_dict)
        bn_2_input_size = len(bn_2_dict['bias'])
        bn_2 = torch.nn.BatchNorm2d(bn_2_input_size)
        bn_2.load_state_dict(bn_2_dict)

        conv_1_input_size = conv1x1_dict['weight'].shape[1]
        conv_1_output_size = conv1x1_dict['weight'].shape[0]
        conv_1_kernel_size = 1
        if conv1x1_dict['bias'] == None:
            del conv1x1_dict['bias']
            conv1x1 = torch.nn.Conv2d(conv_1_input_size, conv_1_output_size, conv_1_kernel_size, bias=False)
            conv1x1.load_state_dict(conv1x1_dict)
        else:
            conv1x1 = torch.nn.Conv2d(conv_1_input_size, conv_1_output_size, conv_1_kernel_size)
            conv1x1.load_state_dict(conv1x1_dict)

        conv_2_input_size = convNxN_dict['weight'].shape[1]
        conv_2_output_size = convNxN_dict['weight'].shape[0]
        conv_2_kernel_size = convNxN_dict['weight'].shape[2]
        if convNxN_dict['bias'] == None:
            del convNxN_dict['bias']
            convNxN = torch.nn.Conv2d(conv_2_input_size, conv_2_output_size, conv_2_kernel_size, bias=False)
            convNxN.load_state_dict(convNxN_dict)
        else:
            convNxN = torch.nn.Conv2d(conv_2_input_size, conv_2_output_size, conv_2_kernel_size)
            convNxN.load_state_dict(convNxN_dict)

        #Form layers
        layers = [("ReLU1", relu_1), ("BN1", bn_1), ("Conv1x1", conv1x1), ("ReLU2", relu_2), ("BN2", bn_2), ("ConvNxN", convNxN)]

        #Return
        return BasicConvBlock(layers)

class NonReduceChannelDimsBlock(SavableSequential):
    """
    Inherits from torch.nn.Sequential, and is a PyTorch wrapper on ReduceChannelDimsBlock.
    Simple block that reduces channel dimensionality.
    """
    def __init__(self, in_channels, out_channels, inplace=True):
        super().__init__()
        self.add_module("0", torch.nn.BatchNorm2d(in_channels))
        self.add_module("1", torch.nn.ReLU(inplace=inplace))
        self.add_module("2", torch.nn.Conv2d(in_channels, out_channels, 1, padding=0, bias=False))

        self.kwargs = {'in_channels': in_channels, 'out_channels': out_channels}

    @classmethod
    def load_from_dict(cls, properties):
       return super().load_from_dict(properties, kwarg_keys=('in_channels', 'out_channels'))

class NonUpscaleBlock(SavableSequential):
    """
    Inherits from torch.nn.Sequential, and is a PyTorch wrapper on UpscaleBlock.
    Simple block that upscales the input.
    """
    def __init__(self, in_channels, scale_factor=2):
        super().__init__()
        self.add_module("0", torch.nn.Upsample(scale_factor=scale_factor))

        self.kwargs = {'in_channels': in_channels, 'scale_factor': scale_factor}

    @classmethod
    def load_from_dict(cls, properties):
         return super().load_from_dict(properties, kwarg_keys=('in_channels', 'scale_factor'))

class NonDownscaleBlock(SavableSequential):
    """
    Inherits from torch.nn.Sequential, and is a PyTorch wrapper on DownscaleBlock.
    Simple block that downscales the input.
    """
    def __init__(self, in_channels, scale_factor=2):
        super().__init__()
        self.add_module("0", torch.nn.MaxPool2d(kernel_size=scale_factor, stride=1, dilation=1))
        self.add_module("1", antialiased_cnns.BlurPool(in_channels, stride=scale_factor))

        self.kwargs = {'in_channels': in_channels, 'scale_factor': scale_factor}

    @classmethod
    def load_from_dict(cls, properties):
            return super().load_from_dict(properties, kwarg_keys=('in_channels', 'scale_factor'))

class NonGroupPoolingBlock(SavableSequential):
    """
    Simple wrapper on MaxPoolChannels, a block implemented by e2cnn to provide 
    a PyTorch equivalent of orientation pooling.
    """
    def __init__(self, input_channels):
        super().__init__()
        self.add_module("0", nn.MaxPoolChannels(input_channels))

        self.kwargs = {'input_channels': input_channels}

    @classmethod
    def load_from_dict(cls, properties):
        return super().load_from_dict(properties, kwarg_keys=('input_channels'))

class NonDenseBlock(SavableSequential):#torch.nn.Sequential):#torch.nn.Sequential):
    def __init__(self, in_channels, out_channels, n_sub_blocks=4, kernel_size=3):
        super().__init__()

        out_channel_basic = (out_channels-in_channels)//n_sub_blocks
        out_to_in = out_channels//in_channels
        reduce_type = out_channel_basic//out_to_in

        for i in range(n_sub_blocks):
            in_channels_basic = in_channels + i*out_channel_basic
            """
            self.add_module(str(i), BasicConvBlock([ 
                ("BN1", torch.nn.BatchNorm2d(in_channels_basic)),
                ("ReLU1", torch.nn.ReLU(inplace=True)),
                ("Conv1x1", torch.nn.Conv2d(in_channels_basic, reduce_type, 1, padding=0, bias=False)),
                ("BN2", torch.nn.BatchNorm2d(reduce_type)),
                ("ReLU2", torch.nn.ReLU(inplace=True)),
                ("ConvNxN", torch.nn.Conv2d(reduce_type, out_channel_basic, kernel_size, padding=kernel_size//2, bias=False))]))
            """
            self.add_module(str(i),BasicConvBlock(in_channels_basic, out_channel_basic, out_to_in, kernel_size)) 
        self.kwargs = {'in_channels': in_channels, 'out_channels': out_channels, 'n_sub_blocks': n_sub_blocks, 'kernel_size': kernel_size}

    @classmethod
    def load_from_dict(cls, properties):
        return super().load_from_dict(properties, kwarg_keys=('in_channels', 'out_channels', 'n_sub_blocks', 'kernel_size'))

class NonTwoStepConvBlock(SavableSequential):
    """
    Inherits from torch.nn.Sequential, and is a PyTorch wrapper on ReduceChannelDimsBlock.
    Simple block that reduces channel dimensionality.
    """
    def __init__(self, in_channels, out_channels, kernel_size=2, bias=True):
        super().__init__()
        self.add_module("0", torch.nn.BatchNorm2d(in_channels))
        self.add_module("1", torch.nn.ReLU(inplace=True))
        self.add_module("2", torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2, bias=bias))
        self.add_module("3", torch.nn.BatchNorm2d(out_channels))
        self.add_module("4", torch.nn.ReLU(inplace=True))
        self.add_module("5", torch.nn.Conv2d(out_channels, out_channels, kernel_size, padding=kernel_size//2, bias=bias))

        self.kwargs = {'in_channels': in_channels, 'out_channels': out_channels, 'kernel_size': kernel_size, 'bias': bias}

    @classmethod
    def load_from_dict(cls, properties):
            return super().load_from_dict(properties, kwarg_keys=('in_channels', 'out_channels', 'kernel_size', 'bias'))

class ConvUNeXtConv(SavableSequential):
    """
    SavableSequential wrapper on Conv block for ConvUNeXt.
    """
    def __init__(self, dim):
        super().__init__()
        #self.dwconv = torch.nn.Conv2d(dim, dim, kernel_size=7, padding=3, stride=1, groups=dim, padding_mode='reflect') # depthwise conv
        self.add_module("0", torch.nn.Conv2d(dim, dim, kernel_size=7, padding=3, stride=1, groups=dim//8, padding_mode='reflect')) # depthwise conv
        #self.norm1 = torch.nn.BatchNorm2d(dim)
        self.add_module("1", torch.nn.BatchNorm2d(dim))
        #self.pwconv1 = torch.nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        #self.add_module("2", torch.nn.Linear(dim, 4 * dim))  # pointwise/1x1 convs, implemented with linear layers
        self.add_module("2", torch.nn.Conv2d(dim, 4*dim, kernel_size=1, padding=0))
        #self.act1 = torch.nn.GELU()
        self.add_module("3", torch.nn.GELU())
        #self.pwconv2 = torch.nn.Linear(4 * dim, dim)
        #self.add_module("4", torch.nn.Linear(4 * dim, dim))
        self.add_module("4", torch.nn.Conv2d(4*dim, dim, kernel_size=1, padding=0))
        #self.norm2 = torch.nn.BatchNorm2d(dim)
        self.add_module("5", torch.nn.BatchNorm2d(dim))
        #self.act2 = torch.nn.GELU()
        self.add_module("6", torch.nn.GELU())

        self.kwargs = {'dim': dim}

    @classmethod
    def load_from_dict(cls, properties):
        return super().load_from_dict(properties, kwarg_keys=('dim',))

    def forward(self, x):
        residual = x
        x = self._modules['0'](x)
        x = self._modules['1'](x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self._modules['2'](x)
        x = self._modules['3'](x)
        x = self._modules['4'](x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = self._modules['5'](x)
        x = self._modules['6'](residual + x)

        return x



"""
This module implements some basic blocks that are useful
for C8-invariant and similarly flavored models.
"""
from abc import ABC, abstractmethod
import math 
import numpy as np
import torch
from e2cnn import nn, gspaces
from scripts.basic_blocks import SavableModule, BasicConvBlock, NonReduceChannelDimsBlock
from scripts.basic_blocks import NonUpscaleBlock, NonDownscaleBlock, NonDenseBlock 
from scripts.basic_blocks import NonTwoStepConvBlock, ConvUNeXtConv 
from scripts.escnn_extension import InteroperableR2Conv, InteroperableBatchNorm
from scripts.escnn_extension import InteroperableMaxBlurPool, InteroperableUpsample
from scripts.escnn_extension import InteroperableReLU, InteroperableGELU

class EquivariantSavableModule(SavableModule, nn.EquivariantModule):
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

    def evaluate_output_shape(self, input_shape):
        """
        Evaluate output shape.

        Arguments
        -----------
            input_shape: tuple
                size-4 tuple (size of inv group, channels, x, y)

        Returns
        --------
            output_shape: tuple
                size-4 tuple (size of inv group, channels, x, y)
        """
        out_shape = (input_shape[0], self.out_type.size, input_shape[2], input_shape[3])
        return out_shape

    @abstractmethod
    def export(self):
        """
        Exports this module to a torch.nn.Module.

        Returns
        ----------
           module: torch.nn.Sequential
                PyTorch version of this module
        """

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
        #assert '.' not in prefix #Having '.' in prefix causes issues when loading from dict

        #Save each layer
        properties = {}
        for key, val in self.block._modules.items():
            if '.' not in prefix:
                layer_dict = val.get_save_dict(prefix=f"{prefix}.{key}")
            else:
                layer_dict = val.get_save_dict(prefix=f"{prefix}{key}")
            for layer_key, layer_val in layer_dict.items():
                properties[layer_key] = layer_val

        return properties

class ReduceChannelDimsBlock(EquivariantSavableModule):
    """
    Inherits from EquivariantSavableModule. Simple block that reduces channel
    dimensionality.

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
        inplace: boolean
            whether or not ReLU activation is inplace
        block: nn.SequentialModule
            actual layers to use in forward method
    """
    def __init__(self, in_type, out_type, inplace=True):
        super().__init__()

        self.inplace = inplace
        self.in_type = in_type
        self.out_type = out_type

        #BN -> ReLU -> 1x1 conv to reduce # of channels
        reduce_channels_block = nn.SequentialModule(
                InteroperableBatchNorm(in_type),
                InteroperableReLU(in_type, inplace=True),
                InteroperableR2Conv(in_type, out_type, kernel_size=1, padding=0, bias=False)
                )
        self.block = reduce_channels_block

    @classmethod
    def load_from_dict(cls, properties):
        """
        Load this module from a state dictionary.

        Arguments
        ----------
            properties: dict
                dictionary containing properties

        Returns
        --------
            reduce_module: ReduceChannelDimsBlock
                module implementing properties
        """
        #Split up layer dictionaries
        bn_dict = {} #0
        relu_dict = {} #1
        conv_dict = {} #2
        for key in properties.keys():
            if key[0] == '.':
                new_key = key[1:]
            else:
                new_key = key
            if int(new_key.split('.')[0]) == 0:
                bn_dict[key] = properties[key]
            elif int(new_key.split('.')[0]) == 1:
                relu_dict[key] = properties[key]
            elif int(new_key.split('.')[0]) == 2:
                conv_dict[key] = properties[key]


        #Load layers and form block
        bn_layer = InteroperableBatchNorm.load_from_dict(bn_dict)
        relu_layer = InteroperableReLU.load_from_dict(relu_dict)
        conv_layer = InteroperableR2Conv.load_from_dict(conv_dict)

        block = nn.SequentialModule(bn_layer, relu_layer, conv_layer)

        #Initialize and return ReduceChannelDimsBlock
        in_type = bn_layer.in_type
        out_type = conv_layer.out_type
        inplace = relu_layer._inplace

        reduce_module = ReduceChannelDimsBlock(in_type, out_type, inplace=inplace)
        reduce_module.block = block

        return reduce_module

    def export(self):
        self.eval()
        in_channels = len(self.in_type)
        
        if self.in_type.representations[0].name == 'regular':
            in_channels *= self.in_type.representations[0].size

        out_channels = len(self.out_type)
        if self.out_type.representations[0].name == 'regular':
            out_channels *= self.out_type.representations[0].size

        new_block = NonReduceChannelDimsBlock(in_channels, out_channels, self.inplace)
        new_block.load_state_dict(self.block.export().state_dict())
        return new_block


class DownscaleBlock(EquivariantSavableModule):
    """
    Inherits from EquivariantSavableModule. Simple block that provides spatial
    downscaling.

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
    def __init__(self, in_type, out_type, kernel_size=2, padding=0):
        super().__init__()

        assert in_type == out_type #in/out types are the same, only change is spatial
        self.in_type = in_type
        self.out_type = out_type

        #Use PointwiseMaxPoolAntialiased to avoid aliasing. This does a pointwise max
        #followed by blur pool
        downscale_block = nn.SequentialModule(
                InteroperableMaxBlurPool(in_type, kernel_size=kernel_size, padding=padding)
                )
        self.block = downscale_block

    @classmethod
    def load_from_dict(cls, properties):
        """
        Load this module from a state dictionary.

        Arguments
        ----------
            properties: dict
                dictionary containing properties

        Returns
        --------
            downscale_module: DownscaleBlock
                module implementing properties
        """
        #Only layer is max blur pool
        max_pool = InteroperableMaxBlurPool.load_from_dict(properties)
        downscale_block = nn.SequentialModule(max_pool)

        #Get properties
        in_type = downscale_block.in_type
        out_type = downscale_block.out_type
        kernel_size = max_pool.kernel_size[0]
        padding = max_pool.padding[0]

        #Init + return
        downscale_module = DownscaleBlock(in_type, out_type,
                kernel_size=kernel_size, padding=padding)
        downscale_module.block = downscale_block

        return downscale_module
    
    def export(self):
        self.eval()
        in_channels = len(self.in_type)
        if self.in_type.representations[0].name == 'regular':
            in_channels *= self.in_type.representations[0].size

        kernel_size = self.block._modules['0'].kernel_size[0]
        new_block = NonDownscaleBlock(in_channels, kernel_size)
        return new_block

class UpscaleBlock(EquivariantSavableModule):
    """
    Inherits from EqivariantSavableModule. Simple block that provides spatial
    upsampling.

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
    def __init__(self, in_type, out_type, scale_factor=2):
        super().__init__()

        self.in_type = in_type
        self.out_type = out_type
        #Wrapper on torch.nn.functional.interpolate()
        upscale_block = nn.SequentialModule(
                InteroperableUpsample(in_type, scale_factor=scale_factor)
                )
        self.block = upscale_block

    @classmethod
    def load_from_dict(cls, properties):
        """
        Load this module from a state dictionary.

        Arguments
        ----------
            properties: dict
                dictionary containing properties

        Returns
        --------
            upscale_module: UpscaleBlock
                module implementing properties
        """
        #Only layer is interp
        interp = InteroperableUpsample.load_from_dict(properties)
        upscale_block = nn.SequentialModule(interp)

        #Get properties
        in_type = upscale_block.in_type
        out_type = upscale_block.out_type

        #Init + return
        upscale_module = UpscaleBlock(in_type, out_type)
        upscale_module.block = upscale_block

        return upscale_module
    
    def export(self):
        self.eval()
        in_channels = len(self.in_type)
        
        if self.in_type.representations[0].name == 'regular':
            in_channels *= self.in_type.representations[0].size

        scale_factor = self.block._modules['0']._scale_factor

        new_block = NonUpscaleBlock(in_channels, scale_factor)
        return new_block




class EquivariantBasicConvBlock(EquivariantSavableModule):
    """
    Inherits from EquivariantSavableModule. Describes a two-step basic block used in the
    DenseBlock and C8InvariantCNN class.

    Parameters
    -----------
        in_type: escnn.nn.FieldType
            input type of basic block, giving invariant representation of block input
        out_type: escnn.nn.FieldType
            output type of basic block EXCLUDING CONCATENATION to input. This is NOT the same as
                the out_type attribute
        out_to_in: int (optional)
            ratio of # of channels in dense block output to # of channels in dense block
                input (default is 2). May not be the same as out-to-in ratio of channels
                for a specific EquivariantBasicConvBlock

    Attributes
    ------------
        in_type: escnn.nn.FieldType
            input type of basic block, giving invariant representation of block input
        out_type: escnn.nn.FieldType
            output type of basic block, giving invariant rep/dims of block output
        _layers: list
            list of (name, escnn.nn.SequentialModule) pairs representing layers in order:
                BN, ReLU, 1x1 Conv, BN, ReLU, 3x3 Conv
        batch_norm: escnn.nn.InnerBatchNorm
            batch norm layer performed on input
        relu: escnn.nn.ReLu
            relu layer prior to first convolutional layer
        conv_1x1: InteroperableR2Conv
            C8-invariant 1x1 convolutional layer done after first batch norm + ReLu
        batch_norm_2: escnn.nn.InnerBatchNorm
            batch norm layer on output of 1x1 convolution
        relu_2: escnn.nn.ReLu
            relu layer done on output of batch norm
        conv_NxN: InteroperableR2Conv
            final C8-invariant NxN convolutional layer to give output of basic block
    """
    def __init__(self, in_type, out_type, out_to_in=2, kernel_size=3):
        super().__init__()

        assert kernel_size%2 == 1, "Kernel side length must be odd."

        #Input and output type (output size is different due to concatenation)
        self.in_type = in_type
        #Divide by 8 because it's C8-invariant
        self.out_type = nn.FieldType(in_type.gspace,
                round((in_type.size+out_type.size)/8)*[in_type.gspace.regular_repr])
        self.out_to_in = out_to_in

        #Layers (in order)
        self.batch_norm = InteroperableBatchNorm(in_type)
        self.relu = InteroperableReLU(in_type, inplace=True)
        assert round(out_type.size/8) > 0, "Too few input channels"
        reduce_type = nn.FieldType(in_type.gspace,
                round(out_type.size/(8))*[in_type.gspace.regular_repr])
        self.reduce_type = reduce_type
        self.final_conv_type = out_type
        self.conv_1x1 = InteroperableR2Conv(in_type, reduce_type,
                kernel_size=1, padding=0, bias=False)
        self.batch_norm_2 = InteroperableBatchNorm(reduce_type)
        self.relu_2 = InteroperableReLU(reduce_type, inplace=True)
        self.conv_NxN = InteroperableR2Conv(reduce_type, out_type,
                kernel_size=kernel_size, padding=math.floor(kernel_size/2), bias=False)

        #Add layers to self._layers
        self._layers = []
        self._layers.append(("BN1", self.batch_norm))
        self._layers.append(("ReLU1", self.relu))
        self._layers.append(("Conv1x1", self.conv_1x1))
        self._layers.append(("BN2", self.batch_norm_2))
        self._layers.append(("ReLU2", self.relu_2))
        self._layers.append(("ConvNxN", self.conv_NxN))

    def forward(self, *args):
        """
        Runs the EquivariantBasicConvBlock forwards, putting an input tensor
        through each layer (in order).

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
        out = x
        for _, layer in self._layers:
            out = layer(out)
        return nn.tensor_directsum([x, out])

    def export(self):
        """
        Export this module to a normal PyTorch module and set to "eval" mode.

        Returns
        ----------
            module: BasicConvBlock
                torch export of this module
        """
        self.eval()
        in_channels = len(self.in_type)
        
        if self.in_type.representations[0].name == 'regular':
            in_channels *= self.in_type.representations[0].size

        out_channels = len(self.out_type)
        if self.out_type.representations[0].name == 'regular':
            out_channels *= self.out_type.representations[0].size

        out_channels = out_channels - in_channels

        self.kernel_size = self._layers[5][1].kernel_size

        #TODO clean up. gross but works for now
        new_block = BasicConvBlock(in_channels, out_channels, kernel_size=self.kernel_size,
                out_to_in=self.out_to_in)
        new_sequential = torch.nn.Sequential()
        for name, layer in self._layers:
            new_sequential.add_module(name, layer.export())
        transfer_state = torch.nn.Sequential(new_sequential).state_dict(prefix='')
        clean_transfer_state = {}
        for key, value in transfer_state.items():
            if key.startswith('0.'):
                clean_transfer_state[key[2:]] = value
        new_block.load_state_dict(clean_transfer_state)
        return new_block


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
        properties = {}
        for layer in self._layers:
            layer_dict = layer[1].get_save_dict(prefix=f"{prefix}.{layer[0]}")
            for key, val in layer_dict.items():
                properties[key] = val
        properties[f"{prefix}.out_to_in"] = self.out_to_in

        return properties

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return self.get_save_dict(prefix=prefix)

    @classmethod
    def load_from_dict(cls, properties):
        """
        Load this module from a state dictionary.

        Arguments
        ----------
            properties: dict
                dictionary containing properties

        Returns
        --------
            basic_module: EquivariantBasicConvBlock
                module implementing properties
        """
        #Get dictionaries with properties for each submodule
        relu_1_dict = {}
        relu_2_dict = {}
        bn_1_dict = {}
        bn_2_dict = {}
        conv1x1_dict = {}
        convNxN_dict = {}
        for key in properties.keys():
            if "BN1" in key:
                bn_1_dict[key] = properties[key]
            if "BN2" in key:
                bn_2_dict[key] = properties[key]
            if "ReLU1" in key:
                relu_1_dict[key] = properties[key]
            if "ReLU2" in key:
                relu_2_dict[key] = properties[key]
            if "Conv1x1" in key:
                conv1x1_dict[key] = properties[key]
            if "ConvNxN" in key:
                convNxN_dict[key] = properties[key]

        #Initialize first + last submodules
        relu_1 = InteroperableReLU.load_from_dict(relu_1_dict)
        conv_NxN = InteroperableR2Conv.load_from_dict(convNxN_dict)

        #Get in_type and out_type
        out_to_in = 2
        for key in properties.keys():
            if key.split('.')[-1] == "out_to_in":
                out_to_in = properties[key]

        #Create new EquivariantBasicConvBlock
        basic_module = EquivariantBasicConvBlock(relu_1.in_type, conv_NxN.out_type,
                out_to_in=out_to_in)

        #Set properties appropriately (+ init remaining submodules)
        basic_module.batch_norm = InteroperableBatchNorm.load_from_dict(bn_1_dict)
        basic_module.relu = relu_1
        basic_module.conv_1x1 = InteroperableR2Conv.load_from_dict(conv1x1_dict)
        basic_module.batch_norm_2 = InteroperableBatchNorm.load_from_dict(bn_2_dict)
        basic_module.relu_2 = InteroperableReLU.load_from_dict(relu_2_dict)
        basic_module.conv_NxN = conv_NxN

        #Add layers to self._layers
        basic_module._layers = []
        basic_module._layers.append(("BN1", basic_module.batch_norm))
        basic_module._layers.append(("ReLU1", basic_module.relu))
        basic_module._layers.append(("Conv1x1", basic_module.conv_1x1))
        basic_module._layers.append(("BN2", basic_module.batch_norm_2))
        basic_module._layers.append(("ReLU2", basic_module.relu_2))
        basic_module._layers.append(("ConvNxN", basic_module.conv_NxN))

        #Return
        return basic_module

class TwoStepConvBlock(EquivariantSavableModule):
    """
    Inherits from EquivariantSavableModule. Describes a two-step convolution
    block used in the VanillaC8InvariantCNN class.

    Parameters
    -----------
        in_type: escnn.nn.FieldType
           input type, giving invariant representation and dims of block input
        out_type: escnn.nn.FieldType
              output type, giving invariant representation and dims of block output
        kernel_size: int (optional)
            size of convolution kernel

    Attributes
    ------------
        block: e2cnn.nn.SequentialModule
            implementation of the two-step convolution block
    """
    def __init__(self, in_type, out_type, kernel_size=2, has_bias=True):
        super().__init__()

        self.in_type = in_type
        self.out_type = out_type

        #BN -> ReLU -> Conv -> BN -> ReLU -> Conv
        conv_block = nn.SequentialModule(
                InteroperableBatchNorm(in_type), 
                InteroperableReLU(in_type, inplace=True),
                InteroperableR2Conv(in_type, out_type, kernel_size=kernel_size,
                    padding=math.floor(kernel_size/2), bias=has_bias),
                InteroperableBatchNorm(out_type),
                InteroperableReLU(out_type, inplace=True),
                InteroperableR2Conv(out_type, out_type, kernel_size=kernel_size,
                    padding=math.floor(kernel_size/2), bias=has_bias)
                )
        self.block = conv_block

    @classmethod
    def load_from_dict(cls, properties):
        """
        Load this module from a state dictionary.

        Arguments
        ----------
            properties: dict
                dictionary containing properties

        Returns
        --------
            conv_module: TwoStepConvBlock
                module implementing properties
        """
        #Get Conv2d + BatchNorm2d properties, and relu slope
        bn_dict_1 = {}
        relu_dict_1 = {} 
        conv_dict_1 = {}
        bn_dict_2 = {}
        relu_dict_2 = {}
        conv_dict_2 = {}
        for key in properties.keys():
            if key[0] == '.':
                new_key = key[1:]
            else:
                new_key = key
            if new_key.split('.')[0] == '0':
                bn_dict_1[new_key] = properties[key]
            if new_key.split('.')[0] == '1':
                relu_dict_1[new_key] = properties[key]
            if new_key.split('.')[0] =='2':
                conv_dict_1[new_key] = properties[key]
            if new_key.split('.')[0] == '3':
                bn_dict_2[new_key] = properties[key]
            if new_key.split('.')[0] == '4':
                relu_dict_2[new_key] = properties[key]
            if new_key.split('.')[0] =='5':
                conv_dict_2[new_key] = properties[key]

        #Load layers and form block
        bn_1 = InteroperableBatchNorm.load_from_dict(bn_dict_1)
        relu_1 = InteroperableReLU.load_from_dict(relu_dict_1)
        conv_1 = InteroperableR2Conv.load_from_dict(conv_dict_1)
        bn_2 = InteroperableBatchNorm.load_from_dict(bn_dict_2)
        relu_2 = InteroperableReLU.load_from_dict(relu_dict_2)
        conv_2 = InteroperableR2Conv.load_from_dict(conv_dict_2)

        block = nn.SequentialModule(bn_1, relu_1, conv_1, bn_2, 
                relu_2, conv_2)

        #Initialize and return TwoStepConvBlock
        in_type = bn_1.in_type
        out_type = conv_2.out_type
        kernel_size = conv_1.kernel_size
        bias = conv_1.bias is not None

        two_step_conv_block = TwoStepConvBlock(in_type, out_type, kernel_size,
                bias)
        two_step_conv_block.block = block
        return two_step_conv_block

    def export(self):
        self.eval()
        in_channels = len(self.in_type)
        
        if self.in_type.representations[0].name == 'regular':
            in_channels *= self.in_type.representations[0].size

        out_channels = len(self.out_type)
        if self.out_type.representations[0].name == 'regular':
            out_channels *= self.out_type.representations[0].size

        self.kernel_size = self.block._modules['2'].kernel_size
        self.bias = self.block._modules['2'].bias is not None

        new_block = NonTwoStepConvBlock(in_channels, out_channels, kernel_size=self.kernel_size,
                bias=self.bias)
        new_block.load_state_dict(self.block.export().state_dict())
        return new_block



class DenseBlock(EquivariantSavableModule):
    """
    Inherits from EquivariantSavableModule. Describes a dense block used in
    the C8InvariantCNN class.

    Parameters
    ------------
        in_type: escnn.nn.FieldType
            input type of basic block, giving invariant representation of block input. Must have
                fewer channels than out_type
        out_type: escnn.nn.FieldType
            output type of basic block, giving invariant rep/dims of block output
        n_sub_blocks: int (optional)
            number of sub-blocks in each dense block
            default is 4
        kernel_size: int (optional)
            sidelength for non-1x1 convolution in EquivariantBasicConvBlock. default is 3

    Attributes
    -----------
        block: escnn.nn.SequentialModule
            implementation of the DenseBlock
    """
    def __init__(self, in_type, out_type, n_sub_blocks=4, kernel_size=3):
        super().__init__()

        #Assert # out channels > # in channels
        assert in_type.size < out_type.size,\
            "DenseBlock must have more output channels than input channels"

        #Set attributes
        self.in_type = in_type
        self.out_type = out_type
        self.out_to_in = out_type.size/in_type.size
        self.n_sub_blocks = n_sub_blocks
        self.kernel_size = kernel_size

        inv_group = in_type.gspace
        raw_in_channels = round(in_type.size/8)
        basic_out_channels_no_concat = (out_type.size - in_type.size)/(n_sub_blocks*8)

        #Assert that # of output channels (excluding concatenation) in each sub-block is an int
        assert np.fabs(math.floor(basic_out_channels_no_concat)\
                - basic_out_channels_no_concat) <= 1e-9,\
                "Each sub-block of DenseBlock must have an integer number of channels"

        #Make layers
        self.block = self._make_layer(n_sub_blocks, raw_in_channels,\
                basic_out_channels_no_concat, inv_group)

    def _make_layer(self, n_sub_blocks, raw_in_channels, basic_out_channels_no_concat, inv_group):
        """
        Actually builds the DenseBlock out of EquivariantBasicConvBlocks.

        Arguments
        -----------
            n_sub_blocks: int
                number of sub-blocks in each dense block
            raw_in_channels: int
                number of input channels into DenseBlock
            basic_out_channels_no_concat: int
                number of output channels of each sub-block (excluding concatenated input)
            inv_group: escnn.gspaces.GSpace
                space where the input signal lives

        Returns
        ---------
            layers: escnn.nn.SequentialModule
                group equivariant sequential module that implements the DenseBlock
        """
        layers = []
        #Make each of n sub blocks
        for i in range(n_sub_blocks):
            #Get input and output channels for this specific EquivariantBasicConvBlock
            in_channels = raw_in_channels + i*basic_out_channels_no_concat
            out_channels = basic_out_channels_no_concat #Output of conv block WITHOUT concatenation

            #Add EquivariantBasicConvBlock to layers
            in_type = nn.FieldType(inv_group, round(in_channels)*[inv_group.regular_repr])
            out_type = nn.FieldType(inv_group, round(out_channels)*[inv_group.regular_repr])
            layers.append(EquivariantBasicConvBlock(in_type, out_type, out_to_in=self.out_to_in, kernel_size=self.kernel_size))

        return nn.SequentialModule(*layers)

    @classmethod
    def load_from_dict(cls, properties):
        """
        Load this module from a state dictionary.

        Arguments
        ----------
            properties: dict
                dictionary containing properties

        Returns
        --------
            basic_module: DenseBlock
                module implementing properties
        """
        #Get number of sub blocks
        num_sub_blocks = 0
        for key in properties.keys():
            if key[0] == '.':
                new_key = key[1:]
            else:
                new_key = key
            num_sub_blocks = max(num_sub_blocks, int(new_key.split('.')[0]))

        #Load each sub block
        modules = []
        for i in range(num_sub_blocks+1):
            i_dict = {}
            for key in properties.keys():
                if key[0] == '.':
                    new_key = key[1:]
                else:
                    new_key = key
                if int(new_key.split('.')[0]) == i:
                    i_dict[new_key] = properties[key]
            i_module = EquivariantBasicConvBlock.load_from_dict(i_dict)
            modules.append(i_module)

        #Get input type and output type of dense block
        out_type = modules[-1].out_type
        in_type = modules[0].in_type

        #Initialize DenseBlock and its layers, and return
        dense_block = DenseBlock(in_type, out_type, n_sub_blocks=len(modules))
        dense_block.block = nn.SequentialModule(*modules)
        return dense_block

    def export(self):
        self.eval()

        in_channels = len(self.in_type)
        
        if self.in_type.representations[0].name == 'regular':
            in_channels *= self.in_type.representations[0].size

        out_channels = len(self.out_type)
        if self.out_type.representations[0].name == 'regular':
            out_channels *= self.out_type.representations[0].size

        new_block = NonDenseBlock(in_channels, out_channels, self.n_sub_blocks, self.kernel_size)

        current_state_dict = {}
        for i in range(self.n_sub_blocks):
            i_state_dict = self.block.eval().export()._modules[str(i)].state_dict(prefix=f'{str(i)}.')
            for key in i_state_dict.keys():
                if key[1] =='.' and key[2]=='.':
                    new_key = key[0:2] + key[3:]
                else:
                    new_key = key
                if 'base' not in key:
                    current_state_dict[new_key] = i_state_dict[key]
        new_block.load_state_dict(current_state_dict)
        return new_block

class EquivariantConvUNeXtConv(EquivariantSavableModule):
    """
    Inherits from EquivariantSavableModule. Describes rotation invariant version 
    of convolution block used in ConvUNeXt paper.

    Parameters
    -----------
        in_type: escnn.nn.FieldType
           input type, giving invariant representation and dims of block input
        out_type: escnn.nn.FieldType
              output type, giving invariant representation and dims of block output
        kernel_size: int (optional)
            size of convolution kernel

    Attributes
    ------------
        block: e2cnn.nn.SequentialModule
            implementation of the two-step convolution block
    """
    def __init__(self, in_type, out_type):
        super().__init__()

        self.in_type = in_type
        dim = len(self.in_type)
        r2_act = gspaces.Rot2dOnR2(N=8)
        four_dim_type = nn.FieldType(gspaces.Rot2dOnR2(N=8), dim*4*[r2_act.regular_repr])

        self.out_type = out_type
        assert in_type == out_type, "Input and output types must be the same"

        conv_block = nn.SequentialModule(
                #Depthwise convolution
                InteroperableR2Conv(in_type, in_type, kernel_size=7, padding=3, stride=1, 
                    groups=dim, padding_mode='reflect'),
                #Norm 1
                InteroperableBatchNorm(in_type),
                #Pointwise convolution
                InteroperableR2Conv(in_type, four_dim_type, kernel_size=1, padding=0, stride=1),
                #Activation
                InteroperableGELU(four_dim_type),
                #PW conv 2
                InteroperableR2Conv(four_dim_type, out_type, kernel_size=1, padding=0, stride=1),
                #Norm 2
                InteroperableBatchNorm(out_type),
                #Activation 2
                InteroperableGELU(out_type)
                )
        self.block = conv_block

    def forward(self, x):
        #Skip permutations because they can't really be done cleanly 
        #for rotationally invariant representations, and because 
        #they only really exist to speed up the convnext computation 
        #anyway
        residual = x
        x = self.block._modules['0'](x)
        x = self.block._modules['1'](x)
        x = self.block._modules['2'](x)
        x = self.block._modules['3'](x)
        x = self.block._modules['4'](x)
        x = self.block._modules['5'](x)
        x = self.block._modules['6'](residual + x)
        return x


    @classmethod
    def load_from_dict(cls, properties):
        """
        Load this module from a state dictionary.

        Arguments
        ----------
            properties: dict
                dictionary containing properties

        Returns
        --------
            conv_module: TwoStepConvBlock
                module implementing properties
        """
        dw_conv_dict = {}
        bn_dict_1 = {}
        pw_conv_dict_1 = {}
        gelu_dict_1 = {}
        pw_conv_dict_2 = {}
        bn_dict_2 = {}
        gelu_dict_2 = {}
        for key in properties.keys():
            if key[0] == '.':
                new_key = key[1:]
            else:
                new_key = key
            if new_key.split('.')[0] == '0':
                dw_conv_dict[new_key] = properties[key]
            if new_key.split('.')[0] == '1':
                bn_dict_1[new_key] = properties[key]
            if new_key.split('.')[0] =='2':
                pw_conv_dict_1[new_key] = properties[key]
            if new_key.split('.')[0] == '3':
                gelu_dict_1[new_key] = properties[key]
            if new_key.split('.')[0] == '4':
                pw_conv_dict_2[new_key] = properties[key]
            if new_key.split('.')[0] =='5':
                bn_dict_2[new_key] = properties[key]
            if new_key.split('.')[0] == '6':
                gelu_dict_2[new_key] = properties[key]

        #Load layers and form block
        dw_conv = InteroperableR2Conv.load_from_dict(dw_conv_dict)
        bn_1 = InteroperableBatchNorm.load_from_dict(bn_dict_1)
        pw_conv_1 = InteroperableR2Conv.load_from_dict(pw_conv_dict_1)
        gelu_1 = InteroperableGELU.load_from_dict(gelu_dict_1)
        pw_conv_2 = InteroperableR2Conv.load_from_dict(pw_conv_dict_2)
        bn_2 = InteroperableBatchNorm.load_from_dict(bn_dict_2)
        gelu_2 = InteroperableGELU.load_from_dict(gelu_dict_2)

        block = nn.SequentialModule(dw_conv, bn_1, pw_conv_1, gelu_1, pw_conv_2, bn_2, gelu_2)

        #Initialize and return TwoStepConvBlock
        in_type = dw_conv.in_type
        out_type = gelu_2.out_type

        eq_convunext_conv = EquivariantConvUNeXtConv(in_type, out_type)
        eq_convunext_conv.block = block
        return eq_convunext_conv

    def export(self):
        self.eval()
        dims = len(self.in_type)*8

        new_block = ConvUNeXtConv(dims)
        new_block.load_state_dict(self.block.export().state_dict())

        return new_block

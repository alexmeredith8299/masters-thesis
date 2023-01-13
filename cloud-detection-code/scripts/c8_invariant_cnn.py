"""
This module implements an E(8)-equivariant steerable CNN as described in General E(2)-Equivariant
Steerable CNNs (Weiler & Cesa 2019).
"""
from abc import ABC, abstractmethod
import torch
from e2cnn import gspaces, nn
from scripts.escnn_extension import InteroperableGroupPooling
from scripts.equivariant_basic_blocks import DenseBlock, ReduceChannelDimsBlock
from scripts.equivariant_basic_blocks import DownscaleBlock, UpscaleBlock
from scripts.equivariant_basic_blocks import TwoStepConvBlock 
from scripts.basic_blocks import ActivationBlock 
from scripts.block_builder import ReduceDimsBlockBuilder, DenseBlockBuilder
from scripts.block_builder import TwoStepConvBlockBuilder, DownscaleBlockBuilder
from scripts.block_builder import UpscaleBlockBuilder, GroupPoolingBlockBuilder
from scripts.block_builder import ActivationBlockBuilder, InvariantType

class BaseC8InvariantCNN(ABC, torch.nn.Module):
    """
    Abstract implementation of a C8-invariant CNN that is savable and loadable.

    Parameters
    -----------
        kernel_size: int
            side length of convolutions in conv blocks (default=3)
        input_channels: int
            number of input channels (default=4)
    """
    @abstractmethod
    def __init__(self, *args):
        super().__init__()
        self.kernel_size = 3
        self.input_channels = 4
        self.f_1 = 8
        self.inv_group_type = InvariantType.C8
        self.inv_group = gspaces.Rot2dOnR2(N=8)
        self.named_blocks = {}

    @abstractmethod
    def forward(self, *args):
        """
        Forward pass of the CNN. Called during model evaluation and testing.

        Arguments
        -----------
            x: escnn.nn.GeometricTensor
                tensor with an invariant type given as input
        Returns
        ---------
            out: escnn.nn.GeometricTensor
                tensor with an invariant type given as output
        """

    def __getstate__(self):
        """
        This function is called when pickling the model.
        It returns a dictionary containing all the fields of the model.

        Returns
        ----------
            state: dict
                dictionary containing all the fields of the model
        """
        state = self.state_dict().copy()
        return state

    def __setstate__(self, state):
        """
        This function is called when unpickling the model.
        It sets all the fields of the model from the dictionary.

        Arguments
        ----------
            state: dict
                dictionary containing all the fields of the model
        """
        self.__init__(state['kernel_size'])
        self.reload_from_dict(state)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
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
        return self.get_save_dict()

    def get_save_dict(self):
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

        #Get dicts for each sub-block and update properties
        for name in self.named_blocks:
            block_dict = getattr(self, name).get_save_dict(prefix=name+'.')
            properties.update(block_dict)

        properties["kernel_size"] = self.kernel_size
        properties["f_1"] = self.f_1
        properties["inv_group_type"] = self.inv_group_type

        return properties


    def reload_from_dict(self, properties):
        """
        Load this module from a state dictionary.

        Arguments
        ----------
            properties: dict
                dictionary containing properties

        """
        try:
            inv_group_type = properties.pop("inv_group_type")
            if inv_group_type == InvariantType.C8:
                self.inv_group = gspaces.Rot2dOnR2(N=8)
            elif inv_group_type == InvariantType.NONE:
                self.inv_group = None
        except KeyError: #TODO maybe remove this at some pt
            self.inv_group = gspaces.Rot2dOnR2(N=8)
        #Load blocks
        for name, block_type in self.named_blocks.items():
            #Get properties dict for each block
            block_dict = {}
            for key, val in properties.items():
                if key.split('.')[0] == name:
                    new_key = '.'.join(key.split('.')[1:])
                    block_dict[new_key] = val

            #Hacky
            block = block_type.build_block_from_dict(self.inv_group_type, block_dict)
            setattr(self, name, block)


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
            module: C8InvariantCNN
                module implementing properties
        """
        #Initialize C8InvariantCNN
        cnn = cls(kernel_size=properties["kernel_size"])
        cnn.reload_from_dict(properties)
        return cnn

#TODO change name
class DenseC8InvariantCNN(BaseC8InvariantCNN, torch.nn.Module):
    """
    This class implements a C8-invariant Dense CNN that takes in 144x144xN tensor inputs and
    returns a 144x144x1 cloud mask.

    Parameters
    ------------
        kernel_size: int
            side length of convolutions in DenseBlocks (default=3)
        input_channels: int
            number of input channels (default=4)
    """
    def __init__(self, kernel_size=3, input_channels=4, f_1=64, inv_group_type=InvariantType.C8, n_classes=2):
        super().__init__()

        if self.inv_group_type != InvariantType.NONE:
            f_1 = f_1//8

        multiclass = n_classes > 2
        n_out_classes = n_classes if multiclass else 1

        #Model is equivariant under rotations by 45 degrees (C8)
        self.inv_group_type = inv_group_type
        self.f_1, self.f_2, self.f_3, self.f_4, self.f_5 = f_1, f_1*2, f_1*4, f_1*8, f_1*16
        self.kernel_size = kernel_size

        #First convolution (144x144x4 -> 144x144x16)
        self.input_conv_block = ReduceDimsBlockBuilder.build_block(self.inv_group_type, input_channels, self.f_1//2, input_regular=False)

        #Second convolution (dense) + downscale (144x144x16 -> 72x72x32)
        self.dense_conv_block_1 = DenseBlockBuilder.build_block(self.inv_group_type, self.f_1//2, self.f_1, kernel_size=self.kernel_size)
        self.downsample_1 = DownscaleBlockBuilder.build_block(self.inv_group_type, self.f_1)

        #Third convolution (dense) + downscale (72x72x32 -> 36x36x64)
        self.dense_conv_block_2 = DenseBlockBuilder.build_block(self.inv_group_type, self.f_1, self.f_2, kernel_size=self.kernel_size)
        self.downsample_2 = DownscaleBlockBuilder.build_block(self.inv_group_type, self.f_2)

        #Fourth convolution (dense) + downscale (36x36x64 -> 18x18x128)
        self.dense_conv_block_3 = DenseBlockBuilder.build_block(self.inv_group_type, self.f_2, self.f_3, kernel_size=self.kernel_size)
        self.downsample_3 = DownscaleBlockBuilder.build_block(self.inv_group_type, self.f_3)

        #Fifth convolution (dense) + downscale (18x18x128 -> 9x9x256)
        self.dense_conv_block_4 = DenseBlockBuilder.build_block(self.inv_group_type, self.f_3, self.f_4, kernel_size=self.kernel_size)
        self.downsample_4 = DownscaleBlockBuilder.build_block(self.inv_group_type, self.f_4)

        #Sixth convolution (dense bottleneck) (9x9x256 -> 9x9x512)
        self.dense_conv_block_5 = DenseBlockBuilder.build_block(self.inv_group_type, self.f_4, self.f_5, kernel_size=self.kernel_size)

        #Now time to upscale! (9x9x512 -> 18x18x256)
        self.upscale_1 = UpscaleBlockBuilder.build_block(self.inv_group_type, self.f_5)

        #Reduce channels with 1x1 conv before concatenation (18x18x512 -> 18x18x256)
        self.reduce_1a = ReduceDimsBlockBuilder.build_block(self.inv_group_type, self.f_5, self.f_4)

        #1x1 conv block on concatenated inputs (18x18x512 -> 18x18x128)
        self.reduce_1 = ReduceDimsBlockBuilder.build_block(self.inv_group_type, self.f_5, self.f_4//2)

        #Dense block (18x18x128 -> 18x18x256)
        self.dense_conv_block_6 = DenseBlockBuilder.build_block(self.inv_group_type, self.f_4//2, self.f_4, kernel_size=self.kernel_size)

        #Repeat upscale, concat, conv, dense (18x18x256 -> 36x36x128)
        self.upscale_2 = UpscaleBlockBuilder.build_block(self.inv_group_type, self.f_4)
        self.reduce_2a = ReduceDimsBlockBuilder.build_block(self.inv_group_type, self.f_4, self.f_3)
        self.reduce_2 = ReduceDimsBlockBuilder.build_block(self.inv_group_type, self.f_4, self.f_3//2)
        self.dense_conv_block_7 = DenseBlockBuilder.build_block(self.inv_group_type, self.f_3//2, self.f_3, kernel_size=self.kernel_size)

        #Repeat upscale, concat, conv, dense (36x36x128 -> 72x72x64)
        self.upscale_3 = UpscaleBlockBuilder.build_block(self.inv_group_type, self.f_3)
        self.reduce_3a = ReduceDimsBlockBuilder.build_block(self.inv_group_type, self.f_3, self.f_2)
        self.reduce_3 = ReduceDimsBlockBuilder.build_block(self.inv_group_type, self.f_3, self.f_2//2)
        self.dense_conv_block_8 = DenseBlockBuilder.build_block(self.inv_group_type, self.f_2//2, self.f_2, kernel_size=self.kernel_size)

        #Repeat upscale, conv, dense (72x72x64 -> 144x144x32)
        self.upscale_4 = UpscaleBlockBuilder.build_block(self.inv_group_type, self.f_2)
        self.reduce_4 = ReduceDimsBlockBuilder.build_block(self.inv_group_type, self.f_2, self.f_1//2)
        self.dense_conv_block_9 = DenseBlockBuilder.build_block(self.inv_group_type, self.f_1//2, self.f_1, kernel_size=self.kernel_size)

        #Final output (1x1 conv to go to 144x144x1, then group pooling)
        self.group_reduce_block = ReduceDimsBlockBuilder.build_block(self.inv_group_type, self.f_1, n_out_classes)
        if self.inv_group_type == InvariantType.NONE:
            self.group_pool_block = GroupPoolingBlockBuilder.build_block(self.inv_group_type, 1)
        else:
            self.group_pool_block = GroupPoolingBlockBuilder.build_block(self.inv_group_type, n_out_classes)
        #Normal torch block (not group invariant) for final 1x1 conv/BN/Relu
        self.plain_torch_block = ActivationBlockBuilder.build_block(InvariantType.NONE, relu_slope=0.1, n_classes=n_out_classes)

        #Dictionary mapping sub-blocks to types (for loading/saving from dictionaries)
        self.named_blocks = {}
        self.named_blocks['input_conv_block'] = ReduceDimsBlockBuilder 
        self.named_blocks['dense_conv_block_1'] = DenseBlockBuilder
        self.named_blocks['dense_conv_block_2'] = DenseBlockBuilder
        self.named_blocks['dense_conv_block_3'] = DenseBlockBuilder
        self.named_blocks['dense_conv_block_4'] = DenseBlockBuilder
        self.named_blocks['dense_conv_block_5'] = DenseBlockBuilder
        self.named_blocks['dense_conv_block_6'] = DenseBlockBuilder
        self.named_blocks['dense_conv_block_7'] = DenseBlockBuilder
        self.named_blocks['dense_conv_block_8'] = DenseBlockBuilder
        self.named_blocks['dense_conv_block_9'] = DenseBlockBuilder
        self.named_blocks['downsample_1'] = DownscaleBlockBuilder
        self.named_blocks['downsample_2'] = DownscaleBlockBuilder
        self.named_blocks['downsample_3'] = DownscaleBlockBuilder
        self.named_blocks['downsample_4'] = DownscaleBlockBuilder
        self.named_blocks['upscale_1'] = UpscaleBlockBuilder
        self.named_blocks['upscale_2'] = UpscaleBlockBuilder
        self.named_blocks['upscale_3'] = UpscaleBlockBuilder
        self.named_blocks['upscale_4'] = UpscaleBlockBuilder
        self.named_blocks['reduce_1a'] = ReduceDimsBlockBuilder
        self.named_blocks['reduce_1'] = ReduceDimsBlockBuilder
        self.named_blocks['reduce_2a'] = ReduceDimsBlockBuilder
        self.named_blocks['reduce_2'] = ReduceDimsBlockBuilder
        self.named_blocks['reduce_3a'] = ReduceDimsBlockBuilder
        self.named_blocks['reduce_3'] = ReduceDimsBlockBuilder
        self.named_blocks['reduce_4'] = ReduceDimsBlockBuilder
        self.named_blocks['group_reduce_block'] = ReduceDimsBlockBuilder
        self.named_blocks['group_pool_block'] = GroupPoolingBlockBuilder
        self.named_blocks['plain_torch_block'] = ActivationBlockBuilder

        #Save to dict + then reload to ensure that all fields are initialized
        self.reload_from_dict(self.get_save_dict())

    def forward(self, input_tensor):
        """
        This function runs the C8 invariant CNN forwards.

        Arguments
        ----------
            input_tensor: torch.Tensor
                RGB+IR images to put in the model

        Returns
        --------
            output_tensor: torch.tensor
                cloud map
        """
        #Transform input into C8-invariant GeometricTensor
        if self.inv_group_type != InvariantType.NONE:
            x = nn.GeometricTensor(input_tensor, self.input_conv_block.in_type)
        else:
            x = input_tensor

        #Ingest layer
        x = self.input_conv_block(x) #144x144x16

        #Repeatedly downscale
        x = self.dense_conv_block_1(x) #144x144x32
        x = self.downsample_1(x) #72x72x32

        x_cat_72x72 = self.dense_conv_block_2(x) #72x72x64
        x = self.downsample_2(x_cat_72x72) #36x36x64

        x_cat_36x36 = self.dense_conv_block_3(x) #36x36x128
        x = self.downsample_3(x_cat_36x36) #18x18x128

        x_cat_18x18 = self.dense_conv_block_4(x) #18x18x256
        x = self.downsample_4(x_cat_18x18) #9x9x256

        #Bottleneck layer
        x = self.dense_conv_block_5(x) #9x9x512

        #Repeatedly upscale
        x = self.upscale_1(x) #18x18x512
        x = self.reduce_1a(x) #18x18x256

        #Important note: concatenations happen in dimension 1
        #because GeometricTensors have shape (inv_rep, channels, x, y)
        if self.inv_group_type != InvariantType.NONE:
            x = nn.tensor_directsum([x, x_cat_18x18]) #18x18x512
        else:
            x = torch.cat((x, x_cat_18x18), dim=1)
        x = self.reduce_1(x) #18x18x128
        x = self.dense_conv_block_6(x) #18x18x256

        x = self.upscale_2(x) #36x36x256
        x = self.reduce_2a(x) #36x36x128
        if self.inv_group_type != InvariantType.NONE:
            x = nn.tensor_directsum([x, x_cat_36x36])
        else:
            x = torch.cat((x, x_cat_36x36), dim=1)
        x = self.reduce_2(x) #36x36x64
        x = self.dense_conv_block_7(x) #36x36x128

        x = self.upscale_3(x) #72x72x128
        x = self.reduce_3a(x) #72x72x64
        if self.inv_group_type != InvariantType.NONE:
            x = nn.tensor_directsum([x, x_cat_72x72])
        else:
            x = torch.cat((x, x_cat_72x72), dim=1)
        x = self.reduce_3(x) #72x72x32
        x = self.dense_conv_block_8(x) #72x72x64

        x = self.upscale_4(x) #144x144x64
        x = self.reduce_4(x) #144x144x16
        x = self.dense_conv_block_9(x) #144x144x32

        #Final C8-invariant layer (goes to torch.tensor)
        x = self.group_reduce_block(x) #144x144x1
        x = self.group_pool_block(x) #144x144x1 torch.tensor

        #Final torch layer
        if self.inv_group_type != InvariantType.NONE:
            x = x.tensor #Unwrap GeometricTensor to get PyTorch tensor
        x = self.plain_torch_block(x) #144x144x1 torch.tensor

        return x

class C8InvariantCNN(BaseC8InvariantCNN, torch.nn.Module):
    """
    This class implements a C8-invariant CNN that takes in 144x144xN tensor inputs and
    returns a 144x144x1 cloud mask.

    Parameters 
    ------------
        kernel_size: int 
            side length of convolutions in DenseBlocks (default=3)
        input_channels: int 
            number of input channels (default=4)
    """
    def __init__(self, kernel_size=3, input_channels=4, has_bias=True, f_1=64, inv_group_type=InvariantType.C8, n_classes=2):
        super().__init__()

        self.inv_group_type = inv_group_type

        if self.inv_group_type != InvariantType.NONE:
            f_1 = f_1//8

        multiclass = n_classes > 2

        n_out_classes = n_classes if multiclass else 1

        #Model is equivariant under rotations by 45 degrees (C8)
        self.f_1, self.f_2, self.f_3, self.f_4, self.f_5 = f_1, f_1*2, f_1*4, f_1*8, f_1*16
        self.kernel_size = kernel_size 

        #First convolution (144x144x4 -> 144x144x16)
        self.input_conv_block = ReduceDimsBlockBuilder.build_block(self.inv_group_type, input_channels, self.f_1, input_regular=False)

        #Second convolution (dense) + downscale (144x144x16 -> 72x72x32)
        self.conv_block_1 = TwoStepConvBlockBuilder.build_block(self.inv_group_type, self.f_1, self.f_1, kernel_size=self.kernel_size)
        self.downsample_1 = DownscaleBlockBuilder.build_block(self.inv_group_type, self.f_1)

        #Third convolution (dense) + downscale (72x72x32 -> 36x36x64)
        self.conv_block_2 = TwoStepConvBlockBuilder.build_block(self.inv_group_type, self.f_1, self.f_2, kernel_size=self.kernel_size)
        self.downsample_2 = DownscaleBlockBuilder.build_block(self.inv_group_type, self.f_2)

        #Fourth convolution (dense) + downscale (36x36x64 -> 18x18x128)
        self.conv_block_3 = TwoStepConvBlockBuilder.build_block(self.inv_group_type, self.f_2, self.f_3, kernel_size=self.kernel_size)
        self.downsample_3 = DownscaleBlockBuilder.build_block(self.inv_group_type, self.f_3)

        #Fifth convolution (dense) + downscale (18x18x128 -> 9x9x256)
        self.conv_block_4 = TwoStepConvBlockBuilder.build_block(self.inv_group_type, self.f_3, self.f_4, kernel_size=self.kernel_size)
        self.downsample_4 = DownscaleBlockBuilder.build_block(self.inv_group_type, self.f_4)

        #Sixth convolution (dense bottleneck) (9x9x256 -> 9x9x512)
        self.conv_block_5 = TwoStepConvBlockBuilder.build_block(self.inv_group_type, self.f_4, self.f_5, kernel_size=self.kernel_size)

        #Now time to upscale! (9x9x512 -> 18x18x256)
        self.upscale_1 = UpscaleBlockBuilder.build_block(self.inv_group_type, self.f_5)

        #Reduce channels with 1x1 conv before concatenation (18x18x512 -> 18x18x256)
        self.reduce_1a = ReduceDimsBlockBuilder.build_block(self.inv_group_type, self.f_5, self.f_4)

        #1x1 conv block on concatenated inputs (18x18x512 -> 18x18x128)
        self.reduce_1 = ReduceDimsBlockBuilder.build_block(self.inv_group_type, self.f_5, self.f_4//2)

        #Dense block (18x18x128 -> 18x18x256)
        self.conv_block_6 = TwoStepConvBlockBuilder.build_block(self.inv_group_type, self.f_4//2, self.f_4, kernel_size=self.kernel_size)

        #Repeat upscale, concat, conv, dense (18x18x256 -> 36x36x128)
        self.upscale_2 = UpscaleBlockBuilder.build_block(self.inv_group_type, self.f_4)
        self.reduce_2a = ReduceDimsBlockBuilder.build_block(self.inv_group_type, self.f_4, self.f_3)
        self.reduce_2 = ReduceDimsBlockBuilder.build_block(self.inv_group_type, self.f_4, self.f_3//2)
        self.conv_block_7 = TwoStepConvBlockBuilder.build_block(self.inv_group_type, self.f_3//2, self.f_3, kernel_size=self.kernel_size)

        #Repeat upscale, concat, conv, dense (36x36x128 -> 72x72x64)
        self.upscale_3 = UpscaleBlockBuilder.build_block(self.inv_group_type, self.f_3)
        self.reduce_3a = ReduceDimsBlockBuilder.build_block(self.inv_group_type, self.f_3, self.f_2)
        self.reduce_3 = ReduceDimsBlockBuilder.build_block(self.inv_group_type, self.f_3, self.f_2//2)
        self.conv_block_8 = TwoStepConvBlockBuilder.build_block(self.inv_group_type, self.f_2//2, self.f_2, kernel_size=self.kernel_size)

        #Repeat upscale, concat, conv, dense (72x72x64 -> 144x144x32)
        self.upscale_4 = UpscaleBlockBuilder.build_block(self.inv_group_type, self.f_2)
        self.reduce_4 = ReduceDimsBlockBuilder.build_block(self.inv_group_type, self.f_2, self.f_1//2)
        self.conv_block_9 = TwoStepConvBlockBuilder.build_block(self.inv_group_type, self.f_1//2, self.f_1, kernel_size=self.kernel_size)

        #Final output (1x1 conv to go to 144x144x1, then group pooling)
        self.group_reduce_block = ReduceDimsBlockBuilder.build_block(self.inv_group_type, self.f_1, n_out_classes)
        if self.inv_group_type == InvariantType.NONE:
            self.group_pool_block = GroupPoolingBlockBuilder.build_block(self.inv_group_type, 1)
        else:
            self.group_pool_block = GroupPoolingBlockBuilder.build_block(self.inv_group_type, n_out_classes)

        #Normal torch block (not group invariant) for final 1x1 conv/BN/Relu
        self.plain_torch_block = ActivationBlockBuilder.build_block(None, relu_slope=0.1, n_classes=n_out_classes)

        #Dictionary mapping sub-blocks to types (for loading/saving from dictionaries)
        self.named_blocks = {}
        self.named_blocks['input_conv_block'] = ReduceDimsBlockBuilder 
        self.named_blocks['conv_block_1'] = TwoStepConvBlockBuilder
        self.named_blocks['conv_block_2'] = TwoStepConvBlockBuilder
        self.named_blocks['conv_block_3'] = TwoStepConvBlockBuilder
        self.named_blocks['conv_block_4'] = TwoStepConvBlockBuilder
        self.named_blocks['conv_block_5'] = TwoStepConvBlockBuilder
        self.named_blocks['conv_block_6'] = TwoStepConvBlockBuilder
        self.named_blocks['conv_block_7'] = TwoStepConvBlockBuilder
        self.named_blocks['conv_block_8'] = TwoStepConvBlockBuilder
        self.named_blocks['conv_block_9'] = TwoStepConvBlockBuilder
        self.named_blocks['downsample_1'] = DownscaleBlockBuilder
        self.named_blocks['downsample_2'] = DownscaleBlockBuilder
        self.named_blocks['downsample_3'] = DownscaleBlockBuilder
        self.named_blocks['downsample_4'] = DownscaleBlockBuilder
        self.named_blocks['upscale_1'] = UpscaleBlockBuilder
        self.named_blocks['upscale_2'] = UpscaleBlockBuilder
        self.named_blocks['upscale_3'] = UpscaleBlockBuilder
        self.named_blocks['upscale_4'] = UpscaleBlockBuilder 
        self.named_blocks['reduce_1a'] = ReduceDimsBlockBuilder 
        self.named_blocks['reduce_1'] = ReduceDimsBlockBuilder
        self.named_blocks['reduce_2a'] = ReduceDimsBlockBuilder
        self.named_blocks['reduce_2'] = ReduceDimsBlockBuilder
        self.named_blocks['reduce_3a'] = ReduceDimsBlockBuilder
        self.named_blocks['reduce_3'] = ReduceDimsBlockBuilder
        self.named_blocks['reduce_4'] = ReduceDimsBlockBuilder
        self.named_blocks['group_reduce_block'] = ReduceDimsBlockBuilder
        self.named_blocks['group_pool_block'] = GroupPoolingBlockBuilder
        self.named_blocks['plain_torch_block'] = ActivationBlockBuilder

        #Save to dict + then reload to ensure that all fields are initialized
        self.reload_from_dict(self.get_save_dict())

    def forward(self, input_tensor):
        """
        This function runs the C8 invariant CNN forwards.

        Arguments
        ----------
            input_tensor: torch.Tensor
                RGB+IR images to put in the model

        Returns
        --------
            output_tensor: torch.tensor
                cloud map
        """
        #Transform input into C8-invariant GeometricTensor if needed
        if self.inv_group_type != InvariantType.NONE:
            x = nn.GeometricTensor(input_tensor, self.input_conv_block.in_type)
        else:
            x = input_tensor

        #Ingest layer
        x = self.input_conv_block(x) #144x144x16

        #Repeatedly downscale
        x = self.conv_block_1(x) #144x144x32
        x = self.downsample_1(x) #72x72x32

        x_cat_72x72 = self.conv_block_2(x) #72x72x64
        x = self.downsample_2(x_cat_72x72) #36x36x64

        x_cat_36x36 = self.conv_block_3(x) #36x36x128
        x = self.downsample_3(x_cat_36x36) #18x18x128

        x_cat_18x18 = self.conv_block_4(x) #18x18x256
        x = self.downsample_4(x_cat_18x18) #9x9x256

        #Bottleneck layer
        x = self.conv_block_5(x) #9x9x512

        #Repeatedly upscale
        x = self.upscale_1(x) #18x18x512
        x = self.reduce_1a(x) #18x18x256

        #Important note: concatenations happen in dimension 1
        #because GeometricTensors have shape (inv_rep, channels, x, y)
        if self.inv_group_type != InvariantType.NONE:
            x = nn.tensor_directsum([x, x_cat_18x18]) #18x18x512
        else:
            x = torch.cat([x, x_cat_18x18], dim=1)
        x = self.reduce_1(x) #18x18x128
        x = self.conv_block_6(x) #18x18x256

        x = self.upscale_2(x) #36x36x256
        x = self.reduce_2a(x) #36x36x128
        if self.inv_group_type != InvariantType.NONE:
            x = nn.tensor_directsum([x, x_cat_36x36])
        else:
            x = torch.cat([x, x_cat_36x36], dim=1)
        x = self.reduce_2(x) #36x36x64
        x = self.conv_block_7(x) #36x36x128

        x = self.upscale_3(x) #72x72x128
        x = self.reduce_3a(x) #72x72x64
        if self.inv_group_type != InvariantType.NONE:
            x = nn.tensor_directsum([x, x_cat_72x72])
        else:
            x = torch.cat([x, x_cat_72x72], dim=1)
        x = self.reduce_3(x) #72x72x32
        x = self.conv_block_8(x) #72x72x64

        x = self.upscale_4(x) #144x144x64
        x = self.reduce_4(x) #144x144x16
        x = self.conv_block_9(x) #144x144x32
        #Final C8-invariant layer (goes to torch.tensor)
        x = self.group_reduce_block(x) #144x144x1
        x = self.group_pool_block(x) #144x144x1 torch.tensor

        #Final torch layer
        if self.inv_group_type != InvariantType.NONE:
            x = x.tensor #Unwrap GeometricTensor to get PyTorch tensor
        x = self.plain_torch_block(x) #144x144x1 torch.tensor

        return x

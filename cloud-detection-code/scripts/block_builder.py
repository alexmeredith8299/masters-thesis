"""
This module implements different block builders, which generate appropriate 
blocks for use in a model.
"""
from abc import ABC, abstractmethod
from enum import Enum, unique
from e2cnn import gspaces, nn
from scripts.basic_blocks import NonReduceChannelDimsBlock, NonDenseBlock, NonUpscaleBlock
from scripts.basic_blocks import NonDownscaleBlock, NonGroupPoolingBlock, NonTwoStepConvBlock
from scripts.basic_blocks import ActivationBlock, ConvUNeXtConv
from scripts.equivariant_basic_blocks import ReduceChannelDimsBlock, DenseBlock, UpscaleBlock
from scripts.equivariant_basic_blocks import DownscaleBlock, TwoStepConvBlock, EquivariantConvUNeXtConv 
from scripts.escnn_extension import InteroperableGroupPooling

@unique 
class InvariantType(Enum):
    """
    This enum exists to track allowed invariant groups for the BlockBuilder 
    classes and different flavors of CNN.
    """
    NONE = 0
    C8 = 1

class BlockBuilder(ABC):
    @staticmethod
    @abstractmethod
    def build_block(group, **kwargs):
        """
        Build block for use in the model given an invariant group and arguments 
        for the generic block.
        """

    @staticmethod
    def build_block_from_dict(group, properties, non_inv_type, inv_type):
        """
        Build block from properties dict. Default implementation is to just 
        load from dictionary for an invariant type if the group is invariant, 
        otherwise load from dictionary for a non-invariant type.

        Arguments 
        ---------
            group : InvariantType
                Invariant group for the block.
            properties : dict
                Dictionary of properties for the block.
            non_inv_type : type
                Type of block to load if group is not invariant.
            inv_type : type
                Type of block to load if group is invariant.

        Returns
        -------
            block : torch.nn.Module
                Block to use in the model.
        """
        if group == InvariantType.NONE:
            return non_inv_type.load_from_dict(properties)
        else:
            return inv_type.load_from_dict(properties)

class ReduceDimsBlockBuilder(BlockBuilder):
    """
    Build a dimension reduction block with the proper invariant group.
    """
    @staticmethod
    def build_block(group, in_channels, out_channels, inplace=True, input_regular=True):
        if group == InvariantType.NONE:
            return NonReduceChannelDimsBlock(in_channels, out_channels, inplace=inplace)
        elif group == InvariantType.C8:
            r2_act = gspaces.Rot2dOnR2(N=8)
            if input_regular:
                input_type = nn.FieldType(r2_act, in_channels*[r2_act.regular_repr])
            else:
                input_type = nn.FieldType(r2_act, in_channels*[r2_act.trivial_repr])
            output_type = nn.FieldType(r2_act, out_channels*[r2_act.regular_repr])
            return ReduceChannelDimsBlock(input_type, output_type, inplace=inplace)
        else:
            raise ValueError("Invalid invariant group.")

    @staticmethod
    def build_block_from_dict(group, properties):
        return BlockBuilder.build_block_from_dict(group, properties, NonReduceChannelDimsBlock, ReduceChannelDimsBlock)

class DenseBlockBuilder(BlockBuilder):
    """
    Build a dense block with the proper invariant group.
    """
    @staticmethod 
    def build_block(group, in_channels, out_channels, n_sub_blocks=4, kernel_size=3):
        if group == InvariantType.NONE:
            return NonDenseBlock(in_channels, out_channels, n_sub_blocks=n_sub_blocks, kernel_size=kernel_size)
        elif group == InvariantType.C8:
            r2_act = gspaces.Rot2dOnR2(N=8)
            input_type = nn.FieldType(r2_act, in_channels*[r2_act.regular_repr])
            output_type = nn.FieldType(r2_act, out_channels*[r2_act.regular_repr])
            return DenseBlock(input_type, output_type, n_sub_blocks=n_sub_blocks, kernel_size=kernel_size)
        else:
            raise ValueError("Invalid invariant group.")

    @staticmethod
    def build_block_from_dict(group, properties):
        return BlockBuilder.build_block_from_dict(group, properties, NonDenseBlock, DenseBlock)

class UpscaleBlockBuilder(BlockBuilder):
    """
    Build an upscaling block with the proper invariant group.
    """
    @staticmethod
    def build_block(group, in_channels, scale_factor=2):
        out_channels = in_channels
        if group == InvariantType.NONE:
            return NonUpscaleBlock(in_channels, scale_factor=scale_factor)
        elif group == InvariantType.C8:
            r2_act = gspaces.Rot2dOnR2(N=8)
            input_type = nn.FieldType(r2_act, in_channels*[r2_act.regular_repr])
            output_type = nn.FieldType(r2_act, out_channels*[r2_act.regular_repr])
            return UpscaleBlock(input_type, output_type, scale_factor=scale_factor)
        else:
            raise ValueError("Invalid invariant group.")

    @staticmethod
    def build_block_from_dict(group, properties):
        return BlockBuilder.build_block_from_dict(group, properties, NonUpscaleBlock, UpscaleBlock)

class DownscaleBlockBuilder(BlockBuilder):
    """
    Build a downscaling block with the proper invariant group.
    """
    @staticmethod
    def build_block(group, in_channels, scale_factor=2):
        if group == InvariantType.NONE:
            return NonDownscaleBlock(in_channels, scale_factor=scale_factor)
        elif group == InvariantType.C8:
            r2_act = gspaces.Rot2dOnR2(N=8)
            input_type = nn.FieldType(r2_act, in_channels*[r2_act.regular_repr])
            output_type = nn.FieldType(r2_act, in_channels*[r2_act.regular_repr])
            return DownscaleBlock(input_type, output_type, kernel_size=scale_factor)
        else:
            raise ValueError("Invalid invariant group.")

    @staticmethod
    def build_block_from_dict(group, properties):
        return BlockBuilder.build_block_from_dict(group, properties, NonDownscaleBlock, DownscaleBlock)

class GroupPoolingBlockBuilder(BlockBuilder):
    """
    Build a group pooling block with the proper invariant group. 
    Obviously if you're not using an invariant group, this is just 
    going to pool channel-wise rather than over orientations.
    """
    @staticmethod
    def build_block(group, in_channels):
        if group == InvariantType.NONE:
            return NonGroupPoolingBlock(in_channels)
        elif group == InvariantType.C8:
            r2_act = gspaces.Rot2dOnR2(N=8)
            input_type = nn.FieldType(r2_act, in_channels*[r2_act.regular_repr])
            return InteroperableGroupPooling(input_type)
        else:
            raise ValueError("Invalid invariant group.")

    @staticmethod
    def build_block_from_dict(group, properties):
        return BlockBuilder.build_block_from_dict(group, properties, NonGroupPoolingBlock, InteroperableGroupPooling)

class TwoStepConvBlockBuilder(BlockBuilder):
    """
    Build a TwoStepConvBlock with the proper invariant group.
    """
    @staticmethod
    def build_block(group, in_channels, out_channels, kernel_size=3, bias=True):
        if group == InvariantType.NONE:
            return NonTwoStepConvBlock(in_channels, out_channels, kernel_size=kernel_size, bias=bias)
        elif group == InvariantType.C8:
            r2_act = gspaces.Rot2dOnR2(N=8)
            input_type = nn.FieldType(r2_act, in_channels*[r2_act.regular_repr])
            output_type = nn.FieldType(r2_act, out_channels*[r2_act.regular_repr])
            return TwoStepConvBlock(input_type, output_type, kernel_size=kernel_size, has_bias=bias)
        else:
            raise ValueError("Invalid invariant group.")

    @staticmethod
    def build_block_from_dict(group, properties):
        return BlockBuilder.build_block_from_dict(group, properties, NonTwoStepConvBlock, TwoStepConvBlock)

class ConvUNeXtConvBlockBuilder(BlockBuilder):
    """
    Build a ConvUNeXtConv with the proper invariant group.
    """
    @staticmethod
    def build_block(group, in_channels):
        if group == InvariantType.NONE:
            return ConvUNeXtConv(in_channels)
        elif group == InvariantType.C8:
            r2_act = gspaces.Rot2dOnR2(N=8)
            input_type = nn.FieldType(r2_act, in_channels*[r2_act.regular_repr])
            output_type = input_type
            return EquivariantConvUNeXtConv(input_type, output_type)
        else:
            raise ValueError("Invalid invariant group.")

    @staticmethod
    def build_block_from_dict(group, properties):
        return BlockBuilder.build_block_from_dict(group, properties, ConvUNeXtConv, EquivariantConvUNeXtConv)



class ActivationBlockBuilder(BlockBuilder):
    """
    Just a wrapper on ActivationBlock.
    """
    @staticmethod
    def build_block(group, relu_slope=0.1, n_classes=2):
        return ActivationBlock(relu_slope=relu_slope, n_classes=n_classes)

    @staticmethod
    def build_block_from_dict(group, properties):
        return ActivationBlock.load_from_dict(properties)


import os
import torch
from e2cnn import gspaces, nn
from scripts.escnn_extension import InteroperableGroupPooling
from scripts.equivariant_basic_blocks import EquivariantBasicConvBlock, ReduceChannelDimsBlock, UpscaleBlock
from scripts.equivariant_basic_blocks import DownscaleBlock, DenseBlock, TwoStepConvBlock
from scripts.basic_blocks import BasicConvBlock, NonReduceChannelDimsBlock, NonUpscaleBlock
from scripts.basic_blocks import NonDownscaleBlock, NonGroupPoolingBlock, NonDenseBlock
from scripts.basic_blocks import NonTwoStepConvBlock 
from scripts.block_builder import ReduceDimsBlockBuilder, DenseBlockBuilder
from scripts.block_builder import UpscaleBlockBuilder, DownscaleBlockBuilder
from scripts.block_builder import GroupPoolingBlockBuilder, TwoStepConvBlockBuilder
from scripts.block_builder import ConvUNeXtConvBlockBuilder, InvariantType

def test_basic_conv_block_export_and_save():
    """
    Instantiate and export basic conv block to ensure export works.
    Then save + load BasicConvBlock.
    """
    r2_act = gspaces.Rot2dOnR2(N=8)
    in_type = nn.FieldType(r2_act, 4*[r2_act.regular_repr])
    out_type = nn.FieldType(r2_act, 8*[r2_act.regular_repr])

    blk = EquivariantBasicConvBlock(in_type, out_type)
    blk_pt = blk.export()

    #Save model to dict + then reload
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(current_dir,'test_artifacts', 'test_save_basic_conv_block', 'test_save_basic_conv_block.pt')
    torch.save(blk_pt.state_dict(), save_path)

    #Load new block 
    blk_loaded = BasicConvBlock.load_from_dict(torch.load(save_path))

    a = (blk.state_dict()['.ConvNxN.filter'])
    b = (blk_loaded.state_dict()['ConvNxN.weight'])

    assert torch.sum(a-b).item() < 1e-9

def test_reduce_channels_dim_block_export_and_save():
    """
    Insantiate and export ReduceChannelDimsBlock. Then save + reload.
    """
    r2_act = gspaces.Rot2dOnR2(N=8)
    in_type = nn.FieldType(r2_act, 4*[r2_act.regular_repr])
    out_type = nn.FieldType(r2_act, 8*[r2_act.regular_repr])

    #blk = ReduceChannelDimsBlock(in_type, out_type)
    blk = ReduceDimsBlockBuilder.build_block(InvariantType.C8, 4, 8)
    blk_pt = blk.export()

    #Save model to dict + then reload
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(current_dir,'test_artifacts', 'test_convert_reduce_channels_block', 'test_save_reduce_channels_block.pt')
    torch.save(blk_pt.state_dict(), save_path)

    properties = torch.load(save_path)
    blk_loaded = ReduceDimsBlockBuilder.build_block_from_dict(InvariantType.NONE, properties)

    assert torch.sum(blk_loaded.state_dict()['2.weight'] - blk.block._modules['2'].filter).item() < 1e-9

def test_upscale_block_export_and_save():
    """
    Insantiate and export ReduceChannelDimsBlock. Then save + reload.
    """
    r2_act = gspaces.Rot2dOnR2(N=8)
    in_type = nn.FieldType(r2_act, 4*[r2_act.regular_repr])
    out_type = nn.FieldType(r2_act, 8*[r2_act.regular_repr])

    blk = UpscaleBlockBuilder.build_block(InvariantType.C8, 4, 8)
    blk_pt = blk.export()

    #Save model to dict + then reload
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(current_dir,'test_artifacts', 'test_convert_upscale_block', 'test_save_upscale_block.pt')
    torch.save(blk_pt.state_dict(), save_path)

    #Load new block
    state = torch.load(save_path)
    blk_loaded = UpscaleBlockBuilder.build_block_from_dict(InvariantType.NONE, state)

    assert blk_loaded._modules['0'].scale_factor == blk.block._modules['0']._scale_factor

def test_downscale_block_export_and_save():
    """
    Insantiate and export ReduceChannelDimsBlock. Then save + reload.
    """
    r2_act = gspaces.Rot2dOnR2(N=8)
    in_type = nn.FieldType(r2_act, 4*[r2_act.regular_repr])

    blk = DownscaleBlockBuilder.build_block(InvariantType.C8, 4)
    blk_pt = blk.export()

    #Save model to dict + then reload
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(current_dir,'test_artifacts', 'test_convert_downscale_block', 'test_save_downscale_block.pt')
    torch.save(blk_pt.state_dict(), save_path)

    #Load new block
    state = torch.load(save_path)
    blk_loaded = DownscaleBlockBuilder.build_block_from_dict(InvariantType.NONE, state)

    assert True

def test_group_pooling_block_export_and_save():
    """
    Instantiate and export InteroperableGroupPooling. Then save + reload.
    """
    r2_act = gspaces.Rot2dOnR2(N=8)
    in_type = nn.FieldType(r2_act, 4*[r2_act.regular_repr])

    blk = GroupPoolingBlockBuilder.build_block(InvariantType.C8, 4)
    blk_pt = blk.export()

    #Save model to dict + then reload
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(current_dir,'test_artifacts', 'test_convert_group_pooling_block', 'test_save_group_pooling_block.pt')
    torch.save(blk_pt.state_dict(), save_path)

    #Load new block
    state = torch.load(save_path)
    blk_loaded = GroupPoolingBlockBuilder.build_block_from_dict(InvariantType.NONE, state)

    assert True

def test_dense_block_export_and_save():
    """
    Instantiate and export DenseBlock.
    """
    r2_act = gspaces.Rot2dOnR2(N=8)
    in_type = nn.FieldType(r2_act, 8*[r2_act.regular_repr])
    out_type = nn.FieldType(r2_act, 16*[r2_act.regular_repr])

    #blk = DenseBlock(in_type, out_type, n_sub_blocks=4, kernel_size=5)
    blk = DenseBlockBuilder.build_block(InvariantType.C8, 8, 16, 4, 5)
    blk_pt = blk.export()
    blk_pt.eval()

    #Save model to dict + then reload
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(current_dir,'test_artifacts', 'test_convert_dense_block', 'test_save_dense_block.pt')

    torch.save(blk_pt.state_dict(), save_path)

    #Load new block 
    state = torch.load(save_path)
    blk_loaded = DenseBlockBuilder.build_block_from_dict(InvariantType.NONE, state)
    #blk_loaded.load_state_dict(state)

    a = (blk_loaded._modules['1'].state_dict()['ConvNxN.weight'])
    b = blk.block._modules['1'].state_dict()['.ConvNxN.filter']

    assert torch.sum(a-b).item() < 1e-9

def test_two_step_conv_block_export_and_save():
    """
    Insantiate and export ReduceChannelDimsBlock. Then save + reload.
    """
    r2_act = gspaces.Rot2dOnR2(N=8)
    in_type = nn.FieldType(r2_act, 4*[r2_act.regular_repr])
    out_type = nn.FieldType(r2_act, 8*[r2_act.regular_repr])

    blk = TwoStepConvBlockBuilder.build_block(InvariantType.C8, 4, 8, kernel_size=3)
    blk_pt = blk.export()

    #Save model to dict + then reload
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(current_dir,'test_artifacts', 'test_convert_two_step_conv_block', 'test_save_two_step_conv_block.pt')
    torch.save(blk_pt.state_dict(), save_path)

    #Load new block
    state = torch.load(save_path)
    blk_loaded = TwoStepConvBlockBuilder.build_block_from_dict(InvariantType.NONE, state)

    assert torch.sum(blk_loaded.state_dict()['2.weight'] - blk.block._modules['2'].filter).item() < 1e-9

def test_conv_u_next_conv_block_export_and_save():
    """
    Insantiate and export ReduceChannelDimsBlock. Then save + reload.
    """
    r2_act = gspaces.Rot2dOnR2(N=8)
    in_type = nn.FieldType(r2_act, 4*[r2_act.regular_repr])
    out_type = nn.FieldType(r2_act, 4*[r2_act.regular_repr])

    blk = ConvUNeXtConvBlockBuilder.build_block(InvariantType.C8, 4)
    blk_pt = blk.export()

    #Save model to dict + then reload
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(current_dir,'test_artifacts', 'test_convert_conv_u_next_conv_block', 'test_save_conv_u_next_conv_block.pt')
    torch.save(blk_pt.state_dict(), save_path)

    #Load new block
    state = torch.load(save_path)
    blk_loaded = ConvUNeXtConvBlockBuilder.build_block_from_dict(InvariantType.NONE, state)

    assert torch.sum(blk_loaded.state_dict()['0.weight'] - blk.block._modules['0'].filter).item() < 1e-9



import os
from scripts.cloud_dataset import CloudDataset

def test_len():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(current_dir,'..', '..', 'scitech-dataset')

    training_data = CloudDataset(train_path, "train")
    assert len(training_data) == 672

def test_get_k_folds():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(current_dir,'..', '..', 'scitech-dataset')

    training_data = CloudDataset(train_path, "train")
    k_folds = training_data.get_k_folds(7)
    assert len(k_folds) == 7
    for fold in k_folds:
        assert len(fold)==96

def test_resize():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(current_dir,'..', '..', 'sparcs-dataset')

    training_data = CloudDataset(train_path, "train", use_swir=True)
    img = training_data[0]
    rgb_img = img['img'][0:3] 
    lwir_img = img['img'][3]
    swir_img = img['img'][4]
    ref = img['ref']
    rgb_resize = training_data.resize(rgb_img)#, ir_resize, ref_resize = training_data.resize(rgb_img, ir_img, ref)
    assert rgb_resize.shape[1]==144
    assert rgb_resize.shape[2]==144

def test_sparcs_multiclass():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(current_dir,'..', '..', 'sparcs-multiclass-dataset')

    training_data = CloudDataset(train_path, "train", use_swir=True, n_classes=3)
    img = training_data[0]
    rgb_img = img['img'][0:3] 
    lwir_img = img['img'][3]
    swir_img = img['img'][4]
    ref = img['ref']

    assert ref.shape[0]==1
    assert ref.shape[1]==144
    assert ref.shape[2]==144
    assert ref.shape[3]==3


def test_rgb_only():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(current_dir,'..', '..', 'scitech-dataset')

    training_data = CloudDataset(train_path, "train", use_lwir=False)
    img = training_data[0]['img']
    assert img.shape[0] == 3
    assert img.shape[1] == 144
    assert img.shape[2] == 144


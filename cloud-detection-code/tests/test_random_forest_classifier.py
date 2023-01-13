import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from scripts.random_forest_classifier import RandomForest 
from scripts.cloud_dataset import CloudDataset
from scripts.road_dataset import RoadDataset 
from torch.utils.data import DataLoader

def test_random_forest_classifier_init():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(current_dir,'..', '..', 'scitech-dataset')

    train_set = CloudDataset(train_path, "train")
    train_loader = DataLoader(train_set, batch_size=10, shuffle=False)

    lum = RandomForest(train_loader)
    assert True

def test_random_forest_classifier_extract_pixel_features():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(current_dir,'..', '..', 'scitech-dataset')

    train_set = CloudDataset(train_path, "train", use_swir=True)
    train_loader = DataLoader(train_set, batch_size=len(train_set), shuffle=False)

    rf = RandomForest(train_loader)

    for img in train_loader:
        dims, cloud_lum, noncloud_lum = rf.extract_pixel_features(img)
    assert True 

def test_random_forest_classifier_extract_features():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(current_dir,'..', '..', 'scitech-dataset')

    train_set = CloudDataset(train_path, "train", use_swir=True)
    train_loader = DataLoader(train_set, batch_size=len(train_set), shuffle=False)

    rf = RandomForest(train_loader)
    rf.extract_features()
    assert rf.training_mask_features.shape[0] == 13934592

def test_random_forest_classifier_train():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(current_dir,'..', '..', 'scitech-dataset')

    train_set = CloudDataset(train_path, "train-tiny", use_swir=True)
    train_loader = DataLoader(train_set, batch_size=10, shuffle=False)

    rf = RandomForest(train_loader)
    rf.extract_features()
    rf.train()

def test_random_forest_classifier_classify():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(current_dir,'..', '..', 'scitech-dataset')

    train_set = CloudDataset(train_path, "val-tiny", use_swir=True, randomly_flip=False)
    train_loader = DataLoader(train_set, batch_size=10, shuffle=False)
    val_set = CloudDataset(train_path, "validate", use_swir=True, randomly_flip=False)
    val_loader = DataLoader(val_set, batch_size=len(val_set), shuffle=False)


    rf = RandomForest(train_loader)
    rf.extract_features()
    rf.train()
    for img in val_loader:
        cloud_mask = rf.classify(img, plot_on=False)

def test_random_forest_classifier_save_and_load_classifier():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(current_dir,'..', '..', 'scitech-dataset')

    train_set = CloudDataset(train_path, "val-tiny", use_swir=True, randomly_flip=False)
    train_loader = DataLoader(train_set, batch_size=10, shuffle=False)
    val_set = CloudDataset(train_path, "validate", use_swir=True, randomly_flip=False)
    val_loader = DataLoader(val_set, batch_size=len(val_set), shuffle=False)

    rf = RandomForest(train_loader)
    rf.extract_features()
    rf.train()
    original_mask = None
    for img in val_loader:
        original_mask = rf.classify(img, plot_on=False)

    clf_fname = os.path.join(current_dir, 'test_artifacts', 'test_save_and_load_rf', 'rf.pkl')
    rf.save_classifier(clf_fname)
    rf.rf = None
    rf.load_classifier(clf_fname)
    val_loader = DataLoader(val_set, batch_size=len(val_set), shuffle=False)
    new_mask = None
    for img in val_loader:
        new_mask = rf.classify(img, plot_on=False)

    assert torch.abs(new_mask-original_mask).max() < 1e-9

def test_random_forest_classifier_extract_pixel_features_shape():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(current_dir,'..', '..', 'scitech-dataset')

    train_set = CloudDataset(train_path, "val-tiny", use_swir=True, randomly_flip=False, randomly_rotate=False)
    train_loader = DataLoader(train_set, batch_size=10, shuffle=False)

    rf = RandomForest(train_loader)
    original_img = None
    original_mask = None
    dims, X, y = None, None, None
    for img in train_loader:
        original_img = img['img']
        original_mask = img['ref']
        dims, X, y = rf.extract_pixel_features(img)
    """
    plt.imshow(original_img[0, 0, :, :])
    plt.show()
    x_reshape = X.reshape((144, 144, 45))
    plt.imshow(x_reshape[:, :, 5].T)
    plt.show()
    plt.imshow(original_mask[0, 0, :, :])
    plt.show()
    plt.imshow(y.reshape(144, 144, 1)[:, :, 0].T)
    plt.show()
    """


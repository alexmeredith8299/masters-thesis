import os
import numpy as np
from scripts.luminosity_classifier import LuminosityClassifier 
from scripts.cloud_dataset import CloudDataset
from scripts.road_dataset import RoadDataset 
from torch.utils.data import DataLoader

def test_luminosity_classifier_init():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(current_dir,'..', '..', 'scitech-dataset')

    train_set = CloudDataset(train_path, "train")
    train_loader = DataLoader(train_set, batch_size=10, shuffle=False)

    lum = LuminosityClassifier(train_loader)
    assert True

def test_luminosity_classifier_extract_pixel_lum():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(current_dir,'..', '..', 'scitech-dataset')

    train_set = CloudDataset(train_path, "train", use_swir=True)
    train_loader = DataLoader(train_set, batch_size=10, shuffle=False)

    lum = LuminosityClassifier(train_loader)

    for img in train_loader:
        cloud_lum, noncloud_lum = lum.extract_pixel_luminosities(img)
    assert True 

def test_luminosity_classifier_unpack_pixels():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(current_dir,'..', '..', 'scitech-dataset')

    train_set = CloudDataset(train_path, "train", use_swir=True)
    train_loader = DataLoader(train_set, batch_size=10, shuffle=False)

    lum = LuminosityClassifier(train_loader)
    lum.unpack_pixels(train_loader)

    assert len(lum.cloud_luminosities[0]) == 4204662 

def test_luminosity_classifier_train():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(current_dir,'..', '..', 'scitech-dataset')

    train_set = CloudDataset(train_path, "train", use_swir=True)
    train_loader = DataLoader(train_set, batch_size=10, shuffle=False)

    lum = LuminosityClassifier(train_loader)
    lum.train()

    assert (lum.thresholds[0]-0.5176) < 1e-3
    assert (lum.thresholds[1]-0.470588) < 1e-3
    assert (lum.thresholds[2]-0.4235294) < 1e-3
    assert (lum.thresholds[3]-0.63137257) < 1e-3
    assert (lum.thresholds[4]-0.823529422) < 1e-3

def test_luminosity_classifier_classify():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(current_dir,'..', '..', 'scitech-dataset')

    train_set = CloudDataset(train_path, "val-tiny", use_swir=True)
    train_loader = DataLoader(train_set, batch_size=10, shuffle=False)

    lum = LuminosityClassifier(train_loader)
    lum.train()
    for img in train_loader:
        cloud_mask = lum.classify(img['img'])

    assert True

def test_luminosity_classifier_save_and_load():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(current_dir,'..', '..', 'scitech-dataset')

    train_set = CloudDataset(train_path, "train", use_swir=True)
    train_loader = DataLoader(train_set, batch_size=len(train_set), shuffle=False)

    lum = LuminosityClassifier(train_loader)
    lum.train()

    clf_fname = os.path.join(current_dir, 'test_artifacts', 'test_save_and_load_lum', 'lum.pkl')
    cloud_clf_fname = os.path.join(current_dir, 'test_artifacts', 'test_save_and_load_lum', 'lum_cloud.pkl')
    noncloud_clf_fname = os.path.join(current_dir, 'test_artifacts', 'test_save_and_load_lum', 'lum_noncloud.pkl')
    lum.save_thresholds(clf_fname, cloud_clf_fname, noncloud_clf_fname)
    lum.thresholds = None
    lum.load_thresholds(clf_fname)

    assert (lum.thresholds[0]-0.5176) < 1e-3
    assert (lum.thresholds[1]-0.470588) < 1e-3
    assert (lum.thresholds[2]-0.4235294) < 1e-3
    assert (lum.thresholds[3]-0.63137257) < 1e-3
    assert (lum.thresholds[4]-0.823529422) < 1e-3

#TODO move elsewhere.
def plot_pixel_luminosity_histogram_cloud():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(current_dir,'..', '..', 'scitech-dataset')

    train_set = CloudDataset(train_path, "train", use_swir=True)
    train_loader = DataLoader(train_set, batch_size=10, shuffle=False)

    lum = LuminosityClassifier(train_loader)
    lum.plot_pixel_luminosity_histogram(threshold=True)
    assert False

#TODO move elsewhere.
def plot_pixel_luminosity_histogram_road():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(current_dir,'..', '..', 'massachusetts-roads-dataset')

    train_set = RoadDataset(train_path, "train")
    train_loader = DataLoader(train_set, batch_size=10, shuffle=False)

    lum = LuminosityClassifier(train_loader)
    lum.plot_pixel_luminosity_histogram(class_label='road', non_class_label='non-road', bands=(0, 1, 2), band_labels=('(a) Red', '(b) Green', '(c) Blue'), threshold=True)
    assert False
 

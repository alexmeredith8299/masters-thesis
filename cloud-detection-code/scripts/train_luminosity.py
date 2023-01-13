#Dirty hack to import modules from relative path
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
#For MIT Engaging only
sys.path.append('/home/ameredit/masters-thesis/cloud-detection-code')
sys.path.append('/home/ameredit/masters-thesis/cloud-detection-code/scripts')
from scripts.luminosity_classifier import LuminosityClassifier
from scripts.cloud_dataset import CloudDataset
from scripts.road_dataset import RoadDataset 
from torch.utils.data import DataLoader

def train_and_save_luminosity_model(train_set, val_set, save_fname, save_target_fname, save_background_fname):
    train_loader = DataLoader(train_set, batch_size=len(train_set), shuffle=False)
    val_loader = DataLoader(val_set, batch_size=len(val_set), shuffle=False)

    lum = LuminosityClassifier(train_loader)
    lum.reverse_keys = ()
    if train_set.use_lwir:
        lum.reverse_keys = (3,)
    lum.train()
    original_mask = None
    for img in val_loader:
        original_mask = lum.classify(img['img'])


    clf_fname = os.path.join(current_dir, 'saved_models', 'luminosity', save_fname)
    target_fname = os.path.join(current_dir, 'saved_models', 'luminosity', save_target_fname)
    background_fname = os.path.join(current_dir, 'saved_models', 'luminosity', save_background_fname)
    lum.save_thresholds(clf_fname, target_fname, background_fname)



current_dir = os.path.dirname(os.path.abspath(__file__))
cloud_train_path = os.path.join(current_dir,'..', '..', 'sparcs-dataset')
road_train_path = os.path.join(current_dir,'..', '..', 'massachusetts-roads-dataset')

cloud_rgb_train_set = CloudDataset(cloud_train_path, "train", use_lwir=False, use_swir=False, randomly_flip=False)
cloud_rgb_val_set = CloudDataset(cloud_train_path, "validate", use_lwir=False, use_swir=False, randomly_flip=False)
train_and_save_luminosity_model(cloud_rgb_train_set, cloud_rgb_val_set, 'lum_cloud_rgb.pkl', 'lum_cloud_rgb_cloud_lum.pkl', 'lum_cloud_rgb_noncloud_lum.pkl')

cloud_swir_train_set = CloudDataset(cloud_train_path, "train", use_lwir=False, use_swir=True, randomly_flip=False)
cloud_swir_val_set = CloudDataset(cloud_train_path, "validate", use_lwir=False, use_swir=True, randomly_flip=False)
train_and_save_luminosity_model(cloud_swir_train_set, cloud_swir_val_set, 'lum_cloud_swir.pkl', 'lum_cloud_swir_cloud_lum.pkl', 'lum_cloud_swir_noncloud_lum.pkl')

cloud_lwir_train_set = CloudDataset(cloud_train_path, "train", use_lwir=True, use_swir=False, randomly_flip=False)
cloud_lwir_val_set = CloudDataset(cloud_train_path, "validate", use_lwir=True, use_swir=False, randomly_flip=False)
train_and_save_luminosity_model(cloud_lwir_train_set, cloud_lwir_val_set, 'lum_cloud_lwir.pkl', 'lum_cloud_lwir_cloud_lum.pkl', 'lum_cloud_lwir_noncloud_lum.pkl')

cloud_lwir_swir_train_set = CloudDataset(cloud_train_path, "train", use_lwir=True, use_swir=True, randomly_flip=False)
cloud_lwir_swir_val_set = CloudDataset(cloud_train_path, "validate", use_lwir=True, use_swir=True, randomly_flip=False)
train_and_save_luminosity_model(cloud_lwir_swir_train_set, cloud_lwir_swir_val_set, 'lum_cloud_lwir_swir.pkl', 'lum_cloud_lwir_swir_cloud_lum.pkl', 'lum_cloud_lwir_swir_noncloud_lum.pkl')

road_train_set = RoadDataset(road_train_path, "train", randomly_flip=False, randomly_crop=False, randomly_rotate=False)
road_val_set = RoadDataset(road_train_path, "validate", randomly_flip=False, randomly_crop=False, randomly_rotate=False)
#train_and_save_luminosity_model(road_train_set, road_val_set, 'lum_road.pkl', 'lum_road_road_lum.pkl', 'lum_road_nonroad_lum.pkl')


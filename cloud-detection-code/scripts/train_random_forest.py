#Dirty hack to import modules from relative path
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from scripts.random_forest_classifier import RandomForest
from scripts.cloud_dataset import CloudDataset
from scripts.road_dataset import RoadDataset 
from torch.utils.data import DataLoader
from torch.utils.data import random_split

def train_and_save_random_forest_model(train_set, val_set, save_fname):
    #lengths = []
    #for i in range(14):
    #    lengths.append(774)
    #for i in range(11):
    #    lengths.append(773)
    #list_sets = random_split(train_set, lengths) 

    train_loader = DataLoader(train_set, batch_size=len(train_set), shuffle=False)
    rf = RandomForest(train_loader)
    rf.extract_features()
    rf.train()

    clf_fname = os.path.join(current_dir, 'saved_models', 'random_forest', save_fname)
    rf.save_classifier(clf_fname)

    """
    for i in range(25):
        train_loader = DataLoader(list_sets[i], batch_size=len(list_sets[i]), shuffle=False)

        rf = RandomForest(train_loader)
        rf.extract_features()
        rf.train()

        clf_fname = os.path.join(current_dir, 'saved_models', 'random_forest', f"round_{i}"+save_fname)
        rf.save_classifier(clf_fname)
    """



current_dir = os.path.dirname(os.path.abspath(__file__))
cloud_train_path = os.path.join(current_dir,'..', '..', 'sparcs-dataset')
road_train_path = os.path.join(current_dir,'..', '..', 'massachusetts-roads-dataset')

cloud_rgb_train_set = CloudDataset(cloud_train_path, "train", use_lwir=False, use_swir=False, randomly_flip=False)
cloud_rgb_val_set = CloudDataset(cloud_train_path, "validate", use_lwir=False, use_swir=False, randomly_flip=False)
train_and_save_random_forest_model(cloud_rgb_train_set, cloud_rgb_val_set, 'rf_cloud_rgb_only.pkl')

cloud_swir_train_set = CloudDataset(cloud_train_path, "train", use_lwir=False, use_swir=True, randomly_flip=False)
cloud_swir_val_set = CloudDataset(cloud_train_path, "validate", use_lwir=False, use_swir=True, randomly_flip=False)
train_and_save_random_forest_model(cloud_swir_train_set, cloud_swir_val_set, 'rf_cloud_swir.pkl')

cloud_lwir_train_set = CloudDataset(cloud_train_path, "train", use_lwir=True, use_swir=False, randomly_flip=False)
cloud_lwir_val_set = CloudDataset(cloud_train_path, "validate", use_lwir=True, use_swir=False, randomly_flip=False)
train_and_save_random_forest_model(cloud_lwir_train_set, cloud_lwir_val_set, 'rf_cloud_lwir.pkl')

cloud_lwir_swir_train_set = CloudDataset(cloud_train_path, "train", use_lwir=True, use_swir=True, randomly_flip=False)
cloud_lwir_swir_val_set = CloudDataset(cloud_train_path, "validate", use_lwir=True, use_swir=True, randomly_flip=False)
train_and_save_random_forest_model(cloud_lwir_swir_train_set, cloud_lwir_swir_val_set, 'rf_cloud_lwir_swir.pkl')

road_train_set = RoadDataset(road_train_path, "train", randomly_flip=False, randomly_crop=False, randomly_rotate=False)
road_val_set = RoadDataset(road_train_path, "validate", randomly_flip=False, randomly_crop=False, randomly_rotate=False)
#train_and_save_random_forest_model(road_train_set, road_val_set, 'rf_road.pkl')


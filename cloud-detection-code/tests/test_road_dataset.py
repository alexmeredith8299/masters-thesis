import os
from scripts.road_dataset import RoadDataset

def test_len():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(current_dir,'..', '..', 'massachusetts-roads-dataset')

    training_data = RoadDataset(train_path, "train")
    assert len(training_data) == 19339 

def test_get_k_folds():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(current_dir,'..', '..', 'massachusetts-roads-dataset')

    training_data = RoadDataset(train_path, "train")
    k_folds = training_data.get_k_folds(10)
    assert len(k_folds) == 10 
    for fold in k_folds:
        assert len(fold)==1933


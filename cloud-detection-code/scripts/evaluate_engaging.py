#Dirty hack to import modules from relative path
import os
import sys
import itertools
sys.path.append('../../cloud-detection-code')
sys.path.append('../../cloud-detection-code/scripts')
import numpy as np
import torch
import torchvision
import pytest
from e2cnn import gspaces, nn
from scripts.c8_invariant_cnn import C8InvariantCNN, DenseC8InvariantCNN
from scripts.cloud_dataset import CloudDataset 
from scripts.road_dataset import RoadDataset
from scripts.train_pytorch_model import evaluate_rotational_equivariance, train_model 
from scripts.train_pytorch_model import save_model_at_checkpoint, load_model_from_checkpoint 
from scripts.evaluate_model import ClassifierValidator
from scripts.luminosity_classifier import LuminosityClassifier
from scripts.block_builder import InvariantType
from scripts.evaluate_utils import load_model_for_eval
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt



def save_classifier_validator(model, val_set, model_name, epoch, verbose=False):
    """
    This function saves the classifier validator for a model.

    Arguments
    ----------
        model: torch.nn.Module
            model to save the classifier validator for
        val_set: CloudDataset
            validation set
        model_name: string
            name of the model
        epoch: int
            epoch number of the model
        verbose: bool (optional)
            if True, print out the model name

    Returns
    --------
        success: bool 
            True if successful, False otherwise
        model_tester: ClassifierValidator
            classifier validator for the model
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dir_path = os.path.join(current_dir, 'saved_models', model_name)
    pickle_path = f"{directory}/model_classifier_validator_{epoch}.pkl"

    if verbose:
        print("Saving classifier validator to {}".format(pickle_path))

    val_loader = DataLoader(val_set, batch_size=1)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_tester = ClassifierValidator(model, val_loader, device, val_set=[])

    try:
        with open(pickle_path, "wb") as f:
            pickle.dump(model_tester, f)

        if verbose:
            print("Successfully saved classifier validator to {}".format(pickle_path))

        return True, model_tester #success
    except:
        if verbose:
            print("Failed to save classifier validator to {}".format(pickle_path))
        return False, model_tester #failure

def compare_model_tables(models, val_sets, buffer=4, buffer_save_name=None, tholds=[]):
    """
    This function prints the comparison tables of each model.

    Arguments 
    ---------
        models: list of [string, torch.nn.Module]
            name of each model and the model itself
        val_set: CloudDataset 
            validation set 

    Returns 
    --------
        model_tables: dictionary
            dictionary of comp tables for each model. Keys 
            are the model names.
    """
    model_tables = {}
    for i, named_model in enumerate(models):
        if tholds == []:
            thold = 0.5
        else:
            thold = tholds[i]
        buffer_name = buffer_save_name
        (name, model) = named_model

        if 'lum' not in name and 'rf' not in name:
            model.eval()
        val_loader = DataLoader(val_sets[i], batch_size=1)
        print("Dataloader created")
        if 'lum' in name or 'rf' in name:
            model_tester = ClassifierValidator(model, val_loader)
        else:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model_tester = ClassifierValidator(model, val_loader, device)
        print("Model tester created")
        os.makedirs(os.path.join(buffer_name, name), exist_ok=True)
        buffer_name = os.path.join(buffer_name, name)
        model_table, latex_table = model_tester.generate_comparison_table(cloud_threshold=thold, buffer=0)
        model_table_buffer, latex_table_buffer = model_tester.generate_comparison_table(cloud_threshold=thold, buffer=buffer)#25)
        fp_classes, latex_fp_classes = model_tester.generate_fp_class_table()
        fp_classes_buffer, latex_fp_classes_buffer = model_tester.generate_fp_class_table(buffer=buffer)

        model_tables[name] = (model_table, latex_table, model_table_buffer, latex_table_buffer, fp_classes, latex_fp_classes, fp_classes_buffer, latex_fp_classes_buffer)
 
    return model_tables



def compare_model_rocs(models, val_sets, buffer=4):
    """
    This function compares the roc curves of the models and plots 
    them against each other.

    Arguments 
    ----------
        models: list of [string, torch.nn.Module]
            name of each model and the model itself
        val_sat: CloudDataset
            validation set 

    Returns
    --------
        model_rocs: dictionary 
            dictionary of roc curves for each model. Keys 
            are the model names and values are [fpr, tpr]
    """
    model_rocs = {}
    model_rocs_buffer = {}
    for i,named_model in enumerate(models):
        (name, model) = named_model
        if 'lum' not in name and 'rf' not in name:
            #Evaluate model and get roc curve
            model.eval()
            val_loader = DataLoader(val_sets[i], batch_size=len(val_sets[i]))
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model_tester = ClassifierValidator(model, val_loader, device)
        else:
            val_loader = DataLoader(val_sets[i], batch_size=1)#len(val_sets[i]))
            model_tester = ClassifierValidator(model, val_loader)
        roc_fprates, roc_tprates, AUC, thold = model_tester.generate_roc_curve()
        roc_fprates_buffer, roc_tprates_buffer, AUC_buffer, thold_buffer = model_tester.generate_roc_curve(buffer=buffer)
        #Save roc curve
        model_rocs[name] = (roc_fprates, roc_tprates, AUC)
        with open(f"saved_tables/{name}_roc_{epoch}_epochs_buffer_0.csv", "w") as text_file:
            np.savetxt(text_file, np.array([roc_fprates, roc_tprates]).T, delimiter=',')
        with open(f"saved_tables/{name}_buffer_0_AUC.txt", "w") as text_file:
            text_file.write(f"AUC, {AUC}") 
        model_rocs_buffer[name] = (roc_fprates_buffer, roc_tprates_buffer, AUC_buffer)
        with open(f"saved_tables/{name}_roc_{epoch}_epochs_buffer_{buffer}.csv", "w") as text_file:
            np.savetxt(text_file, np.array([roc_fprates_buffer, roc_tprates_buffer]).T, delimiter=',')
        with open(f"saved_tables/{name}_buffer_{buffer}_AUC.txt", "w") as text_file:
            text_file.write(f"AUC, {AUC_buffer}") 
        with open(f"saved_tables/{name}_buffer_{buffer}_thold.txt", "w") as text_file:
            text_file.write(f"thold, {thold_buffer}") 

    return model_rocs, model_rocs_buffer

def get_model_dir_paths(exp_num, pattern="_", root_path='saved_models'):
    """
    This function gets the directory paths of the models to load 
    for evaluation.
    """
    #TODO handle parent paths properly (for train err, etc.)
    if exp_num == 'cloud_final' or exp_num == 'road_final' or exp_num == 'luminosity':
        exp_root = exp_num
    else:
        exp_root = f"exp_{exp_num}"
    exp_folder = os.path.join(root_path, exp_root)
    model_dir_paths = []
    model_target_classes = []
    model_bands = []
    for model_dir in os.listdir(exp_folder):
        if ('model_checkpoints' in model_dir or 'lum' in model_dir or 'rf' in model_dir) and pattern in model_dir:
            model_path = os.path.join(exp_root, model_dir)
            model_dir_paths.append((model_path, [model_path]))
            bands = ['r', 'g', 'b']
            if 'lwir' in model_dir or not any(band_type in model_dir for band_type in ['lwir', 'swir', 'rgb']): 
                bands.append('lwir')
            if 'swir' in model_dir: 
                bands.append('swir')
            if exp_num == 'road_final':
                bands = ['r', 'g', 'b']
            model_bands.append(bands)
            model_target = 'road' if ('road' in model_dir or exp_num =='road_final') else 'multiclass' if 'clouds_and_shadow' in model_dir else 'shadow' if 'shadow' in model_dir else 'cloud'
            model_target_classes.append(model_target)
    return model_dir_paths, model_target_classes, model_bands, exp_root

def get_model_val_sets(model_classes, model_bands):
    """
    Get validation set for each model given classes and bands.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    cloud_train_path = os.path.join(current_dir,'..', '..', 'sparcs-dataset')
    shadow_train_path = os.path.join(current_dir,'..', '..', 'sparcs-shadow-dataset')
    multiclass_train_path = os.path.join(current_dir,'..', '..', 'sparcs-multiclass-dataset')
    road_train_path = os.path.join(current_dir,'..', '..', 'massachusetts-roads-dataset')
    val_sets = []
    for i in range(len(model_classes)):
        if model_classes[i] == 'road':
            val_sets.append(RoadDataset(road_train_path, "test", randomly_flip=False, randomly_rotate=False))
        elif model_classes[i] == 'cloud':
            use_lwir = 'lwir' in model_bands[i]
            use_swir = 'swir' in model_bands[i]
            val_sets.append(CloudDataset(cloud_train_path, "test", randomly_flip=False, randomly_rotate=False, use_lwir=use_lwir, use_swir=use_swir))
        elif model_classes[i] == 'shadow':
            use_lwir = 'lwir' in model_bands[i]
            use_swir = 'swir' in model_bands[i]
            val_sets.append(CloudDataset(shadow_train_path, "test", randomly_flip=False, randomly_rotate=False, use_lwir=use_lwir, use_swir=use_swir))
        elif model_classes[i] == 'multiclass':
            use_lwir = 'lwir' in model_bands[i]
            use_swir = 'swir' in model_bands[i]
            val_sets.append(CloudDataset(multiclass_train_path, "test", randomly_flip=False, randomly_rotate=False, use_lwir=use_lwir, use_swir=use_swir, n_classes=3))
        else:
            raise ValueError("Model target class must be `cloud` or `road`.")
    return val_sets 

model_dir_paths = []
epoch = 90#995#500#995#900#700#1500#05#990#95#200#5#400#485#1995#500#755
input_channels = 5 
lr = 0.001 #Doesn't actually matter here.

exp_num = 'road_final'#95#62#131#62#131#62#131#62#'road_final'#131#'cloud_final'#134#'cloud_final'#'road_final'#110#'cloud_final'#107#98#'road_final'#'cloud_final'#104#'road_final'#101#'cloud_final' 
#model_dir_paths, model_classes, model_bands, exp_root = get_model_dir_paths(exp_num, pattern="clouds_and_shadow")
model_dir_paths, model_classes, model_bands, exp_root = get_model_dir_paths(exp_num, pattern="dice")
model_val_sets = get_model_val_sets(model_classes, model_bands)
#Validation set
current_dir = os.path.dirname(os.path.abspath(__file__))
#train_path = os.path.join(current_dir,'..', '..', 'scitech-dataset')
#train_path = os.path.join(current_dir,'..', '..', 'massachusetts-roads-dataset')
#val_set = CloudDataset(train_path, "validate", use_lwir=True, use_swir=True, randomly_flip=False, randomly_rotate=True)#True)#False, use_swir=True)

#Load models
models = []
training_errs = []
validation_errs = []
for i, (model_dir_path, parent_train_err_path) in enumerate(model_dir_paths):
    input_channels = len(model_bands[i])
    model = load_model_for_eval(model_dir_path, epoch, lr, verbose=True, input_channels=input_channels)
    models.append((model_dir_path, model))
    if 'lum' not in model_dir_path and 'rf' not in model_dir_path:
        train_err = np.genfromtxt(os.path.join(current_dir, 'saved_models', model_dir_path, 'train_err.csv'), delimiter=',')
        val_err = np.genfromtxt(os.path.join(current_dir, 'saved_models', model_dir_path, 'val_err.csv'), delimiter=',')

        if model_dir_path == parent_train_err_path[0]: #Didn't resume 
            pass
        else: #Started from last epoch of parent
            for parent_train_path in reversed(parent_train_err_path):
                parent_train_err = np.genfromtxt(os.path.join(current_dir, 'saved_models', parent_train_path, 'train_err.csv'), delimiter=',')
                parent_val_err = np.genfromtxt(os.path.join(current_dir, 'saved_models', parent_train_path, 'val_err.csv'), delimiter=',')
                train_err = np.concatenate((parent_train_err, train_err))
                val_err = np.concatenate((parent_val_err, val_err))

        training_errs.append(train_err)
        validation_errs.append(val_err)


#Plot training error rates 
plt.figure()
for i, model in enumerate(model_dir_paths):
    if 'lum' not in model_dir_paths[i][0] and 'rf' not in model_dir_paths[i][0]:
        plt.plot(training_errs[i], label=model[0])
#plt.plot(training_errs[0], label='3x3_mse_0_0001')
#plt.plot(training_errs[1], label='7x7_mse_0_0001')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Training error')
plt.title('Training error rates')
plt.show()

#Plot validation error rates
plt.figure()
#plt.plot(validation_errs[0], label='3x3_mse_0_0001')
#plt.plot(validation_errs[1], label='7x7_mse_0_0001')
for i, model in enumerate(model_dir_paths):
    if 'lum' not in model_dir_paths[i][0] and 'rf' not in model_dir_paths[i][0]:
        plt.plot(validation_errs[i], label=model[0])
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Validation error')
plt.title('Validation error rates')
plt.show()

#lum = LuminosityClassifier(train_loader)
#lum.train()
#models = [('lum', lum.classify)]


os.makedirs(os.path.join(current_dir, 'saved_figs'), exist_ok=True)
tholds=[]#0.46875]#tholds = [0.5625]#[0.5, 0.625, 0.421875, 0.46875, 0.4375, 0.4375]#[0.5, 0.375, 0.359375, 0.375, 0.3125, 0.25]
model_tables = compare_model_tables(models, model_val_sets, buffer_save_name=os.path.join(current_dir, 'saved_figs'), tholds=tholds)
os.makedirs(os.path.join(current_dir, 'saved_tables', exp_root), exist_ok=True)
for model_name, model_tables in model_tables.items():
    (model_table, latex_table, model_table_buffer, latex_table_buffer, fp_table, fp_latex_table, fp_table_buffer, fp_latex_table_buffer) = model_tables
    with open(f"saved_tables/{model_name}_tbl_thresh_opt_{epoch}_epochs.txt", "w") as text_file:
        text_file.write(model_table)
    with open(f"saved_tables/{model_name}_tbl_thresh_opt_{epoch}_epochs_texformat.txt", "w") as text_file:
        text_file.write(latex_table)
    with open(f"saved_tables/{model_name}_tbl_thresh_opt_{epoch}_epochs_buffer.txt", "w") as text_file:
        text_file.write(model_table_buffer)
    with open(f"saved_tables/{model_name}_tbl_thresh_opt_{epoch}_epochs_texformat_buffer.txt", "w") as text_file:
        text_file.write(latex_table_buffer)
    with open(f"saved_tables/{model_name}_tbl_thresh_opt_{epoch}_epochs_fp.txt", "w") as text_file:
        text_file.write(fp_table)
    with open(f"saved_tables/{model_name}_tbl_thresh_opt_{epoch}_epochs_fp_texformat.txt", "w") as text_file:
        text_file.write(fp_latex_table)
    with open(f"saved_tables/{model_name}_tbl_thresh_opt_{epoch}_epochs_fp_buffer.txt", "w") as text_file:
        text_file.write(fp_table_buffer)
    with open(f"saved_tables/{model_name}_tbl_thresh_opt_{epoch}_epochs_fp_texformat_buffer.txt", "w") as text_file:
        text_file.write(fp_latex_table_buffer)

    print(model_name)
    print(model_table)
    print(model_table_buffer)
    print()


#Compare model rocs
model_rocs, model_rocs_buffer = compare_model_rocs(models, model_val_sets, buffer=4)

def save_roc_figs(model_rocs, buffer):
    #Plot rocs
    plt.figure()
    ref = [0, 1]
    for model_name, model_roc in model_rocs.items():
        with open(f"saved_tables/{model_name}_roc_{epoch}_epochs_buffer_{buffer}.csv", "w") as text_file:
            np.savetxt(text_file, np.array([model_roc[0], model_roc[1]]).T, delimiter=',')
        with open(f"saved_tables/{model_name}_buffer_{buffer}_AUC.txt", "w") as text_file:
            text_file.write(f"AUC, {str(model_roc[2]).replace('.', '_')}") 
        plt.plot(model_roc[0], model_roc[1], label=f'{model_name} (AUC={model_roc[2]})')
    plt.plot(ref, ref, 'k--', label='Reference (y=x)')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.legend()
    plt.tight_layout()
    plt.grid()
    plt.show()

save_roc_figs(model_rocs, 0)
save_roc_figs(model_rocs_buffer, 4)

for model_name, model_roc in model_rocs.items():
    print(model_name)
    print(f'AUC = {model_roc[2]}')
    print()



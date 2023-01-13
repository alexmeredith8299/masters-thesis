import os
import numpy as np
import torch
import torchvision
import pytest
from e2cnn import gspaces, nn
from scripts.c8_invariant_cnn import C8InvariantCNN
from scripts.cloud_dataset import CloudDataset 
from scripts.train_pytorch_model import evaluate_rotational_equivariance, train_model 
from scripts.train_pytorch_model import save_model_at_checkpoint, load_model_from_checkpoint 
from scripts.evaluate_model import ClassifierValidator
from torch.utils.data import DataLoader

@pytest.fixture(scope='session')  # one model_tester only (for speed)
def model_tester():
    """
    Generate model_tester to use in other tests.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(current_dir,'..', '..', 'scitech-dataset')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_set = CloudDataset(train_path, "train-tiny", randomly_flip=False)
    val_set = CloudDataset(train_path, "validate", randomly_flip=False)
    loss_fn = torch.nn.MSELoss()

    #Load model from checkpoint
    dir_path = os.path.join(current_dir, 'test_permanent', 'test_evaluate_model')
    model_loaded = C8InvariantCNN().to(device)
    opt_loaded = torch.optim.Adam
    model_loaded, opt_loaded, epoch_loaded, train_err_loaded = load_model_from_checkpoint(model_loaded, opt_loaded, 6, 0.001, dir_path)

    #Create ClassifierValidator
    model_loaded.eval()
    val_loader = DataLoader(val_set, batch_size=10, shuffle=False)
    model_tester = ClassifierValidator(model_loaded, val_loader, device)
 
    return model_tester 

def test_generate_comparison_table(model_tester):
    """
    Make sure we can generate comparison table correctly.
    """
    tbl, latex_tbl = model_tester.generate_comparison_table()

    #Compare to expected output 
    expected_tbl= "| Category   |      Acc |   Bal Acc |   Sens |   Spec |   Prec |   Recall |   F1 |   IoU |\n\
|------------+----------+-----------+--------+--------+--------+----------+------+-------|\n\
| arid       | 0.738055 |       0.5 |      0 |      1 |      0 |        0 |    0 |     0 |\n\
| coastline  | 0.604714 |       0.5 |      0 |      1 |      0 |        0 |    0 |     0 |\n\
| forest     | 0.761002 |       0.5 |      0 |      1 |      0 |        0 |    0 |     0 |\n\
| ocean      | 0.733914 |       0.5 |      0 |      1 |      0 |        0 |    0 |     0 |\n\
| plains     | 0.611258 |       0.5 |      0 |      1 |      0 |        0 |    0 |     0 |\n\
| snow       | 0.82347  |       0.5 |      0 |      1 |      0 |        0 |    0 |     0 |\n\
| urban      | 0.773727 |       0.5 |      0 |      1 |      0 |        0 |    0 |     0 |\n\
| Total      | 0.729765 |       0.5 |      0 |      1 |      0 |        0 |    0 |     0 |"
    expected_latex_tbl = ""
    expected_latex_tbl="\\begin{table}\n\
        \centering\n\
        \caption{Insert caption here}\n\
        \label{table:insert_label_here}\n\
        \\begin{tabular}{@{}lrrrrrrrr@{}}\n\
        \\toprule\n\
        \\textbf{Scene Type} & \\textbf{Acc.} & \\textbf{Bal. Acc.} & \\textbf{Sens.} & \\textbf{Spec.} & \\textbf{Prec.} & \\textbf{Recall}  &"
    expected_latex_tbl += "\n \\textbf{$\\textrm{F}_1$} & \\textbf{IoU} \\\\" 
    expected_latex_tbl += "\n\\midrule\n\
Arid & 73.81\% & 50.00\% & 0.00\% & 100.00\% & 0.00\% & 0.00\% & 0.0000 & 0.0000 \\\ \n\
Coastline & 60.47\% & 50.00\% & 0.00\% & 100.00\% & 0.00\% & 0.00\% & 0.0000 & 0.0000 \\\ \n\
Forest & 76.10\% & 50.00\% & 0.00\% & 100.00\% & 0.00\% & 0.00\% & 0.0000 & 0.0000 \\\ \n\
Ocean & 73.39\% & 50.00\% & 0.00\% & 100.00\% & 0.00\% & 0.00\% & 0.0000 & 0.0000 \\\ \n\
Plains & 61.13\% & 50.00\% & 0.00\% & 100.00\% & 0.00\% & 0.00\% & 0.0000 & 0.0000 \\\ \n\
Snow & 82.35\% & 50.00\% & 0.00\% & 100.00\% & 0.00\% & 0.00\% & 0.0000 & 0.0000 \\\ \n\
Urban & 77.37\% & 50.00\% & 0.00\% & 100.00\% & 0.00\% & 0.00\% & 0.0000 & 0.0000 \\\ \n\
\midrule \n\
\\textbf{Overall} & \\textbf{72.98\%} & \\textbf{50.00\%} & \\textbf{0.00\%} & \\textbf{100.00\%} & \\textbf{0.00\%} & \\textbf{0.00\%} & \\textbf{0.0000} & \\textbf{0.0000} \\\ \n\
\\bottomrule\n\
        \end{tabular}\n\
        \end{table}"
    assert tbl == expected_tbl
    assert "".join(latex_tbl.split()) == "".join(expected_latex_tbl.split()) #Ignore whitespace

def test_generate_confusion_matrix(model_tester):
    """
    Make sure we can generate confusion matrix correctly.
    """
    conf_mat, fig, ax = model_tester.generate_confusion_matrix()
    conf_mat_expected = np.array([[0, 244958],
                   [0, 107554]])
    assert conf_mat.all() == conf_mat_expected.all()

def test_generate_roc_curve(model_tester):
    """
    Make sure we can generate ROC curve correctly.
    """
    roc_fprates, roc_tprates, AUC = model_tester.generate_roc_curve()
    roc_fprates_expected = [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    roc_tprates_expected = [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert np.fabs(AUC-0.5) < 1e-6
    assert np.fabs(np.array(roc_fprates) - np.array(roc_fprates_expected)).sum() < 1e-6
    assert np.fabs(np.array(roc_tprates) - np.array(roc_tprates_expected)).sum() < 1e-6


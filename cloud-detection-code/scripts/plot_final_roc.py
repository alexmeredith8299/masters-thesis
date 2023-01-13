import os
import sys
sys.path.append('../../cloud-detection-code')
sys.path.append('../../cloud-detection-code/scripts')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scripts.cloud_dataset import CloudDataset
from scripts.evaluate_utils import load_img_from_fname, plot_model_comparison, load_road_img_from_fname
from scripts.evaluate_utils import load_model_for_eval, evaluate_models_on_img, plot_band_comparison, plot_buffer_difference
from scipy.interpolate import make_interp_spline, BSpline
from scipy.interpolate import interp1d
sns.set_palette('colorblind')

def remove_dupes(x, y):
    #Remove duplicates
    unique_inds = []
    x_set = set()
    for i, xi in enumerate(x):
        if xi not in x_set:
            unique_inds.append(i)
        x_set.add(xi)
    new_x, new_y = [], []
    for ind in unique_inds:
        new_x.append(x[ind])
        new_y.append(y[ind])

    #Sort xs
    zipped = zip(new_x, new_y)
    zipped_sorted = sorted(zipped, key=lambda x:x[0])
    final_x, final_y = [], []
    for z in zipped_sorted:
        final_x.append(z[0])
        final_y.append(z[1])
    return np.array(final_x), np.array(final_y)

cloud=False
if cloud:
    exp_folder = 'cloud_final'
    bands = 'lwir_swir'
    buffers = ['buffer_0', 'buffer_2']
else:
    exp_folder = 'road_final'
    bands = 'rgb'
    buffers = ['buffer_0', 'buffer_4']
#checkpoint_name = 'model_checkpoint_995.pt'
epoch = 500 
if not cloud:
    epoch = 90

current_dir = os.path.dirname(os.path.abspath(__file__))
if cloud:
    train_path = os.path.join(current_dir,'..', '..', 'sparcs-dataset')
else:
    train_path = os.path.join(current_dir,'..', '..', 'massachusetts-roads-dataset')
#val_set = CloudDataset(train_path, "validate", use_lwir=True, use_swir=True, randomly_flip=False, randomly_rotate=True)#True)#False, use_swir=True)
#img = None
#for val_img in val_set:
#    img = val_img
#plains_fname = 'plains_35.79695094294484_-86.16782201433625'
#snow_fname = 'snow_46.12935804682721_10.722238830040727'
plains_fname = 'LC82210662014229LGN00_18_23'
snow_fname = 'LC81480352013195LGN00_32_22'
road_fname_easy = '1024_1024_test_img-8'
road_fname_hard = '1024_1024_test_img-4'
img_fname = plains_fname#snow_fname#plains_fname#plains_fname#plains_fname#snow_fname
img_root = 'plains'#'snow'#'plains'
#img_fname = snow_fname
#img_root = 'snow'
if not cloud:
    img_fname = road_fname_easy
    img_root = 'road_easy'
    #img_fname = road_fname_hard
    #img_root = 'road_hard'
#img_fname = snow_fname#snow_fname
if cloud:
    img = load_img_from_fname(os.path.join(train_path, 'test'), img_fname)
else:
    img = load_road_img_from_fname(os.path.join(train_path, 'test'), img_fname)
    loss= 'dice'
band_comparison = False


for buffer in buffers:
    if band_comparison:
        density = '_dense'
        group = 'C8'
        band_order = ['rgb_only', f'0_002_flip', '0_002_swir', 'lwir_swir']
        order = [(density, band, group, 'flip') for band in band_order]
        order_names = ['VIS (AUC = 0.99876)', 'VIS + LWIR (AUC = 0.99901)', 'VIS + SWIR (AUC = 0.99896)', 'VIS + LWIR + SWIR (AUC = 0.99940)']
        if buffer == 'buffer_0':
            order_names = ['VIS (AUC = 0.97775)', 'VIS + LWIR (AUC = 0.98375)', 'VIS + SWIR (AUC = 0.98121)', 'VIS + LWIR + SWIR (AUC = 0.98984)']
        channel_inputs = [['r', 'g', 'b'], ['r', 'g', 'b', 'lwir'], ['r', 'g', 'b', 'swir'], ['r', 'g', 'b', 'lwir', 'swir']]
        epochs = [500, 500, 500, 500]
    else:
        order = [('lum', bands), ('rf', bands), ('nondense', bands, 'NONE', buffer), ('dense', bands, 'NONE', buffer), ('nondense', bands, 'C8', buffer), ('dense', bands, 'C8', buffer)]
        order_names = ['Luminosity (AUC = 0.86570)', 'Random Forest (AUC = 0.99773)', 'U-Net (AUC = 0.99850)', 'Dense U-Net (AUC = 0.99952)', '$C_8$-Equivariant U-Net (AUC = 0.99948)', '$C_8$-Equivariant Dense U-Net (AUC = 0.99952)']
        if buffer == 'buffer_0':
            order_names = ['Luminosity (AUC = 0.75609)', 'Random Forest (AUC = 0.97412)', 'U-Net (AUC = 0.98847)', 'Dense U-Net (AUC = 0.99016)', '$C_8$-Equivariant U-Net (AUC = 0.98553)', '$C_8$-Equivariant Dense U-Net (AUC = 0.98984)']
        if cloud:
            channel_inputs = [['r', 'g', 'b', 'lwir', 'swir'],['r', 'g', 'b', 'lwir', 'swir'],['r', 'g', 'b', 'lwir', 'swir'],['r', 'g', 'b', 'lwir', 'swir'], ['r', 'g', 'b', 'lwir', 'swir'], ['r', 'g', 'b', 'lwir', 'swir']]
            epochs = [500, 500, 505, 500, 505, 500]
        else:
            order_names = ['U-Net (AUC = 0.99486)', 'Dense U-Net (AUC = 0.99006)', '$C_8$-Equivariant U-Net (AUC = 0.98821)', '$C_8$-Equivariant Dense U-Net (AUC = 0.98443)']
            if buffer == 'buffer_0':
                order_names = ['U-Net (AUC = 0.84340)', 'Dense U-Net (AUC = 0.83609)', '$C_8$-Equivariant U-Net (AUC = 0.80781)', '$C_8$-Equivariant Dense U-Net (AUC = 0.80849)']
            order = [('nondense', loss, bands, 'NONE', buffer), ('dense', loss, bands, 'NONE', buffer), ('nondense', loss, bands, 'C8', buffer), ('dense', loss, bands, 'C8', buffer)]
            channel_inputs = [ ['r', 'g', 'b'],['r', 'g', 'b'], ['r', 'g', 'b'], ['r', 'g', 'b']]
            epochs = [90, 90, 90, 90]

    #Get inputs in order
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_checkpoint_files = os.listdir(os.path.join(current_dir, 'saved_models', exp_folder))
    model_checkpoint_names = []
    roc_input_files = os.listdir(os.path.join(current_dir, 'saved_tables', exp_folder))
    roc_fnames = []
    roc_output_dirs = os.listdir(os.path.join(current_dir, 'saved_figs', exp_folder))
    roc_outnames = []
    buffer_png_outnames = []

    for name in order:
        for file in roc_input_files:
            match_name = True
            for token in name:
                if token not in file or 'roc' not in file:
                    match_name = False
            if match_name:
                full_name = os.path.join(current_dir, 'saved_tables', exp_folder, file)
                if full_name not in roc_fnames:
                    roc_fnames.append(full_name)

    for name in order:
        for file in roc_output_dirs:
            match_name = True
            for token in name:
                if token not in file and 'buffer' not in token:# or 'roc' not in file:
                    match_name = False
            if match_name:
                if band_comparison:
                    full_name = os.path.join(current_dir, 'saved_figs', exp_folder, file, f'bands_roc_{buffer}.png')
                else:
                    full_name = os.path.join(current_dir, 'saved_figs', exp_folder, file, f'models_roc_{buffer}.png')
                buffer_name = os.path.join(current_dir, 'saved_figs', exp_folder, file, f'{img_root}_buffer_fig.png')
                if full_name not in roc_outnames:
                    roc_outnames.append(full_name)
                if buffer_name not in buffer_png_outnames:
                    buffer_png_outnames.append(buffer_name)

    for name in order:
        for file in model_checkpoint_files:
            match_name = True
            for token in name:
                if token not in file and 'buffer' not in token:# or 'roc' not in file:
                    match_name = False
            if match_name:
                full_name = os.path.join(current_dir, 'saved_models', exp_folder, file)
                if full_name not in model_checkpoint_names:
                    model_checkpoint_names.append(full_name)
    models = []
    for i, model_name in enumerate(model_checkpoint_names):
        models.append(load_model_for_eval(model_name, epochs[i], 0.001, input_channels=len(channel_inputs[i])))
    if cloud:
        model_masks = evaluate_models_on_img(img, models, channel_inputs)
    else:
        model_masks = evaluate_models_on_img(img, models, channel_inputs, img_size=256, has_ir=False)

    if band_comparison:
        [rgb_mask, rgb_lwir_mask, rgb_swir_mask, rgb_lwir_swir_mask] = model_masks
        band_comparison_fname = os.path.join(current_dir, 'saved_figs', exp_folder, f'{epoch}_{group}{density}_{img_fname}_band_comparison.png')
        plot_band_comparison(img, rgb_mask, rgb_lwir_mask, rgb_swir_mask, rgb_lwir_swir_mask, band_comparison_fname)
        for i, mask in enumerate(model_masks):
            if cloud:
                plot_buffer_difference(img, mask, buffer_png_outnames[i])
            else:
                plot_buffer_difference(img, mask, buffer_png_outnames[i], rgb_only=True, buffer=4)
    else:
        if cloud:
            [lum_mask, rf_mask, unet_mask, dense_mask, rot_unet_mask, rot_dense_mask] = model_masks
        else:
            [unet_mask, dense_mask, rot_unet_mask, rot_dense_mask] = model_masks
        for i, mask in enumerate(model_masks):
            if cloud:
                plot_buffer_difference(img, mask, buffer_png_outnames[i])
            else:
                plot_buffer_difference(img, mask, buffer_png_outnames[i], rgb_only=True, buffer=4)
        #lum_mask, rf_mask = unet_mask, unet_mask
        model_comparison_fname = os.path.join(current_dir, 'saved_figs', exp_folder, f'{epoch}_{bands}_{img_fname}_model_comparison.png')
        if cloud:
            plot_model_comparison(img, lum_mask, rf_mask, unet_mask, dense_mask, rot_unet_mask, rot_dense_mask, model_comparison_fname)
        else:
            plot_model_comparison(img, None, None, unet_mask, dense_mask, rot_unet_mask, rot_dense_mask, model_comparison_fname, rgb_only=True, unets_only=True)

    roc_names = [fname for fname in roc_fnames if buffer in fname]# and str(epoch) in fname]
    #Plot ROCs as needed
    for i in range(len(roc_names)):
        plt.figure()
        ref = [0, 1]
        for j in range(i+1):
            roc = np.genfromtxt(roc_names[j], delimiter=',')
            roc = roc.T
            plt.plot(roc[0], roc[1], label=order_names[j])
        plt.plot(ref, ref, 'k--', label='Reference')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.tight_layout()
        plt.grid()
        #plt.show()
        plt.savefig(roc_outnames[i])

        plt.figure()
        ref = [0, 1]
        for j in range(i+1):
            roc = np.genfromtxt(roc_names[j], delimiter=',')
            roc = roc.T
            xnew = np.linspace(roc[0].min(), roc[0].max(), 1500)
            roc0, roc1 = remove_dupes(roc[0], roc[1])
            #spline_fit = make_interp_spline(roc0, roc1, k=3)
            f_cubic = interp1d(roc0, roc1, kind='cubic')
            ynew = f_cubic(xnew)
            plt.plot(roc[0], roc[1], label=order_names[j])
        plt.plot(ref, ref, 'k--', label='Reference')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        if cloud or not buffer=='buffer_0':
            plt.xlim((0.0, 0.1))
            plt.ylim((0.9, 1.0))
        else:
            plt.xlim((0.0, 0.4))
            plt.ylim((0.6, 1.0))
        plt.legend()
        plt.tight_layout()
        plt.grid()
        #plt.show()
        fname_split = roc_outnames[i].split('.')
        zoom_fname = f'{fname_split[0]}_zoom.png'
        plt.savefig(zoom_fname)

    #Save to appropriate files

import os
import math
import random 
import shutil
from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import ToTensor

def rename_images(parent_dir, categories=('arid', 'coastline', 'mixed', 'snow', 'verdant')):
    for category in categories:
        category_files = os.listdir(f'{parent_dir}/{category}')
        for file in category_files:
            if category not in file:
                file_split = file.split('_')
                file_split.insert(2, category)
                new_fname = '_'.join(file_split)
                os.system(f'cp {parent_dir}/{category}/{file} {parent_dir}/{new_fname}')

def split_images(parent_dir, categories=('arid', 'coastline', 'mixed', 'snow', 'verdant'), val_per_cat=1):
    files = os.listdir(f'{parent_dir}')
    val_files = []
    train_files = []
    for category in categories:
        category_data = [f for f in files if category in f and 'data' in f]
        random.shuffle(category_data)
        val_files.append(category_data[0])
        file_root = '_'.join(category_data[0].split('_')[0:2])
        for f in files:
            if file_root in f and f not in val_files:
                val_files.append(f)
    for f in files:
        if f not in val_files and ('tif' in f or 'png' in f):
            train_files.append(f)
    for file in val_files:
        os.system(f'mv {parent_dir}/{file} {parent_dir}/validate/{file}')
    for file in train_files:
        os.system(f'mv {parent_dir}/{file} {parent_dir}/train/{file}')

def process_masks(parent_dir):
    mask_files = [f for f in os.listdir(parent_dir) if 'mask.png' in f]
    for mask in mask_files:
        img = Image.open(os.path.join(parent_dir, mask))
        breakpoint()
        #cloud_mask = (np.array(img) == 5)*255
        #im = Image.fromarray(np.uint8(cloud_mask))
        #new_name = '_'.join(mask.split('_')[0:2]) + '_ref.tif'
        #im.save(os.path.join(parent_dir, new_name))

def int16_to_int8(arr):
    return (arr/256).astype(np.uint8)

def process_data(parent_dir):
    data_files = [f for f in os.listdir(parent_dir) if 'data.tif' in f]
    for data in data_files:
        img = gdal.Open(os.path.join(parent_dir, data)).ReadAsArray()
        img = np.array(int16_to_int8(img))
        blue = img[1, :, :].reshape(1000, 1000, 1) #B2
        green = img[2, :, :].reshape(1000, 1000, 1) #B3
        red = img[3, :, :].reshape(1000, 1000, 1) #B4
        swir = img[5, :, :].reshape(1000, 1000) #B6
        lwir = img[9, :, :].reshape(1000, 1000) #B10
        rgb = np.concatenate((red, green, blue), 2)

        #Save
        new_name_rgb = '_'.join(data.split('_')[0:2]) + '_rgb.tif'
        new_name_lwir = '_'.join(data.split('_')[0:2]) + '_lwir.tif'
        new_name_swir = '_'.join(data.split('_')[0:2]) + '_swir.tif'

        Image.fromarray(np.uint8(rgb)).save(os.path.join(parent_dir, new_name_rgb))
        Image.fromarray(np.uint8(lwir)).save(os.path.join(parent_dir, new_name_lwir))
        Image.fromarray(np.uint8(swir)).save(os.path.join(parent_dir, new_name_swir))



def find_missing_images(dir_1, dir_2):
    files_1 = os.listdir(dir_1)
    files_2 = os.listdir(dir_2)
    for file_1 in files_1:
        file_root = '_'.join(file_1.split('_')[0:2])
        if f'{file_root}_data.tif' not in files_2 and 'txt' not in file_1:
            print(f'{file_1}')

def distribute_images(parent_dir, categories=('arid', 'coastline', 'mixed', 'snow', 'verdant')):
    files = os.listdir(parent_dir)
    category_roots = {}
    for category in categories:
        cat_roots = []
        category_files = os.listdir(f'{parent_dir}/{category}')
        for file in category_files:
            if file == '.DS_Store':
                rm_r(f'{parent_dir}/{category}/{file}')
            else:
                cat_roots.append('_'.join(file.split('_')[0:2]))
        category_roots[category] = cat_roots
    for file in files:
        if file.endswith('tif') or file.endswith('png'):
            file_root = '_'.join(file.split('_')[0:2])
            for category in categories:
                if file_root in category_roots[category]:
                    print(f'Moving file {file} to category {category}')
                    shutil.copyfile(f'{parent_dir}/{file}', f'{parent_dir}/{category}/{file}')

def rm_r(path):
    if not os.path.exists(path):
        return
    if os.path.isfile(path) or os.path.islink(path):
        os.unlink(path)
    else:
        shutil.rmtree(path)

def fix_data(parent_dir, subfolder):
    mask_files = [f for f in os.listdir(os.path.join(parent_dir, subfolder)) if 'mask' in f]
    for file in mask_files:
        file_root = '_'.join(file.split('_')[0:2])
        category = file.split('_')[2]
        data_file = file_root + '_data.tif'
        new_data_file = file_root + f'_{category}_data.tif'
        os.system(f'mv {parent_dir}/{data_file} {parent_dir}/{subfolder}/{new_data_file}')

def patch_images(parent_dir, new_sidelength=250):
    mask_files = [f for f in os.listdir(os.path.join(parent_dir)) if 'ref' in f]
    mask_roots = ['_'.join(f.split('_')[0:2]) for f in mask_files]

    convert_tensor = ToTensor()
    for root in mask_roots:
        lwir_name = f'{root}_lwir.tif'
        swir_name = f'{root}_swir.tif'
        rgb_name = f'{root}_rgb.tif'
        ref_name = f'{root}_ref.tif'

        rgb_img = convert_tensor(Image.open(os.path.join(parent_dir, rgb_name)))
        ref_img = convert_tensor(Image.open(os.path.join(parent_dir, ref_name)))
        lwir_img = convert_tensor(Image.open(os.path.join(parent_dir, lwir_name)))
        swir_img = convert_tensor(Image.open(os.path.join(parent_dir, swir_name)))

        width, height = rgb_img.shape[1], rgb_img.shape[2]
        n_cols = int(math.floor(width/new_sidelength))
        n_rows = int(math.floor(height/new_sidelength))

        for i in range(n_cols):
            start_col = i*new_sidelength
            end_col = (i+1)*new_sidelength
            for j in range(n_rows):
                start_row = j*new_sidelength
                end_row = (j+1)*new_sidelength
                rgb_patch = rgb_img[:, start_col:end_col, start_row:end_row].permute(1, 2, 0).numpy()
                lwir_patch = lwir_img[:, start_col:end_col, start_row:end_row].numpy().reshape(new_sidelength, new_sidelength)
                swir_patch = swir_img[:, start_col:end_col, start_row:end_row].numpy().reshape(new_sidelength, new_sidelength)
                ref_patch = ref_img[:, start_col:end_col, start_row:end_row].numpy().reshape(new_sidelength, new_sidelength)

                rgb_new_name = f'{root}_{i}{j}_rgb.tif'
                lwir_new_name = f'{root}_{i}{j}_lwir.tif'
                swir_new_name = f'{root}_{i}{j}_swir.tif'
                ref_new_name = f'{root}_{i}{j}_ref.tif'

                Image.fromarray(np.uint8(rgb_patch*255)).save(os.path.join(parent_dir, 'crops', rgb_new_name))
                Image.fromarray(np.uint8(lwir_patch*255)).save(os.path.join(parent_dir, 'crops', lwir_new_name))
                Image.fromarray(np.uint8(swir_patch*255)).save(os.path.join(parent_dir, 'crops', swir_new_name))
                Image.fromarray(np.uint8(ref_patch*255)).save(os.path.join(parent_dir, 'crops', ref_new_name))

def patch_masks(parent_dir, new_sidelength=250):
    mask_files = [f for f in os.listdir(os.path.join(parent_dir)) if 'mask.png' in f]
    mask_roots = ['_'.join(f.split('_')[0:2]) for f in mask_files]

    convert_tensor = ToTensor()
    for root in mask_roots:
        mask_name = f'{root}_mask.png'

        mask_img = convert_tensor(Image.open(os.path.join(parent_dir, mask_name)))

        width, height = mask_img.shape[1], mask_img.shape[2]
        n_cols = int(math.floor(width/new_sidelength))
        n_rows = int(math.floor(height/new_sidelength))

        for i in range(n_cols):
            start_col = i*new_sidelength
            end_col = (i+1)*new_sidelength
            for j in range(n_rows):
                start_row = j*new_sidelength
                end_row = (j+1)*new_sidelength
                mask_patch = mask_img[:, start_col:end_col, start_row:end_row].numpy().reshape(new_sidelength, new_sidelength)
                mask_new_name = f'{root}_{i}{j}_mask.png'
                Image.fromarray(np.uint8(mask_patch*255)).save(os.path.join(parent_dir, 'crops', mask_new_name))




#Move all "valid" images to raw/
#reallocate_images('training', 'train')
#reallocate_images('testing', 'test')

#Patch images
#distribute_images('.')
#rename_images('.')
#split_images('.')
#patch_images('test')
#patch_images('train')
#patch_images('validate')
#process_masks('validate')
#process_masks('train')
#process_masks('test')
#fix_data('.', 'train')
#process_data('validate')
#process_data('test')
#process_data('train')
#patch_images('old/validate')
#patch_images('old/test')
#patch_images('old/train')
patch_masks('.')
#process_masks("crops")
#patch_images('test')
#find_missing_images('/Users/alexmeredith/Downloads/sending', 'old')
#clean_unnecessary_masks('test_clean')
#clean_unnecessary_masks('validate_clean')
#clean_unnecessary_masks('train_clean')
#patch_images('train-tiny', overlap=256)
#patch_images('val-tiny')
#patch_images('train', overlap=256)
#patch_images('test')
#patch_images('validate')


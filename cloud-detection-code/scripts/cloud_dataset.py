"""
This module creates a dataset for detecting clouds with PyTorch models.
"""
import os
import random
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
from torchvision.transforms import ToTensor
from torchvision import transforms
from PIL import Image
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
#torch.multiprocessing.set_start_method('spawn')

class CloudDataset(Dataset):
    """
    Wrapper for torch.utils.data.Dataset that can randomly flip, shuffle,
    and resize images for use in PyTorch models.

    Parameters
    -----------
        root: string
            filepath to directory containing folder with images
        folder: string
            name of folder containing images
        filenames: list (optional)
            list of filenames

    Attributes
    ------------
        root: string
            filepath to directory containing folder with images
        dataset_folder: string
            full path (incl. folder name) to images
        images_list: list
            list of filenames of images in dataset_folder
    """
    def __init__(self, root, folder, use_lwir=True, use_swir=False, randomly_flip=True, randomly_crop=False, filenames=None, randomly_rotate=False, n_classes=2):
        self.root = root
        self.use_lwir = use_lwir
        self.use_swir = use_swir
        self.randomly_flip_samples = randomly_flip
        self.randomly_rotate_samples = randomly_rotate
        self.randomly_crop_samples = randomly_crop
        self.n_classes = n_classes

        if filenames is not None:
            self.images_list = filenames
            self.dataset_folder = os.path.join(root, folder)
        else:
            self.dataset_folder =  os.path.join(root, folder)
            self.images_list = list(set(["_".join(x.split("_")[0:-1])
                for x in os.listdir(self.dataset_folder)]))

        #Sort filenames so images are deterministically ordered
        self.images_list = sorted(self.images_list)
        self.len = len(self.images_list)

        #Can cache for faster loading if not cropping randomly 
        if not self.randomly_crop_samples:
            self.cache = self.cache_images()
            if not self.randomly_flip_samples:
                self.samples = self.cache_samples(self.cache)

    def __len__(self):
        """
        Return number of images in dataset.

        Returns
        ---------
            int
                number of images in datase
        """
        return self.len

    def get_k_folds(self, k_folds):
        """
        Break dataset into k subsets and return a list of subsets, for k-fold cross-validation.

        Arguments
        ----------
            k_folds: int
                number of folds

        Returns
        --------
            sets: list
                list of k lists containing strings representing image filenames
        """
        sets = []
        for k in range(k_folds):
            kth_filenames = self.images_list[k *
                (len(self)//k_folds): (k+1) * (len(self)//k_folds) ]
            sets.append(CloudDataset(self.root, self.dataset_folder, filenames=kth_filenames))

        return sets

    @staticmethod
    def randomly_rotate(rgb_img, ir_img, swir_img, ref, class_mask):
        """
        Randomly flip an image horizontally and/or vertically (probability 0.5
        for horizontal flip, and probability
        0.5 for vertical flip, independent of each other).

        Arguments
        -----------
            rgb_img: np array
                np array representing an RGB image
            ir_img: np array
                np array representing an IR image (corresponds to rgb_img)
            swir_img: np array
                np array representing an SWIR image (corresponds to rgb_img)
            ref: np array
                np array representing a cloud mask corresponding to rgb_img and ir_img

        Returns
        ------
            rgb_img: np array
                RGB image (randomly flipped)
            ir_img: np array
                IR image (randomly flipped, same flips as RGB image)
            swir_img: np array
                SWIR image (randomly flipped, same flips as RGB image)
            ref: np array
                reference cloud mask (undergone same transformations as rgb_img and ir_img
        """
        angle = random.random()*360
        rgb_img = F.rotate(rgb_img, angle)
        ir_img = F.rotate(ir_img, angle)
        swir_img = F.rotate(swir_img, angle)
        ref = F.rotate(ref, angle)
        class_mask = F.rotate(class_mask, angle)
        return rgb_img, ir_img, swir_img, ref, class_mask



    @staticmethod
    def randomly_flip(rgb_img, ir_img, swir_img, ref, class_mask):
        """
        Randomly flip an image horizontally and/or vertically (probability 0.5
        for horizontal flip, and probability
        0.5 for vertical flip, independent of each other).

        Arguments
        -----------
            rgb_img: np array
                np array representing an RGB image
            ir_img: np array
                np array representing an IR image (corresponds to rgb_img)
            swir_img: np array
                np array representing an SWIR image (corresponds to rgb_img)
            ref: np array
                np array representing a cloud mask corresponding to rgb_img and ir_img

        Returns
        ------
            rgb_img: np array
                RGB image (randomly flipped)
            ir_img: np array
                IR image (randomly flipped, same flips as RGB image)
            swir_img: np array
                SWIR image (randomly flipped, same flips as RGB image)
            ref: np array
                reference cloud mask (undergone same transformations as rgb_img and ir_img
        """
        # Random horizontal flipping
        if random.random() > 0.5:
            rgb_img = F.hflip(rgb_img)
            ir_img = F.hflip(ir_img)
            swir_img = F.hflip(swir_img)
            ref = F.hflip(ref)
            class_mask = F.hflip(class_mask)

        # Random vertical flipping
        if random.random() > 0.5:
            rgb_img = F.vflip(rgb_img)
            ir_img = F.vflip(ir_img)
            swir_img = F.vflip(swir_img)
            ref = F.vflip(ref)
            class_mask = F.vflip(class_mask)

        return rgb_img, ir_img, swir_img, ref, class_mask

    def resize(self, img,  size=144):
        """
        Resize an image (RGB image, IR image, and corresponding cloud mask).

        Arguments
        ----------
            img: np array
                np array representing an image
            size: int (optional)
                size to crop image sidelength to, default is 144

        Returns
        ---------
            img: np array
                np array representing an image (resized)
       """
        if self.randomly_crop_samples:
            top = random.randint(0, img.shape[1] - size)
            left = random.randint(0, img.shape[2] - size)
            return F.crop(img, top, left, size, size)
        else: #Resize if not random
            return F.resize(img, size, interpolation=Image.NEAREST)

    def shuffle(self):
        """
        Shuffle order of images in self.images_list
        """
        random.shuffle(self.images_list)


    # returns both raw and validation
    def __getitem__(self, idx, get_class_mask=True):
        """
        Get image from self.images_list for training or testing a PyTorch model.

        Arguments
        ----------
            idx: int
                index of image to get from self.images_list

        Returns
        ---------
            sample: dictionary
                dictionary where 'img' maps to np array with image, 'ref' maps
                to np array with mask, 'category' maps to string with image
                scene type (e.g. snow, ocean, etc)
        """
        #Simple case: no transformations -> return from cache
        if not self.randomly_crop_samples and not self.randomly_flip_samples and not get_class_mask:
            return self.samples[idx]

        #If cropping, need to reload for random crop
        if self.randomly_crop_samples:
            trainimage_name = self.images_list[idx]
            category = trainimage_name.split("_")[0] if 'scitech' in self.dataset_folder else 'cloud'

            convert_tensor = ToTensor()

            rgbimg_name = f"{trainimage_name}_rgb.tif"
            irimg_name = f"{trainimage_name}_lwir.tif"
            swirimg_name = f"{trainimage_name}_swir.tif"
            ref_name = f"{trainimage_name}_ref.tif"

            rgb_img = convert_tensor(Image.open(os.path.join(self.dataset_folder, rgbimg_name)))
            refmask = convert_tensor(Image.open(os.path.join(self.dataset_folder, ref_name)))*(self.n_classes-1)
            ir_img = convert_tensor(Image.open(os.path.join(self.dataset_folder, irimg_name)))
            swir_img = convert_tensor(Image.open(os.path.join(self.dataset_folder, swirimg_name)))

            rgb_img = self.resize(rgb_img)
            refmask = self.resize(refmask)
            ir_img = self.resize(ir_img)
            swir_img = self.resize(swir_img)
            if get_class_mask:
                class_mask_name = f'{trainimage_name}_mask.png'
                class_mask = convert_tensor(Image.open(os.path.join(self.dataset_folder, class_mask_name)))
                class_mask = self.resize(class_mask)
        else: #otherwise can load cropped image from cache
            (rgb_img, refmask, ir_img, swir_img, category) = self.cache[idx]
            if get_class_mask:
                trainimage_name = self.images_list[idx]
                convert_tensor = ToTensor()
                class_mask_name = f'{trainimage_name}_mask.png'
                class_mask = convert_tensor(Image.open(os.path.join(self.dataset_folder, class_mask_name)))
                class_mask = self.resize(class_mask)
        if not get_class_mask:
            class_mask = refmask 
        #Flip if needed
        if self.randomly_flip_samples:
            rgb_img, ir_img, swir_img, refmask, class_mask = self.randomly_flip(rgb_img, ir_img, swir_img, refmask, class_mask)
        if self.randomly_rotate_samples:
            rgb_img, ir_img, swir_img, refmask, class_mask = self.randomly_rotate(rgb_img, ir_img, swir_img, refmask, class_mask)
        #Create sample and return
        if self.use_lwir and self.use_swir:
            img = torch.cat((rgb_img, ir_img, swir_img), 0)
        elif self.use_lwir:
            img = torch.cat((rgb_img, ir_img), 0)
        elif self.use_swir:
            img = torch.cat((rgb_img, swir_img), 0)
        else:
            img = rgb_img

        #One-hot
        if self.n_classes > 2:
            refmask = torch.nn.functional.one_hot(torch.round(refmask).to(torch.int64), self.n_classes)
        sample = {'img': img, 'ref': refmask, 'classmask': class_mask, 'category': category}

        return sample

    #TODO document (and maybe refactor so img loading has SPOT)
    def cache_images(self):
        cache = {}
        for idx in range(len(self)):
            trainimage_name = self.images_list[idx]
            category = trainimage_name.split("_")[0] if 'scitech' in self.dataset_folder else 'cloud'

            convert_tensor = ToTensor()

            rgbimg_name = f"{trainimage_name}_rgb.tif"
            irimg_name = f"{trainimage_name}_lwir.tif"
            swirimg_name = f"{trainimage_name}_swir.tif"
            ref_name = f"{trainimage_name}_ref.tif"

            rgb_img = convert_tensor(Image.open(os.path.join(self.dataset_folder, rgbimg_name)))
            refmask = convert_tensor(Image.open(os.path.join(self.dataset_folder, ref_name)))*(self.n_classes-1)
            ir_img = convert_tensor(Image.open(os.path.join(self.dataset_folder, irimg_name)))
            swir_img = convert_tensor(Image.open(os.path.join(self.dataset_folder, swirimg_name)))
            #breakpoint()

            rgb_img = self.resize(rgb_img)
            refmask = self.resize(refmask)
            ir_img = self.resize(ir_img)
            swir_img = self.resize(swir_img)
            cache[idx] = (rgb_img, refmask, ir_img, swir_img, category)
        return cache

    def cache_samples(self, cache):
        samples = []
        for idx in range(len(self)):
            (rgb_img, refmask, ir_img, swir_img, category) = cache[idx]
            if self.use_lwir and self.use_swir:
                img = torch.cat((rgb_img, ir_img, swir_img), 0)
            elif self.use_lwir:
                img = torch.cat((rgb_img, ir_img), 0)
            elif self.use_swir:
                img = torch.cat((rgb_img, swir_img), 0)
            else:
                img = rgb_img

            sample = {'img': img, 'ref': refmask, 'category': category}
            samples.append(sample)
        return samples



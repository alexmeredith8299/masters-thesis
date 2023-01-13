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

class RoadDataset(Dataset):
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
    def __init__(self, root, folder, randomly_crop=False, randomly_flip=True, randomly_rotate=True, filenames=None, cache=False):
        self.root = root
        self.randomly_flip_samples = randomly_flip#True#randomly_flip
        self.randomly_rotate_samples = randomly_rotate#True#randomly_rotate
        self.randomly_crop_samples = 'train' in folder

        if filenames is not None:
            self.images_list = filenames
            self.dataset_folder = os.path.join(root, folder)
        else:
            self.dataset_folder =  os.path.join(root, folder)
            self.images_list = [f for f in os.listdir(self.dataset_folder) if 'mask' not in f]

        #Sort filenames so images are deterministically ordered
        self.images_list = sorted(self.images_list)
        self.len = len(self.images_list)

        #Can cache images for faster loading if not cropping randomly
        self.caching = cache
        if cache and not self.randomly_crop_samples:
            self.cache = self.cache_images()
            if cache and not self.randomly_flip_samples:
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
            sets.append(RoadDataset(self.root, self.dataset_folder, filenames=kth_filenames))

        return sets

    @staticmethod
    def randomly_rotate(img, ref):
        """
        Randomly rotate an image (RGB image, IR image, and corresponding cloud mask).

        Arguments
        ----------
            img: np array
                np array representing an image
            ref: np array
                np array representing a cloud mask

        Returns
        ---------
            img: np array
                np array representing an image (rotated)
            ref: np array
                np array representing a cloud mask (rotated)
        """
        angle = random.randint(0, 3)
        return F.rotate(img, angle*90), F.rotate(ref, angle*90)

    @staticmethod
    def randomly_flip(img, ref):
        """
        Randomly flip an image horizontally and/or vertically (probability 0.5
        for horizontal flip, and probability
        0.5 for vertical flip, independent of each other).

        Arguments
        -----------
            img: np array
                np array representing an RGB image
            ref: np array
                np array representing a cloud mask corresponding to rgb_img and ir_img

        Returns
        ------
            img: np array
                RGB image (randomly flipped)
            ref: np array
                reference cloud mask (undergone same transformations as rgb_img and ir_img
        """
        # Random horizontal flipping
        if random.random() > 0.5:
            img = F.hflip(img)
            ref = F.hflip(ref)

        # Random vertical flipping
        if random.random() > 0.5:
            img = F.vflip(img)
            ref = F.vflip(ref)
        
        return img, ref

    def crop(self, img, ref, size=256):
        """
        Crop an image (RGB image, IR image, and corresponding cloud mask).

        Arguments
        ----------
            img: np array
                np array representing an image
            ref: np array
                np array representing a cloud mask
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
            return F.crop(img, top, left, size, size), F.crop(ref, top, left, size, size)
        else: #Resize if not random
            #return F.crop(img, 0, 0, size, size), F.crop(ref, 0, 0, size, size)
            return F.resize(img, size, interpolation=Image.BILINEAR), F.resize(ref, size, interpolation=Image.BILINEAR)

    def shuffle(self):
        """
        Shuffle order of images in self.images_list
        """
        random.shuffle(self.images_list)


    # returns both raw and validation
    def __getitem__(self, idx):
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
                to np array with mask
        """
        #If not flipping or cropping, just return from cache
        if self.caching and not self.randomly_crop_samples and not self.randomly_flip_samples and not self.randomly_rotate_samples:
            return self.samples[idx]

        #If cropping or flipping, need to load for transformation
        if not self.caching or self.randomly_crop_samples:
            img_name = self.images_list[idx]

            convert_tensor = ToTensor()

            ref_split = img_name.split("_")
            ref_split.insert(3, "mask")
            ref_name = "_".join(ref_split)

            img = convert_tensor(Image.open(os.path.join(self.dataset_folder, img_name)))
            refmask = convert_tensor(Image.open(os.path.join(self.dataset_folder, ref_name)))
        else:
            (img, refmask) = self.cache[idx]

        #Resize
        img, refmask = self.crop(img, refmask)

        #Flip if needed
        if self.randomly_flip_samples:
            img, refmask = self.randomly_flip(img, refmask)

        if self.randomly_rotate_samples:
            img, refmask = self.randomly_rotate(img, refmask)

        sample = {'img': img, 'ref': refmask, 'category': 'road', 'name': self.images_list[idx]}
        return sample

    #TODO document and maybe refactor so img loading logic has single point of truth
    #TODO make version of dataset that contains pre-cropped/scaled images to speed things up
    def cache_images(self):
        cache = {}
        for idx in range(len(self)):
            print(f"Caching img with idx {idx}....out of {len(self)}")
            img_name = self.images_list[idx]

            convert_tensor = ToTensor()

            ref_split = img_name.split("_")
            ref_split.insert(3, "mask")
            ref_name = "_".join(ref_split)

            img = convert_tensor(Image.open(os.path.join(self.dataset_folder, img_name)))
            refmask = convert_tensor(Image.open(os.path.join(self.dataset_folder, ref_name)))

            #img, refmask = self.crop(img, refmask)
            cache[idx] = (img, refmask)
        return cache

    def cache_samples(self, cache):
        samples = {} 
        for idx in range(len(self)):
            (img, refmask) = cache[idx]
            sample = {'img': img, 'ref': refmask}
            samples[idx] = sample
        return samples



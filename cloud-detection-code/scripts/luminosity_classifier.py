import numpy as np
import torch
import matplotlib.pyplot as plt
from bisect import bisect
import pickle

class LuminosityClassifier():
    
    def __init__(self, train_set, test_set=None):
        self.training_dataset = train_set
        #Validation set is optional
        if test_set != None:
            self.validation_dataset = test_set
        self.cloud_luminosities = {}
        self.non_cloud_luminosities = {}
        self.thresholds = {}
        self.reverse_keys = None 
    
    def train(self):
        """
        Trains the classifier by finding the optimal thresholds for each band.
        """
        reverse_keys = self.reverse_keys
        self.unpack_pixels(self.training_dataset)
        for key in self.cloud_luminosities:
            if key in reverse_keys: #Cloud/road if below threshold instead of above.
                thold = self.get_thresholds(self.non_cloud_luminosities[key], self.cloud_luminosities[key])
            else:
                thold = self.get_thresholds(self.cloud_luminosities[key], self.non_cloud_luminosities[key])
            self.thresholds[key] = thold[0]
        self.reverse_keys = reverse_keys

    def save_thresholds(self, save_fname, cloud_lum_fname, noncloud_lum_fname):
        """
        Saves thresholds to a file.

        Arguments 
        -----------
            save_fname: str
                full pathname to saved luminosity classifier
            lum_fn
        """
        with open(save_fname, 'wb') as f:
            pickle.dump(self.thresholds, f)
        with open(cloud_lum_fname, 'wb') as f:
            pickle.dump(self.cloud_luminosities, f)
        with open(noncloud_lum_fname, 'wb') as f:
            pickle.dump(self.non_cloud_luminosities, f)

    def load_thresholds(self, load_fname):
        """
        Loads thresholds from a file.

        Arguments 
        ----------
            load_fname: str
                full pathname to pickled luminosity classifier
        """
        with open(load_fname, 'rb') as f:
            self.thresholds = pickle.load(f)

    def classify(self, img, thresholds=None, reverse_keys=None, plot_on=False):
        """
        Creates a cloud mask based on the thresholds.

        Arguments
        -----------
            img: torch.Tensor
                image to classify
            thresholds: dict
                dictionary of thresholds for each band
            reverse_keys: tuple
                tuple of keys for which the cloud/road classification is reversed
            plot_on: bool
                whether to plot the image and mask

        Returns
        -----------
            mask: torch.Tensor
                cloud mask
        """
        if thresholds == None:
            thresholds = self.thresholds
        if reverse_keys == None:
            reverse_keys = self.reverse_keys
        img_shape = img.shape
        mask = np.zeros((img_shape[0], img_shape[2], img_shape[3]))
        for band, threshold in thresholds.items():
            if band in reverse_keys:
                band_mask = img[:, band, :, :].numpy() < threshold
                mask += band_mask
            else:
                band_mask = img[:, band, :, :].numpy() > threshold
                mask += band_mask
        mask = mask/len(thresholds)
        if plot_on:
            plt.imshow(mask[0, :, :])
            plt.show()
        return torch.Tensor(mask.reshape((img_shape[0], 1, img_shape[2], img_shape[3])))

    def unpack_pixels(self, dataset):
        """
        Unpacks all pixels from a dataset into a dictionary of cloud and non-cloud pixel luminosities for each band.

        Arguments
        -----------
            dataset: torch.utils.data.DataLoader
                DataLoader object containing images and masks
        """
        num_img = len(dataset)
        for img in dataset:
            cloud_luminosities, non_cloud_luminosities = self.extract_pixel_luminosities(img)
            for key in cloud_luminosities:
                if key in self.cloud_luminosities:
                    self.cloud_luminosities[key] = np.append(self.cloud_luminosities[key], cloud_luminosities[key])
                else:
                    self.cloud_luminosities[key] = cloud_luminosities[key]
            for key in non_cloud_luminosities:
                if key in self.non_cloud_luminosities:
                    self.non_cloud_luminosities[key] = np.append(self.non_cloud_luminosities[key], non_cloud_luminosities[key])
                else:
                    self.non_cloud_luminosities[key] = non_cloud_luminosities[key]

    def plot_pixel_luminosity_histogram(self, class_label='cloud', non_class_label='non-cloud', bands=(0,1,2,3,4), band_labels=('(a) Red', '(b) Green', '(c) Blue', '(d) LWIR', '(e) SWIR'), threshold=False):
        """
        Plots a histogram of pixel luminosities for each band.

        Arguments
        -----------
            class_label: str
                label for the class of pixels
            non_class_label: str
                label for the non-class of pixels
            bands: tuple
                tuple of bands to plot
            band_labels: tuple
                tuple of labels for each band
        """
        #Train if needed
        if len(self.cloud_luminosities) == 0:
            self.train()

        #Plot histograms
        plt.figure()
        if len(bands) == 5:
            ax1 = plt.subplot2grid(shape=(2,6), loc=(0,0), colspan=2)
            ax2 = plt.subplot2grid((2,6), (0,2), colspan=2)
            ax3 = plt.subplot2grid((2,6), (0,4), colspan=2)
            ax4 = plt.subplot2grid((2,6), (1,1), colspan=2)
            ax5 = plt.subplot2grid((2,6), (1,3), colspan=2)
            plt.subplots_adjust(hspace=0.5, wspace=1)
            axes = [ax1, ax2, ax3, ax4, ax5]
        if len(bands) == 3:
            ax1 = plt.subplot2grid(shape=(2,4), loc=(0,0), colspan=2)
            ax2 = plt.subplot2grid((2,4), (0,2), colspan=2)
            ax3 = plt.subplot2grid((2,4), (1,1), colspan=2)
            plt.subplots_adjust(hspace=0.5, wspace=1)
            axes = [ax1, ax2, ax3]
        for i, band in enumerate(bands):
            axes[i].hist(self.non_cloud_luminosities[band], bins=100, label=non_class_label, alpha=0.8)
            axes[i].hist(self.cloud_luminosities[band], bins=100, label=class_label, alpha=0.8)
            if threshold:
                axes[i].axvline(x=self.thresholds[band], color='r', linestyle='--', label='luminosity threshold')
            axes[i].legend()
            axes[i].set_title(f'{band_labels[i]}')
            axes[i].set_xlabel('Pixel luminosity (normalized)')
            axes[i].set_ylabel('Frequency')
        plt.show()
        
    def extract_pixel_luminosities(self, img):
        """
        Vectorized function to extract luminosities of cloud and non-cloud pixels for an image or batch of images.

        Arguments 
        -----------
            img: dict 
                dict with img with dimensions batch_size x bands x height x width and mask with dimensions 
                 batch_size x 1 x height x width

        Returns
        -----------
            cloud_luminosities: dict 
                dict with cloud luminosities for each band
            non_cloud_luminosities: dict
                dict with non-cloud luminosities for each band
        """
        #Loop over all pixels in image in r, g, b channels and add each pixel's luminosity to the appropriate lists
        #based on whether or not it's a cloud
        cloud_luminosities = {}
        noncloud_luminosities = {}

        for band in range(img['img'].size()[1]):
            cloud_luminosities[band] = np.array([])
            noncloud_luminosities[band] = np.array([])
        

        for band in range(img['img'].size()[1]):
            cloud_lums = np.where(img['ref'][:, 0, :, :] == 1, img['img'][:, band, :, :], -1).flatten()
            cloud_lums = cloud_lums[cloud_lums != -1]
            noncloud_lums = np.where(img['ref'][:, 0, :, :] == 0, img['img'][:, band, :, :], -1).flatten()
            noncloud_lums = noncloud_lums[noncloud_lums != -1]

            cloud_luminosities[band] = np.append(cloud_luminosities[band], cloud_lums)
            noncloud_luminosities[band] = np.append(noncloud_luminosities[band], noncloud_lums)
                  
        return cloud_luminosities, noncloud_luminosities
    
    def test_threshold(self, cloud, non_cloud, thold):
        """
        Tests a threshold on a set of cloud and non-cloud pixel luminosities. Considers a pixel to be 
        correctly classified if it's cloud and above the thold, or if it's non-cloud and below the thold.

        Arguments
        -----------
            cloud: np.array
                array of cloud pixel luminosities
            non_cloud: np.array
                array of non-cloud pixel luminosities
            thold: float
                threshold to test

        Returns
        -----------
            accuracy: int 
                number of correctly classified pixels
        """
        cloud_below = bisect(cloud, thold)
        cloud_above = len(cloud)-cloud_below
        noncloud_below = bisect(non_cloud, thold)
        noncloud_above = len(non_cloud) - noncloud_below
        return cloud_above + noncloud_below

    def get_thresholds(self, cloud_luminosity, noncloud_luminosity):
        """
        Finds the optimal threshold for a set of cloud and non-cloud pixel luminosities, such that the number 
        of cloud pixels above the threshold and the number of non-cloud pixels below the threshold are maximized.

        Arguments
        -----------
            cloud_luminosity: np.array
                array of cloud pixel luminosities
            noncloud_luminosity: np.array
                array of non-cloud pixel luminosities

        Returns
        -----------
            output: tuple 
                tuple of (threshold, number of px correctly classified)  
        """
        cloud_luminosity.sort()
        noncloud_luminosity.sort()
        threshold_set = set()
        for num in cloud_luminosity:
            threshold_set.add(num)
        for num in noncloud_luminosity:
            threshold_set.add(num)
        distinct_sorted_thresholds = list(threshold_set)
        distinct_sorted_thresholds.sort()
        best_i = -1
        best_thold = 0
        best_correct_ids = 0
        for i, thold in enumerate(distinct_sorted_thresholds):
            correct_ids = self.test_threshold(cloud_luminosity, noncloud_luminosity, thold)
            if correct_ids > best_correct_ids:
                best_i = i 
                best_thold = thold
                best_correct_ids = correct_ids
        #Try intermediate thresholds
        if best_i < (len(distinct_sorted_thresholds)-1):
            int_above = (best_thold + distinct_sorted_thresholds[best_i+1])/2
            above_correct = self.test_threshold(cloud_luminosity, noncloud_luminosity, int_above)
        else:
            above_correct = -1
        if best_i > 0:
            int_below = (best_thold + distinct_sorted_thresholds[best_i-1])/2
            below_correct = self.test_threshold(cloud_luminosity, noncloud_luminosity, int_below)
        else:
            below_correct = -1
        best_ids = max(best_correct_ids, above_correct, below_correct)
        if best_ids == best_correct_ids:
            return (best_thold, best_correct_ids)
        elif best_ids == above_correct:
            return (int_above, best_ids)
        else:
            return (int_below, best_ids)



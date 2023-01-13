import numpy as np
import math
import itertools
import torch
import pickle
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

class RandomForest():
    """
    Wrapper around sklearn.ensemble.RandomForestClassifier that quickly
    loads a PyTorch dataset and reshapes it to be fed to a random forest model,
    and can also train and validate a random forest model.
    """
    def __init__(self, train_set, kernel_dims=3, tf=np.zeros([0]), tmf=np.zeros([0]), clf=RandomForestClassifier(max_depth=25, n_estimators=25), test_set=None):
        self.training_dataset = train_set
        self.kernel_dims = kernel_dims
        self.rf = clf
        self.training_features = tf
        self.training_mask_features = tmf

    def save_classifier(self, clf_fname):
        """
        Saves RandomForestClassifier in file.

        Arguments 
        ----------
            clf_fname: str
                full pathname to save pickled RandomForestClassifier
        """
        with open(clf_fname, 'wb') as f:
            pickle.dump(self.rf, f)

    def load_classifier(self, new_clf_fname):
        """
        Loads RandomForestClassifier from file.

        Arguments 
        -----------
            new_clf_fname: str
                full pathname to pickled RandomForestClassifier
        """
        with open(new_clf_fname, 'rb') as f:
            self.rf = pickle.load(f)
        
    def extract_features(self):
        """
        Extracts features for all images in training dataset in 
        preparation for training random forest.
        """
        X_rf, y_rf = None, None
        x_fill = False
        for img in self.training_dataset:
            _, X, y = self.extract_pixel_features(img)
            if x_fill == False:
                X_rf = X
                y_rf = y
                x_fill = True
            else:
                X_rf = np.concatenate((X_rf, X))
                y_rf = np.concatenate((y_rf, y))
        self.training_features = X_rf
        self.training_mask_features = y_rf
        
    def train(self, max_depth=25, n_estimators=25, retrain=False):
        """
        Trains random forest classifier.
        """
        if retrain:
            clf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators)
            self.rf = clf
        self.rf.fit(self.training_features, self.training_mask_features)
    
    
    def classify(self, img, plot_on=False):
        """
        Classifies an image or images and returns a probability mask.

        Arguments 
        -----------
            img: dictionary 
                cloud or road sample

        Returns 
        ---------
            proba_img: torch.Tensor
                predicted probability mask for img
        """
        #dims, X, y = self.extract_pixel_features(img, self.kernel_dims)
        X = self.extract_image_features(img, self.kernel_dims)
        dims = img.shape
        predict_probs = self.rf.predict_proba(X)
        proba_img = 1-predict_probs[:, 0].reshape(dims[0], dims[2], dims[3])

        #Plot if necessary
        if plot_on:
            #Reference mask
            plt.figure()
            plt.imshow(img['ref'][0, 0, :, :])

            #Reconstructed reference mask from y
            y_reconstructed = y.reshape(dims[0], dims[2], dims[3])
            plt.figure()
            plt.imshow(y_reconstructed[0])

            #Predicted mask
            predictions = self.rf.predict(X)
            predictions_img = predictions.reshape(dims[0], dims[2], dims[3])
            plt.figure()
            plt.imshow(predictions_img[0])

            #Predicted probability map
            plt.figure()
            plt.imshow(proba_img[0])
            plt.show()

        return F.to_tensor(proba_img).reshape((dims[0], 1, dims[2], dims[3])) 

    def shift_tensor(self, image, up, left):
        """
        Shifts a tensor up by up pixels and left by left pixels.
        Up and left can be negative to shift down and left.

        Arguments
        ----------
            image: torch.Tensor
                image to shift
            up: int
                number of px to shift up (can be -, 0, or +)
            left: int 
                number of px to shift left (can be -, 0, or +)

        Returns 
        --------
            shifted_img: torch.Tensor
                shifted image
        """
        max_pad = max(abs(up), abs(left))
        if max_pad > 0:
            pad_transform = torch.nn.ReplicationPad2d(max_pad)
            rolled_img = pad_transform(image).roll(-up, 2).roll(-left, 3)
            return rolled_img[:, :, max_pad:-(max_pad), max_pad:-(max_pad)]
        else:
            return image
        
        
    def extract_pixel_features(self, img, kernel_dims=3):
        """
        Extracts pixel features for an image + mask.

        Arguments 
        ----------
            image: torch.Tensor
                image to classify of size n x b x w x h, where n is batch size,
                b is number of bands, w is width and h is height
            kernel_dims: int
                optional size of kernel (default=3)

        Returns 
        --------
            dims: list
                list [n, 1, w, h] giving dimensions of the mask so that random forest
                outputs can be properly reconstructed later
            X: np.array
                np array of size n*w*h x b*kernel_dims**2, where first dimension represents
                pixel and 2nd dimension represents features for that pixel

            y: np.array
                np array of size n*w*h, representing expected output for all pixels

        """
        image = img['img']
        mask = img['ref']
        X = self.extract_image_features(image, kernel_dims=kernel_dims)
        dims, y = self.extract_mask_features(mask)
        return dims, X, y

    def extract_image_features(self, image, kernel_dims=3):
        """
        Extracts pixel features for an image with no mask.

        Arguments 
        ----------
            image: torch.Tensor
                image to classify of size n x b x w x h, where n is batch size,
                b is number of bands, w is width and h is height
            kernel_dims: int
                optional size of kernel (default=3)

        Returns 
        --------
            X: np.array
                np array of size n*w*h x b*kernel_dims**2, where first dimension represents
                pixel and 2nd dimension represents features for that pixel
        """
        kdims = [i for i in range(math.ceil(-kernel_dims/2), math.floor(kernel_dims/2) + 1)]
        kdims_combos = ((a,b) for (a, b) in itertools.product(kdims, kdims))
        img_stack_kernel = torch.stack([self.shift_tensor(image, a, b) for (a, b) in kdims_combos])

        (n, b, w, h) = image.shape
        X = img_stack_kernel.permute(0, 2, 1, 3, 4).T.reshape(n*w*h, b*kernel_dims**2)
        return X.numpy()

    def extract_mask_features(self, mask):
        """
        Extracts pixel features for an image mask.

        Arguments 
        ----------
            mask: torch.Tensor
                mask to classify of size n x 1 x w x h, where n is batch size,
                w is width and h is height

        Returns 
        --------
            dims: list
                list [n, 1, w, h] giving dimensions of the mask so that random forest
                outputs can be properly reconstructed later
            y: np.array
                np array of size n*w*h, representing expected output for all pixels
        """
        dims = list(mask.size())

        (n, _, w, h) = mask.shape
        y = mask.T.reshape(n*w*h)
        return dims, y.numpy().astype(int)



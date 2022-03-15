from feature import RectangleRegion, HaarFeature
from classifier import WeakClassifier
import utils
import numpy as np
import math
from sklearn.feature_selection import SelectPercentile, f_classif
import pickle

class Adaboost:
    def __init__(self, T = 10):
        """
          Parameters:
            T: The number of weak classifiers which should be used.
        """
        self.T = T
        self.alphas = []
        self.clfs = []

    def train(self, dataset):
        """
        Trains the Viola Jones classifier on a set of images.
          Parameters:
            dataset: A list of tuples. The first element is the numpy 
              array with shape (m, n) representing the image. The second
              element is its classification (1 or 0).
        """
        print("Computing integral images")
        posNum, negNum = 0, 0
        iis, labels = [], []
        for i in range(len(dataset)):
            iis.append(utils.integralImage(dataset[i][0]))
            labels.append(dataset[i][1])
            if dataset[i][1] == 1:
                posNum += 1
            else:
                negNum += 1
        print("Building features")
        print(iis[0].shape)
        features = self.buildFeatures(iis[0].shape)
        print("Applying features to dataset")
        featureVals = self.applyFeatures(features, iis)
        print("Selecting best features")
        indices = SelectPercentile(f_classif, percentile=10).fit(featureVals.T, labels).get_support(indices=True)
        featureVals = featureVals[indices]
        features = features[indices]
        print("Selected %d potential features" % len(featureVals))
        
        print("Initialize weights")
        weights = np.zeros(len(dataset))
        for i in range(len(dataset)):
            if labels[i] == 1:
                weights[i] = 1.0 / (2 * posNum)
            else:
                weights[i] = 1.0 / (2 * negNum)
        for t in range(self.T):
            print("Run No. of Iteration: %d" % (t+1))
            # Normalize weights
            weights = weights / np.linalg.norm(weights)
            # Compute error and select best classifiers
            clf, error = self.selectBest(featureVals, iis, labels, features, weights)
            #update weights
            accuracy = []
            for x, y in zip(iis, labels):
                correctness = abs(clf.classify(x) - y)
                accuracy.append(correctness)
            beta = error / (1.0 - error)
            for i in range(len(accuracy)):
                weights[i] = weights[i] * (beta ** (1 - accuracy[i]))
            alpha = math.log(1.0/beta)
            self.alphas.append(alpha)
            self.clfs.append(clf)
            print("Chose classifier: %s with accuracy: %f and alpha: %f" % (str(clf), len(accuracy) - sum(accuracy), alpha))
    
    def buildFeatures(self, imageShape):
        """
        Builds the possible features given an image shape.
          Parameters:
            imageShape: A tuple of form (height, width).
          Returns:
            A numpy array of HaarFeature class.
        """
        height, width = imageShape
        features = []
        for w in range(1, width+1):
            for h in range(1, height+1):
                i = 0
                while i + w < width:
                    j = 0
                    while j + h < height:
                        #2 rectangle features
                        immediate = RectangleRegion(i, j, w, h)
                        right = RectangleRegion(i+w, j, w, h)
                        if i + 2 * w < width: #Horizontally Adjacent
                            features.append(HaarFeature([right], [immediate]))

                        bottom = RectangleRegion(i, j+h, w, h)
                        if j + 2 * h < height: #Vertically Adjacent
                            features.append(HaarFeature([immediate], [bottom]))
                        
                        right_2 = RectangleRegion(i+2*w, j, w, h)
                        #3 rectangle features
                        if i + 3 * w < width: #Horizontally Adjacent
                            features.append(HaarFeature([right], [right_2, immediate]))

                        bottom_2 = RectangleRegion(i, j+2*h, w, h)
                        if j + 3 * h < height: #Vertically Adjacent
                            features.append(HaarFeature([bottom], [bottom_2, immediate]))

                        #4 rectangle features
                        bottom_right = RectangleRegion(i+w, j+h, w, h)
                        if i + 2 * w < width and j + 2 * h < height:
                            features.append(HaarFeature([right, bottom], [immediate, bottom_right]))

                        j += 1
                    i += 1
        return np.array(features)
    
    def applyFeatures(self, features, iis):
        """
        Maps features onto the training dataset.
          Parameters:
            features: A numpy array of HaarFeature class.
            iis: A list of numpy array with shape (m, n) representing the integral images.
          Returns:
            featureVals: A numpy array of shape (len(features), len(dataset)).
              Each row represents the values of a single feature for each training sample.
        """
        featureVals = np.zeros((len(features), len(iis)))
        for j in range(len(features)):
            for i in range(len(iis)):
                featureVals[j, i] = features[j].computeFeature(iis[i])
        return featureVals
    
    def selectBest(self, featureVals, iis, labels, features, weights):
        """
        Finds the appropriate weak classifier for each feature.
        Selects the best weak classifier for the given weights.
          Parameters:
            featureVals: A numpy array of shape (len(features), len(dataset)).
              Each row represents the values of a single feature for each training sample.
            iis: A list of numpy array with shape (m, n) representing the integral images.
            labels: A list of integer.
              The ith element is the classification of the ith training sample.
            features: A numpy array of HaarFeature class.
            weights: A numpy array with shape(len(dataset)).
              The ith element is the weight assigned to the ith training sample.
          Returns:
            bestClf: The best WeakClassifier Class
            bestError: The error of the best classifer
        """
        # Begin your code (Part 2)
        # raise NotImplementedError("To be implemented")

        print("featureVals", featureVals)
        print("========================")
        # print("iis", iis)
        # print("========================")
        # print("labels", labels)
        # print("========================")
        # print("features", features)
        # print("========================")
        # print("weights", weights)


      # featureVals [[ -256.   -97.   -67. ...   -95.  -171.  -128.]
      # [ -227.   -96.   -95. ...   -97.  -210.  -123.]
      # [ -211.   -97.  -121. ...   -99.  -108.  -127.]
      # ...
      # [-1208.  2710. -2588. ...   128.  1495.   303.]
      # [ 3015.  2296.  -359. ...   344.  3668.   354.]
      # [-1947.  3281. -3794. ...   169.  1763.    83.]]
      # ========================
      #   iis
      #   [ 1760.,  3506.,  5249.,  7008.,  8741., 10474., 12216., 13957.,
      #   15661., 17380., 19107., 20827., 22513., 24211., 25899., 27579.,
      #   29228., 30885., 32572., 34232., 35869., 37530., 39215., 40890.,
      #   42540., 44188., 45857., 47493., 49097., 50742., 52407., 54057.,
      #   55452., 57757., 60737., 62508.],
      #  [ 1888.,  3761.,  5631.,  7517.,  9377., 11238., 13105., 14968.,
      #   16793., 18631., 20477., 22317., 24125., 25945., 27752., 29552.,
      #   31320., 33095., 34902., 36682., 38438., 40220., 42024., 43816.,
      #   45584., 47351., 49139., 50892., 52614., 54379., 56165., 57941.,
      #   59412., 61926., 65101., 67006.],
      #  [ 2021.,  4025.,  6025.,  8044., 10036., 12026., 14019., 16009.,
      #   17960., 19921., 21891., 23856., 25789., 27736., 29669., 31593.,
      #   33487., 35386., 37314., 39216., 41098., 43005., 44934., 46848.,
      #   48739., 50629., 52539., 54412., 56254., 58141., 60048., 61950.,
      #   63504., 66231., 69603., 71629.],
      #  [ 2157.,  4293.,  6430.,  8585., 10715., 12834., 14957., 17076.,
      #   19155., 21245., 23344., 25438., 27499., 29573., 31632., 33681.,
      #   35700., 37724., 39776., 41804., 43815., 45851., 47906., 49947.,
      #   51967., 53986., 56021., 58018., 59983., 61996., 64031., 66061.,
      #   67730., 70673., 74241., 76372.]])]
      #   ========================
      #   labels [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
      #   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 
      #   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
      #   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
      #   ========================
      #   features [<feature.HaarFeature object at 0x0000025861D9FE50>
      #   <feature.HaarFeature object at 0x0000025861DA16D0>
      #   <feature.HaarFeature object at 0x0000025861DA8610> ...
      #   <feature.HaarFeature object at 0x000002586A04C4F0>
      #   <feature.HaarFeature object at 0x000002586A04C610>
      #   <feature.HaarFeature object at 0x000002586A04CA90>]
      #   ========================
      #   weights [0.04082483 0.04082483 0.04082483 0.04082483 0.04082483 0.04082483
      #   0.04082483 0.04082483 0.04082483 0.04082483 0.04082483 0.04082483
      #   0.04082483 0.04082483 0.04082483 0.04082483 0.04082483 0.04082483
      #   0.04082483 0.04082483 0.04082483 0.04082483 0.04082483 0.04082483
      #   0.04082483 0.04082483 0.04082483 0.04082483 0.04082483 0.04082483
      #   0.04082483 0.04082483 0.04082483 0.04082483 0.04082483 0.04082483
      #   0.04082483 0.04082483 0.04082483 0.04082483 0.04082483 0.04082483
      #   0.04082483 0.04082483 0.04082483 0.04082483 0.04082483 0.04082483
      #   0.04082483 0.04082483 0.04082483 0.04082483 0.04082483 0.04082483
      #   0.04082483 0.04082483 0.04082483 0.04082483 0.04082483 0.04082483
      #   0.04082483 0.04082483 0.04082483 0.04082483 0.04082483 0.04082483
      #   0.04082483 0.04082483 0.04082483 0.04082483 0.04082483 0.04082483
      #   0.04082483 0.04082483 0.04082483 0.04082483 0.04082483 0.04082483
      #   0.04082483 0.04082483 0.04082483 0.04082483 0.04082483 0.04082483
      #   0.04082483 0.04082483 0.04082483 0.04082483 0.04082483 0.04082483
      #   0.04082483 0.04082483 0.04082483 0.04082483 0.04082483 0.04082483
      #   0.04082483 0.04082483 0.04082483 0.04082483 0.04082483 0.04082483
      #   0.04082483 0.04082483 0.04082483 0.04082483 0.04082483 0.04082483
      #   0.04082483 0.04082483 0.04082483 0.04082483 0.04082483 0.04082483
      #   0.04082483 0.04082483 0.04082483 0.04082483 0.04082483 0.04082483
      #   0.04082483 0.04082483 0.04082483 0.04082483 0.04082483 0.04082483
      #   0.04082483 0.04082483 0.04082483 0.04082483 0.04082483 0.04082483
      #   0.04082483 0.04082483 0.04082483 0.04082483 0.04082483 0.04082483
      #   0.04082483 0.04082483 0.04082483 0.04082483 0.04082483 0.04082483
      #   0.04082483 0.04082483 0.04082483 0.04082483 0.04082483 0.04082483
      #   0.04082483 0.04082483 0.04082483 0.04082483 0.04082483 0.04082483
      #   0.04082483 0.04082483 0.04082483 0.04082483 0.04082483 0.04082483
      #   0.04082483 0.04082483 0.04082483 0.04082483 0.04082483 0.04082483
      #   0.04082483 0.04082483 0.04082483 0.04082483 0.04082483 0.04082483
      #   0.04082483 0.04082483 0.04082483 0.04082483 0.04082483 0.04082483
      #   0.04082483 0.04082483 0.04082483 0.04082483 0.04082483 0.04082483
      #   0.04082483 0.04082483 0.04082483 0.04082483 0.04082483 0.04082483
      #   0.04082483 0.04082483 0.04082483 0.04082483 0.04082483 0.04082483
      #   0.04082483 0.04082483 0.04082483 0.04082483 0.04082483 0.04082483
      #   0.04082483 0.04082483 0.04082483 0.04082483 0.04082483 0.04082483
      #   0.04082483 0.04082483 0.04082483 0.04082483 0.04082483 0.04082483
      #   0.04082483 0.04082483 0.04082483 0.04082483 0.04082483 0.04082483
      #   0.04082483 0.04082483 0.04082483 0.04082483 0.04082483 0.04082483
      #   0.04082483 0.04082483 0.04082483 0.04082483 0.04082483 0.04082483
      #   0.04082483 0.04082483 0.04082483 0.04082483 0.04082483 0.04082483
      #   0.04082483 0.04082483 0.04082483 0.04082483 0.04082483 0.04082483
      #   0.04082483 0.04082483 0.04082483 0.04082483 0.04082483 0.04082483
      #   0.04082483 0.04082483 0.04082483 0.04082483 0.04082483 0.04082483
      #   0.04082483 0.04082483 0.04082483 0.04082483 0.04082483 0.04082483
      #   0.04082483 0.04082483 0.04082483 0.04082483 0.04082483 0.04082483
      #   0.04082483 0.04082483 0.04082483 0.04082483 0.04082483 0.04082483
      #   0.04082483 0.04082483 0.04082483 0.04082483 0.04082483 0.04082483
      #   0.04082483 0.04082483 0.04082483 0.04082483 0.04082483 0.04082483
      #   0.04082483 0.04082483 0.04082483 0.04082483 0.04082483 0.04082483
      #   0.04082483 0.04082483 0.04082483 0.04082483 0.04082483 0.04082483
      #   0.04082483 0.04082483 0.04082483 0.04082483 0.04082483 0.04082483
      #   0.04082483 0.04082483 0.04082483 0.04082483 0.04082483 0.04082483
      #   0.04082483 0.04082483 0.04082483 0.04082483 0.04082483 0.04082483
      #   0.04082483 0.04082483 0.04082483 0.04082483 0.04082483 0.04082483
      #   0.04082483 0.04082483 0.04082483 0.04082483 0.04082483 0.04082483
      #   0.04082483 0.04082483 0.04082483 0.04082483 0.04082483 0.04082483
      #   0.04082483 0.04082483 0.04082483 0.04082483 0.04082483 0.04082483
      #   0.04082483 0.04082483 0.04082483 0.04082483 0.04082483 0.04082483
      #   0.04082483 0.04082483 0.04082483 0.04082483 0.04082483 0.04082483
      #   0.04082483 0.04082483 0.04082483 0.04082483 0.04082483 0.04082483
      #   0.04082483 0.04082483 0.04082483 0.04082483 0.04082483 0.04082483
      #   0.04082483 0.04082483 0.04082483 0.04082483 0.04082483 0.04082483
      #   0.04082483 0.04082483 0.04082483 0.04082483 0.04082483 0.04082483
      #   0.04082483 0.04082483 0.04082483 0.04082483 0.04082483 0.04082483
      #   0.04082483 0.04082483 0.04082483 0.04082483 0.04082483 0.04082483
      #   0.04082483 0.04082483 0.04082483 0.04082483 0.04082483 0.04082483
      #   0.04082483 0.04082483 0.04082483 0.04082483 0.04082483 0.04082483
      #   0.04082483 0.04082483 0.04082483 0.04082483 0.04082483 0.04082483
      #   0.04082483 0.04082483 0.04082483 0.04082483 0.04082483 0.04082483
      #   0.04082483 0.04082483 0.04082483 0.04082483 0.04082483 0.04082483
      #   0.04082483 0.04082483 0.04082483 0.04082483 0.04082483 0.04082483
      #   0.04082483 0.04082483 0.04082483 0.04082483 0.04082483 0.04082483
      #   0.04082483 0.04082483 0.04082483 0.04082483 0.04082483 0.04082483
      #   0.04082483 0.04082483 0.04082483 0.04082483 0.04082483 0.04082483
      #   0.04082483 0.04082483 0.04082483 0.04082483 0.04082483 0.04082483
      #   0.04082483 0.04082483 0.04082483 0.04082483 0.04082483 0.04082483
      #   0.04082483 0.04082483 0.04082483 0.04082483 0.04082483 0.04082483
      #   0.04082483 0.04082483 0.04082483 0.04082483 0.04082483 0.04082483
      #   0.04082483 0.04082483 0.04082483 0.04082483 0.04082483 0.04082483
      #   0.04082483 0.04082483 0.04082483 0.04082483 0.04082483 0.04082483
      #   0.04082483 0.04082483 0.04082483 0.04082483 0.04082483 0.04082483
      #   0.04082483 0.04082483 0.04082483 0.04082483 0.04082483 0.04082483
      #   0.04082483 0.04082483 0.04082483 0.04082483 0.04082483 0.04082483
      #   0.04082483 0.04082483 0.04082483 0.04082483 0.04082483 0.04082483
      #   0.04082483 0.04082483 0.04082483 0.04082483 0.04082483 0.04082483
      #   0.04082483 0.04082483 0.04082483 0.04082483 0.04082483 0.04082483
      #   0.04082483 0.04082483 0.04082483 0.04082483 0.04082483 0.04082483
      #   0.04082483 0.04082483 0.04082483 0.04082483 0.04082483 0.04082483
      #   0.04082483 0.04082483 0.04082483 0.04082483 0.04082483 0.04082483
      #   0.04082483 0.04082483 0.04082483 0.04082483 0.04082483 0.04082483
      #   0.04082483 0.04082483 0.04082483 0.04082483 0.04082483 0.04082483
      #   0.04082483 0.04082483 0.04082483 0.04082483 0.04082483 0.04082483
      #   0.04082483 0.04082483 0.04082483 0.04082483 0.04082483 0.04082483
      #   0.04082483 0.04082483 0.04082483 0.04082483 0.04082483 0.04082483
      #   0.04082483 0.04082483 0.04082483 0.04082483 0.04082483 0.04082483
      #   0.04082483 0.04082483 0.04082483 0.04082483 0.04082483 0.04082483
      #   0.04082483 0.04082483 0.04082483 0.04082483 0.04082483 0.04082483
      #   0.04082483 0.04082483 0.04082483 0.04082483 0.04082483 0.04082483
      #   0.04082483 0.04082483 0.04082483 0.04082483 0.04082483 0.04082483
      #   0.04082483 0.04082483 0.04082483 0.04082483 0.04082483 0.04082483]

        # End your code (Part 2)
        return bestClf, bestError
    
    def classify(self, image):
        """
        Classifies an image
          Parameters:
            image: A numpy array with shape (m, n). The shape (m, n) must be
              the same with the shape of training images.
          Returns:
            1 if the image is positively classified and 0 otherwise
        """
        total = 0
        ii = utils.integralImage(image)
        for alpha, clf in zip(self.alphas, self.clfs):
            total += alpha * clf.classify(ii)
        return 1 if total >= 0.5 * sum(self.alphas) else 0
    
    def save(self, filename):
        """
        Saves the classifier to a pickle
          Args:
            filename: The name of the file (no file extension necessary)
        """
        with open(filename+".pkl", 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """
        A static method which loads the classifier from a pickle
          Args:
            filename: The name of the file (no file extension necessary)
        """
        with open(filename+".pkl", 'rb') as f:
            return pickle.load(f)
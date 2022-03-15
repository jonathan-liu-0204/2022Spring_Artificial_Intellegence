import os
import cv2
import numpy as np

def loadImages(dataPath):
    """
    Load all Images in the folder and transfer a list of tuples. 
    The first element is the numpy array of shape (m, n) representing the image.
    (remember to resize and convert the parking space images to 36 x 16 grayscale images.) 
    The second element is its classification (1 or 0)
      Parameters:
        dataPath: The folder path.
      Returns:
        dataset: The list of tuples.
    """
    # Begin your code (Part 1)
    # raise NotImplementedError("To be implemented")

    dataset = []

    for foldername in os.listdir(dataPath):
      if(foldername == "car"):
        np_label = np.array([1])
      elif(foldername == "non-car"):
        np_label = np.array([0])

      for filename in os.listdir(os.path.join(dataPath, foldername)):
          img = cv2.imread(os.path.join(dataPath, foldername, filename))
          if img is not None:
            height, width, channel = img.shape
            np_shape = np.array([height, width])
            np_img = np.array(img)
            np_data = np.array([np_img, np_shape, np_label], dtype=object)
            dataset.append(np_data)
            # print(np_data)

    # End your code (Part 1)
    
    return dataset

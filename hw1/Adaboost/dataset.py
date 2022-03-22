import os
import cv2

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
        label = 1
      elif(foldername == "non-car"):
        label = 0

      for filename in os.listdir(os.path.join(dataPath, foldername)):
          img = cv2.imread(os.path.join(dataPath, foldername, filename), 0)
          img = cv2.resize(img, (16, 36))
          tp_data = tuple()
          tp_data = (img, label)
          dataset.append(tp_data)
    
    # End your code (Part 1)
    
    return dataset

import os
from turtle import Turtle
import cv2
import matplotlib.pyplot as plt
import numpy as np
import utils
from os import walk
from os.path import join
from datetime import datetime

def crop(x1, y1, x2, y2, x3, y3, x4, y4, img) :
    """
    This function ouput the specified area (parking space image) of the input frame according to the input of four xy coordinates.
    
      Parameters:
        (x1, y1, x2, y2, x3, y3, x4, y4, frame)
        
        (x1, y1) is the lower left corner of the specified area
        (x2, y2) is the lower right corner of the specified area
        (x3, y3) is the upper left corner of the specified area
        (x4, y4) is the upper right corner of the specified area
        frame is the frame you want to get it's parking space image
        
      Returns:
        parking_space_image (image size = 360 x 160)
      
      Usage:
        parking_space_image = crop(x1, y1, x2, y2, x3, y3, x4, y4, img)
    """
    left_front = (x1, y1)
    right_front = (x2, y2)
    left_bottom = (x3, y3)
    right_bottom = (x4, y4)
    src_pts = np.array([left_front, right_front, left_bottom, right_bottom]).astype(np.float32)
    dst_pts = np.array([[0, 0], [0, 160], [360, 0], [360, 160]]).astype(np.float32)
    projective_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    croped = cv2.warpPerspective(img, projective_matrix, (360,160))
    return croped


def detect(dataPath, clf):
    """
    Please read detectData.txt to understand the format. 
    Use cv2.VideoCapture() to load the video.gif.
    Use crop() to crop each frame (frame size = 1280 x 800) of video to get parking space images. (image size = 360 x 160) 
    Convert each parking space image into 36 x 16 and grayscale.
    Use clf.classify() function to detect car, If the result is True, draw the green box on the image like the example provided on the spec. 
    Then, you have to show the first frame with the bounding boxes in your report.
    Save the predictions as .txt file (Adaboost_pred.txt), the format is the same as GroundTruth.txt. 
    (in order to draw the plot in Yolov5_sample_code.ipynb)
    
      Parameters:
        dataPath: the path of detectData.txt
      Returns:
        No returns.
    """
  
    # Begin your code (Part 4)

    with open(dataPath) as f:
      lines = f.readlines()

    cap = cv2.VideoCapture("data/detect/video.gif")

    now_at = 0

    while(cap.isOpened()):
      ret, frame = cap.read()

      tmp_classify_result = []

      if ret == True:

        now_at += 1
        print("we are now at frame no.", now_at)
        
        for i in range(int(lines[0])):
          tmp_object = list(map(int, lines[i+1].split(" ")))
          tmp = np.fromiter(tmp_object, dtype=np.int)

          cropped_frame = crop(tmp[0], tmp[1], tmp[2], tmp[3], tmp[4], tmp[5], tmp[6], tmp[7], frame)
          # cropped_frame = np.rot90(cropped_frame)
          cropped_frame = cv2.resize(cropped_frame, (36, 16))
          cropped_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)

          classify_result = clf.classify(cropped_frame)

          if classify_result == 1:
            tmp_classify_result.append(1)

            if now_at == 1:
              draw = np.array([[tmp[4], tmp[5]], [tmp[6], tmp[7]], [tmp[2], tmp[3]], [tmp[0], tmp[1]]])
              cv2.polylines(frame, [draw], True, (0,255,0), 2)
          else:
            tmp_classify_result.append(0)

        file = open("Adaboost_pred.txt", "w+")
        for i in range(len(tmp_classify_result)):
          file.write(str(tmp_classify_result[i]))
          if i != (len(tmp_classify_result)-1):
            file.write(" ")

        file.write("\n")
        file.close()

        if now_at == 1:
          frame = frame[:, :, [2,1,0]]
          plt.imshow(frame)
          plt.show()
          
      else:
        break;
    
    cap.release()

    # raise NotImplementedError("To be implemented")
    # End your code (Part 4)

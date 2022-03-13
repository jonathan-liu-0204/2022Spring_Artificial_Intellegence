import cv2
import numpy as np
from PIL import Image

image = cv2.imread("image.png")

with open("bounding_box.txt") as f:
    lines = f.readlines()

for i in lines:
    tmp_object = list(map(int, i.split(" ")))
    tmp = np.fromiter(tmp_object, dtype=np.int)

    cv2.rectangle(image, (tmp[0], tmp[1]), (tmp[2], tmp[3]), (0,0,255), 2)

cv2.imwrite("hw0_0716304_1.png",image)
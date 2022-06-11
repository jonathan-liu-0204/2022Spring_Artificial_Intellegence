import cv2
import numpy as np
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt

cap = cv2.VideoCapture("video.mp4")

count = 0
ret, frame = cap.read()
pre_frame = frame

while cap.isOpened():
    ret, frame = cap.read()

    # if frame is read correctly ret is True
    if not ret:
        break
    
    if(pre_frame is None): 
        pre_frame = frame

    img = cv2.absdiff(pre_frame, frame)
    # img = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # img_green = img[:, :, 0] = int(0)
    # img_green = img[:, :, 2] = int(0)

    # plt.imshow(img_green)

    pre_frame = frame

    # print(img.shape)
    #size = (2160, 3840)

    if count == 20:
        # Create output image
        output_image = Image.new("RGB", (2160, 3840))
        draw = ImageDraw.Draw(output_image)

        # Generate image
        for x in range(output_image.width):
            for y in range(output_image.height):
                # print(img[x, y])
                r, g, b = img[x, y]
                r = int(0)
                g = int(g)
                b = int(0)
                draw.point((x, y), (r, g, b))

        output_image.save("hw0_0716304_2.png")

        img = cv2.imread('hw0_0716304_2.png')
        img_rotate_90_clockwise = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

        final_image = np.concatenate((frame, img_rotate_90_clockwise), axis = 1)
        cv2.imwrite('hw0_0716304_2.png', final_image)

        break

    count = count + 1
        
    # cv2.imwrite("hw0_0716304_2.png",img)
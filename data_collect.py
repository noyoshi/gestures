import cv2
import numpy as np
import copy
import math
import sys
import os

n = 0
if len(sys.argv) < 3:
    print("Please give the destination file and file template name next time") 
    sys.exit(1)

dest_dir, file_name = sys.argv[1:3]
dir_path = os.path.dirname(os.path.realpath(__file__))
if dest_dir[0] != "/":
    dest_dir = "/" + dest_dir

if dest_dir[-1] != "/":
    dest_dir += "/"

OUTPUT_TEMPLATE = dir_path + dest_dir + file_name + "-{}.jpg" 
print("Outputting to \n{}\n\tCorrect? [y/n]".format(OUTPUT_TEMPLATE))
x = input()
x = x.strip().lower()
if x != "y":
    sys.exit(1)

if len(sys.argv) > 3:
    n = int(sys.argv[3])

# Global Parameters
cap_region_x_begin = 0.5  	# Start Point / Total Width
cap_region_y_end = 0.8  	# Start Point / Total Height
threshold = 60  			# BINARY Threshold
blurValue = 41  			# GaussianBlur Parameter
bgSubThreshold = 50
learningRate = 0

# Variables
isBgCaptured = 0   			# Whether the Background is Captured

# Remove Background from Frame using Model Built by BackgroundSubtractorMOG2
def removeBG(frame):
    fgmask = bgModel.apply(frame,learningRate=learningRate)
    # Apply the Model to a Frame
    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)

    # Return Frame w BG Removed
    return res

# Camera
camera = cv2.VideoCapture(0)
camera.set(10,200)

while camera.isOpened():
    ret, frame = camera.read()
    frame = cv2.bilateralFilter(frame, 5, 50, 100)  # smoothing filter
    frame = cv2.flip(frame, 1)  # flip the frame horizontally
    cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),
                 (frame.shape[1], int(cap_region_y_end * frame.shape[0])), (255, 0, 0), 2)
    cv2.imshow('original', frame)

    #  Main Operation
    if isBgCaptured == 1:
    	# Remove Background + # Clip the ROI
        img = removeBG(frame)
        img = img[0:int(cap_region_y_end * frame.shape[0]),
                    int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]
        # Convert the Image into a Binary Image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)
        _ ,img_bw = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        cv2.imshow('blur', img_bw)

    # Keyboard Operations
    k = cv2.waitKey(10)
    if k == 27: # Press ESC to Exit
        break

    elif k == ord('b'):  # Press 'b' to Capture the Background
        bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)
        isBgCaptured = 1
        print( '* Background Captured')

    elif k == ord('x'):  # Press 'r' to Reset the Background
        cv2.imshow('got this', img_bw)
        img_path = OUTPUT_TEMPLATE.format(n)
        cv2.imwrite(img_path, img_bw)
        n += 1
        print ('* got image {}'.format(img_path))

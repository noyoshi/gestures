import cv2
import numpy as np
import copy
import math

# TODO get rid of the global variables, put this ito a class??

def printThreshold(thr):
    print("! Changed Threshold to " + str(thr))

def removeBG(f, bgModel):
    # Remove Background from Frame using Model Built by BackgroundSubtractorMOG2
    # To make sure we do not change the original frame
    frame = f.copy()
    # Build a BG Subtractor Model
    fgmask = bgModel.apply(frame, learningRate=LEARNING_RATE)

    # Apply the Model to a Frame
    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)

    # Return Frame w BG Removed
    return res

def extract_features(filtered_img, og_img):
    _, contours, hierarchy = cv2.findContours(filtered_img, 
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return

    cnt = max(contours, key = lambda x: cv2.contourArea(x))
    
    # Math stuff to approximate stuff
    epsilon = 0.0005*cv2.arcLength(cnt,True)
    approx= cv2.approxPolyDP(cnt,epsilon,True)

    # Make a hull around the hand
    hull = cv2.convexHull(cnt)

    # Get the area hull and the area of the countour
    areahull = cv2.contourArea(hull)
    areacnt = cv2.contourArea(cnt)

    # Get the percentage of hull not covered by hand
    arearatio=((areahull-areacnt)/areacnt)*100

    # Get the defects
    hull = cv2.convexHull(approx, returnPoints=False)
    defects = cv2.convexityDefects(approx, hull)

    # Calculate the number of defects
    l = 0
    defect_distances = []

    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]
        start = tuple(approx[s][0])
        end = tuple(approx[e][0])
        far = tuple(approx[f][0])
        pt= (100,180)


        # find length of all sides of triangle
        a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
        c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
        s = (a+b+c)/2
        ar = math.sqrt(s*(s-a)*(s-b)*(s-c))

        #distance between point and convex hull
        d=(2*ar)/a
        # print(ar, d)

        # apply cosine rule here
        angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57

        # ignore angles > 90 and ignore points very close to convex hull(they generally come due to noise)
        if angle <= 90 and d>30:
            defect_distances.append(d)
            l += 1

        #draw lines around hand
        cv2.line(og_img,start, end, [0,255,0], 2)
    if len(defect_distances) == 0:
        average_d = 0
    else:
        average_d = sum(defect_distances) / len(defect_distances)
    l+=1

    return l, arearatio, average_d

def convex_hull_classifier(filtered_img, og_img):
    #print corresponding gestures which are in their ranges
    l, areacnt, average_d = extract_features(filtered_img, og_img)
    font = cv2.FONT_HERSHEY_SIMPLEX

    if l==1:
        if areacnt<2000:
            cv2.putText(og_img,'Put hand in the box',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
        else:
            if d<15:
                cv2.putText(og_img,'good job!',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
            else:
                cv2.putText(og_img,'shaka',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
    elif l==2:
        cv2.putText(og_img,'peace',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
    elif l==3:
        if areacnt>2000:
            cv2.putText(og_img,'OK',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
        else:
            cv2.putText(og_img,'Rock On',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)   
    elif l==6:
        cv2.putText(og_img,'reposition',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
    else :
        cv2.putText(og_img,'reposition',(10,50), font, 2, (0,0,255), 3, cv2.LINE_AA)

    cv2.imshow('cont?', og_img)

if __name__ == '__main__':
    # Instructions
    # press 'b' to capture the background model (No Hand!)
    # press 'r' to reset the backgroud model
    # press 'ESC' to exit

    # Global Parameters
    cap_region_x_begin = 0.5  	# Start Point / Total Width
    cap_region_y_end = 0.8  	# Start Point / Total Height
    threshold = 60  			# BINARY Threshold
    blurValue = 41  			# GaussianBlur Parameter
    bgSubThreshold = 50
    LEARNING_RATE  = 0

    # Variables
    isBgCaptured = 0   			# Whether the Background is Captured
    triggerSwitch = False  		# Allow Keyboard simulator works
    bgModel = None

    camera = cv2.VideoCapture(0)
    camera.set(10,200)
    cv2.namedWindow('trackbar')
    cv2.createTrackbar('trh1', 'trackbar', threshold, 100, printThreshold)

    while camera.isOpened():
        ret, frame = camera.read()
        threshold = cv2.getTrackbarPos('trh1', 'trackbar')
        frame = cv2.bilateralFilter(frame, 5, 50, 100)  # smoothing filter
        frame = cv2.flip(frame, 1)  # flip the frame horizontally
        # TODO actually get the ROI and dont do this computation twice
        cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),
                     (frame.shape[1], int(cap_region_y_end * frame.shape[0])), (255, 0, 0), 2)
        cv2.imshow('original', frame)

        #  Main Operation
        if isBgCaptured == 1:
            # Remove Background + # Clip the ROI
            img = removeBG(frame, bgModel)
            img = img[0:int(cap_region_y_end * frame.shape[0]),
                        int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]

            # Convert the Image into a Binary Image
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)
            _ ,img_bw = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            cv2.imshow('blur', img_bw)

            # Run the classifiers? TODO implement 2 more
            convex_hull_classifier(img_bw, img)

        k = cv2.waitKey(10)
        if k == 27: # Press ESC to Exit
            break
        elif k == ord('b'):  # Press 'b' to Capture the Background
            bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)
            isBgCaptured = 1
            print( '* Background Captured')
        elif k == ord('r'):  # Press 'r' to Reset the Background
            bgModel = None
            triggerSwitch = False
            isBgCaptured = 0
            print ('* Reset BackGround')

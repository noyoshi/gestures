import cv2
import numpy as np
import copy
import math

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
learningRate = 0

# Variables
isBgCaptured = 0   			# Whether the Background is Captured
triggerSwitch = False  		# Allow Keyboard simulator works

def printThreshold(thr):
    print("! Changed Threshold to " + str(thr))

# Remove Background from Frame using Model Built by BackgroundSubtractorMOG2
def removeBG(frame):
	# Build a BG Subtractor Model
    fgmask = bgModel.apply(frame,learningRate=learningRate)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # res = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    # Apply the Model to a Frame
    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)

    # Return Frame w BG Removed
    return res

# Function Calculates ANgle Between Two Fingertips
def innerAngle(px1, py1, px2, py2, cx1, cy1):
    dist1 = math.sqrt(  (px1-cx1)*(px1-cx1) + (py1-cy1)*(py1-cy1) )
    dist2 = math.sqrt(  (px2-cx1)*(px2-cx1) + (py2-cy1)*(py2-cy1) )

    Ax = Ay = Bx = By = 0

    # Find closest point to C  
    Cx = cx1
    Cy = cy1
    if dist1 < dist2: 
        Bx = px1
        By = py1
        Ax = px2
        Ay = py2
    else:
        Bx = px2
        By = py2
        Ax = px1
        Ay = py1

    Q1 = Cx - Ax
    Q2 = Cy - Ay
    P1 = Bx - Ax
    P2 = By - Ay


    A = math.acos( (P1*Q1 + P2*Q2) / ( math.sqrt(P1*P1+P2*P2) * math.sqrt(Q1*Q1+Q2*Q2) ) )

    return A*180/np.pi

# Function Returns Bool for if Valid and Finger Count
def calculateFingers(res,drawing):

    # Convexity: Create Hull and Detect Defects
    hull = cv2.convexHull(res, returnPoints=False)
    finger_points = set()
    if len(hull) > 2:
        defects = cv2.convexityDefects(res, hull)
        if type(defects) != type(None):
            cnt = 0
            for i in range(defects.shape[0]):  # calculate the angle
                s, e, f, d = defects[i][0]
                start = tuple(res[s][0])
                end = tuple(res[e][0])
                far = tuple(res[f][0])
                a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # cosine theorem
                if angle <= math.pi / 2:  # angle less than 90 degree, treat as fingers
                    cnt += 1
                    cv2.circle(drawing, start, 8, [211, 84, 0], -1)
                    finger_points.add(start)
                    if end not in finger_points:
                        cv2.circle(drawing, end, 8, [84, 211, 0], -1)
                        finger_points.add(end)

            return True, cnt+1
    return False, 0


# Camera
camera = cv2.VideoCapture(0)
camera.set(10,200)
cv2.namedWindow('trackbar')
cv2.createTrackbar('trh1', 'trackbar', threshold, 100, printThreshold)


while camera.isOpened():
    ret, frame = camera.read()
    threshold = cv2.getTrackbarPos('trh1', 'trackbar')
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
        cv2.imshow('mask', img)

        # Convert the Image into a Binary Image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)
        cv2.imshow('blur', blur)
        ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)
        cv2.imshow('ori', thresh)

        # Get the Goutours
        thresh1 = copy.deepcopy(thresh)
        _,contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        length = len(contours)
        maxArea = -1
        if length > 0:

        	# Find the Biggest Contour (According to Area)
            for i in range(length):
                temp = contours[i]
                area = cv2.contourArea(temp)
                if area > maxArea:
                    maxArea = area
                    ci = i

            res = contours[ci]
            hull = cv2.convexHull(res)
            drawing = np.zeros(img.shape, np.uint8)
            cv2.drawContours(drawing, [res], 0, (0, 255, 0), 2)
            cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 3)

            isFinishCal,cnt = calculateFingers(res,drawing)
            
            # TODO: Do Something Here
            print(chr(27) + "[2J")
            print('Finger Count:', cnt)
                    

        cv2.imshow('output', drawing)

    # Keyboard Operations
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

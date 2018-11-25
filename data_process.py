import os
import cv2
import numpy as np
import copy
import math

def convex_hull_classifier(filtered_img):
    # Find contours
    _,contours,hierarchy= cv2.findContours(filtered_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        print("im guna return now")
        return

    # Get the max contour (hand)
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
    l+=1

    # features for SVM: # of defects; area ratio; distance bw point and convex hull 
    average_d = sum(defect_distances) / len(defect_distances)
    return l, arearatio, average_d


if __name__ == '__main__':
    datafile = open("data_features_svm.txt", "w")
    datafile.write("defects,arearatio,distance,label\n")

    gestures = ['ok','peace','rockon','shaka','thumbsup']
    for gesture in gestures:
        dirname = "data\\train\\"+gesture
        for imagefile in os.listdir(dirname):
            imagefilepath = dirname + "\\" + imagefile
            filtered_img = cv2.imread(imagefilepath,0)
            l, ar, d = convex_hull_classifier(filtered_img)
            label = imagefile.split('-')[0]
            content = "{},{},{},{}\n".format(l,ar,d,label)
            datafile.write(content)

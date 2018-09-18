#!/usr/bin/env python3

import numpy as np
import cv2
import sys

def hello(file_name):
    img = cv2.imread(file_name, 1)
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    file_name = sys.argv[1]
    hello(file_name)

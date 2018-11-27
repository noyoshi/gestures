#!/usr/bin/env python3

import argparse
import cv2

def hello(file_name):
    img = cv2.imread(file_name, 1)
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument(dest='image_file')
    ARGS = PARSER.parse_args()
    hello(ARGS.image_file)

import imutils
import glob
import string
import random
import cv2 as cv
import os
import numpy as np


def name_generator(size=10, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))


def change_main_dataset_names():
    dirs = glob.glob('mainDataset/*')
    for dir in dirs:
        imgs = glob.glob(dir+'/*')
        for img in imgs:
            new_name = dir+'/'+name_generator()+'.jpg'
            os.rename(img, new_name)


def rotate(src, dir):
    angles = [2, -5, 5, -10, 10]
    for angle in angles:
        img_new = imutils.rotate(src, angle)
        dst = dir+'/'+name_generator()+'.jpg'
        cv.imwrite(dst, img_new)

def blur(src, dir):    
    blurs = [(3, 3), (9, 9), (15, 15), (25, 25), (35, 35)]
    for blur in blurs:
        img_new = cv.blur(src, blur)
        dst = dir+'/'+name_generator()+'.jpg'
        cv.imwrite(dst, img_new)

def add_padding(src, dir):
    padding = 150

    img_new = cv.copyMakeBorder(src, padding, 0, 0, 0, cv.BORDER_CONSTANT)
    dst = dir+'/'+name_generator()+'.jpg'
    cv.imwrite(dst, img_new)

    img_new = cv.copyMakeBorder(src, 0, padding, 0, 0, cv.BORDER_CONSTANT)
    dst = dir+'/'+name_generator()+'.jpg'
    cv.imwrite(dst, img_new)

    img_new = cv.copyMakeBorder(src, 0, 0, padding, 0, cv.BORDER_CONSTANT)
    dst = dir+'/'+name_generator()+'.jpg'
    cv.imwrite(dst, img_new)

    img_new = cv.copyMakeBorder(src, 0, 0, 0, padding, cv.BORDER_CONSTANT)
    dst = dir+'/'+name_generator()+'.jpg'
    cv.imwrite(dst, img_new)

    img_new = cv.copyMakeBorder(src, padding, 0, padding, 0, cv.BORDER_CONSTANT)
    dst = dir+'/'+name_generator()+'.jpg'
    cv.imwrite(dst, img_new)

def brightness(src, dir):
    values = [0.3, 0.4, 0.5, 0.6, 0.7]
    for value in values:
        hsv = cv.cvtColor(src, cv.COLOR_BGR2HSV)
        hsv = np.array(hsv, dtype = np.float64)
        hsv[:,:,1] = hsv[:,:,1]*value
        hsv[:,:,1][hsv[:,:,1]>255]  = 255
        hsv[:,:,2] = hsv[:,:,2]*value
        hsv[:,:,2][hsv[:,:,2]>255]  = 255
        hsv = np.array(hsv, dtype = np.uint8)
        img_new = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
        dst = dir+'/'+name_generator()+'.jpg'
        cv.imwrite(dst, img_new)

def horizontal_shift(src, dir):
    ratios = [0.1, -0.2, 0.2, -0.3, 0.3]
    for ratio in ratios:
        h, w = src.shape[:2]
        to_shift = w*ratio
        if ratio > 0:
            img_new = src[:, :int(w-to_shift), :]
        if ratio < 0:
            img_new = src[:, int(-1*to_shift):, :]
        img_new = cv.resize(img_new, (h, w), cv.INTER_CUBIC)

        dst = dir+'/'+name_generator()+'.jpg'
        cv.imwrite(dst, img_new)


def do_augs():
    dirs = glob.glob('mainDataset/*')
    for dir in dirs:
        imgs = glob.glob(dir+'/*.jpg')
        for img in imgs:
            src = cv.imread(img)
            
            rotate(src, dir)
            blur(src, dir)
            add_padding(src, dir)
            brightness(src, dir)
            horizontal_shift(src, dir)


# change_main_dataset_names()
# do_augs()




# # TEST
# SIZE = 720
# while cv.waitKey(1) != ord('0'):
#     img = cv.resize(src, (SIZE, SIZE))
#     cv.imshow("Rotated", img)
# while cv.waitKey(1) != ord('0'):
#     img = cv.resize(img_new, (SIZE, SIZE))
#     cv.imshow("Rotated", img)

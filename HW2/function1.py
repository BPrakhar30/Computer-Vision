#!/usr/bin/env python
# coding: utf-8

# In[2]:



# Do Not Modify
import nbimporter
from util import display_filter_responses
import numpy as np
import multiprocess
import scipy.ndimage
import skimage
import sklearn.cluster
import scipy.spatial.distance
import os, time
import matplotlib.pyplot as plt
import random
import cv2

from skimage import io

def extract_filter_responses(image):
    '''
    Extracts the filter responses for the given image.

    [input]
    * image: numpy.ndarray of shape (H, W) or (H, W, 3)

    [output]
    * filter_responses: numpy.ndarray of shape (H, W, 3F)
    '''
    
    if(len(image.shape) == 2):
        image = np.stack((image, image, image), axis=-1)

    if(image.shape == 3 and image.shape[2] == 1):
        image = np.concatenate((image, image, image), axis=-1)

    if(image.shape[2] == 4):
        image = image[:, :, 0:3]

    image = skimage.color.rgb2lab(image)

    filter_responses = []
    '''
    HINTS: 
    1.> Iterate through the scales (5) which can be 1, 2, 4, 8, 8$\sqrt{2}$
    2.> use scipy.ndimage.gaussian_* to create filters
    3.> Iterate over each of the three channels independently
    4.> stack the filters together to (H, W,3F) dim
    '''
    # ----- TODO -----
    # YOUR CODE HERE
    #print(image.shape)
    for i in [1., 2., 4., 8., (8*2**(0.5))]:
        for j in range(3):
            filter_responses.append((scipy.ndimage.gaussian_filter(image[:,:,j], sigma = i)))
        for j in range(3):
            filter_responses.append((scipy.ndimage.gaussian_laplace(image[:,:,j], sigma = i)))
        for j in range(3):
            filter_responses.append((scipy.ndimage.gaussian_filter(image[:,:,j], sigma = i, order = [0,1])))
        for j in range(3):
            filter_responses.append((scipy.ndimage.gaussian_filter(image[:,:,j], sigma = i, order = [1,0])))
    
    #print(filter_responses[0])
    filter_responses_stacked = np.array(filter_responses[0])
    for i in range(1,60):
        filter_responses_stacked = np.dstack([filter_responses_stacked, filter_responses[i]])   
    
    filter_responses = filter_responses_stacked
    #print(filter_responses.shape)
    #raise NotImplementedError()
    return filter_responses


def get_harris_corners(image, alpha, k = 0.05):
    '''
    Compute points of interest using the Harris corner detector

    [input]
    * image: numpy.ndarray of shape (H, W) or (H, W, 3)
    * alpha: number of points of interest desired
    * k: senstivity factor 

    [output]
    * points_of_interest: numpy.ndarray of shape (2, alpha) that contains interest points
    '''
    
    '''
    HINTS:
    (1) Visualize and Compare results with cv2.cornerHarris() for debug (DO NOT SUBMIT cv2's implementation)
    '''
    # ----- TODO -----
    
    ######### Actual Harris #########
    from skimage.color import rgb2gray
    from scipy import ndimage

    bw_img = rgb2gray(image)  # To get grayscale image
    '''
    HINTS:
    1.> For derivative images we can use cv2.Sobel filter of 3x3 kernel size
    2.> Multiply the derivatives to get Ix * Ix, Ix * Iy, etc.
    '''
    # YOUR CODE HERE
    dx = cv2.Sobel(bw_img, cv2.CV_16S, dx=1, dy=0, borderType = cv2.BORDER_CONSTANT)
    dy = cv2.Sobel(bw_img, cv2.CV_16S, dx=0, dy=1, borderType = cv2.BORDER_CONSTANT)
    
    # calculate covariance matrix of each point
    dxx = dx ** 2
    dxy = dx * dy
    dyx = dy * dx
    dyy = dy ** 2
    
    ## Debugging with openCV
    #dst = cv2.cornerHarris(bw_img.astype(np.float32), 3, 3, 0.04)
    #plt.imshow(dst)
    #plt.show()
    
    #raise NotImplementedError()
    '''
    HINTS:
    1.> Think of R = det - trace * k
    2.> We can use ndimage.convolve
    3.> sort (argsort) the values and pick the alpha larges ones
    3.> points_of_interest should have this structure [[x1,x2,x3...],[y1,y2,y3...]] (2,alpha)
        where x_i is across H and y_i is across W
    '''
    # YOUR CODE HERE

    sum_filter = np.array([[1., 1., 1.],
                           [1., 1., 1.], 
                           [1., 1., 1.]])
    
    sum_xx = ndimage.convolve(dxx, sum_filter, mode='constant')
    sum_xy = ndimage.convolve(dxy, sum_filter, mode='constant')
    sum_yx = ndimage.convolve(dyx, sum_filter, mode='constant')
    sum_yy = ndimage.convolve(dyy, sum_filter, mode='constant')

    R = ((sum_xx * sum_yy) - (sum_xy * sum_yx)) - k * ((sum_xx + sum_yy) ** 2)
    top_alpha_R = (R.argsort(axis=None)[-1: -(alpha+1): -1])[-1::-1]
    
    poi_x, poi_y = top_alpha_R // R.shape[1], top_alpha_R % R.shape[1]
    
    points_of_interest = np.array([poi_x, poi_y])
    #raise NotImplementedError()
    
    ######### Actual Harris #########
    return points_of_interest



# In[ ]:





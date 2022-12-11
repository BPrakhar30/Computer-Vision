
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
import threading
import queue
import math

def helper_func(args):
    file_path, dictionary, layer_num, K, trained_features, train_labels = args
    _, feature = get_image_feature(file_path, dictionary, layer_num, K)
    print('hello')
    distances = distance_to_set(feature, trained_features)
    nearest_image_idx = np.argmax(distances)
    pred_label = train_labels[nearest_image_idx]   
    
    #return [file_path, pred_label, nearest_image_idx]
    
    # YOUR CODE HERE
    #raise NotImplementedError()
    return [file_path, pred_label, nearest_image_idx]


def get_image_feature(file_path, dictionary, layer_num, K):
    '''
    Extracts the spatial pyramid matching feature.

    [input]
    * file_path: path of image file to read
    * dictionary: numpy.ndarray of shape (K,3F)
    * layer_num: number of spatial pyramid layers
    * K: number of clusters for the word maps

    [output]
    * feature: numpy.ndarray of shape (K*(4^layer_num-1)/3)
    '''
    # ----- TODO -----
    # YOUR CODE HERE
    
    import skimage
    from skimage import io
    
    image = io.imread(file_path) 
    image = image.astype('float') / 255.
    wordmap = get_visual_words(image, dictionary)
    
    feature = get_feature_from_wordmap_SPM(wordmap, layer_num, K)
    
    #raise NotImplementedError()
    return [file_path, feature]



def distance_to_set(word_hist, histograms):
    '''
    Compute similarity between a histogram of visual words with all training image histograms.

    [input]
    * word_hist: numpy.ndarray of shape (K)
    * histograms: numpy.ndarray of shape (N,K)

    [output]
    * sim: numpy.ndarray of shape (N)
    '''
    '''
    HINTS:
    (1) Consider A = [0.1,0.4,0.5] and B = [[0.2,0.3,0.5],[0.8,0.1,0.1]] then \
        similarity between element A and set B could be represented as [[0.1,0.3,0.5],[0.1,0.1,0.1]]   
    '''
    # ----- TODO -----
    # YOUR CODE HERE
    sim = np.sum(np.minimum(word_hist, histograms), axis = 1)
    #raise NotImplementedError()
    return sim




def get_visual_words(image, dictionary):
    '''
    Compute visual words mapping for the given image using the dictionary of visual words.

    [input]
    * image: numpy.ndarray of shape (H,W) or (H,W,3)

    [output]
    * wordmap: numpy.ndarray of shape (H,W)
    '''
    '''
    HINTS:
    (1) Use scipy.spatial.distance.cdist to find closest match from dictionary
    (2) Wordmap represents the indices of the closest element (np.argmin) in the dictionary
    '''
    filter_responses = extract_filter_responses(image)
    
    h, w, _ = filter_responses.shape
    filter_responses = np.reshape(filter_responses, [-1, filter_responses.shape[-1]])
    # ----- TODO -----
    
    # YOUR CODE HERE
    distances = scipy.spatial.distance.cdist(filter_responses, dictionary)
    
    min_dist = np.argmin(distances, axis=1)
    wordmap = min_dist.reshape(h, w)
    #raise NotImplementedError()
    return wordmap





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


def get_feature_from_wordmap_SPM(wordmap, layer_num, dict_size):
    '''
    Compute histogram of visual words using spatial pyramid matching.

    [input]
    * wordmap: numpy.ndarray of shape (H,W)
    * layer_num: number of spatial pyramid layers
    * dict_size: dictionary size K

    [output]
    * hist_all: numpy.ndarray of shape (K*(4^layer_num-1)/3)
    '''
    '''
    HINTS:
    (1) Take care of Weights 
    (2) Try to build the pyramid in Bottom Up Manner
    (3) the output array should first contain the histogram for Level 0 (top most level) , followed by Level 1, and then Level 2.
    '''
    # ----- TODO -----
    h, w = wordmap.shape
    L = layer_num - 1
    patch_width = math.floor(w / (2**L))
    patch_height = math.floor(h / (2**L))
    
    '''
    HINTS:
    1.> create an array of size (dict_size, (4**(L + 1) -1)/3) )
    2.> pre-compute the starts, ends and weights for the SPM layers L 
    '''
    # YOUR CODE HERE
    
    
    hist_all = np.zeros((dict_size, int((4**(L + 1) - 1) / 3)))
    weights = np.array([2 ** (l - L - 1) / (4 ** l) if l > 1 else 2 ** (-L) / (4 ** l)
                        for l in range(L + 1)])
    
    index = 0
    starts = np.zeros(L + 1, dtype=np.int16)
    ends = np.zeros(L + 1, dtype=np.int16)
    
    for l in range(L + 1):
        elements_l = 4 ** l
        
        starts[l] = index
        ends[l] = index + elements_l
        index += elements_l
    
    #raise NotImplementedError()
    '''
    HINTS:
    1.> Loop over the layers from L to 0
    2.> Handle the base case (Layer L) separately and then build over that
    3.> Normalize each histogram separately and also normalize the final histogram
    '''
    # YOUR CODE HERE
    
    for l in range(L + 1):
        patch_w = math.floor(w / (2**l))
        patch_h = math.floor(h / (2**l))
        for i in range(4 ** l):
            h_l = (i // (2 ** l)) * patch_h
            w_l = (i % (2 ** l)) * patch_w
            h_l_end = h_l + patch_h 
            w_l_end = w_l + patch_w
            norm_patch_hist = get_feature_from_wordmap(wordmap[h_l: h_l_end, 
                                                               w_l: w_l_end],
                                                       dict_size)
            hist_all[:, starts[l] + i] = (norm_patch_hist * weights[l])

    hist_all /= hist_all.sum()    
    #raise NotImplementedError()
    return hist_all.flatten()


def get_feature_from_wordmap(wordmap, dict_size):
    '''
    Compute histogram of visual words.

    [input]
    * wordmap: numpy.ndarray of shape (H,W)
    * dict_size: dictionary size K

    [output]
    * hist: numpy.ndarray of shape (K)
    '''
    
    '''
    HINTS:
    (1) We can use np.histogram with flattened wordmap
    '''
    # ----- TODO -----
    # YOUR CODE HERE
    
    hist, _ = np.histogram(wordmap.flatten(), bins=range(dict_size + 1), range=(0, dict_size))
    hist = hist/ np.sum(hist)
    
    #raise NotImplementedError()
    return hist

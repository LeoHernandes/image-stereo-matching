import math
import sys
from typing import Tuple
import cv2 as cv
import numpy as np
import re

def rms_result_evaluation(result: np.ndarray, ground_thruth: np.ndarray, win_size: int) -> float:
    """
    Evaluate the result depth map and the ground thruth with root mean square
    :param result: result depth map
    :param ground_thruth: ground thruth depth map
    :param win_size: window size (N x N)
    :return: root mean square
    """
    apothem = int(win_size/2)
    height = result.shape[0]
    width = result.shape[1]
    
    square_diff = (result - ground_thruth)**2
    # sum only the pixels with depth information, disconsidering the borders based on window size
    square_sum = np.sum(square_diff[apothem : height - apothem, apothem : width - apothem])
    rms = math.sqrt(square_sum/float(height * width))
    return rms


def bad_pixels_result_evaluation(result: np.ndarray, ground_thruth: np.ndarray, win_size: int) -> float:
    """
    Evaluate the result depth map and the ground thruth with the percentage of bad pixels
    :param result: result depth map
    :param ground_thruth: ground thruth depth map
    :param win_size: window size (N x N)
    :return: percentage of bad pixels
    """
    THRESHOLD = 5
    apothem = int(win_size/2)
    height = result.shape[0]
    width = result.shape[1]
    
    diff = np.abs(result - ground_thruth)
    bad_pixels_mat = np.where(diff[apothem : height - apothem, apothem : width - apothem] > THRESHOLD, 1, 0)
    percentage = (np.sum(bad_pixels_mat)/float(height * width))*100
    return percentage


def sum_squared_distances(l_im: np.ndarray, r_im: np.ndarray, l_win_center: Tuple[int, int], r_win_center: Tuple[int, int], win_size: int) -> int:
    """
    Calculate the sum squared distances between two windows of pixels in two images
    :param l_im: left image of the stereo matching
    :param r_im: right image of the stereo matching
    :param l_win_center: the window center of left image
    :param r_win_center: the window center of right image
    :param win_size: window size (N x N)
    :return: sum squared distances
    """
    apothem = int(win_size/2)
    l_win = l_im[l_win_center[0] - apothem : l_win_center[0] + apothem + 1, l_win_center[1] - apothem : l_win_center[1] + apothem + 1]
    r_win = r_im[r_win_center[0] - apothem : r_win_center[0] + apothem + 1, r_win_center[1] - apothem : r_win_center[1] + apothem + 1]
    
    ssd = np.sum(np.subtract(l_win, r_win)**2)
    return ssd


def robust_error_function(l_im: np.ndarray, r_im: np.ndarray, l_win_center: Tuple[int, int], r_win_center: Tuple[int, int], win_size: int) -> int:
    """
    Calculate a robust error function based on ssd between two windows of pixels in two images
    :param l_im: left image of the stereo matching
    :param r_im: right image of the stereo matching
    :param l_win_center: the window center of left image
    :param r_win_center: the window center of right image
    :param win_size: window size (N x N)
    :return: robust error function cost
    """
    apothem = int(win_size/2)
    l_win = l_im[l_win_center[0] - apothem : l_win_center[0] + apothem + 1, l_win_center[1] - apothem : l_win_center[1] + apothem + 1]
    r_win = r_im[r_win_center[0] - apothem : r_win_center[0] + apothem + 1, r_win_center[1] - apothem : r_win_center[1] + apothem + 1]
    
    l_win_mean = np.sum(l_win)/(l_win.shape[0] * l_win.shape[1])
    r_win_mean = np.sum(r_win)/(r_win.shape[0] * r_win.shape[1])
    zero_mean_l_win = l_win - l_win_mean
    zero_mean_r_win = r_win - r_win_mean
    zssd = np.sum((zero_mean_l_win - zero_mean_r_win)**2)
    
    return zssd


def min_epipolar_line_distance(l_im: np.ndarray, r_im: np.ndarray, cur_pixel: Tuple[int, int], win_size: int, err_type: int) -> Tuple[int, int]:
    """
    Calculate the error between the blocks around each pixel in the epipolar line 
    :param l_im: left image of the stereo matching
    :param r_im: right image of the stereo matching
    :param cur_pixel: coordinates of the current pixel in left image
    :param win_size: window size (N x N)
    :param err_type: type of error function. 0 = default SSD, 1 = robust error function
    :return: tuple with the minimum distance and its corresponding column coordinate 
    """
    MAX_DISTANCE = 64
    
    distance = (sys.maxsize, 0)
    cur_row, cur_col = cur_pixel
    # only search at a maximum amount of pixel distance from current block position
    min_col = int(win_size/2) if cur_col <= MAX_DISTANCE + int(win_size/2) else cur_col - MAX_DISTANCE
    
    # Only search the matching block on the left side of the current pixel
    for col in range(min_col, cur_pixel[1] + 1):
        error = 0
        if err_type == 0:
            error = sum_squared_distances(l_im, r_im, cur_pixel, (cur_row, col), win_size)
        elif err_type == 1:
            error = robust_error_function(l_im, r_im, cur_pixel, (cur_row, col), win_size)
        if(error < distance[0]):
            distance = (error, col)
    
    return distance


def stereo_matching(l_image_path: str, r_image_path: str, gt_image_path: str, win_size: int, err_type: int):
    """
    Calculate the depth mapping by stereo matching
    :param l_image_path: left image path of the stereo matching
    :param r_image_path: right image path of the stereo matching
    :param gt_image_path: ground thruth image of the depth mapping with left image as reference
    :param err_type: type of error function. 0 = default SSD, 1 = robust error function
    :param win_size: window size (N x N)
    """
    left_image = cv.cvtColor(cv.imread(l_image_path), cv.COLOR_BGR2Lab) 
    right_image = cv.cvtColor(cv.imread(r_image_path), cv.COLOR_BGR2Lab)
    ground_thruth = cv.imread(gt_image_path, cv.IMREAD_GRAYSCALE)
    
    image_height = left_image.shape[0]
    image_width = left_image.shape[1]
    result = np.zeros((image_height,image_width), dtype=np.uint8)
    
    for row in range(int(win_size/2), image_height - int(win_size/2)):
        for col in range(int(win_size/2), image_width - int(win_size/2)):
            min_distance = min_epipolar_line_distance(left_image, right_image, (row, col), win_size, err_type)
            result[row, col] = abs(col - min_distance[1])
            
    image_name = re.split('/|-', l_image_path)[1]
    file_name = "depth_maps/" + image_name + "-window" + str(win_size) + "-err" + str(err_type)
    cv.imwrite(file_name + ".png", result)
    log_result_evaluations(result, ground_thruth, win_size, file_name)


def log_result_evaluations(result: np.ndarray, ground_thruth: np.ndarray, win_size: int, file_name:str):
    rms = rms_result_evaluation(result, ground_thruth//4, win_size)
    percentage = bad_pixels_result_evaluation(result, ground_thruth//4, win_size)
    
    f = open(file_name + ".txt", "w")    
    f.write("Results:\n")
    f.write("-----------------\n")
    f.write("Root Mean Square: " + str(rms) + "\n")
    f.write("Percentage of bad pixels: " + str(percentage) + "%")
    f.close()  
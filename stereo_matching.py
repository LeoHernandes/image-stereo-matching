import sys
from typing import Tuple
import cv2 as cv
import numpy as np

def is_in_bounds(im_shape: int, im_width: int, row: int, col: int) -> bool:
    """
    Test if a pixel coordinate is in bounds of a image
    :param im_height: image height
    :param im_width: image width
    :param row: y coordinate of a pixel
    :param col: x coordinate of a pixel
    :return: True if is in bounds, False if is not
    """
    if(row < 0 or row > im_shape - 1): return False
    if(col < 0 or col > im_width - 1): return False
    return True


def square_euclidian_distance(pixel_1: np.ndarray, pixel_2: np.ndarray) -> int:
    """
    Calculate the squared euclidian distance between two pixels
    :param pixel_1: array representing the values of the first pixel
    :param pixel_2: array representing the values of the second pixel
    :return: squared euclidian distance
    """
    L_1, a_1, b_1 = pixel_1
    L_2, a_2, b_2 = pixel_2
    return (L_1 - L_2)**2 + (a_1 - a_2)**2 + (b_1 - b_2)**2


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
    height = l_im.shape[0]
    width = l_im.shape[1]
    left_row, left_col = l_win_center
    right_row, right_col = r_win_center
    
    ssd = 0
    for window_row in range (-int(win_size/2), int(win_size/2) + 1):
        for window_col in range (-int(win_size/2), int(win_size/2) + 1):
            left_pixel = np.zeros(3) # Represents the zero padding when window goes out of bounds
            right_pixel = np.zeros(3)
            if is_in_bounds(height, width, window_row + left_row, window_col + left_col):
                left_pixel = l_im[window_row + left_row, window_col + left_col]
            if is_in_bounds(height, width, window_row + right_row, window_col + right_col):
                right_pixel = r_im[window_row + right_row, window_col + right_col]
            ssd += square_euclidian_distance(left_pixel, right_pixel)

    return ssd


def min_epipolar_line_distance(l_im: np.ndarray, r_im: np.ndarray, cur_pixel: Tuple[int, int], win_size: int) -> Tuple[int, int]:
    """
    Calculate the sum squared distances between the blocks around each pixel in the epipolar line 
    :param l_im: left image of the stereo matching
    :param r_im: right image of the stereo matching
    :param cur_pixel: coordinates of the current pixel in left image
    :param win_size: window size (N x N)
    :return: list of the sum squared distances with the corresponding column coordinate 
    """
    distance = (sys.maxsize, 0)
    image_width = l_im.shape[1]
    
    for col in range(0, image_width):
        ssd = sum_squared_distances(l_im, r_im, cur_pixel, (cur_pixel[0], col), win_size)
        if(ssd < distance[0]):
            distance = (ssd, col)
    
    return distance


def stereo_matching(l_image_path: str, r_image_path: str, gt_image_path: str, win_size: int):
    """
    Calculate the depth mapping by stereo matching
    :param l_image_path: left image path of the stereo matching
    :param r_image_path: right image path of the stereo matching
    :param gt_image_path: ground thruth image of the depth mapping with left image as reference
    :param win_size: window size (N x N)
    """
    left_image = cv.cvtColor(cv.imread(l_image_path), cv.COLOR_BGR2Lab) 
    right_image = cv.cvtColor(cv.imread(r_image_path), cv.COLOR_BGR2Lab)
    ground_thruth = cv.imread(gt_image_path)
    
    image_height = left_image.shape[0]
    image_width = left_image.shape[1]
    result = np.zeros((image_height,image_width), dtype=np.uint8)
    
    for row in range(0, image_height):
        for col in range(0, image_width):
            min_distance = min_epipolar_line_distance(left_image, right_image, (row, col), win_size)
            result[row, col] = abs(col - min_distance[1])
            
    cv.imshow("result", result)
    cv.imshow("gt", ground_thruth)
    
    cv.waitKey(0)

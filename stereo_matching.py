import sys
from typing import Tuple
import cv2 as cv
import numpy as np

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
    
    ssd = np.sum((l_win - r_win)**2)
    return ssd


def min_epipolar_line_distance(l_im: np.ndarray, r_im: np.ndarray, cur_pixel: Tuple[int, int], win_size: int) -> Tuple[int, int]:
    """
    Calculate the sum squared distances between the blocks around each pixel in the epipolar line 
    :param l_im: left image of the stereo matching
    :param r_im: right image of the stereo matching
    :param cur_pixel: coordinates of the current pixel in left image
    :param win_size: window size (N x N)
    :return: tuple with the minimum distance and its corresponding column coordinate 
    """
    distance = (sys.maxsize, 0)
    cur_row, cur_col = cur_pixel
    # only search at a maximum 64 pixel distance from current block position
    min_col = int(win_size/2) if cur_col <= 64 + int(win_size/2) else cur_col - 64
    
    # Only search the matching block on the left side of the current pixel
    for col in range(min_col, cur_pixel[1] + 1):
        ssd = sum_squared_distances(l_im, r_im, cur_pixel, (cur_row, col), win_size)
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
    
    for row in range(int(win_size/2), image_height - int(win_size/2)):
        for col in range(int(win_size/2), image_width - int(win_size/2)):
            min_distance = min_epipolar_line_distance(left_image, right_image, (row, col), win_size)
            result[row, col] = abs(col - min_distance[1])
            
    cv.imshow("result", result)
    cv.imshow("gt", ground_thruth)
    
    cv.waitKey(0)

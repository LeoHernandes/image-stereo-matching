import argparse

from stereo_matching import stereo_matching

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image Stereo Matching.')
    
    parser.add_argument('left_image', type=str,
                        help='Path to left image of stereo matching')
    
    parser.add_argument('right_image', type=str,
                        help='Path to right image of stereo matching')

    parser.add_argument('ground_truth', type=str,
                        help='Path to ground truth image of stereo matching')

    parser.add_argument('-w', '--window-size', type=int, dest='window_size', default=1, metavar='window-size',
                        help='Neighbourhood used to match pixels.')

    args = parser.parse_args()

    stereo_matching(args.left_image, args.right_image, args.ground_truth, args.window_size)
import numpy as np
import cv2
import sys
import os

from scipy.ndimage.interpolation import map_coordinates
from data_processing import generate_training_list

WRITE_DIR="data"

# Function to distort image
def elastic_transform(image, alpha, sigma, alpha_affine, random_state=None):
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]

    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size,
        [center_square[0]+square_size, center_square[1]-square_size],
        center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine,
        size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1],
        borderMode=cv2.BORDER_REFLECT)

    blur_size = int(4*sigma) | 1
    dx = alpha * cv2.GaussianBlur((random_state.rand(*shape) * 2 - 1),
        ksize=(blur_size, blur_size), sigmaX=sigma)
    dy = alpha * cv2.GaussianBlur((random_state.rand(*shape) * 2 - 1),
        ksize=(blur_size, blur_size), sigmaX=sigma)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))

    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)

def deform_images(direc, filenames):
    directory = os.path.join(WRITE_DIR, direc)
    if not os.path.exists(directory):
        os.mkdir(directory)
        print("Created directory %s" % directory)

    for fname in filenames:
        im = cv2.imread("train/images/" + fname + ".tif", 0)
        im_mask = cv2.imread("train/masks/" + fname + "_mask.tif", 0)

        im_merge = np.concatenate((im[...,None], im_mask[...,None]), axis=2)
        im_merge_t = elastic_transform(im_merge, im_merge.shape[1] * 2,
            im_merge.shape[1] * 0.08, im_merge.shape[1] * 0.08)

        im_t = im_merge_t[...,0]
        im_mask_t = im_merge_t[...,1]
        cv2.imwrite(os.path.join(directory, fname + "-" + direc + ".tif"), im_t)
        cv2.imwrite(os.path.join(directory, fname + "-" + direc + "_mask.tif"), im_mask_t)

if __name__ == '__main__':
    dir_name = sys.argv[1]
    filenames = generate_training_list()
    deform_images(dir_name, filenames)

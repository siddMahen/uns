import numpy as np
import cv2

import sys
import re

def rle_from_image(img, order='F', format=True):
    bytes = img.reshape(img.shape[0] * img.shape[1], order=order)
    runs = [] ## list of run lengths
    r = 0     ## the current run length
    pos = 1   ## count starts from 1 per WK
    for c in bytes:
        if ( c == 0 ):
            if r != 0:
                runs.append((pos, r))
                pos+=r
                r=0
            pos+=1
        else:
            r+=1

    #if last run is unsaved (i.e. data ends with 1)
    if r != 0:
        runs.append((pos, r))
        pos += r
        r = 0

    if format:
        z = ''

        for rr in runs:
            z+='{} {} '.format(rr[0],rr[1])
        return z[:-1]
    else:
        return runs


def create_rle(rle_fname, filenames):
    f = open(rle_fname, 'w')
    pat = re.compile(".*/([0-9]+)_mask.tif")
    fnames_sorted = sorted(filenames, key=lambda x: int(pat.match(x).group(1)))

    f.write("img,pixels\n")
    for fname in fnames_sorted:
        im = cv2.imread(fname, 0)
        rle = rle_from_image(im)
        num = pat.match(fname).group(1)

        f.write("%s, %s\n" % (num, rle))

    f.close()

if __name__ == '__main__':
    rle_name = sys.argv[1]
    filenames = sys.argv[2:]
    path = create_rle(rle_name, filenames)

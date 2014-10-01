from __future__ import division

import numpy as np
import PIL.Image as Im
import os
import sys

"""
Take a set of images of some action sequence and combine them into a single image showing all of the poses of the moving body. We do this by finding the median image of all the frames, then extracting only the pixels which do *not* match the median image from each source file.

usage:
    python action_sequence.py foldername

This will find all images in the directory given by foldername and produce a new file in that directory called out.png which contains the multiple-exposure action sequence.
"""

folder = sys.argv[1]
source_fnames = reversed(sorted(os.listdir(folder)))
images = []
for f in source_fnames:
    img = Im.open(os.path.join(folder, f))
    images.append(np.expand_dims(np.array(img), 3))


stack = np.concatenate(images, axis=3)

med_img = np.array(np.median(stack, axis=3), dtype=np.uint8)

out_img = med_img.copy()
for img in images:
    img = np.squeeze(img)
    mask = img != med_img
    out_img[mask] = img[mask]
out_img = Im.fromarray(out_img)
out_img.save(os.path.join(folder, 'out.png'))

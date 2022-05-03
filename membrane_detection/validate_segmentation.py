import numpy as np
from pathlib import Path
from PIL import Image
from membranes import segment_membranes


def dice_coeff(im1, im2, empty_score=1.0):

    im1 = np.asarray(im1).astype('bool')
    im2 = np.asarray(im2).astype('bool')

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return (2. * intersection.sum() / im_sum)


if __name__ == "__main__":
    raw = []
    seg = []
    for i in range(125):
        im = Image.open(Path('./CREMI_images/raw/sample_C_20160501/') / ('sample_C_20160501_' + str(i) + '.png'))
        raw.append(im)
        im = Image.open(Path('./CREMI_images/seg/sample_C_20160501/') / ('sample_C_20160501_' + str(i) + '.png'))
        seg.append(im)
    raw = np.stack(raw, axis=2)
    seg = np.stack(seg, axis=2)

    membranes = segment_membranes(raw, device_s='gpu')
    print(dice_coeff(seg, membranes))

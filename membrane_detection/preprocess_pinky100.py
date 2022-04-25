from pathlib import Path
from intern import array
from tqdm import tqdm
import cv2 as cv
import numpy as np


def preprocess_pinky100(dir_out):
    p_out = Path(dir_out)
    p_raw = p_out / 'raw'
    p_seg = p_out / 'seg'
    p_raw.mkdir(parents=True, exist_ok=True)
    p_seg.mkdir(parents=True, exist_ok=True)

    em = array("bossdb://microns/pinky100/em", resolution=1)
    seg = array("bossdb://microns/pinky100_8x8x40/segmentation")

    rng = np.random.default_rng(seed=0)
    im_size = np.array((1, 1024, 1024))
    num_images = 10000
    num = 0
    while num < num_images:
        ul = np.asarray(rng.random(3) * seg.shape, dtype="int")
        lr = ul + im_size
        if np.all(lr < seg.shape):
            em_data = em[ul[0]:lr[0], ul[1]:lr[1], ul[2]:lr[2]]
            seg_data = seg[ul[0]:lr[0], ul[1]:lr[1], ul[2]:lr[2]]
            if (np.any(seg_data > 0)) and (np.any(em_data > 0)):
                fn = str(ul[0]) + '_' + str(ul[1]) + '_' + str(ul[2]) + '.png'
                cv.imwrite(str(p_raw / fn), em_data)
                contours, hierarchy = cv.findContours(seg_data, cv.RETR_FLOODFILL, cv.CHAIN_APPROX_SIMPLE)
                seg_data = np.zeros([im_size[1], im_size[2]], np.uint8)
                cv.drawContours(seg_data, contours, -1, (255, 255, 255), 3)
                cv.imwrite(str(p_seg / fn), seg_data)
                num += 1


if __name__ == "__main__":
    preprocess_pinky100('pinky100_images')

from pathlib import Path
import h5py
from tqdm import tqdm
import cv2 as cv
import numpy as np


def preprocess_CREMI(dir_in, dir_out):
    p_out = Path(dir_out)
    p_raw = p_out / 'raw'
    p_seg = p_out / 'seg'
    p_raw.mkdir(parents=True, exist_ok=True)
    p_seg.mkdir(parents=True, exist_ok=True)

    for p in Path(dir_in).glob('sample_?_20160501.hdf'):
        dataset = h5py.File(p, 'r')
        Z, X, Y = dataset['volumes']['labels']['neuron_ids'].shape
        for z in tqdm(range(Z)):
            img = dataset['volumes']['raw'][z, :, :]
            fn = p_raw / (p.stem + '_' + str(z) + '.png')
            cv.imwrite(str(fn), img)
            seg = dataset['volumes']['labels']['neuron_ids'][z, :, :]
            contours, hierarchy = cv.findContours(seg, cv.RETR_FLOODFILL, cv.CHAIN_APPROX_SIMPLE)
            out = np.zeros([X, Y], np.uint8)
            cv.drawContours(out, contours, -1, (255, 255, 255), 1)
            fn = p_seg / (p.stem + '_' + str(z) + '.png')
            cv.imwrite(str(fn), out)


if __name__ == "__main__":
    preprocess_CREMI('.', 'images')

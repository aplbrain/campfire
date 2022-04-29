from pathlib import Path
import h5py
from tqdm import tqdm
import cv2 as cv
import numpy as np


def detect_membranes(seg):
    Z, X, Y = seg.shape
    outs = []
    for z in tqdm(range(Z)):
        contours, hierarchy = cv.findContours(seg[z, :, :], cv.RETR_FLOODFILL, cv.CHAIN_APPROX_SIMPLE)
        out = np.zeros([X, Y], np.uint8)
        cv.drawContours(out, contours, -1, (255, 255, 255), 1)
        outs.append(out)

    return np.stack(outs)


def preprocess_CREMI(dir_in, dir_out):
    p_out = Path(dir_out)
    p_raw = p_out / 'raw'
    p_seg = p_out / 'seg'
    p_raw.mkdir(parents=True, exist_ok=True)
    p_seg.mkdir(parents=True, exist_ok=True)

    for p in Path(dir_in).glob('sample_?_20160501.hdf'):
        dataset = h5py.File(p, 'r')
        raw = dataset['volumes']['raw'][:, ::2, ::2]
        neuron_ids = dataset['volumes']['labels']['neuron_ids'][:, ::2, ::2]

        outs = []
        for i in range(3):
            out = detect_membranes(np.moveaxis(neuron_ids, 0, i))
            outs.append(np.moveaxis(out, i, 0).astype('bool'))
        outs = np.any(np.stack(outs), axis=0)

        for z in range(outs.shape[0]):
            fn = p_raw / (p.stem + '_' + str(z) + '.png')
            cv.imwrite(str(fn), raw[z, :, :])
            fn = p_seg / (p.stem + '_' + str(z) + '.png')
            cv.imwrite(str(fn), outs[z, :, :].astype('uint8') * 255)


if __name__ == "__main__":
    preprocess_CREMI('images', 'images')

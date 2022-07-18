from pathlib import Path
import h5py
from intern import array
from cloudvolume import CloudVolume
from tqdm import tqdm
import cv2 as cv
import numpy as np


def detect_membranes(seg):
    Z, X, Y = seg.shape
    outs = []
    for z in tqdm(range(Z)):
        contours, hierarchy = cv.findContours(
            seg[z, :, :], cv.RETR_FLOODFILL, cv.CHAIN_APPROX_NONE
        )
        out = np.zeros([X, Y], np.uint8)
        cv.drawContours(out, contours, -1, (255, 255, 255), 1)
        outs.append(out)

    return np.stack(outs)


def preprocess_CREMI(dir_in, dir_out):
    p_out = Path(dir_out)
    p_raw = p_out / "raw"
    p_seg = p_out / "seg"
    p_raw.mkdir(parents=True, exist_ok=True)
    p_seg.mkdir(parents=True, exist_ok=True)

    for p in Path(dir_in).glob("sample_?_20160501.hdf"):
        dataset = h5py.File(p, "r")
        raw = dataset["volumes"]["raw"][:, ::2, ::2]
        neuron_ids = dataset["volumes"]["labels"]["neuron_ids"][:, ::2, ::2]

        outs = []
        for i in range(3):
            out = detect_membranes(np.moveaxis(neuron_ids, 0, i))
            outs.append(np.moveaxis(out, i, 0).astype("bool"))
        outs = np.any(np.stack(outs), axis=0)

        for z in range(outs.shape[0]):
            fn = p_raw / (p.stem + "_" + str(z) + ".png")
            cv.imwrite(str(fn), raw[z, :, :])
            fn = p_seg / (p.stem + "_" + str(z) + ".png")
            cv.imwrite(str(fn), outs[z, :, :].astype("uint8") * 255)


def preprocess_pinky100(dir_out, bounds):
    p_out = Path(dir_out)
    p_raw = p_out / "raw"
    p_seg = p_out / "seg"
    p_raw.mkdir(parents=True, exist_ok=True)
    p_seg.mkdir(parents=True, exist_ok=True)

    em = array("bossdb://microns/pinky100/em", resolution=1)
    seg = array("bossdb://microns/pinky100_8x8x40/segmentation")

    em_data = em[bounds[0] : bounds[1], bounds[2] : bounds[3], bounds[4] : bounds[5]]
    seg_data = seg[bounds[0] : bounds[1], bounds[2] : bounds[3], bounds[4] : bounds[5]]

    outs = []
    for i in range(3):
        out = detect_membranes(np.moveaxis(seg_data, 0, i))
        outs.append(np.moveaxis(out, i, 0).astype("bool"))
    outs = np.any(np.stack(outs), axis=0)

    for z in range(outs.shape[0]):
        fn = p_raw / (str(z) + ".png")
        cv.imwrite(str(fn), em_data[z, :, :])
        fn = p_seg / (str(z) + ".png")
        cv.imwrite(str(fn), outs[z, :, :].astype("uint8") * 255)


def preprocess_flyem(dir_out, bounds):
    p_out = Path(dir_out)
    p_raw = p_out / "raw"
    p_seg = p_out / "seg"
    p_raw.mkdir(parents=True, exist_ok=True)
    p_seg.mkdir(parents=True, exist_ok=True)

    em = CloudVolume(
        "gs://neuroglancer-janelia-flyem-hemibrain/emdata/raw/jpeg",
        use_https=True,
        mip=0,
    )
    seg = CloudVolume(
        "gs://neuroglancer-janelia-flyem-hemibrain/v1.2/segmentation",
        use_https=True,
        mip=0,
    )

    em_data = np.squeeze(
        em[bounds[0] : bounds[1] : 5, bounds[2] : bounds[3], bounds[4] : bounds[5]]
    )
    seg_data = np.squeeze(
        seg[bounds[0] : bounds[1] : 5, bounds[2] : bounds[3], bounds[4] : bounds[5]]
    )

    outs = []
    for i in range(3):
        out = detect_membranes(np.moveaxis(seg_data, 0, i))
        outs.append(np.moveaxis(out, i, 0).astype("bool"))
    outs = np.any(np.stack(outs), axis=0)

    for z in range(outs.shape[0]):
        fn = p_raw / (str(z) + ".png")
        cv.imwrite(str(fn), em_data[z, :, :])
        fn = p_seg / (str(z) + ".png")
        cv.imwrite(str(fn), outs[z, :, :].astype("uint8") * 255)


if __name__ == "__main__":
    # preprocess_CREMI('./CREMI_images', './CREMI_images')
    # preprocess_pinky100('./pinky100_images', (1000, 1500, 20000, 21024, 30000, 31024))
    preprocess_flyem("./flyem_images", (1000, 3500, 30000, 31024, 30000, 31024))

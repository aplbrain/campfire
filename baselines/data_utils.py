from logging import warning
from cloudvolume import CloudVolume
from cloudvolume.exceptions import OutOfBoundsError, EmptyVolumeException
import os
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
import h5py
FILENAME_ENCODING="{point}_{vol_size}.{fmt}"
def encode_filename(**kwargs):
    for k,v in kwargs.items():
        if type(v) != str:
            v = "_".join([str(int(i)) for i in v])
            kwargs[k] = v
    ret = FILENAME_ENCODING.format(**kwargs)
    return ret 
def load_point(point, vol, loc, vol_size, no_cache):
    """Loads a region around a point

    Args:
        point (3 tuple): x,y,z point, in voxels
        vol (CloudVolume): The CloudVolume containing the point
        loc (pathlike): Where to load cached volumes from
        vol_size (Union[int,tuple]): Either a single number, in which case a cube around the point is grabbed, or a 3 tuple descirbing the dimensions of a rectuangular prism around the point
        no_cache (bool): Don't read from cache.

    Returns:
        np.array: volume around the point
    """
    npy_fn = os.path.join(loc, encode_filename(point=point, vol_size=vol_size, fmt='npy'))
    h5_fn = os.path.join(loc, encode_filename(point=point, vol_size=vol_size, fmt='h5'))
    if not no_cache:
        if os.path.exists(npy_fn) and not no_cache:
            return np.load(npy_fn, allow_pickle=True)
        elif os.path.exists(h5_fn) and not no_cache:
            try:
                with h5py.File(h5_fn,'r') as f:
                    ret = np.array(f['raw'])
                return ret
            except Exception as e:
                warnings.warn(f"Problem while reading {h5_fn}: {e}, redownloading")
                os.remove(h5_fn)
                return load_point(point, vol, loc, vol_size, no_cache)
    try:
        return vol.download_point(point, size=vol_size,mip=0)
    except (OutOfBoundsError, EmptyVolumeException) as e:
        warnings.warn(f"Unable to download with volume size {vol_size} around {point}, failed with {e}")
        return None
def download_data(df, em_dir, seg_dir, vol_size=(256,256,32), resolution_scale=(2,2,1), output_fmt='h5', no_cache=False):
    """Downloads segmentation and EM for an evaluation dataframe

    Args:
        df (pd.DataFrame): Dataframe with at least a column called EP
        em_dir (pathlike): path to where EM data is saved. Will be created if doesn't exist
        seg_dir (pathlike): path to where segmentation data is saved. Will be created if doesn't exist.
        vol_size (Union[tuple, int]): Size of volume to download around point
        resolution_scale (tuple): scaling factor for cloudvolume points. Default cloudvolume resolution is 8x8x40, but this dataset is 4x4x40, so resolution_scale needs to be (2,2,1)
        output_fmt (str): output format, either npy or h5

    """
    os.makedirs(em_dir, exist_ok=True)
    os.makedirs(seg_dir, exist_ok=True)
    em_vol = CloudVolume("precomputed://https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/minnie65/em", use_https=True, mip=0)
    seg_vol = CloudVolume("precomputed://https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/minnie65/seg", use_https=True, mip=0)

    def save(loc, dat):
        if dat is None:
            return
        if output_fmt == 'h5':
            try:
                with h5py.File(loc, 'w') as f:
                    f.create_dataset('raw', data=dat, compression='gzip')
            except Exception as e:
                warnings.warn(f'Unable to write {loc} for {e}')
                print(dat)
                os.remove(loc)
                raise
        elif output_fmt == 'npy':
            np.save(loc, dat, allow_pickle=False)
    for i,row in tqdm(df.iterrows(),desc='Points', total=len(df)):
        
        
        point = np.divide(row['EP'], resolution_scale)
        em_loc = os.path.join(em_dir, encode_filename(point=point, vol_size=vol_size, fmt=output_fmt))
        seg_loc = os.path.join(seg_dir, encode_filename(point=point, vol_size=vol_size, fmt=output_fmt))
        
        
        em = load_point(point, em_vol, em_dir,vol_size=vol_size, no_cache=no_cache)
        save(em_loc, em)
        del em
        seg = load_point(point, seg_vol, seg_dir,vol_size=vol_size, no_cache=no_cache)
        save(seg_loc, seg)
        del seg


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('evaluation_file', default='expanded_gt.pkl', help="The evalation file to grab points from. Must contain the column 'EP' that contains (x,y,z) positions of each point")
    parser.add_argument('--seg-dir',default='seg', help="The directory to download segmenation volumes to. Created if not already existing.")
    parser.add_argument('--em-dir',default='em', help="The directory to download EM volumes to. Created if not already existing.")
    parser.add_argument('--vol-size',default="256,256,32",help='Size of volume around point. Comma seperated size in x,y,z or a single number for uniformity')
    parser.add_argument('--output-fmt', choices=['npy','h5'], default='h5', help="Output file format. For FFN use h5.")
    parser.add_argument('--no-cache', help="Do not load from cache, force download and overwrite.", action='store_true')
    
    args = parser.parse_args()
    kwargs = vars(args)
    df = pd.read_pickle(kwargs.pop('evaluation_file'))
    kwargs['vol_size'] =  [int(i) for i in args.vol_size.split(',')]
    kwargs['df'] = df
    download_data(**kwargs)

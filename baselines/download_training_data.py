from pickletools import int4
from data_utils import *
import fastremap
def main(point=None, radius=(1500,1500,150), scale=(2,2,1), output_dir='training_data'):
    em_vol = CloudVolume("precomputed://https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/minnie65/em", use_https=True, mip=0)
    seg_vol = CloudVolume("precomputed://https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/minnie65/seg", use_https=True, mip=0)
    if point is None:
        point = np.array(em_vol.bounds.center())
    os.makedirs('training_data', exist_ok=True)
    print(scale)
    point = np.divide(point, scale).astype(int)
    em = em_vol.download_point(point, size=radius,mip=0)
    
    with h5py.File(os.path.join(output_dir, "em_volume"),'w') as f:
        f.create_dataset("raw",data=np.array(em))
    seg = seg_vol.download_point(point, size=radius,mip=0)
    seg = np.array(seg)
    remapped_seg, mapping = fastremap.renumber(seg, in_place=False)
    print(mapping)
    with h5py.File(os.path.join(output_dir, "labels"),'w') as f:
        f.create_dataset("groundtruth",data=seg)
        f.create_dataset("remapped",data=remapped_seg)
        f.create_dataset("mapping",data=mapping)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--point", help="Point to download from")
    parser.add_argument('--radius', help="Size of downloaded volume")
    parser.add_argument("--scale", help="Resolution scale")
    parser.add_argument("--output-dir", help='Output directory' )
    args = parser.parse_args()
    kwargs = vars(args)
    kwargs = {k:v for k,v in kwargs.items() if v is not None}
    main(**kwargs)
from cloudvolume import CloudVolume
from caveclient import CAVEclient

def get_em(x_pre, x_post, y_pre, y_post, z_pre, z_post):
    em = CloudVolume("s3://bossdb-open-data/iarpa_microns/minnie/minnie65/em", use_https=True, mip=0)
    return em[x_pre:x_post, y_pre:y_post, z_pre:z_post]
    
def get_seg(x_pre, x_post, y_pre, y_post, z_pre, z_post):
    seg = CloudVolume("s3://bossdb-open-data/iarpa_microns/minnie/minnie65/seg", use_https=True, mip=0)
    return seg[x_pre:x_post, y_pre:y_post, z_pre:z_post]

def supervoxels(x_pre, x_post, y_pre, y_post, z_pre, z_post, simplify_supervoxels=True):
    cave_client = CAVEclient('minnie65_phase3_v1')
    sv_ids = CloudVolume("s3://bossdb-open-data/iarpa_microns/minnie/minnie65/ws", use_https=True, mip=0)
    sv_ids = sv_ids[x_pre:x_post, y_pre:y_post, z_pre:z_post]
    for sv_id in sv_ids.unique():
        sv_ids[sv_ids==sv_id] = cave_client.chunkedgraph.get_root_id(supervoxel_id=sv_id)

    return sv_ids
    
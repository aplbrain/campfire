from cloudvolume import CloudVolume
from caveclient import CAVEclient
import numpy as np

def get_em(x_pre, x_post, y_pre, y_post, z_pre, z_post):
    em = CloudVolume("s3://bossdb-open-data/iarpa_microns/minnie/minnie65/em", use_https=True, mip=0)
    return em[x_pre:x_post, y_pre:y_post, z_pre:z_post]
    
def get_seg(x_pre, x_post, y_pre, y_post, z_pre, z_post):
    seg = CloudVolume("s3://bossdb-open-data/iarpa_microns/minnie/minnie65/seg", use_https=True, mip=0)
    return seg[x_pre:x_post, y_pre:y_post, z_pre:z_post]

def get_nuclei(x_pre, x_post, y_pre, y_post, z_pre, z_post):
    seg = CloudVolume("s3://bossdb-open-data/iarpa_microns/minnie/minnie65/nuclei", use_https=True, mip=0)
    return seg[x_pre:x_post, y_pre:y_post, z_pre:z_post]

def supervoxels(x_pre, x_post, y_pre, y_post, z_pre, z_post, simplify_supervoxels=True):
    cave_client = CAVEclient('minnie65_phase3_v1')
    sv_ids = CloudVolume("s3://bossdb-open-data/iarpa_microns/minnie/minnie65/ws", use_https=True, mip=0)
    sv_ids = sv_ids[x_pre:x_post, y_pre:y_post, z_pre:z_post]
    
    sv_ids = np.asarray(np.squeeze(sv_ids))
    if simplify_supervoxels:
        unique_sv_ids, inv = np.unique(sv_ids,return_inverse = True)

        segs = cave_client.chunkedgraph.get_roots(supervoxel_ids=unique_sv_ids)
        seg_dict = dict(zip(unique_sv_ids, segs))

        sv_ids = np.array([seg_dict[x] for x in unique_sv_ids])[inv].reshape(sv_ids.shape)

    return sv_ids

def get_syn_counts(root_id:str):
    cave_client = CAVEclient('minnie65_phase3_v1')
    pre_synapses = cave_client.materialize.query_table(
        "synapses_pni_2", 
        filter_in_dict={"pre_pt_root_id": [root_id]},
        select_columns=['ctr_pt_position', 'pre_pt_root_id']
    )

    post_synapses = cave_client.materialize.query_table(
        "synapses_pni_2", 
        filter_in_dict={"post_pt_root_id": [root_id]},
        select_columns=['ctr_pt_position', 'post_pt_root_id']
    )

    return len(pre_synapses), len(post_synapses)

def get_num_soma(root_id:str):
    cave_client = CAVEclient('minnie65_phase3_v1')
    soma = cave_client.materialize.query_table(
        "nucleus_neuron_svm",
        filter_equal_dict={'pt_root_id':root_id}
    )
    return len(soma)

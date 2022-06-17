from caveclient import CAVEclient
import numpy as np
import pandas as pd

def get_syn_cts_mult(root_ids: list):
    cave_client = CAVEclient('minnie65_phase3_v1')
    pre_synapses = cave_client.materialize.query_table(
        "synapses_pni_2", 
        filter_in_dict={"pre_pt_root_id": root_ids},
        select_columns=['ctr_pt_position', 'pre_pt_root_id']
    )

    post_synapses = cave_client.materialize.query_table(
        "synapses_pni_2", 
        filter_in_dict={"post_pt_root_id": root_ids},
        select_columns=['ctr_pt_position', 'post_pt_root_id']
    )

    return pre_synapses, post_synapses


def get_num_soma_mult(root_ids: list):
    cave_client = CAVEclient('minnie65_phase3_v1')
    soma = cave_client.materialize.query_table(
        "nucleus_neuron_svm",
        materialization_version=117,
        filter_in_dict={'pt_root_id':root_ids},
        select_columns=['id', 'pt_root_id', 'classification_system', 'cell_type']
    )

    return soma


def axon_dendrite_conditions(pre_syn, post_syn):
    if pre_syn > post_syn:
        return 'axon'
    elif pre_syn < post_syn:
        return 'dendrite'
    else:
        return 'unconfirmed'


def multi_soma_count(root_ids: list) -> dict:
    root_ids_str = list(map(str, root_ids))
    soma_exists_df = get_num_soma_mult(root_ids_str)
    # Drop non-neuronal types
    soma_exists_df = soma_exists_df[(soma_exists_df.classification_system != 'is_neuron') & (soma_exists_df.cell_type != 'neuron')]

    num_soma_sr = soma_exists_df['pt_root_id'].value_counts()
    for i in root_ids:
        if i not in num_soma_sr.index:
            num_soma_sr[int(i)] = 0
    
    num_soma_dict = num_soma_sr.to_dict()

    return num_soma_dict
    

def multi_proc_type(root_ids: list) -> dict:
    root_ids_str = list(map(str, root_ids))
    len_pre, len_post = get_syn_cts_mult(root_ids_str)

    pre_counts_sr = len_pre['pre_pt_root_id'].value_counts()
    post_counts_sr = len_post['post_pt_root_id'].value_counts()

    all_cts_df = pd.DataFrame({'pre_cts': pre_counts_sr, 'post_cts': post_counts_sr})
    
    for i in root_ids:
        if i not in all_cts_df.index:
            all_cts_df.loc[i] = [0, 0]
    
    all_cts_df.fillna(0, inplace=True)

    vec_conditions_func = np.vectorize(axon_dendrite_conditions)

    all_cts_df = all_cts_df.assign(proc_type=vec_conditions_func(all_cts_df['pre_cts'], all_cts_df['post_cts']))
    all_cts_dict = all_cts_df['proc_type'].to_dict()

    return all_cts_dict


if __name__ == "__main__":
    some_list = [864691136109063864, 864691135699441698, 864691135521264882, 864691135368930546, 864691135918483376, 864691135804594461, 864691136914365806, 864691135395943378, 864691135478343235, 864691135648168388, 864691136000419720, 864691135449037042, 864691136181973462, 864691135582201586, 864691133035107425, 864691136913731182, 864691135407650258, 864691135968211902, 864691132647778343, 864691135401901778, 864691135104027483, 864691133716301031, 864691132647775271, 864691132647776807, 864691135648134852, 864691135407637970, 864691133035106657, 864691135648947396, 864691133456842377, 864691132647776295, 864691133716301287, 864691133035107169, 864691132647777575, 864691135804092189, 864691135793272349, 864691133456842633, 864691132647777063]
    # some_list = list(map(str, some_list))
    # act = multi_soma_count(some_list)
    actt = get_num_soma_mult(some_list)
    print(actt)
    # print(act)
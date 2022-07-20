from caveclient import CAVEclient
import numpy as np
import pandas as pd


def get_syn_cts_mult(root_ids: list):
    cave_client = CAVEclient("minnie65_phase3_v1")
    pre_synapses = cave_client.materialize.query_table(
        "synapses_pni_2",
        filter_in_dict={"pre_pt_root_id": root_ids},
        select_columns=["ctr_pt_position", "pre_pt_root_id"],
    )

    post_synapses = cave_client.materialize.query_table(
        "synapses_pni_2",
        filter_in_dict={"post_pt_root_id": root_ids},
        select_columns=["ctr_pt_position", "post_pt_root_id"],
    )

    return pre_synapses, post_synapses


def get_num_soma_mult(root_ids: list):
    cave_client = CAVEclient("minnie65_phase3_v1")
    soma = cave_client.materialize.query_table(
        "nucleus_neuron_svm",
        filter_in_dict={"pt_root_id": root_ids},
        select_columns=["id", "pt_root_id", "classification_system", "cell_type"],
    )

    return soma


def axon_dendrite_conditions(pre_syn, post_syn):
    """
    Pre-established utility function used in multi_proc_type. Contain in numpy.vectorize.
    """
    if pre_syn > post_syn:
        return "axon"
    elif pre_syn < post_syn:
        return "dendrite"
    else:
        return "unconfirmed"


def multi_soma_count(root_ids: list) -> dict:
    """
    #### Gets number of somas in arbitrary amount of segs.

    Parameter: root_ids: input list of seg ids

    Returns:num_soma_dict: dictionary where key is seg id and value is number of somas
    """
    root_ids_str = list(map(str, root_ids))
    soma_exists_df = get_num_soma_mult(root_ids_str)
    # Drop non-neuronal types
    # non_neuron_df = soma_exists_df[soma_exists_df.cell_type == 'not-neuron']
    # soma_exists_df = soma_exists_df[soma_exists_df.cell_type == 'neuron']

    num_soma_sr = soma_exists_df["pt_root_id"].value_counts()
    for i in root_ids:
        if (
            int(i) not in num_soma_sr.index
        ):  # and int(i) not in list(non_neuron_df['pt_root_id']):
            num_soma_sr[int(i)] = 0

    num_soma_dict = num_soma_sr.to_dict()

    return num_soma_dict


def multi_proc_type(root_ids: list) -> dict:
    root_ids_str = list(map(str, root_ids))
    len_pre, len_post = get_syn_cts_mult(root_ids_str)

    pre_counts_sr = len_pre["pre_pt_root_id"].value_counts()
    post_counts_sr = len_post["post_pt_root_id"].value_counts()

    all_cts_df = pd.DataFrame({"pre_cts": pre_counts_sr, "post_cts": post_counts_sr})

    for i in root_ids:
        if i not in all_cts_df.index:
            all_cts_df.loc[i] = [0, 0]

    all_cts_df.fillna(0, inplace=True)

    vec_conditions_func = np.vectorize(axon_dendrite_conditions)

    all_cts_df = all_cts_df.assign(
        proc_type=vec_conditions_func(all_cts_df["pre_cts"], all_cts_df["post_cts"])
    )
    all_cts_dict = all_cts_df["proc_type"].to_dict()

    return all_cts_dict


def get_tables(datastack: str):
    """
    Helper function to grab tables in a datastack.
    """
    client = CAVEclient(datastack)
    tables = client.annotation.get_tables()

    return tables


if __name__ == "__main__":
    nonneuron_list = [864691136909215598]
    some_list = list(map(str, nonneuron_list))
    act = multi_soma_count(some_list)
    actt = get_num_soma_mult(some_list)
    # print(act)
    scdf = get_syn_cts_mult([864691136025170522, 864691136020184410])
    print(scdf)

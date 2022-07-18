from caveclient import CAVEclient
import pcg_skel


def find_endpoints(root_id, **kwargs):
    """Find euclidean-space skeleton end point vertices from the pychunkedgraph
    Parameters
    ----------
    root_id : uint64
        Root id of the neuron to skeletonize
    client : caveclient.CAVEclientFull or None, optional
        Pre-specified cave client for the pcg. If this is not set, datastack_name must be provided. By default None
    datastack_name : str or None, optional
        If no client is specified, a CAVEclient is created with this datastack name, by default None
    cv : cloudvolume.CloudVolume or None, optional
        Prespecified cloudvolume instance. If None, uses the client info to make one, by default None
    refine : 'all', 'ep', 'bp', 'epbp', 'bpep', or None, optional
        Selects how to refine vertex locations by downloading mesh chunks. Unrefined vertices are placed in the
        center of their chunk in euclidean space.
        * 'all' refines all vertex locations. (Default)
        * 'ep' refines end points only
        * 'bp' refines branch points only
        * 'bpep' or 'epbp' refines both branch and end points.
        * None refines no points.
        * 'chunk' Keeps things in chunk index space.
    root_point : array-like or None, optional
        3 element xyz location for the location to set the root in units set by root_point_resolution,
        by default None. If None, a distal tip is selected.
    root_point_resolution : array-like, optional
        Resolution in euclidean space of the root_point, by default [4, 4, 40]
    root_point_search_radius : int, optional
        Distance in euclidean space to look for segmentation when finding the root vertex, by default 300
    collapse_soma : bool, optional,
        If True, collapses vertices within a given radius of the root point into the root vertex, typically to better
        represent primary neurite branches. Requires a specified root_point. Default if False.
    collapse_radius : float, optional
        Max distance in euclidean space for soma collapse. Default is 10,000 nm (10 microns).
    invalidation_d : int, optional
        Invalidation radius in hops for the mesh skeletonization along the chunk adjacency graph, by default 3
    return_mesh : bool, optional
        If True, returns the mesh in chunk index space, by default False
    return_l2dict : bool, optional
        If True, returns the tuple (l2dict, l2dict_r), by default False.
        l2dict maps all neuron level2 ids to skeleton vertices. l2dict_r maps skeleton indices to their direct level 2 id.
    return_l2dict_mesh : bool, optional
        If True, returns the tuple (l2dict_mesh, l2dict_mesh_r), by default False.
        l2dict_mesh maps neuron level 2 ids to mesh vertices, l2dict_r maps mesh indices to level 2 ids.
    return_missing_ids : bool, optional
        If True, returns level 2 ids that were missing in the chunkedgraph, by default False. This can be useful
        for submitting remesh requests in case of errors.
    nan_rounds : int, optional
        Maximum number of rounds of smoothing to eliminate missing vertex locations in the event of a
        missing level 2 mesh, by default 20. This is only used when refine=='all'.
    segmentation_fallback : bool, optional
        If True, uses the segmentation in cases of missing level 2 meshes. This is slower but more robust.
        Default is True.
    cache : str or None, optional
        If set to 'service', uses the online l2cache service (if available). Otherwise, this is the filename of a sqlite database with cached lookups for l2 ids. Optional, default is None.
    n_parallel : int, optional
        Number of parallel downloads passed to cloudvolume, by default 1
    Returns
    -------
    end_points : TrackedArray
        Skeleton end point vertices in euclidean space
    """
    sk_l2 = pcg_skel.pcg_skeleton(root_id, **kwargs)
    end_points = sk_l2.vertices[sk_l2.end_points, :]
    return end_points


if __name__ == "__main__":
    client = CAVEclient("minnie65_phase3_v1")
    oid = 864691135761488438  # Root id
    endpoints = find_endpoints(
        oid,
        client=client,
        refine="all",
        root_point_resolution=[4, 4, 40],
        collapse_soma=True,
        n_parallel=8,
    )
    print(endpoints)

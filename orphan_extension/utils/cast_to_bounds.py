from collections.abc import Iterable
import numpy as np
from caveclient import CAVEclient
from cloudvolume import CloudVolume


def cast_points_within_bounds(
    point: Iterable,
    bounds: Iterable = None,
    ds_res: Iterable = [2, 2, 1],
    boxrad: Iterable = [100, 100, 10],
) -> list:
    """
    Casts a point within specified bounds.

    Parameters
    ----------
    point -> Iterable: 1-dimensional iterable containing the coordinates of the point in Cartesian space
    ds_res -> Iterable: 1-dimensional iterable representing scaled dimension in case of downscaling
    bounds -> Iterable: 1-dimensional iterable containing low-high bounds to cast from

    Returns
    ----------
    casted_region -> list: list containing the axis bounds of the casted region
    """
    if not bounds:
        client = CAVEclient("minnie65_phase3_v1")
        vol = CloudVolume(
            client.info.segmentation_source(),
            use_https=True,
            progress=False,
            bounded=False,
        )
        v_b = vol.bounds.to_list()
        bounds = [v_b[0], v_b[3], v_b[1], v_b[4], v_b[2], v_b[5]]

    if len(bounds) != len(point) * 2:
        raise ValueError("Dimensions not proportional between bounds and point.")

    if len(boxrad) != len(point):
        raise ValueError("Dimensions not proportional between radius and point.")

    if any(i % j != 0 for (i, j) in zip(boxrad, ds_res)):
        print(
            "\n   Warning: Casting radius not perfectly divisible into current resolution. Result will be truncated.\n"
        )

    # boxrad = list(np.divide(boxrad, ds_res))
    point = list(np.divide(point, ds_res))
    # ds_res.extend(list(ds_res))
    # bounds = list(np.divide(bounds, ds_res))

    db_high = bounds[1::2]
    db_low = bounds[::2]

    casted_region = []

    th_bounds = np.add(db_high, np.ceil(boxrad))
    tl_bounds = np.subtract(db_low, np.ceil(boxrad))
    # th_bounds = [db_high[i] + np.ceil(boxrad[i]) for i in range(len(boxrad))]
    # tl_bounds = [db_low[i] - np.ceil(boxrad[i]) for i in range(len(boxrad))]

    for i in range(len(point)):
        if point[i] >= th_bounds[i] or point[i] <= tl_bounds[i]:
            raise ValueError("You are attempting to cast outside absolute boundaries.")

        if point[i] >= db_high[i]:
            p_low = point[i] - boxrad[i]
            p_high = db_high[i]
        elif point[i] <= db_low[i]:
            p_low = db_low[i]
            p_high = point[i] + boxrad[i]
        else:
            p_low = point[i] - boxrad[i]
            p_high = point[i] + boxrad[i]

        casted_region.extend([p_low, p_high])

    return casted_region


if __name__ == "__main__":
    point = [107820, 233757, 21960]
    casted = cast_points_within_bounds(point=point, boxrad=[99, 100, 10])
    print(casted)

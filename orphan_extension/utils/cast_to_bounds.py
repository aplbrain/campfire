from collections.abc import Iterable


class SubmoduleException(Exception):
    pass


def cast_points_within_bounds(point: Iterable, bounds: Iterable, boxrad: Iterable = [100,100,10]) -> list:
    """
    Casts a point within specified bounds.

    Parameters
    ----------
    point -> Iterable: 1-dimensional iterable containing the coordinates of the point in Cartesian space
    bounds -> Iterable: 1-dimensional iterable containing low-high bounds to cast from

    Returns
    ----------
    casted_region -> list: list containing the axis bounds of the casted region
    """
    if len(bounds) != len(point)*2:
        raise SubmoduleException('Dimensions not proportional between bounds and point.')
        
    db_high = bounds[1::2]
    db_low = bounds[::2]

    casted_region = []

    th_bounds = [db_high[i] + boxrad[i] for i in range(len(boxrad))]
    tl_bounds = [db_low[i] - boxrad[i] for i in range(len(boxrad))]
    
    for i in range(len(point)):
        if point[i] >= th_bounds[i] or point[i] <= tl_bounds[i]:
            raise SubmoduleException('You are attempting to cast outside absolute boundaries.')

        if point[i] >= db_high[i]:
            p_low = point[i]-boxrad[i]
            p_high = db_high[i]
        elif point[i] <= db_low[i]:
            p_low = db_low[i]
            p_high = point[i]+boxrad[i]
        else:
            p_low = point[i]-boxrad[i]
            p_high = point[i]+boxrad[i]

        casted_region.extend([p_low, p_high])
    
    return casted_region


if __name__ == "__main__":
    data_bounds = [26000,220608,30304,161376,14825,27881]
    point = [220707, 35000, 16000]
    casted = cast_points_within_bounds(point=point, bounds=data_bounds)
    print(casted)
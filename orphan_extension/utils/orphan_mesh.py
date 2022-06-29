
import numpy as np
    
def to_unit_vector(vector, validate=False, thresh=None, thresh_scalar=100):
    vec_arr = np.asanyarray(vector)

    if not thresh:
        thresh = np.finfo(np.float64).resolution
        thresh *= thresh_scalar

    if len(vec_arr.shape) == 2:
        nm = np.sqrt(np.dot(vec_arr*vec_arr, [1.0]*vec_arr.shape[1]))
        non_zero_norms = nm > thresh

        nm[non_zero_norms] **= -1
        unitized_vector = vec_arr * nm.reshape((-1,1))
    
    elif len(vec_arr.shape) == 1:
        nm = np.sqrt(np.dot(vec_arr, vec_arr))
        non_zero_norms = nm > thresh

        if non_zero_norms:
            unitized_vector = vec_arr / nm
        else:
            unitized_vector = vec_arr.copy()
    
    else:
        raise ArithmeticError('Internal vector dimension error.')
    
    if validate:
        return unitized_vector[non_zero_norms], non_zero_norms

    else:
        return unitized_vector
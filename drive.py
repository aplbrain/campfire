import pandas as pd
import numpy as np
from extension import Extension as Ext
import time 

def membrane_seg(radius=(200,200,20), resolution=(2,2,1), unet_bound_mult=1.5, save='pd',device='cpu',
                   nucleus_id=0, time_point=0, threshold=8):
    
    gt_df = pd.read_csv("./expanded_gt.csv")
    for i in range(103, gt_df.shape[0]):
        tic = time.time()
        row = gt_df.iloc[i]
        endpoint = eval(row.EP)
        ext = Ext(-1, resolution, radius, unet_bound_mult, 
        device, save, nucleus_id, time_point, endpoint)
        ext.get_bounds(endpoint)
        success = ext.membrane_seg_save()
        print(i, endpoint, time.time()-tic, success)
if __name__ == "__main__":
    membrane_seg()

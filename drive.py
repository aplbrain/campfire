import pandas as pd
import numpy as np
from tip_finding.tip_finding import tip_finder_decimation
from extension import Extension as Ext

def membrane_seg(radius=(200,200,20), resolution=(2,2,1), unet_bound_mult=1.5, save='pd',device='cpu',
                   nucleus_id=0, time_point=0, threshold=8):
    
    gt_df = pd.read_csv("./expanded_gt.csv")
    for i in range(gt_df.shape[0]):
        row = gt_df.iloc[i]
        endpoint = eval(row.EP)
        ext = Ext(-1, resolution, radius, unet_bound_mult, 
        device, save, nucleus_id, time_point, endpoint)
        ext.membrane_seg_save(endpoint)

if __name__ == "__main__":
    membrane_seg()

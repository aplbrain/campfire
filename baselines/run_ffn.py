from google.protobuf import text_format
from ffn.inference import inference
from ffn.inference import inference_pb2
import ffn
import argparse
import pandas as pd
from data_utils import encode_filename
import numpy as np
import os
import tempfile
import shutil
import tqdm
ffn_basepath = os.path.join(os.path.dirname(ffn.__file__),"..")
REQUEST_TEMPLATE = '''image {{
  hdf5: "{vol_path}:raw"
}}
image_mean: 128
image_stddev: 33
checkpoint_interval: 1800
seed_policy: "PolicyPeaks"
model_checkpoint_path: "{ffn_basepath}/models/fib25/model.ckpt-27465036"
model_name: "convstack_3d.ConvStack3DFFNModel"
model_args: "{{\\"depth\\": 12, \\"fov_size\\": [33, 33, 33], \\"deltas\\": [8, 8, 8]}}"
segmentation_output_dir: "{output_dir}"
inference_options {{
  init_activation: 0.95
  pad_value: 0.05
  move_threshold: 0.9
  min_boundary_dist {{ x: 1 y: 1 z: 1}}
  segment_threshold: 0.6
  min_segment_size: 1000
}}'''
def process_point(row, em_dir, vol_size, output_dir,resolution_scale=(2,2,1)):
  os.makedirs(output_dir, exist_ok=True)
  point = np.divide(row['EP'], resolution_scale)
  vol_size = np.array(vol_size)
  em_loc = os.path.join(em_dir, encode_filename(point=point, vol_size=vol_size, fmt='h5'))
  with tempfile.TemporaryDirectory() as temp_output_dir:
    config = REQUEST_TEMPLATE.format(vol_path=em_loc,output_dir=temp_output_dir, ffn_basepath=ffn_basepath)
    req = inference_pb2.InferenceRequest()
    _ = text_format.Parse(config, req)
    runner = inference.Runner()
    runner.start(req)
    bbox = np.array([(point - vol_size)[::-1], (point+vol_size)[::-1]])
    
    # This should make data?
    
    runner.run((0,0,0), (vol_size*2)[::-1])
    for dir, subdir, fns in os.walk(temp_output_dir):
      for fn in fns:
        _, ext = os.path.splitext(fn)
        shutil.copy(os.path.join(dir,fn), os.path.join(output_dir,encode_filename(point=point, vol_size=vol_size, fmt=ext[1:])))
def main(evaluation_file, row=None,**kwargs):
  df = evaluation_file
  if row is None:
    # Run everything
    for i, row in tqdm.tqdm(df.iterrows(), desc='Points', total=len(df)):
        process_point(row,**kwargs)
  else:
    row = df.iloc[row]
    process_point(row,**kwargs)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('evaluation_file', default='expanded_gt.pkl',help="The evalation file to grab points from. Must contain the column 'EP' that contains (x,y,z) positions of each point")
    parser.add_argument('--em-dir',default='em', help="The directory where EM volumes have been stored")
    parser.add_argument('--output-dir', default='seg_out', help="The directory where segmented volumes should go")
    parser.add_argument('--row',type=int, help="Process a particular row. If left out, will process the whole volume in a single thread.")
    parser.add_argument('--vol-size', default="256,256,32", help='Size of volume around point. Comma seperated size in x,y,z or a single number for uniformity. Should match that of downloaded data.')
    args = parser.parse_args()
    kwargs = vars(args)
    kwargs['evaluation_file'] = pd.read_pickle(args.evaluation_file)
    kwargs['vol_size'] = np.array(args.vol_size.split(","), dtype=int)
    main(**kwargs)

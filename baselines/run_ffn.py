from google.protobuf import text_format
from ffn.inference import inference
from ffn.inference import inference_pb2
import argparse
import pandas as pd
from utls import FILENAME_ENCODING
import numpy as np
import os

REQUEST_TEMPLATE = '''image {
  hdf5: "{vol_path}"
}
image_mean: 128
image_stddev: 33
checkpoint_interval: 1800
seed_policy: "PolicyPeaks"
model_checkpoint_path: "models/fib25/model.ckpt-27465036"
model_name: "convstack_3d.ConvStack3DFFNModel"
model_args: "{\\"depth\\": 12, \\"fov_size\\": [33, 33, 33], \\"deltas\\": [8, 8, 8]}"
segmentation_output_dir: "{output_dir}"
inference_options {
  init_activation: 0.95
  pad_value: 0.05
  move_threshold: 0.9
  min_boundary_dist { x: 1 y: 1 z: 1}
  segment_threshold: 0.6
  min_segment_size: 1000
}'''
def process_point(row, em_dir, radius, output_dir,resolution_scale=(2,2,1)):
  point = np.divide(row['EP'], resolution_scale)
  radius = np.array(radius)
  em_loc = os.path.join(em_dir, FILENAME_ENCODING.format(point=point, radius=radius, fmt='h5'))
  config = REQUEST_TEMPLATE.format(vol_path=em_loc,output_dir=output_dir)
  req = inference_pb2.InferenceRequest()
  _ = text_format.Parse(config, req)
  runner = inference.Runner()
  runner.start(config)
  bbox = point - radius, point+radius
  # This should make data?
  runner.run((bbox[0][::-1]),
             (bbox[1][::-1]))
def main(df):
  
  for i, row in df:
      process_point(row)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('evaluation_file', default='expanded_gt.pkl')
    parser.add_argument('--seg-dir',default='seg')
    parser.add_argument('--em-dir',default='em')
    parser.add_argument('--base-dir',default='.')
    parser.add_argument('--output', default='seg_out')
    args = parser.parse_args()
    df = pd.read_pickle(args.evaluation_file)

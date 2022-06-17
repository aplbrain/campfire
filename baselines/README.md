# Automated baselines

This folder contains baselines to compare agents against.

The baselines will be compared to agents from points in `expanded_gt.pkl` which is a pickled dataframe where each row is an endpoint with the location of the endpoint, the root ID and the "ground truth" segment ID that should be merged.

There are a few useful scripts:

`data_utils` contains data downloading utilities, running from the command line will download all the data for every point when provided with the dataframe. 

```
usage: data_utils.py [-h] [--seg-dir SEG_DIR] [--em-dir EM_DIR]
                     [--vol-size VOL_SIZE] [--output-fmt {npy,h5}]
                     [--no-cache]
                     evaluation_file

positional arguments:
  evaluation_file       The evalation file to grab points from. Must contain
                        the column 'EP' that contains (x,y,z) positions of
                        each point

optional arguments:
  -h, --help            show this help message and exit
  --seg-dir SEG_DIR     The directory to download segmenation volumes to.
                        Created if not already existing.
  --em-dir EM_DIR       The directory to download EM volumes to. Created if
                        not already existing.
  --vol-size VOL_SIZE   Size of volume around point. Comma seperated size in
                        x,y,z or a single number for uniformity
  --output-fmt {npy,h5}
                        Output file format. For FFN use h5.
  --no-cache            Do not load from cache, force download and overwrite.
```

`download_training_data.py` downloads the training data for the ffn.

```
usage: download_training_data.py [-h] [--point POINT] [--vol_size VOL_SIZE]
                                 [--scale SCALE] [--output-dir OUTPUT_DIR]

optional arguments:
  -h, --help            show this help message and exit
  --point POINT         Point to download from
  --vol_size VOL_SIZE   Size of downloaded volume
  --scale SCALE         Resolution scale
  --output-dir OUTPUT_DIR
                        Output directory
```

`run_ffn.py` Runs the ffn on all the points defined in the evaluation file. Can also be used for a single point if you provide a row number of the evaluation file.

```
usage: run_ffn.py [-h] [--em-dir EM_DIR] [--output-dir OUTPUT_DIR] [--row ROW]
                  [--vol-size VOL_SIZE]
                  evaluation_file

positional arguments:
  evaluation_file       The evalation file to grab points from. Must contain
                        the column 'EP' that contains (x,y,z) positions of
                        each point

optional arguments:
  -h, --help            show this help message and exit
  --em-dir EM_DIR       The directory where EM volumes have been stored
  --output-dir OUTPUT_DIR
                        The directory where segmented volumes should go
  --row ROW             Process a particular row. If left out, will process
                        the whole volume in a single thread.
  --vol-size VOL_SIZE   Size of volume around point. Comma seperated size in
                        x,y,z or a single number for uniformity. Should match
                        that of downloaded data.
```

`abyss_ffn.bash`

Submits an array job to SLURM. Make sure the argments match what you used for downloading files! If a file other than the provided evaluation file is used, you must modify the size of the job array to match. 

usage:

`sbatch abyss_ffn.bash`
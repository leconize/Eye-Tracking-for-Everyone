import sys
import argparse
import pickle
import h5py
import keras

from tensorflow.python.lib.io import file_io

def train_model(train_file='', job_dir='', **args):

    f = file_io.FileIO()

if __name__ == "__main__":
    # Parse the input arguments for common Cloud ML Engine options
    parser = argparse.ArgumentParser()
    parser.add_argument(
      '--train-file',
      help='Cloud Storage bucket or local path to training data')
    parser.add_argument(
      '--job-dir',
      help='Cloud storage bucket to export the model and store temp files')
    args = parser.parse_args()
    arguments = args.__dict__
    train_model(**arguments)

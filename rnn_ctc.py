#!/data/mm0105.chen/anaconda2/bin/python
from train import train_multi_process
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str)
    cmd_args = parser.parse_args()
    train_multi_process('rnn_ctc', log_dir=cmd_args.log_dir)

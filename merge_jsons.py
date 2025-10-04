import os
import argparse
import time
import json
import glob

from tqdm import tqdm
from functools import partial
import multiprocessing



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='../../data/shapes')
    parser.add_argument('-j', '--json', type=str, default='../resources/')
    parser.add_argument('-w', '--workers', type=int, default=multiprocessing.cpu_count() - 1)
    parser.add_argument('-o', '--output', type=str, default='../output/')
    parser.add_argument('-t', '--threshold', type=int, default=-1)
    parser.add_argument('--statistics', action=argparse.BooleanOptionalAction)
    parser.add_argument('--annotate', action=argparse.BooleanOptionalAction)
    parser.add_argument('--target', type=int, default=-1)

    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()

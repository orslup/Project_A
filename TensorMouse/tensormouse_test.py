import argparse
import os
from tensormouse import workers_test

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-src', '--source', dest='video_source', type=int,
                        default=0, help='Device index of the camera. (default: 0)')
    args = parser.parse_args()
    
    workers_test.main_worker(args.video_source)
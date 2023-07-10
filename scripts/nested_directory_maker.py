import argparse
from datetime import datetime
import os

parser = argparse.ArgumentParser()

parser.add_argument('--strapn', type=int,default=0, help='Index of the bootstrap to run. 0 means no bootstrap')
parser.add_argument('--run_id', type=int, default=0, help='integer ID to index a run')



flags = parser.parse_args()

date_time = datetime.now().strftime("%d%m%Y_%H:%M:%S")
folder = '../nested_dir/'+date_time+'_gpu_'+str(flags.run_id)+'_'+str(flags.strapn)
if not os.path.exists(folder):
    os.makedirs(folder)
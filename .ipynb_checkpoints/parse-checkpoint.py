import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--config', default='config_omnifold.json', help='Basic config file containing general options')
parser.add_argument('--plot_folder', default='../plots', help='Folder used to store plots')
parser.add_argument('--file_path', default='/global/cfs/cdirs/m3246/bnachman/LEP/aleph/processed/', help='Folder containing input files')
parser.add_argument('--nevts', type=float,default=-1, help='Dataset size to use during training')
parser.add_argument('--strapn', type=int,default=0, help='Index of the bootstrap to run. 0 means no bootstrap')
parser.add_argument('--verbose', action='store_true', default=False,help='Run the scripts with more verbose output')
parser.add_argument('--run_id', type=int, default=0, help='integer ID to index a run')



flags = parser.parse_args()

print(flags.run_id)
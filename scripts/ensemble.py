import numpy as np
import matplotlib.pyplot as plt
import argparse
import random
import os
import tensorflow as tf
import utils
from omnifold import  Multifold,LoadJson
import tensorflow.keras.backend as K
from glob import glob
from tqdm import tqdm
from pprint import pprint


# utils.SetStyle()

parser = argparse.ArgumentParser()

parser.add_argument('--config', default='config_omnifold.json', help='Basic config file containing general options')
parser.add_argument('--plot_folder', default='../plots', help='Folder used to store plots')
parser.add_argument('--file_path', default='/global/cfs/cdirs/m3246/bnachman/LEP/aleph/processed', help='Folder containing input files')
parser.add_argument('--nevts', type=float, default=-1, help='Dataset size to use during training')
parser.add_argument('--verbose', action='store_true', default=False,help='Run the scripts with more verbose output')

flags = parser.parse_args()
nevts=int(flags.nevts)
opt = LoadJson(flags.config)
itrials=0

data, mc_reco,mc_gen,reco_mask,gen_mask = utils.DataLoader(flags.file_path,opt,nevts)

weights_paths = glob('/global/homes/m/mavaylon/phys/OmniFold/weights/*')

# check that all the weights are completed
for folder in weights_paths:
    num_weights=len(glob(folder+'/*'))
    if num_weights!=10:
        break
    else:
        continue
        
# create sets of 10
rem = len(weights_paths)%10
if rem!=0: # gurantee even sets of 10
    number_of_sessions = len(weights_paths)-rem
    weights_paths = random.choices(weights_paths,k=number_of_sessions)
    sets_of_10 = [weights_paths[i:i + 10] for i in range(0, len(weights_paths), 10)]
else:
    sets_of_10 = [weights_paths[i:i + 10] for i in range(0, len(weights_paths), 10)]
    
group_num=1
session_dict={}
for group in tqdm(sets_of_10):
    hist_sessions_data =[]
    bins_=[]
    for session in tqdm(group):
        mfold = Multifold(version='{}_trial{}'.format(opt['NAME'],itrials),verbose=flags.verbose)
        mfold.PrepareModel(nvars=data.shape[1]) # sets the dims for the MLP model used and defines the 2 models (step 1 and 2)
        mfold.LoadModel(iteration=opt['NITER']-1, weights_folder_path=session) # loads weights for model 2
        omnifold_weights = mfold.reweight(mc_gen[gen_mask],mfold.model2) 

        feed_dict_data=1-mc_gen[gen_mask][:,0]
        output_data, bins = np.histogram(feed_dict_data, bins=utils.binning, weights = omnifold_weights[gen_mask], density=False)
        hist_sessions_data.append(output_data)
        bins_.append(bins)

    # sd = np.std(hist_sessions_data,axis=0)
    ave = np.mean(hist_sessions_data,axis=0)
    session_dict['group_'+str(group_num)] = ave
    # sd_over_mean = sd/ave
    group_num=group_num+1

# pprint(session_dict)

# calculate sd of the averages
sd = np.std(list(session_dict.values()),axis=0)

# keys to present sd of ave per bin
bins = ['bin_'+str(i) for i in utils.binning]
sd_dict_of_ave_bins = dict(zip(bins, sd))
pprint(sd_dict_of_ave_bins)
import numpy as np
import matplotlib.pyplot as plt
import argparse
import random
import os
import tensorflow as tf
import utils
# from omnifold import  Multifold,LoadJson
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

#weights_paths = glob('/pscratch/sd/m/mavaylon/phys/weights_tuned_mlp/*')
# weights_paths= glob('/pscratch/sd/m/mavaylon/phys/weights_tuned_mlp_old_mlp2/*')
weights_paths= glob('/pscratch/sd/m/mavaylon/phys_bootstrap/OmniFold/weights_strapn_2/*')[:40]
boot_folders=['/pscratch/sd/m/mavaylon/phys_bootstrap/OmniFold/weights_strapn_'+str(i)+'/*' for i in range(1,11)]
for i in boot_folders:
    set_40 = i
    hist_sessions_data =[]
    bins_=[]
    for session in tqdm(set_40):
        print(session)
        mfold = Multifold(version='{}_trial{}'.format(opt['NAME'],itrials),verbose=flags.verbose)
        mfold.PrepareModel(nvars=data.shape[1]) # sets the dims for the MLP model used and defines the 2 models (step 1 and 2)
        mfold.LoadModel(iteration=opt['NITER']-1, weights_folder_path=session, strapn='2') # loads weights for model 2
        omnifold_weights = mfold.reweight(mc_gen[gen_mask],mfold.model2) 

        feed_dict_data=1-mc_gen[gen_mask][:,0]
        output_data, bins = np.histogram(feed_dict_data, bins=utils.binning, weights = omnifold_weights[gen_mask], density=False)
        # print(output_data)
        hist_sessions_data.append(output_data)
        bins_.append(bins)

    sd = np.std(hist_sessions_data,axis=0)
    ave = np.mean(hist_sessions_data,axis=0)
    sd_over_mean_40 = sd/ave


# feed_dict = {
#         'ave_40': ave,
#         'mc gen':1-mc_gen[gen_mask][:,0]
#     }

# fig,ax = utils.SD_Plot(feed_dict,
#                            weights = None,
#                            binning=utils.binning,
#                            xlabel='1-T',logy=False,
#                            ylabel='Relative Uncertainty',
#                            reference_name='mc gen',
#                            sub=False,
#                            density=False,
#                            show_data=False,
#                            est_uncertainty=False)
# fig.savefig('{}/{}_{}.pdf'.format(flags.plot_folder,"Bootstrap_strapn_2_ensemble_40",opt['NAME']))

# print(ave)
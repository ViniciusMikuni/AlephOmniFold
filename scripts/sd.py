import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import tensorflow as tf
import utils
from omnifold import  Multifold,LoadJson
import tensorflow.keras.backend as K
from glob import glob
from pprint import pprint

utils.SetStyle()

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
subset_weights_paths = glob('/global/homes/m/mavaylon/phys/OmniFold/subweights/*')

hist_sessions_data =[]
bins_=[]
for session in weights_paths:
    mfold = Multifold(version='{}_trial{}'.format(opt['NAME'],itrials),verbose=flags.verbose)
    mfold.PrepareModel(nvars=data.shape[1]) # sets the dims for the MLP model used and defines the 2 models (step 1 and 2)
    mfold.LoadModel(iteration=opt['NITER']-1, weights_folder_path=session) # loads weights for model 2
    omnifold_weights = mfold.reweight(mc_gen[gen_mask],mfold.model2) 
    
    feed_dict_data=1-mc_gen[gen_mask][:,0]
    output_data, bins = np.histogram(feed_dict_data, bins=utils.binning, weights = omnifold_weights[gen_mask], density=False)
    hist_sessions_data.append(output_data)
    bins_.append(bins)

sd = np.std(hist_sessions_data,axis=0)
ave = np.mean(hist_sessions_data,axis=0)
sd_over_mean = sd/ave
    
sub_hist_sessions_data =[]
sub_bins_=[]
for session in subset_weights_paths:
    mfold = Multifold(version='{}_trial{}'.format(opt['NAME'],itrials),verbose=flags.verbose)
    mfold.PrepareModel(nvars=data.shape[1]) # sets the dims for the MLP model used and defines the 2 models (step 1 and 2)
    mfold.LoadModel(iteration=opt['NITER']-1, weights_folder_path=session) # loads weights for model 2
    omnifold_weights = mfold.reweight(mc_gen[gen_mask],mfold.model2) 
    
    feed_dict_data=1-mc_gen[gen_mask][:,0]
    output_data, bins = np.histogram(feed_dict_data, bins=utils.binning, weights = omnifold_weights[gen_mask], density=False)
    sub_hist_sessions_data.append(output_data)
    sub_bins_.append(bins)    

sub_sd = np.std(sub_hist_sessions_data,axis=0)
sub_ave = np.mean(sub_hist_sessions_data,axis=0)
sub_sd_over_mean = sub_sd/sub_ave

feed_dict = {
        'sd': sd,
        'sub_sd': sub_sd,
        'ave': ave,
        'sub_ave': sub_ave,
        'sd_over_mean': sd_over_mean,
        'sub_sd_over_mean': sub_sd_over_mean,
        'mc gen':1-mc_gen[gen_mask][:,0], # why 1-...
    }

sd_change=feed_dict['sd']/feed_dict['sub_sd']

print("ave:",feed_dict['ave'])
print('sub_ave:', feed_dict['sub_ave'])
print('sd:', feed_dict['sd'])
print('sub_sd:', feed_dict['sub_sd'])
print('sd_change:', sd_change)



# fig,ax = utils.SD_Plot(feed_dict,
#                            weights = None,
#                            binning=utils.binning,
#                            xlabel='1-T',logy=False,
#                            ylabel='Normalized events',
#                            reference_name='mc gen',
#                            sub=True,
#                            density=False)
# fig.savefig('{}/{}_{}.pdf'.format(flags.plot_folder,"SD_Unfolded_Hist_T",opt['NAME']))


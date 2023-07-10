import numpy as np
import matplotlib.pyplot as plt
import argparse
import random
import os
import tensorflow as tf
import utils
from omnifold import Multifold,LoadJson

import tensorflow.keras.backend as K
from glob import glob
from tqdm import tqdm
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
# breakpoint()
# estimate statistical uncertainty 
sim_det_data = 1-mc_reco[reco_mask]
# breakpoint()
output_sim_data, bins = np.histogram(sim_det_data, bins=utils.binning, density=False)
est_sim_stat = 1/np.sqrt(output_sim_data)

real_data = 1-data[:,0]
output_real_data, bins = np.histogram(real_data, bins=utils.binning, density=False)
est_real_stat = 1/np.sqrt(output_real_data)

weights_paths= glob('/pscratch/sd/m/mavaylon/redone_enemble_weights/*')

rem = len(weights_paths)%40
if rem!=0: # gurantee even sets of 40
    number_of_sessions = len(weights_paths)-rem
    random.seed(2)
    weights_paths_sample = random.sample(weights_paths,k=400)
    sets_of_40 = [weights_paths_sample[i:i + 40] for i in range(0, len(weights_paths_sample), 40)]
else:
    sets_of_40 = [weights_paths[i:i + 40] for i in range(0, len(weights_paths), 40)]
    
group_num=1
set_40 = sets_of_40[:10] # 10 sets of 40
session_dict={}
for group in tqdm(set_40):
    hist_sessions_data =[]
    bins_=[]
    print('group: ', group)
    for session in tqdm(group):
        print('session: ', session)
        mfold = Multifold(version='{}_trial{}'.format(opt['NAME'],itrials),verbose=flags.verbose)
        mfold.PrepareModel(nvars=data.shape[1]) # sets the dims for the MLP model used and defines the 2 models (step 1 and 2)
        mfold.LoadModel(iteration=opt['NITER']-1, weights_folder_path=session, strapn=0) # loads weights for model 2
        omnifold_weights = mfold.reweight(mc_gen[gen_mask],mfold.model2) 

        feed_dict_data=1-mc_gen[gen_mask][:,0]
        output_data, bins = np.histogram(feed_dict_data, bins=utils.binning, weights = omnifold_weights[gen_mask], density=False)
        # print(output_data)
        hist_sessions_data.append(output_data)
        bins_.append(bins)

    # sd = np.std(hist_sessions_data,axis=0)
    ave = np.mean(hist_sessions_data,axis=0)
    session_dict['group_'+str(group_num)] = ave
    # sd_over_mean = sd/ave
    group_num=group_num+1

# pprint(session_dict)

# calculate sd of the averages
sd_40 = np.std(list(session_dict.values()),axis=0)
ave_40 = np.mean(list(session_dict.values()),axis=0)
sd_over_mean_40 = sd_40/ave_40

# keys to present sd of ave per bin
bins = ['bin_'+str(i) for i in utils.binning]
sd_dict_of_ave_bins_40 = dict(zip(bins, sd_40))
# pprint(sd_dict_of_ave_bins)

#subset of 40 
subset_40 = sets_of_40[1]

sub_hist_sessions_data =[]
sub_bins_=[]
for session in subset_40:
    mfold = Multifold(version='{}_trial{}'.format(opt['NAME'],itrials),verbose=flags.verbose)
    mfold.PrepareModel(nvars=data.shape[1]) # sets the dims for the MLP model used and defines the 2 models (step 1 and 2)
    mfold.LoadModel(iteration=opt['NITER']-1, weights_folder_path=session, strapn=0) # loads weights for model 2
    omnifold_weights = mfold.reweight(mc_gen[gen_mask],mfold.model2) 
    
    feed_dict_data=1-mc_gen[gen_mask][:,0]
    output_data, bins = np.histogram(feed_dict_data, bins=utils.binning, weights = omnifold_weights[gen_mask], density=False)
    sub_hist_sessions_data.append(output_data)
    sub_bins_.append(bins) 

sub_sd_40 = np.std(sub_hist_sessions_data,axis=0)
sub_ave_40 = np.mean(sub_hist_sessions_data,axis=0)
sub_sd_over_mean_40 = sub_sd_40/sub_ave_40

sd_dict_sub_40 = dict(zip(bins,sub_sd_40))



# #Bootstrap Full n=10
# parser = argparse.ArgumentParser()

# parser.add_argument('--config', default='config_omnifold.json', help='Basic config file containing general options')
# parser.add_argument('--plot_folder', default='../plots', help='Folder used to store plots')
# parser.add_argument('--file_path', default='/global/cfs/cdirs/m3246/bnachman/LEP/aleph/processed', help='Folder containing input files')
# parser.add_argument('--nevts', type=float, default=-1, help='Dataset size to use during training')
# parser.add_argument('--verbose', action='store_true', default=False,help='Run the scripts with more verbose output')


# flags = parser.parse_args()
# nevts=int(flags.nevts)
# opt = LoadJson(flags.config)
# itrials=0

# data, mc_reco,mc_gen,reco_mask,gen_mask = utils.DataLoader(flags.file_path,opt,nevts)
# boot_folders=['/pscratch/sd/m/mavaylon/phys_bootstrap/OmniFold/boot_new_double_check/new_boot_'+str(i)+'/*' for i in range(1,11)]

    
# boot_n = 1
# session_dict={}
# for boot in boot_folders:
#     hist_sessions_data =[]
#     bins_=[]
#     errors =[]
#     for session in tqdm(glob(boot)[:40]):
#         print(session)
#         mfold = Multifold(version='{}_trial{}'.format(opt['NAME'],itrials),verbose=flags.verbose)
#         mfold.PrepareModel(nvars=data.shape[1]) # sets the dims for the MLP model used and defines the 2 models (step 1 and 2)
#         mfold.LoadModel(iteration=opt['NITER']-1, weights_folder_path=session, strapn=str(boot_n)) # loads weights for model 2
#         omnifold_weights = mfold.reweight(mc_gen[gen_mask],mfold.model2) 

#         feed_dict_data=1-mc_gen[gen_mask][:,0]
#         output_data, bins = np.histogram(feed_dict_data, bins=utils.binning, weights = omnifold_weights[gen_mask], density=False)

#         # print(output_data)
#         hist_sessions_data.append(output_data)
#         bins_.append(bins)
#     ave = np.mean(hist_sessions_data,axis=0)
#     session_dict['group_'+str(boot_n)] = ave
#     boot_n=boot_n+1
# # calculate sd of the averages
# sd_full = np.std(list(session_dict.values()),axis=0)
# ave_full = np.mean(list(session_dict.values()),axis=0)
# sd_over_mean_full = sd_full/ave_full

#Bootstrap Full n=40
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
boot_folders=['/pscratch/sd/m/mavaylon/phys_weights_n_40/bootstrap_n_'+str(i)+'/*' for i in range(1,41)]


    
boot_n = 1
session_dict={}
for boot in boot_folders:
    hist_sessions_data =[]
    bins_=[]
    errors =[]
    for session in tqdm(glob(boot)[:40]):
        print(session)
        mfold = Multifold(version='{}_trial{}'.format(opt['NAME'],itrials),verbose=flags.verbose)
        mfold.PrepareModel(nvars=data.shape[1]) # sets the dims for the MLP model used and defines the 2 models (step 1 and 2)
        mfold.LoadModel(iteration=opt['NITER']-1, weights_folder_path=session, strapn=str(boot_n)) # loads weights for model 2
        omnifold_weights = mfold.reweight(mc_gen[gen_mask],mfold.model2) 

        feed_dict_data=1-mc_gen[gen_mask][:,0]
        output_data, bins = np.histogram(feed_dict_data, bins=utils.binning, weights = omnifold_weights[gen_mask], density=False)

        # print(output_data)
        hist_sessions_data.append(output_data)
        bins_.append(bins)
    ave = np.mean(hist_sessions_data,axis=0)
    session_dict['group_'+str(boot_n)] = ave
    boot_n=boot_n+1
# calculate sd of the averages
sd_full_40 = np.std(list(session_dict.values()),axis=0)
ave_full_40 = np.mean(list(session_dict.values()),axis=0)
sd_over_mean_full_40 = sd_full_40/ave_full_40

#Bootstrap Sim Full n=40
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
boot_folders=['/pscratch/sd/m/mavaylon/boot_sim_n_40/boot_sim_'+str(i)+'/*' for i in range(1,41)]


    
boot_n = 1
session_dict={}
for boot in boot_folders:
    hist_sessions_data =[]
    bins_=[]
    errors =[]
    for session in tqdm(glob(boot)[:40]):
        print(session)
        mfold = Multifold(version='{}_trial{}'.format(opt['NAME'],itrials),verbose=flags.verbose)
        mfold.PrepareModel(nvars=data.shape[1]) # sets the dims for the MLP model used and defines the 2 models (step 1 and 2)
        mfold.LoadModel(iteration=opt['NITER']-1, weights_folder_path=session, strapn=str(boot_n)) # loads weights for model 2
        omnifold_weights = mfold.reweight(mc_gen[gen_mask],mfold.model2) 

        feed_dict_data=1-mc_gen[gen_mask][:,0]
        output_data, bins = np.histogram(feed_dict_data, bins=utils.binning, weights = omnifold_weights[gen_mask], density=False)

        # print(output_data)
        hist_sessions_data.append(output_data)
        bins_.append(bins)
    ave = np.mean(hist_sessions_data,axis=0)
    session_dict['group_'+str(boot_n)] = ave
    boot_n=boot_n+1
# calculate sd of the averages
sd_sim_full_40 = np.std(list(session_dict.values()),axis=0)
ave_sim_full_40 = np.mean(list(session_dict.values()),axis=0)
sd_over_mean_full_40_sim = sd_sim_full_40/ave_sim_full_40


# #Bootstrap Quarter
# parser = argparse.ArgumentParser()

# parser.add_argument('--config', default='config_omnifold.json', help='Basic config file containing general options')
# parser.add_argument('--plot_folder', default='../plots', help='Folder used to store plots')
# parser.add_argument('--file_path', default='/global/cfs/cdirs/m3246/bnachman/LEP/aleph/processed', help='Folder containing input files')
# parser.add_argument('--nevts', type=float, default=-1, help='Dataset size to use during training')
# parser.add_argument('--verbose', action='store_true', default=False,help='Run the scripts with more verbose output')


# flags = parser.parse_args()
# nevts=int(flags.nevts)
# opt = LoadJson(flags.config)
# itrials=0

# data, mc_reco,mc_gen,reco_mask,gen_mask = utils.DataLoader(flags.file_path,opt,nevts)
# boot_folders=['/pscratch/sd/m/mavaylon/phys_bootstrap/OmniFold/boot_quarter_stratified_check/boot_quarter_'+str(i)+'/*' for i in range(1,11)]


    
# boot_n = 1
# session_dict={}
# for boot in boot_folders:
#     hist_sessions_data =[]
#     bins_=[]
#     errors =[]
#     random.seed(11)
#     boot_40=random.sample(glob(boot), k=40)
#     for session in tqdm(boot_40):
#     # for session in tqdm(glob(boot)[:40]):
#         print(session)
#         mfold = Multifold(version='{}_trial{}'.format(opt['NAME'],itrials),verbose=flags.verbose)
#         mfold.PrepareModel(nvars=data.shape[1]) # sets the dims for the MLP model used and defines the 2 models (step 1 and 2)
#         mfold.LoadModel(iteration=opt['NITER']-1, weights_folder_path=session, strapn=str(boot_n)) # loads weights for model 2
#         omnifold_weights = mfold.reweight(mc_gen[gen_mask],mfold.model2) 

#         feed_dict_data=1-mc_gen[gen_mask][:,0]
#         output_data, bins = np.histogram(feed_dict_data, bins=utils.binning, weights = omnifold_weights[gen_mask], density=False)

#         # print(output_data)
#         hist_sessions_data.append(output_data)
#         bins_.append(bins)
#     ave = np.mean(hist_sessions_data,axis=0)
#     session_dict['group_'+str(boot_n)] = ave
#     boot_n=boot_n+1
# # calculate sd of the averages
# sd_quarter = np.std(list(session_dict.values()),axis=0)
# ave_quarter = np.mean(list(session_dict.values()),axis=0)
# sd_over_mean_quarter = sd_quarter/ave_quarter


feed_dict = {
        'sd_40': sd_40,
        'sub_sd_40': sub_sd_40,
        'ave_40': ave_40,
        'sub_ave_40': sub_ave_40,
        'sd_over_mean_40': sd_over_mean_40,
        'sub_sd_over_mean_40': sub_sd_over_mean_40,
        # 'sd_over_mean_quarter': sd_over_mean_quarter,
        # 'sd_over_mean_full': sd_over_mean_full,
        'sd_over_mean_full_40': sd_over_mean_full_40,
        'sd_over_mean_full_40_sim' : sd_over_mean_full_40_sim,
        'mc gen':1-mc_gen[gen_mask][:,0],
        'est_sim_unc': est_sim_stat,
        'est_real_unc': est_real_stat
    }

fig,ax = utils.SD_Plot(feed_dict,
                           weights = None,
                           binning=utils.binning,
                           xlabel='1-T',logy=False,
                           ylabel='Uncertainty',
                           reference_name='mc gen',
                           sub=True,
                           density=False,
                           show_data=False,
                           est_uncertainty=True)
fig.savefig('{}/{}_{}.pdf'.format(flags.plot_folder,"Bootstrap_with_Sim_boot_n=40",opt['NAME']))


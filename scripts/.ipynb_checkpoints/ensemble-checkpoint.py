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

weights_paths = glob('/pscratch/sd/m/mavaylon/weights_original_mlp/*')

# # check that all the weights are completed
# for folder in weights_paths:
#     num_weights=len(glob(folder+'/*'))
#     if num_weights!=10:
#         break
#     else:
#         continue
        
# create sets of 20
rem = len(weights_paths)%20
if rem!=0: # gurantee even sets of 20
    number_of_sessions = len(weights_paths)-rem
    random.seed(2)
    weights_paths = random.choices(weights_paths,k=number_of_sessions)
    sets_of_20 = [weights_paths[i:i + 20] for i in range(0, len(weights_paths), 20)]
else:
    sets_of_20 = [weights_paths[i:i + 20] for i in range(0, len(weights_paths), 20)]
    
group_num=1
set_200 = sets_of_20[:200]
session_dict={}
for group in tqdm(set_200):
    hist_sessions_data =[]
    bins_=[]
    for session in tqdm(group):
        # print(session)
        mfold = Multifold(version='{}_trial{}'.format(opt['NAME'],itrials),verbose=flags.verbose)
        mfold.PrepareModel(nvars=data.shape[1]) # sets the dims for the MLP model used and defines the 2 models (step 1 and 2)
        mfold.LoadModel(iteration=opt['NITER']-1, weights_folder_path=session) # loads weights for model 2
        omnifold_weights = mfold.reweight(mc_gen[gen_mask],mfold.model2) 

        feed_dict_data=1-mc_gen[gen_mask][:,0]
        output_data, bins = np.histogram(feed_dict_data, bins=utils.binning, weights = omnifold_weights[gen_mask], density=False)
        # print(output_data)
        hist_sessions_data.append(output_data)
        bins_.append(bins)

    # sd = np.std(hist_sessions_data,axis=0)
    ave = np.median(hist_sessions_data,axis=0)
    session_dict['group_'+str(group_num)] = ave
    # sd_over_mean = sd/ave
    group_num=group_num+1

# pprint(session_dict)

# calculate sd of the averages
sd = np.std(list(session_dict.values()),axis=0)
ave = np.median(list(session_dict.values()),axis=0)
sd_over_mean = sd/ave

# keys to present sd of ave per bin
bins = ['bin_'+str(i) for i in utils.binning]
sd_dict_of_ave_bins = dict(zip(bins, sd))
# pprint(sd_dict_of_ave_bins)

#subset of 20 
subset_weights_paths = glob('/global/homes/m/mavaylon/phys/OmniFold/sub_20_new/*')

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
sub_ave = np.median(sub_hist_sessions_data,axis=0)
sub_sd_over_mean = sub_sd/sub_ave

sd_dict_sub = dict(zip(bins,sub_sd))


# create sets of 10
rem = len(weights_paths)%10
if rem!=0: # gurantee even sets of 20
    number_of_sessions = len(weights_paths)-rem
    random.seed(2)
    weights_paths = random.choices(weights_paths,k=number_of_sessions)
    sets_of_10 = [weights_paths[i:i + 10] for i in range(0, len(weights_paths), 10)]
else:
    sets_of_10 = [weights_paths[i:i + 10] for i in range(0, len(weights_paths), 10)]
    
group_num=1
set_100 = sets_of_10[:100]
session_dict_10={}
for group in tqdm(set_100):
    hist_sessions_data =[]
    bins_=[]
    for session in tqdm(group):
        # print(session)
        mfold = Multifold(version='{}_trial{}'.format(opt['NAME'],itrials),verbose=flags.verbose)
        mfold.PrepareModel(nvars=data.shape[1]) # sets the dims for the MLP model used and defines the 2 models (step 1 and 2)
        mfold.LoadModel(iteration=opt['NITER']-1, weights_folder_path=session) # loads weights for model 2
        omnifold_weights = mfold.reweight(mc_gen[gen_mask],mfold.model2) 

        feed_dict_data=1-mc_gen[gen_mask][:,0]
        output_data, bins = np.histogram(feed_dict_data, bins=utils.binning, weights = omnifold_weights[gen_mask], density=False)
        # print(output_data)
        hist_sessions_data.append(output_data)
        bins_.append(bins)

    # sd = np.std(hist_sessions_data,axis=0)
    ave = np.median(hist_sessions_data,axis=0)
    session_dict_10['group_'+str(group_num)] = ave
    # sd_over_mean = sd/ave
    group_num=group_num+1

# pprint(session_dict_10)

# calculate sd of the averages
sd_10 = np.std(list(session_dict_10.values()),axis=0)
ave_10 = np.median(list(session_dict_10.values()),axis=0)
sd_over_mean_10 = sd_10/ave_10

# keys to present sd of ave per bin
bins = ['bin_'+str(i) for i in utils.binning]
sd_dict_of_ave_bins_10 = dict(zip(bins, sd_10))
# pprint(sd_dict_of_ave_bins)

#subset of 10 
subset_weights_paths = glob('/global/homes/m/mavaylon/phys/OmniFold/subweights_10/*')

sub_hist_sessions_data_10 =[]
sub_bins_10=[]
for session in subset_weights_paths:
    mfold = Multifold(version='{}_trial{}'.format(opt['NAME'],itrials),verbose=flags.verbose)
    mfold.PrepareModel(nvars=data.shape[1]) # sets the dims for the MLP model used and defines the 2 models (step 1 and 2)
    mfold.LoadModel(iteration=opt['NITER']-1, weights_folder_path=session) # loads weights for model 2
    omnifold_weights = mfold.reweight(mc_gen[gen_mask],mfold.model2) 
    
    feed_dict_data=1-mc_gen[gen_mask][:,0]
    output_data, bins = np.histogram(feed_dict_data, bins=utils.binning, weights = omnifold_weights[gen_mask], density=False)
    sub_hist_sessions_data_10.append(output_data)
    sub_bins_10.append(bins) 

sub_sd_10 = np.std(sub_hist_sessions_data_10,axis=0)
sub_ave_10 = np.median(sub_hist_sessions_data_10,axis=0)
sub_sd_over_mean_10 = sub_sd_10/sub_ave_10

sd_dict_sub = dict(zip(bins,sub_sd_10))



# estimate statistical uncertainty 
sim_det_data = 1-mc_reco[reco_mask]
output_sim_data, bins = np.histogram(sim_det_data, bins=utils.binning, density=False)
est_sim_stat = 1/np.sqrt(output_sim_data)
# print(est_sim_stat)

real_data = 1-data[:,0]
output_real_data, bins = np.histogram(real_data, bins=utils.binning, density=False)
est_real_stat = 1/np.sqrt(output_real_data)
# print(est_real_stat)



print('Difference:', np.array(list(sd_dict_sub.values()))/np.array(list(sd_dict_of_ave_bins.values())))

feed_dict = {
        'sd': sd,
        'sub_sd': sub_sd,
        'ave': ave,
        'sub_ave': sub_ave,
        'sd_over_mean': sd_over_mean,
        'sub_sd_over_mean': sub_sd_over_mean,
        'sd_10': sd_10,
        'sub_sd_10': sub_sd_10,
        'ave_10': ave_10,
        'sub_ave_10': sub_ave_10,
        'sd_over_mean_10': sd_over_mean_10,
        'sub_sd_over_mean_10': sub_sd_over_mean_10,
        'mc gen':1-mc_gen[gen_mask][:,0],
        'est_sim_unc': est_sim_stat,
        'est_real_unc': est_real_stat
    }

fig,ax = utils.SD_Plot(feed_dict,
                           weights = None,
                           binning=utils.binning,
                           xlabel='1-T',logy=False,
                           ylabel='Relative Uncertainty',
                           reference_name='mc gen',
                           sub=True,
                           density=False,
                           show_data=False,
                          est_uncertainty=True)
fig.savefig('{}/{}_{}.pdf'.format(flags.plot_folder,"Ensembl_Coef_of_Var_20_median_new_mlp",opt['NAME']))


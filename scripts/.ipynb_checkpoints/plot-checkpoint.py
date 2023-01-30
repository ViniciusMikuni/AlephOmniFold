import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import tensorflow as tf
import utils
from omnifold import  Multifold,LoadJson
import tensorflow.keras.backend as K

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

data, mc_reco,mc_gen,reco_mask,gen_mask = utils.DataLoader(flags.file_path,opt,nevts)

# saved_plots_folder = '../plots/'+date_time
# if not os.path.exists(saved_plots_folder):
#     os.makedirs(saved_plots_folder)

for itrial in range(opt['NTRIAL']):
    mfold = Multifold(version='{}_trial{}'.format(opt['NAME'],itrial),verbose=flags.verbose)
    mfold.PrepareModel(nvars=data.shape[1]) # sets the dims for the MLP model used and defines the 2 models (step 1 and 2)
    mfold.LoadModel(iteration=opt['NITER']-1, weights_folder_path='/global/homes/m/mavaylon/phys/OmniFold/weights/07112022_12:29:49') # loads weights for model 2
    omnifold_weights = mfold.reweight(mc_gen[gen_mask],mfold.model2) #look at code and ask vini, also mc_gen==mc_gen[gen_mask]
    
    # feed_dict_data=1-mc_gen[gen_mask][:,0]
    # o_data, bins = np.histogram(feed_dict_data, bins=utils.binning, weights = omnifold_weights[gen_mask], density=True)
    # hist_sessions_data.append(o_data)
    # print(session)
    
    # print(omnifold_weights)
    if itrial==0:
        weight_dict = {
            'mc gen':np.ones(mc_gen.shape[0]),
            'data': omnifold_weights[gen_mask]
        }

        feed_dict = {
            'mc gen':1-mc_gen[gen_mask][:,0], # why 1-...
            'data':1-mc_gen[gen_mask][:,0],
        }

        fig,ax = utils.HistRoutine(feed_dict,plot_ratio=True,
                                   weights = weight_dict,
                                   binning=utils.binning,
                                   xlabel='1-T',logy=True,
                                   ylabel='Normalized events',
                                   reference_name='mc gen')
        fig.savefig('{}/{}_{}.pdf'.format(flags.plot_folder,"Unfolded_Hist_T",opt['NAME']))

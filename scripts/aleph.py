import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import horovod.tensorflow.keras as hvd
import tensorflow as tf
import utils
from omnifold import  Multifold,LoadJson
import tensorflow.keras.backend as K

utils.SetStyle()

hvd.init()
# Horovod: pin GPU to be used to process local rank (one GPU per process)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')


parser = argparse.ArgumentParser()

parser.add_argument('--config', default='config_omnifold.json', help='Basic config file containing general options')
parser.add_argument('--plot_folder', default='../plots', help='Folder used to store plots')
parser.add_argument('--file_path', default='/global/cfs/cdirs/m3246/bnachman/LEP/aleph/processed/', help='Folder containing input files')
parser.add_argument('--nevts', type=float,default=-1, help='Dataset size to use during training')
parser.add_argument('--strapn', type=int,default=0, help='Index of the bootstrap to run. 0 means no bootstrap')
parser.add_argument('--verbose', action='store_true', default=False,help='Run the scripts with more verbose output')


flags = parser.parse_args()
nevts=int(flags.nevts)
opt = LoadJson(flags.config)

if not os.path.exists(flags.plot_folder):
    os.makedirs(flags.plot_folder)


data, mc_reco,mc_gen,reco_mask,gen_mask = utils.DataLoader(flags.file_path,opt,nevts)

if hvd.rank()==0:
    #Let's make a simple histogram of the feature we want to unfold
    feed_dict={
        'data reco':1-data[:,0],
        'mc reco':1-mc_reco[reco_mask],
        'mc gen':1-mc_gen[gen_mask],
    }
    
    fig,ax = utils.HistRoutine(feed_dict,plot_ratio=True,
                               binning=utils.binning,
                               xlabel='1-T',logy=True,
                               ylabel='Normalized events',
                               reference_name='mc gen')
    
    fig.savefig('{}/{}.pdf'.format(flags.plot_folder,"Hist_T"))

for itrial in range(opt['NTRIAL']):
    K.clear_session()
    mfold = Multifold(version='{}_trial{}_strapn{}'.format(opt['NAME'],itrial,flags.strapn),
                      strapn=flags.strapn,verbose=flags.verbose)
    mfold.mc_gen = mc_gen
    mfold.mc_reco =mc_reco
    mfold.data = data

    tf.random.set_seed(itrial)
    mfold.Preprocessing(pass_reco=reco_mask,pass_gen=gen_mask)
    mfold.Unfold()

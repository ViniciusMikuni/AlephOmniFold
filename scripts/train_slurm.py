'''
Author: Anthony Badea
Date: May 27, 2024
'''

# fix for keras v3.0 update
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1' 

# tensorflow
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K

# python based
import random
from pathlib import Path
import time
import argparse
import json
import submitit
import shutil
import h5py
import numpy as np

# custom code
import dataloader
from omnifold import  Multifold, LoadJson

# set gpu growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

def train(
    conf
):

    print(conf)
    
    # update %j with actual job number
    output_directory = conf["output_directory"]
    try:
        job_env = submitit.JobEnvironment()
        job_id = job_env.job_id
        output_directory = Path(str(output_directory).replace("%j", str(job_id)))
    except:
        job_id = random.randrange(16**8)
        output_directory = Path(str(output_directory).replace("%j", "%08x" % job_id))
        
    os.makedirs(output_directory, exist_ok=True)
    print(output_directory)

    # save configurations to config file
    configPath = Path(output_directory, "config_omnifold.json").resolve()
    with open(str(configPath), "w") as outfile: 
      json.dump(conf, outfile)
    
    # load data
    data, mc_reco, mc_gen, reco_mask, gen_mask = dataloader.DataLoader(conf)

    # create Poisson(1) weights
    weights_mc = np.random.poisson(1, mc_gen.shape[0]) if conf["poisson_weights"] else None
    weights_data = np.random.poisson(1, data.shape[0]) if conf["poisson_weights"] else None

    # make weights directory
    weights_folder = Path(output_directory, "./model_weights").resolve()
    weights_folder.mkdir()
    weights_folder = str(weights_folder)
    
    # launch training
    for itrial in range(conf['NTRIAL']):
      K.clear_session()
      mfold = Multifold(version='{}_trial{}_strapn{}'.format(conf['NAME'],itrial,conf["strapn"]),
                        strapn=conf["strapn"],
                        verbose=conf["verbose"],
                        run_id=job_id,
                        boot=conf["boot"],
                        weights_folder=weights_folder,
                        config_file=configPath
      )
      mfold.mc_gen = mc_gen # the sim pre-detector
      mfold.mc_reco = mc_reco # sim post-detector
      mfold.data = data # experimental real data
      
      # tf.random.set_seed(itrial)
      mfold.Preprocessing(weights_mc=weights_mc, weights_data=weights_data, pass_reco=reco_mask, pass_gen=gen_mask)
      """
      Preprocessing goes as follows
      self.PrepareWeights(weights_mc,weights_data,pass_reco,pass_gen)        
      self.PrepareInputs()
      self.PrepareModel(nvars = self.mc_gen.shape[1])
      
      """
      mfold.Unfold() # note that this will automatically set a random seed based on random_training_seed inside of omnifold.py

      # get weights
      omnifold_weights = mfold.reweight(mc_gen[gen_mask],mfold.model2)
      
      # save weights to h5 file
      outFileName = Path(output_directory, "omnifold_weights.h5").resolve()
      with h5py.File(outFileName, 'w') as hf:
        hf.create_dataset("weights", data=omnifold_weights)
        
if __name__ == "__main__":

    # set up command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--slurm", help="path to json file containing slurm configuration", default=None)
    parser.add_argument("--njobs", help="number of jobs to actually launch. default is all", default=-1, type=int)
    parser.add_argument('--verbose', action='store_true', default=False,help='Run the scripts with more verbose output')
    parser.add_argument('--run_systematics', action='store_true', default=False,help='Run the track and event selection systematic variations')
    parser.add_argument('--run_bootstrap_mc', action='store_true', default=False,help='Run the bootstrapping for MC')
    parser.add_argument('--run_bootstrap_data', action='store_true', default=False,help='Run the bootstrapping for data')
    parser.add_argument('--run_ensembling', action='store_true', default=False,help='Run the ensembling by retraining without changing the inputs')
    args = parser.parse_args()
    
    # read in query
    if Path(args.slurm).resolve().exists():
        query_path = Path(args.slurm).resolve()
    else:
        # throw
        raise ValueError(f"Could not locate {args.slurm} in query directory or as absolute path")
    with open(query_path) as f:
        query = json.load(f)

    # create top level output directory
    top_dir = Path("results", f'./training-{"%08x" % random.randrange(16**8)}', "%j").resolve()

    # shared training configuration
    training_conf = {
      'output_directory' : str(top_dir),
      'FILE_MC':'/home/badea/e+e-/aleph/data/processed/20220514/alephMCRecoAfterCutPaths_1994_ThrustReprocess.npz',
      'FILE_DATA':'/home/badea/e+e-/aleph/data/processed/20220514/LEP1Data1994_recons_aftercut-MERGED_ThrustReprocess.npz',
      'TrackVariation': 0, # nominal track selection
      'EvtVariation': 0, # nominal event selection
      'NITER': 5,
      'NTRIAL':1,
      'LR': 1e-3,
      'BATCH_SIZE': 5000,
      'EPOCHS': 500,
      'NWARMUP': 5,
      'NAME':'toy',
      'NPATIENCE': 10,
      'strapn' : 0,
      'verbose' : args.verbose,
      'boot' : None,
      'poisson_weights' : False
    }
    
    # list of configurations to launch
    confs = []

    # add configurations for track and event selection systematic variations
    if args.run_systematics:
      for TrackVariation in range(0, 9):
        for EvtVariation in range(0, 2):
          temp = training_conf.copy() # copy overall
          temp["TrackVariation"] = TrackVariation
          temp["EvtVariation"] = EvtVariation
          temp["poisson_weights"] = True
          confs.append(temp)

    # add configurations for bootstrap mc or data
    if args.run_bootstrap_mc or args.run_bootstrap_data:
      boot = "mc" if args.run_bootstrap_mc else "data"
      nstraps = 40
      for strapn in range(nstraps):
        temp = training_conf.copy() # copy overall
        temp["boot"] = boot # the combination of boot and strapn will automatically do poisson weights within omnifold.py
        temp["strapn"] = strapn
        temp["poisson_weights"] = False
        confs.append(temp)

    # add configurations for ensembling
    if args.run_ensembling:
      nensemble = 40
      for i in range(nensemble):
        temp = training_conf.copy()
        confs.append(temp)
        
    # if submitit false then just launch job
    if not query.get("submitit", False):
        for iC, conf in enumerate(confs):
            # only launch a single job
            if args.njobs != -1 and (iC+1) > args.njobs:
                continue
            print(conf)
            train(conf)
        exit()
    

    # submission
    executor = submitit.AutoExecutor(folder=top_dir)
    executor.update_parameters(**query.get("slurm", {}))
    # the following line tells the scheduler to only run at most 2 jobs at once. By default, this is several hundreds
    # executor.update_parameters(slurm_array_parallelism=2)
    
    # loop over configurations
    jobs = []
    with executor.batch():
        for iC, conf in enumerate(confs):
            
            # only launch a single job
            if args.njobs != -1 and (iC+1) > args.njobs:
                continue
            
            print(conf)

            job = executor.submit(train, conf) # **conf
            jobs.append(job)

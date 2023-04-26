import numpy as np
import tensorflow as tf
import tensorflow.keras
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense, InputLayer, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint, ReduceLROnPlateau
import sys, os
import horovod.tensorflow.keras as hvd
import json, yaml
import utils
from datetime import datetime
import keras_tuner
from keras_tuner.tuners import RandomSearch, GridSearch

from omnifold import  Multifold, LoadJson, weighted_binary_crossentropy

import argparse

utils.SetStyle()

hvd.init()
# Horovod: pin GPU to be used to process local rank (one GPU per process)
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
# if gpus:
#     tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
#     for gpu in gpus:
#         tf.config.experimental.set_memory_growth(gpu, True)
# else:
#     print("No GPU device found")


parser = argparse.ArgumentParser()

parser.add_argument('--config', default='config_omnifold.json', help='Basic config file containing general options')
parser.add_argument('--plot_folder', default='../plots', help='Folder used to store plots')
parser.add_argument('--file_path', default='/global/homes/m/mavaylon/phys/Data/', help='Folder containing input files')
parser.add_argument('--nevts', type=float,default=-1, help='Dataset size to use during training')
parser.add_argument('--verbose', action='store_true', default=False,help='Run the scripts with more verbose output')
parser.add_argument('--run_id', type=int, default=0, help='integer ID to index a run')


flags = parser.parse_args()
nevts=int(flags.nevts)
opt = LoadJson('/global/homes/m/mavaylon/phys/OmniFold/scripts/config_omnifold.json')

if not os.path.exists(flags.plot_folder):
    os.makedirs(flags.plot_folder)


data, mc_reco,mc_gen,reco_mask,gen_mask = utils.DataLoader(flags.file_path,opt,nevts)

K.clear_session()
mfold = Multifold(version='tune',verbose=flags.verbose, run_id=flags.run_id,tune=True)
mfold.mc_gen = mc_gen
mfold.mc_reco =mc_reco
mfold.data = data

mfold.Preprocessing(pass_reco=reco_mask,pass_gen=gen_mask)
weights_push = np.ones(mfold.weights_mc.shape[0])

NTRAIN, NTEST, train_data, test_data = mfold.PrepareData(
            np.concatenate((mfold.mc_reco, mfold.data)),
            np.concatenate((mfold.labels_mc, mfold.labels_data)),
            np.concatenate((weights_push*mfold.weights_mc,mfold.weights_data)))

def build_MLP_Tune(hp): #tune
    ''' Define a simple fully conneted model to be used during unfolding'''
    model=tf.keras.Sequential()
    model.add(InputLayer((mc_gen.shape[1], )))
    for i in range(hp.Int('num_layers', 2, 4)):
        model.add(Dense(units=hp.Choice('units_'+ str(i),[4,16,32]),
                               activation=hp.Choice("activation", ["relu", "selu"])))
    model.add(Dense(1,activation='sigmoid'))
    
    hvd_lr = (1e-4)*np.sqrt(hvd.size())
    opt = tf.keras.optimizers.Adadelta(learning_rate=hvd_lr)
    opt = hvd.DistributedOptimizer(opt)

    model.compile(loss=weighted_binary_crossentropy,
                  optimizer=opt,experimental_run_tf_function=False)
    return model

tuner = keras_tuner.GridSearch(
    hypermodel=build_MLP_Tune,
    objective="val_loss",
    overwrite=True,
    directory='/pscratch/sd/m/mavaylon/phys/tune_grid',
    project_name='phys'
)

callbacks = [
            # hvd.callbacks.BroadcastGlobalVariablesCallback(0),
            # hvd.callbacks.MetricAverageCallback(),
            EarlyStopping(patience=mfold.opt['NPATIENCE'])
        ]

tuner.search(train_data,
            epochs=opt['EPOCHS'],
            steps_per_epoch=int(NTRAIN/mfold.BATCH_SIZE),
            validation_data=test_data,
            validation_steps=int(NTEST/mfold.BATCH_SIZE),
            callbacks=callbacks)



print(tuner.get_best_hyperparameters()[0].values)
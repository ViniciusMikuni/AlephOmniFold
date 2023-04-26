import numpy as np
import tensorflow as tf
import tensorflow.keras
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint, ReduceLROnPlateau
import sys, os
import horovod.tensorflow.keras as hvd
import json, yaml
import utils
from datetime import datetime
import keras_tuner
from keras_tuner.tuners import RandomSearch



def weighted_binary_crossentropy(y_true, y_pred):
    weights = tf.gather(y_true, [1], axis=1) # event weights
    y_true = tf.gather(y_true, [0], axis=1) # actual y_true for loss    
    # Clip the prediction value to prevent NaN's and Inf's
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    t_loss = -weights * ((y_true) * K.log(y_pred) +
                         (1 - y_true) * K.log(1 - y_pred))
    return K.mean(t_loss) 


def LoadJson(file_name):
    JSONPATH = os.path.join(file_name)
    return yaml.safe_load(open(JSONPATH))


class Multifold():
    def __init__(self,version,config_file='config_omnifold.json',verbose=False, run_id=0, tune=False):
        self.opt = LoadJson(config_file)
        if tune:
            self.niter = 1
            self.hp = keras_tuner.HyperParameters()
        else:
            self.niter = self.opt['NITER']
        self.version=version
        self.mc_gen = None
        self.mc_reco = None
        self.data=None
        self.verbose=verbose
        self.run_id = run_id
                
        
            
    def Unfold(self):
        self.BATCH_SIZE=self.opt['BATCH_SIZE']
        self.EPOCHS=self.opt['EPOCHS']
        
        self.weights_folder = '/global/homes/m/mavaylon/phys/OmniFold/weights'
        if not os.path.exists(self.weights_folder):
            os.makedirs(self.weights_folder)
            
        date_time = datetime.now().strftime("%d%m%Y_%H:%M:%S")
        self.saved_weights_folder = '../weights/'+date_time+'_gpu_'+str(self.run_id)
        if not os.path.exists(self.saved_weights_folder):
            os.makedirs(self.saved_weights_folder)
                                        
        self.weights_pull = np.ones(self.weights_mc.shape[0])
        self.weights_push = np.ones(self.weights_mc.shape[0])
        
        for i in range(self.niter):
            print("ITERATION: {}".format(i + 1))
            if self.niter==1:
                model=self.CompileModel(float(self.opt['LR']),self.model1),
                self.TuneModel(np.concatenate((self.mc_reco, self.data)),
                               np.concatenate((self.labels_mc, self.labels_data)),
                               np.concatenate((self.weights_push*self.weights_mc,self.weights_data)),
                               i,model=model,stepn=1,)
                continue
            else:
                self.CompileModel(float(self.opt['LR']),self.model1)
                self.RunStep1(i)
                self.CompileModel(float(self.opt['LR']),self.model2)
                self.RunStep2(i)
                
    def RunStep1(self,i):
        '''Data versus reco MC reweighting'''
        print("RUNNING STEP 1")
        self.RunModel(
            np.concatenate((self.mc_reco, self.data)),
            np.concatenate((self.labels_mc, self.labels_data)),
            np.concatenate((self.weights_push*self.weights_mc,self.weights_data)),
            i,self.model1,stepn=1,
        )
        

        new_weights=self.reweight(self.mc_reco,self.model1)
        #print(new_weights)
        new_weights[self.not_pass_reco]=1.0
        self.weights_pull = self.weights_push *new_weights
        if self.verbose:            
            print("Plotting the results after step 1")
            weight_dict = {
                'mc reco':self.weights_pull*self.weights_mc,
                'data reco': self.weights_data,
            }

            feed_dict = {
                'mc reco':1-self.mc_reco[:,0],
                'data reco':1-self.data[:,0],
            }

            fig,ax = utils.HistRoutine(feed_dict,plot_ratio=True,
                                       weights = weight_dict,
                                       binning=utils.binning,
                                       xlabel='1-T',logy=True,
                                       ylabel='Normalized events',
                                       reference_name='data reco')
            fig.savefig('../plots/Unfolded_Hist_T_step1_{}_{}.pdf'.format(i,self.opt['NAME']))
        

    def RunStep2(self,i):
        '''Gen to Gen reweighing'''        
        print("RUNNING STEP 2")
            
        self.RunModel(
            np.concatenate((self.mc_gen, self.mc_gen)),
            np.concatenate((self.labels_mc, self.labels_gen)),
            np.concatenate((self.weights_mc, self.weights_mc*self.weights_pull)),
            i,self.model2,stepn=2,
        )


        new_weights=self.reweight(self.mc_gen,self.model2)
        new_weights[self.not_pass_gen]=1.0
        self.weights_push = new_weights
        self.weights_push = self.weights_push/np.average(self.weights_push)

    def TuneModel(self,sample,labels,weights,iteration,model,stepn):
        self.BATCH_SIZE=self.opt['BATCH_SIZE']
        self.EPOCHS=self.opt['EPOCHS']
        tuner = RandomSearch(
            model,
            objective='weighted_binary_crossentropy',
            max_trials=50,
            executions_per_trial=1)
        mask = sample[:,0]!=-10
        # print(weights)
        # input()
        data = tf.data.Dataset.from_tensor_slices((
            sample[mask],
            np.stack((labels[mask],weights[mask]),axis=1))
        ).cache().shuffle(np.sum(mask))

        #Fix same number of training events between ranks
        NTRAIN,NTEST = self.GetNtrainNtest(np.sum(mask))
        test_data = data.take(NTEST).repeat().batch(self.BATCH_SIZE)
        train_data = data.skip(NTEST).repeat().batch(self.BATCH_SIZE)
        
        tuner.search(train_data, max_trials=50, executions_per_trial=1)
        
        return tuner.get_best_hyperparameters()
    
    def PrepareData(self,sample,labels,weights):
        self.BATCH_SIZE=self.opt['BATCH_SIZE']
        self.EPOCHS=self.opt['EPOCHS']
        mask = sample[:,0]!=-10
       
        data = tf.data.Dataset.from_tensor_slices((
            sample[mask],
            np.stack((labels[mask],weights[mask]),axis=1))
        ).cache().shuffle(np.sum(mask))

        #Fix same number of training events between ranks
        NTRAIN,NTEST = self.GetNtrainNtest(np.sum(mask))
        test_data = data.take(NTEST).repeat().batch(self.BATCH_SIZE)
        train_data = data.skip(NTEST).repeat().batch(self.BATCH_SIZE)
        return NTRAIN, NTEST, train_data, test_data

    def RunModel(self,sample,labels,weights,iteration,model,stepn):
        
        mask = sample[:,0]!=-10
       
        data = tf.data.Dataset.from_tensor_slices((
            sample[mask],
            np.stack((labels[mask],weights[mask]),axis=1))
        ).cache().shuffle(np.sum(mask))

        #Fix same number of training events between ranks
        NTRAIN,NTEST = self.GetNtrainNtest(np.sum(mask))
        test_data = data.take(NTEST).repeat().batch(self.BATCH_SIZE)
        train_data = data.skip(NTEST).repeat().batch(self.BATCH_SIZE)

        verbose = 1 if hvd.rank() == 0 else 0
        
        callbacks = [
            hvd.callbacks.BroadcastGlobalVariablesCallback(0),
            hvd.callbacks.MetricAverageCallback(),
            # hvd.callbacks.LearningRateWarmupCallback(
            #     initial_lr=self.hvd_lr, warmup_epochs=self.opt['NWARMUP'],
            #     verbose=verbose),
            # ReduceLROnPlateau(patience=8, min_lr=1e-7,verbose=verbose),
            EarlyStopping(patience=self.opt['NPATIENCE'],restore_best_weights=True)
        ]
        
        
        if hvd.rank() ==0:
            callbacks.append(
                ModelCheckpoint('{}/{}_iter{}_step{}.h5'.format(
                    self.saved_weights_folder,self.version,iteration,stepn),
                                save_best_only=True,mode='auto',period=1,save_weights_only=True))
            
        _ =  model.fit(
            train_data,
            epochs=self.EPOCHS,
            steps_per_epoch=int(NTRAIN/self.BATCH_SIZE),
            validation_data=test_data,
            validation_steps=int(NTEST/self.BATCH_SIZE),
            verbose=verbose,
            callbacks=callbacks)
        

    def Preprocessing(self,weights_mc=None,weights_data=None,pass_reco=None,pass_gen=None):
        self.PrepareWeights(weights_mc,weights_data,pass_reco,pass_gen)
        self.PrepareInputs()
        self.PrepareModel(nvars = self.mc_gen.shape[1])

    def PrepareWeights(self,weights_mc,weights_data,pass_reco,pass_gen):
        
        if pass_reco is None:
            if self.verbose: print("No reco mask provided, making one based on inputs")
            self.not_pass_reco = self.mc_reco[:,0]==-10
        else:
            self.not_pass_reco = pass_reco==0
            self.mc_reco[self.not_pass_reco]=-10
            
        if pass_gen is None:
            if self.verbose: print("No gen mask provided, making one based on inputs")
            self.not_pass_gen = self.mc_gen[:,0]==-10
        else:
            self.not_pass_gen = pass_gen==0
            self.mc_gen[self.not_pass_gen]=-10

        
        if weights_mc is None:
            if self.verbose: print("No MC weights provided, making one filled with 1s")
            self.weights_mc = np.ones(self.mc_reco.shape[0])
        else:
            self.weights_mc = weights_mc

        if weights_data is None:
            if self.verbose: print("No data weights provided, making one filled with 1s")
            self.weights_data = np.ones(self.data.shape[0])
        else:
            self.weights_data =weights_data
            
        #Normalize MC weights to match the sum of data weights
        self.weights_mc = self.weights_mc/np.sum(self.weights_mc[self.not_pass_reco==0])
        self.weights_mc *= 1.0*self.weights_data.shape[0]
    def CompileModel(self,lr,model):
        self.hvd_lr = lr*np.sqrt(hvd.size())
        opt = tf.keras.optimizers.Adadelta(learning_rate=self.hvd_lr)
        opt = hvd.DistributedOptimizer(opt)

        model.compile(loss=weighted_binary_crossentropy,
                      optimizer=opt,experimental_run_tf_function=False)


    def PrepareInputs(self):
        self.labels_mc = np.zeros(len(self.mc_reco))
        self.labels_data = np.ones(len(self.data))
        self.labels_gen = np.ones(len(self.mc_gen))


    def PrepareModel(self,nvars):
        # inputs1,outputs1 = MLP_Tune(nvars,self.hp)
        inputs1,outputs1 = MLP(nvars)
        inputs2,outputs2 = MLP2(nvars)
                                   
        self.model1 = Model(inputs=inputs1, outputs=outputs1)
        self.model2 = Model(inputs=inputs2, outputs=outputs2)


    def GetNtrainNtest(self,nevts):
        NTRAIN=int(0.8*nevts)
        NTEST=int(0.2*nevts)                        
        return NTRAIN,NTEST

    def reweight(self,events,model):
        f = np.nan_to_num(model.predict(events, batch_size=10000)[:,:1],posinf=1,neginf=0) 
        # predict using step 2 model then Replace NaN with zero and infinity with large finite numbers. Here we set positive infinity as 1 and negative infinity as 0.
        weights = f / (1. - f) ## If we interpret f as probs then this is function for "odds". why?
        weights = weights[:,0]
        return np.squeeze(np.nan_to_num(weights,posinf=1))

    def LoadModel(self,iteration, weights_folder_path):
        # the weights_folder_path is the path to specific datetime folder
        model_name = '{}/{}_iter{}_step2.h5'.format(
            weights_folder_path,self.version,iteration)
        self.model2.load_weights(model_name)
    
    # def build_MLP_Tune(self,hp): #tune
    #     ''' Define a simple fully conneted model to be used during unfolding'''
    #     model=tf.keras.Sequential()
    #     model.add(Input((self.mc_gen.shape[1], )))
    #     for i in range(hp.Int('num_layers', 2, 4)):
    #         model.add(Dense(units=hp.Int('units_' + str(i),
    #                                         min_value=4,
    #                                         max_value=32,
    #                                         step=4),
    #                                activation=hp.Choice("activation", ["relu", "selu"])))
    #     model.add(Dense(1,activation='sigmoid'))
    #     return model



def MLP(nvars):
    ''' Define a simple fully conneted model to be used during unfolding'''
    inputs = Input((nvars, ))
    layer = Dense(4,activation='relu')(inputs)
    layer = Dense(4,activation='relu')(layer)
    layer = Dense(4,activation='relu')(layer)
    outputs = Dense(1,activation='sigmoid')(layer)
    return inputs,outputs

def MLP2(nvars):
    inputs = Input((nvars, ))
    layer = Dense(8,activation='selu')(inputs)
    layer = Dense(16,activation='selu')(layer)
    outputs = Dense(1,activation='sigmoid')(layer)
    return inputs,outputs

# def MLP(nvars,nensemb=10):
#     ''' Define a simple fully conneted model to be used during unfolding'''
#     inputs = Input((nvars, ))
#     net_trials=[]
#     for _ in range(nensemb):
#         layer = Dense(8,activation='selu')(inputs)
#         layer = Dense(16,activation='selu')(layer)
#         outputs = Dense(1,activation='sigmoid')(layer)
#         net_trials.append(outputs)

#     outputs = tf.reduce_mean(net_trials,0)
#     return inputs,outputs


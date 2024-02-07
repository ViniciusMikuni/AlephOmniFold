import numpy as np
import json, yaml
import pandas as pd

def DataLoader(config,nevts=-1, half=False):
    # hvd.init()
    if nevts==-1:
        nevts=None

    # load everything
    DATA = np.load(config['FILE_DATA'], allow_pickle=True)
    MC = np.load(config['FILE_MC'], allow_pickle=True)
    # config['TrackVariation'] = track selection variation
    # config['EvtVariation'] = event selection variation
    
    # pick out the correct entries
    data = DATA["t_thrust"][:,config['TrackVariation']]
    data_mask = DATA[f"t_passEventSelection_{config['EvtVariation']}"][:,config['TrackVariation']]
    mc_reco = MC["t_thrust"][:,config['TrackVariation']]
    reco_mask = MC[f"t_passEventSelection_{config['EvtVariation']}"][:,config['TrackVariation']]
    mc_gen = np.stack(MC["tgenBefore_thrust"]).flatten() # little hack because format is array([array([]), array([]), ....]) and we need array([,,,,])
    gen_mask = np.stack(MC[f"tgenBefore_passEventSelection"]).flatten()
    # print(data.shape, data_mask.shape, mc_reco.shape, reco_mask.shape, mc_gen.shape, gen_mask.shape)

    # append to reco to make the same length as tgenbefore
    diff = mc_gen.shape[0] - mc_reco.shape[0]
    mc_reco = np.concatenate([mc_reco, np.ones(diff)])
    reco_mask = np.concatenate([reco_mask, np.ones(diff).astype(bool)])
    # print(data.shape, data_mask.shape, mc_reco.shape, reco_mask.shape, mc_gen.shape, gen_mask.shape)
    
    if half:
        np.random.seed(2)
        df=pd.DataFrame({'data':data*10})
        df['data_mask']=data_mask
        df['bin']=pd.cut(df['data'],10)
        sample=df.groupby('bin',group_keys=False).apply(lambda x:x.sample(frac=.25))
        data=sample['data'].values/10
        data_mask=sample['data_mask'].values

        data = np.expand_dims(data[data_mask],-1)

        df=pd.DataFrame({'mc_reco': mc_reco*10})
        df['reco_mask'] = reco_mask == 1
        df['bin']=pd.cut(df['mc_reco'], 10)
        sample=df.groupby('bin',group_keys=False).apply(lambda x:x.sample(frac=.25))
        mc_reco=sample['mc_reco'].values/10
        reco_mask=sample['reco_mask'].values
        
        df=pd.DataFrame({'mc_gen': mc_gen*10})
        df['gen_mask'] = gen_mask == 1
        df['bin']=pd.cut(df['mc_gen'],10)
        sample=df.groupby('bin',group_keys=False).apply(lambda x:x.sample(frac=.25))
        sample=sample.sort_values(by='bin').iloc[:-1,:]
        mc_gen=sample['mc_gen'].values/10
        gen_mask=sample['gen_mask'].values

        mc_reco = np.expand_dims(mc_reco,-1)
        mc_gen = np.expand_dims(mc_gen,-1)
        # breakpoint()
        return data, mc_reco, mc_gen, reco_mask, gen_mask
    else:
        #We only want data events passing a selection criteria
        data = np.expand_dims(data[data_mask],-1)
        mc_reco = np.expand_dims(mc_reco, -1)
        reco_mask = reco_mask == 1
        mc_gen = np.expand_dims(mc_gen,-1)
        gen_mask = gen_mask == 1

        return data, mc_reco, mc_gen, reco_mask, gen_mask

if __name__ == "__main__":
    opt = yaml.safe_load(open("config_omnifold_test.json"))
    # print(opt)
    data, mc_reco, mc_gen, reco_mask, gen_mask = DataLoader(yaml.safe_load(open("config_omnifold_test.json")), nevts=-1, half=True)
    print(data.shape, mc_reco.shape, reco_mask.shape, mc_gen.shape, gen_mask.shape)

    # data, mc_reco, mc_gen, reco_mask, gen_mask = DataLoader(opt, nevts=-1, half=False)
    # print(data.shape, mc_reco.shape, reco_mask.shape, mc_gen.shape, gen_mask.shape)

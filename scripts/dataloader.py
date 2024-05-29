
import numpy as np
import json, yaml
import pandas as pd
from sklearn.model_selection import train_test_split

def DataLoader(config):

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

    ###### pick up the correct event ID's to use directly tgenbefore
    a = MC['tgen_uniqueID'] # WHES_id
    b = MC['tgenBefore_uniqueID'] # WHOES_id
    intersect, ind_a, ind_b = np.intersect1d(a, b, return_indices=True)
    
    pass_reco = np.zeros(len(b))
    pass_reco[ind_b] = reco_mask[ind_a]
    
    reco_vals = -999.*np.ones(len(b))
    reco_vals[ind_b] = mc_reco[ind_a]

    mc_reco = reco_vals
    reco_mask = pass_reco
    ########
    
    # append to reco to make the same length as tgenbefore
    diff = mc_gen.shape[0] - mc_reco.shape[0]
    mc_reco = np.concatenate([mc_reco, np.ones(diff)])
    reco_mask = np.concatenate([reco_mask, np.ones(diff).astype(bool)])
    print(data.shape, data_mask.shape, mc_reco.shape, reco_mask.shape, mc_gen.shape, gen_mask.shape)
    
    #We only want data events passing a selection criteria
    data = np.expand_dims(data[data_mask],-1)
    mc_reco = np.expand_dims(mc_reco, -1)
    reco_mask = reco_mask == 1
    mc_gen = np.expand_dims(mc_gen,-1)
    gen_mask = gen_mask == 1
    
    return data, mc_reco, mc_gen, reco_mask, gen_mask

if __name__ == "__main__":

    data, mc_reco, mc_gen, reco_mask, gen_mask = DataLoader(yaml.safe_load(open("config_omnifold_test.json")))
    print(data.shape, mc_reco.shape, reco_mask.shape, mc_gen.shape, gen_mask.shape)

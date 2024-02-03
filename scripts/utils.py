import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import argparse
import os
# import horovod.tensorflow.keras as hvd
import tensorflow as tf
import random
import pandas as pd
from sklearn.model_selection import train_test_split


line_style = {
    'data reco':'-',
    'mc gen':'-',
    'mc reco':'-',
    'data':'dotted',
}


colors = {
    'data':'black',
    'data reco':'black',
    'mc gen':'#7570b3',
    'mc reco':'#d95f02',
}

#binning=np.linspace(-4,4,50)
binning=np.linspace(0,0.5,20)
            
def FormatFig(xlabel,ylabel,ax0):
    #Limit number of digits in ticks
    # y_loc, _ = plt.yticks()
    # y_update = ['%.1f' % y for y in y_loc]
    # plt.yticks(y_loc, y_update) 
    ax0.set_xlabel(xlabel,fontsize=20)
    ax0.set_ylabel(ylabel)
        

    # xposition = 0.9
    # yposition=1.03
    # text = 'H1'
    # WriteText(xposition,yposition,text,ax0)


def SetStyle():
    from matplotlib import rc
    rc('text', usetex=True)

    import matplotlib as mpl
    rc('font', family='serif')
    rc('font', size=22)
    rc('xtick', labelsize=15)
    rc('ytick', labelsize=15)
    rc('legend', fontsize=15)

    # #
    mpl.rcParams.update({'font.size': 19})
    #mpl.rcParams.update({'legend.fontsize': 18})
    mpl.rcParams['text.usetex'] = False
    mpl.rcParams.update({'xtick.labelsize': 18}) 
    mpl.rcParams.update({'ytick.labelsize': 18}) 
    mpl.rcParams.update({'axes.labelsize': 18}) 
    mpl.rcParams.update({'legend.frameon': False}) 
    mpl.rcParams.update({'lines.linewidth': 2})
    
    import matplotlib.pyplot as plt
    import mplhep as hep
    hep.set_style(hep.style.CMS)
    hep.style.use("CMS") 

def SetGrid(ratio=True):
    fig = plt.figure(figsize=(9, 9))
    if ratio:
        gs = gridspec.GridSpec(2, 1, height_ratios=[3,1]) 
        gs.update(wspace=0.025, hspace=0.1)
    else:
        gs = gridspec.GridSpec(1, 1)
    return fig,gs

def Generator(nevts,ndim,mean=0,std=1):
    return np.random.normal(size=(nevts,ndim),loc=ndim*[mean],scale=ndim*[std])


def Detector(sample,bias=0,std=0.5):
    '''Assume a simple gaussian smearing for detector effect'''
    #Make a hole in the detector
    xmin=ymin=0.7
    xmax=ymax=1.2

    # xmin=0.7
    # ymin=-10
    # xmax= 10
    # ymax=10


    mask_x = (sample[:,0] > xmin) & (sample[:,0] < xmax)
    # mask_y = (sample[:,1] > ymin) & (sample[:,1] < ymax)
    mask = (mask_x)
    smeared = std*np.random.normal(size=sample.shape) + sample
    # smeared[mask==1] = -10    
    return smeared

    # mask_x = (smeared[:,0] > xmin) & (smeared[:,0] < xmax)
    # mask_y = (smeared[:,1] > ymin) & (smeared[:,1] < ymax)
    # mask = (mask_x) & (mask_y)
    # smeared[mask==1] = -10
    # return smeared


def DataLoader(base_path,config,nevts=-1, half=False):
    # hvd.init()
    if nevts==-1:nevts=None
    data = np.load(os.path.join(base_path,config['FILE_DATA_RECO'])) # [hvd.rank():nevts:hvd.size()]
    data_mask = np.load(os.path.join(base_path,config['FILE_DATA_FLAG_RECO'])) # [hvd.rank():nevts:hvd.size()]
    if half:
        np.random.seed(2)
        df=pd.DataFrame({'data':data*10})
        df['data_mask']=data_mask
        df['bin']=pd.cut(df['data'],10)
        sample=df.groupby('bin',group_keys=False).apply(lambda x:x.sample(frac=.25))
        data=sample['data'].values/10
        data_mask=sample['data_mask'].values
        
        data = np.expand_dims(data[data_mask],-1)
        
        df=pd.DataFrame({'mc_reco': np.load(os.path.join(base_path,config['FILE_MC_RECO']))*10}) # [hvd.rank():nevts:hvd.size()]*10})
        df['reco_mask'] = np.load(os.path.join(base_path,config['FILE_MC_FLAG_RECO']))==1 # [hvd.rank():nevts:hvd.size()]==1
        df['bin']=pd.cut(df['mc_reco'], 10)
        sample=df.groupby('bin',group_keys=False).apply(lambda x:x.sample(frac=.25))
        mc_reco=sample['mc_reco'].values/10
        reco_mask=sample['reco_mask'].values
        
        df=pd.DataFrame({'mc_gen': np.load(os.path.join(base_path,config['FILE_MC_GEN']))*10}) # [hvd.rank():nevts:hvd.size()]*10})
        df['gen_mask'] = np.load(os.path.join(base_path,config['FILE_MC_FLAG_GEN'])) == 1 # [hvd.rank():nevts:hvd.size()]==1
        df['bin']=pd.cut(df['mc_gen'],10)
        sample=df.groupby('bin',group_keys=False).apply(lambda x:x.sample(frac=.25))
        sample=sample.sort_values(by='bin').iloc[:-1,:]
        mc_gen=sample['mc_gen'].values/10
        gen_mask=sample['gen_mask'].values

        mc_reco = np.expand_dims(mc_reco,-1)
        mc_gen = np.expand_dims(mc_gen,-1)
        # breakpoint()
        return data, mc_reco,mc_gen,reco_mask,gen_mask
    else:  
        #We only want data events passing a selection criteria
        data = np.expand_dims(data[data_mask],-1)
        mc_reco = np.expand_dims(np.load(os.path.join(base_path,config['FILE_MC_RECO'])),-1) # [hvd.rank():nevts:hvd.size()],-1)
        mc_gen = np.expand_dims(np.load(os.path.join(base_path,config['FILE_MC_GEN']))-1) # [hvd.rank():nevts:hvd.size()],-1)
        reco_mask = np.load(os.path.join(base_path,config['FILE_MC_FLAG_RECO']))==1 # [hvd.rank():nevts:hvd.size()]==1
        gen_mask = np.load(os.path.join(base_path,config['FILE_MC_FLAG_GEN']))==1 # [hvd.rank():nevts:hvd.size()]==1
        # mc_reco[reco_mask==0]=-10
        # mc_gen[gen_mask==0]=-10

        return data, mc_reco,mc_gen,reco_mask,gen_mask

def Plot_2D(sample,name,use_hist=True,weights=None):
    #cmap = plt.get_cmap('PiYG')
    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad("white")
    # plt.rcParams['pcolor.shading'] ='nearest'

        
    def SetFig(xlabel,ylabel):
        fig = plt.figure(figsize=(8, 6))
        gs = gridspec.GridSpec(1, 1) 
        ax0 = plt.subplot(gs[0])
        ax0.yaxis.set_ticks_position('both')
        ax0.xaxis.set_ticks_position('both')
        ax0.tick_params(direction="in",which="both")    
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.xlabel(xlabel,fontsize=20)
        plt.ylabel(ylabel,fontsize=20)
        
        ax0.minorticks_on()
        return fig, ax0


    fig,ax = SetFig("x","y")

    
    if use_hist:
        if weights is None:
            weights = np.ones(sample.shape[0])
        im=plt.hist2d(sample[:,0],sample[:,1],
                      bins = 50,
                      range=[[-2,2],[-2,2]],
                      weights=weights,
                      cmap =cmap)
        cbar=fig.colorbar(im[3], ax=ax,label='Number of events')
    else:
        x=np.linspace(-2,2,50)
        y=np.linspace(-2,2,50)
        X,Y=np.meshgrid(x,y)
        im=ax.pcolormesh(X,Y,sample, cmap=cmap, shading='auto')
        fig.colorbar(im, ax=ax,label='Standard deviation')
        

    
    plot_folder='../plots'
    fig.savefig('{}/{}.pdf'.format(plot_folder,name))



def HistRoutine(feed_dict,xlabel='',ylabel='',reference_name='Geant4',logy=False,binning=None,label_loc='best',plot_ratio=True,weights=None,uncertainty=None):
    assert reference_name in feed_dict.keys(), "ERROR: Don't know the reference distribution"
    
    fig,gs = SetGrid(ratio=plot_ratio) 
    ax0 = plt.subplot(gs[0])
    if plot_ratio:
        plt.xticks(fontsize=0)
        ax1 = plt.subplot(gs[1],sharex=ax0)

    
    if binning is None:
        binning = np.linspace(np.quantile(feed_dict[reference_name],0.0),np.quantile(feed_dict[reference_name],1),10)
        
    xaxis = [(binning[i] + binning[i+1])/2.0 for i in range(len(binning)-1)]
    reference_hist,_ = np.histogram(feed_dict[reference_name],bins=binning,density=True)
    for ip,plot in enumerate(feed_dict.keys()):
        if weights is not None:
            dist,_,_=ax0.hist(feed_dict[plot],bins=binning,label=plot,linestyle=line_style[plot],color=colors[plot],density=True,histtype="step",weights=weights[plot])
        else:
            dist,_,_=ax0.hist(feed_dict[plot],bins=binning,label=plot,linestyle=line_style[plot],color=colors[plot],density=True,histtype="step")
        
        if plot_ratio:
            if reference_name!=plot:
                ratio = np.ma.divide(dist,reference_hist).filled(0)
                ax1.plot(xaxis,ratio,color=colors[plot],marker='o',ms=10,lw=0,markerfacecolor='none',markeredgewidth=3)
                if uncertainty is not None:
                    for ibin in range(len(binning)-1):
                        xup = binning[ibin+1]
                        xlow = binning[ibin]
                        ax1.fill_between(np.array([xlow,xup]),
                                         uncertainty[ibin],-uncertainty[ibin], alpha=0.3,color='k')    
    if logy:
        ax0.set_yscale('log')

    ax0.legend(loc=label_loc,fontsize=16,ncol=1)        
    if plot_ratio:
        FormatFig(xlabel = "", ylabel = ylabel,ax0=ax0) 
        plt.ylabel('Ratio to Gen.')
        plt.axhline(y=1.0, color='r', linestyle='-',linewidth=1)
        # plt.axhline(y=10, color='r', linestyle='--',linewidth=1)
        # plt.axhline(y=-10, color='r', linestyle='--',linewidth=1)
        plt.ylim([0.8,1.2])
        plt.xlabel(xlabel)
    else:
        FormatFig(xlabel = xlabel, ylabel = ylabel,ax0=ax0) 
        


  
    return fig,ax0

def SD_Plot(feed_dict,xlabel='',ylabel='',reference_name='Geant4',logy=False,binning=None,label_loc='best',weights=None,uncertainty=None, sub=False, density=True, show_data=True, est_uncertainty=False):
    # assert reference_name in feed_dict.keys(), "ERROR: Don't know the reference distribution"
    
    fig,gs = SetGrid() 
    ax0 = plt.subplot(gs[0])
    # ax0.margins(x=0, y=0.05)
    ax0.set_xlim([0,.35])
    ax0.set_ylim([0,.05])

    if binning is None:
        binning = np.linspace(np.quantile(feed_dict[reference_name],0.0),np.quantile(feed_dict[reference_name],1),10)
        binning = np.linspace(np.quantile(data,0.0),np.quantile(data,1),10)
    
    if show_data:
        if density:
            dist,edge,_=ax0.hist(feed_dict[reference_name],bins=binning,label=reference_name, density=True,histtype="step")
            bin_centres = (edge[:-1] + edge[1:]) / 2
        else:
            dist,edge,_=ax0.hist(feed_dict[reference_name],bins=binning,label=reference_name, density=False,histtype="step")
            bin_centres = (edge[:-1] + edge[1:]) / 2
    else:
        output_data, edge = np.histogram(feed_dict[reference_name],bins=binning, density=True)
        bin_centres = (edge[:-1] + edge[1:]) / 2
        
    if not sub:
        plt.plot(bin_centres,feed_dict['sd_over_mean_full'],'o', label="Bootstrap n=10")
    else:
        """
        Bootstrap
        """
        # plt.errorbar(x=bin_centres,y=feed_dict['sd_over_mean_full'],yerr=feed_dict['sd_over_mean_full']/np.sqrt(10), fmt='o', label="Bootstrap Full n=10")
        # plt.errorbar(x=bin_centres,y=feed_dict['sd_over_mean_quarter'],yerr=feed_dict['sd_over_mean_quarter']/np.sqrt(10), fmt='o', label="Bootstrap n=10 Quarter")
        plt.errorbar(x=bin_centres,y=feed_dict['sd_over_mean_full_40'],yerr=feed_dict['sd_over_mean_full_40']/np.sqrt(40), fmt='o', label="Bootstrap Data Full n=40")
        plt.errorbar(x=bin_centres,y=feed_dict['sd_over_mean_full_40_sim'],yerr=feed_dict['sd_over_mean_full_40_sim']/np.sqrt(40), fmt='o', label="Bootstrap Sim Full n=40")
        
        """
        Ensemble
        """
        plt.plot(bin_centres,feed_dict['sd_over_mean_40'],'o', label="Ensembl_40")
        plt.plot(bin_centres,feed_dict['sub_sd_over_mean_40'],'o', label="Sub_40")
        
    if est_uncertainty:
        plt.plot(bin_centres,feed_dict['est_sim_unc'],'*', label="est_sim_unc")
        plt.plot(bin_centres,feed_dict['est_real_unc'],'*', label="est_real_unc")
        
    if logy:
        ax0.set_yscale('log')

    ax0.legend(loc=label_loc,fontsize=16,ncol=1) 
    FormatFig(xlabel = xlabel, ylabel = ylabel,ax0=ax0) 
  
    return fig,ax0

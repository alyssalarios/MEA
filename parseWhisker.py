import miscAnalysis_salt as ma
import numpy as np
import scipy.io
import os
import glob
from sklearn.utils import shuffle
import scipy.optimize
import sklearn
from scipy.stats import f_oneway,ttest_ind, mannwhitneyu,pearsonr
import matplotlib.pyplot as plt
import math

from elephant.statistics import time_histogram, instantaneous_rate
from elephant.kernels import GaussianKernel
from quantities import ms, s, Hz
from neo.core import SpikeTrain


def removeIntanBuffer(path,spikeDict):
    """
    Use this function if there is a collection buffer from Intan acquisition. 
    must have a mat file in 'path' called bufferBoolean which is a vector of samples
    collected by Intan - preprocessing pipeline from the Ginty lab outputs this from 
    rhd files as variable 'board_dig_in_data'. must be concatenated and saved in 
    chronological order. 

    Removes buffer time samples from JRclust output dictionary and subtracts associated
    time steps so goodSamples and goodTimes reflect effective stimulation time, 
    not buffered acquisition time. path = data path. returns cleaned dictionary.
    
    possible that some units only spike during buffer time, data from those units would 
    remain in the rest of dictionary 
    """
    goodSamples = spikeDict['goodSamples']
    goodSpikes = spikeDict['goodSpikes']
    goodTimes = spikeDict['goodTimes']
    sampleRate = spikeDict['sampleRate']

    bufferDict = scipy.io.loadmat(path +  '\\bufferBoolean.mat')
    buffer = bufferDict['bufferBool']

    # create time vector and put 1 where a spike occurs
    allSamples = np.arange(0,np.shape(buffer)[1])
    goodInds = np.zeros(np.shape(allSamples))
    goodInds[goodSamples]= 1

    # filter out buffer samples and subtract the missing times
    filteredSamps= goodInds * buffer
    filtInds = np.asarray(filteredSamps >0).nonzero()[1]
    invertBuffer = np.invert(np.array(buffer[0],dtype = bool))
    missing = np.cumsum(invertBuffer)
    missingFilt = missing[filtInds]
    _ , _, bufferRemoveInd= np.intersect1d(filtInds,goodSamples,return_indices = True)

    goodSamples = goodSamples[bufferRemoveInd] - missingFilt
    goodTimes = goodTimes[bufferRemoveInd] - missingFilt / sampleRate
    goodSpikes = goodSpikes[bufferRemoveInd]

    spikeDict['goodSamples'] = goodSamples
    spikeDict['goodSpikes'] = goodSpikes
    spikeDict['goodTimes'] = goodTimes

    outDict = spikeDict 
    return outDict

def preprocessWhiskerData(path):
    """"
    ** must run matlab function first to get bufferbool.mat in directory 
    imports JRclust data and removes spikes during intan buffer period

    loads associated matlab file and saves stim info 

    saves JRclust data and matlab data in given directory 
    """
    os.chdir(path)

    spikeDict = ma.importJRCLUST(os.getcwd()+'\\alldata\\S0.mat')

    # if buffer time is recorded, if not comment this out
    spikeDict = removeIntanBuffer(os.getcwd(),spikeDict)

    # import matlab metadata file and extract relevant parameters
    matfile = glob.glob('*Metadata.mat')
    mat = scipy.io.loadmat(matfile[0])
    stimDict = {}
    stimDict['preStimTime'] = mat['preStimTime'][0][0]
    stimDict['stimDur'] = mat['stimDur'][0][0]
    stimDict['numRepeats'] = mat['numrepeats'][0][0]

    totalSeconds = int(stimDict['preStimTime']) + int(stimDict['stimDur'])
    numSamples = totalSeconds * spikeDict['sampleRate']
    stimDict['numSamples'] = numSamples
    
    np.save('stimInfo.npy',stimDict)
    np.save('JRclustSpikes.npy',spikeDict)

    return stimDict,spikeDict

def makeSweepPSTH_wh(bin_size, samples, spikes,sample_rate=20000, units=None, duration=None, verbose=False, rate=True, bs_window=[0, 0.25]):
    """
    written by GR - modified by AL 8/3/23
    Use this to convert spike time rasters into PSTHs with user-defined bin

    inputs:
        bin_size - float, bin size in seconds
        samples - list of ndarrays, time of spikes in samples
        spikes- list of ndarrays, spike cluster identities
        sample_rate - int, Hz, default = 20000
        units - None or sequence, list of units to include in PSTH
        duration - None or float, duration of PSTH; if None, inferred from last spike
        verbose - boolean, print information about psth during calculation
        rate - boolean; Output rate (divide by bin_size and # of trials) or total spikes per trial (divide by # trials only)
        bs_window - sequence, len 2; window (in s) to use for baseline subtraction; default = [0, 0.25]
    output: dict with keys:
        psths - ndarray
        bin_size - float, same as input
        sample_rate - int, same as input
        xaxis - ndarray, gives the left side of the bins
        units - ndarray, units included in psth
    """

    bin_samples = bin_size * sample_rate

    if duration is None:
        maxBin = max(np.concatenate(samples))/sample_rate
    else:
        maxBin = duration

    if units is None:  # if user does not specify which units to use (usually done with np.unique(goodSpikes))
        units = np.unique(np.hstack(spikes))
    numUnits = len(units)

    psths = np.zeros([int(np.ceil(maxBin/bin_size)), numUnits])
    if verbose:
        print('psth size is',psths.shape)
    for i in range(len(samples)):
        for stepSample, stepSpike in zip(samples[i], spikes[i]):
            if stepSpike in units:
                if int(np.floor(stepSample/bin_samples)) == psths.shape[0]:
                    psths[int(np.floor(stepSample/bin_samples))-1, np.where(units == stepSpike)[0][0]] += 1 ## for the rare instance when a spike is detected at the last sample of a sweep
                else:
                    if stepSample/bin_samples > duration * (1/bin_size ):
                        if verbose:
                            print(stepSample) 
                    else:
                        psths[int(np.floor(stepSample/bin_samples)), np.where(units == stepSpike)[0][0]] += 1
    psth_dict = {}
    if rate:
        psth_dict['psths'] = psths/bin_size/len(samples) # in units of Hz
    else:
        psth_dict['psths'] = psths/len(samples) # in units of spikes/trial in each bin

    psths_bs = np.copy(np.transpose(psth_dict['psths']))
    for i,psth in enumerate(psths_bs):
        tempMean = np.mean(psth[int(bs_window[0]/bin_size):int(bs_window[1]/bin_size)])
        #print(tempMean)
        psths_bs[i] = psth - tempMean
    psth_dict['psths_bs'] = np.transpose(psths_bs)
    psth_dict['bin_size'] = bin_size # in s
    psth_dict['sample_rate'] = sample_rate # in Hz
    psth_dict['xaxis'] = np.arange(0,maxBin,bin_size)
    psth_dict['units'] = units
    psth_dict['num_sweeps'] = len(samples)
    psth_dict['base_line'] = np.mean(psth[int(bs_window[0]/bin_size):int(bs_window[1]/bin_size)])
    return psth_dict


def aggregate_recordings_wh(pathlist):
    """"
    loads JRclustSpikes.npy from each recording in pathlist
    and appends metadata, spikes trains, 
    """
    numCells = [] # number of cells sorted in recording
    troughPeak = [] # trough to peak ratio for each cell, not separated by recording
    somachannel = [] # channel nearest to soma
    TPlist = [] # trough to peak ratio, separated by recording
    samples = [] # samples where a spike occured
    spikes = [] # cell identifier for who spiked
    numTrials = [] # number of times stim is repeated

    for item in pathlist:
        os.chdir(item)
        spikeDict = np.load('JRclustSpikes.npy',allow_pickle=True)
        spikeDict = spikeDict[()]

        stimDict = np.load('stimInfo.npy',allow_pickle = True)
        stimDict = stimDict[()]

        numCells.append(len(np.unique(spikeDict['goodSpikes'])))
        troughPeak.extend(spikeDict['spikeTroughPeak']*1000)
        TPlist.append(spikeDict['spikeTroughPeak']*1000)
        samples.append(spikeDict['goodSamples'])
        spikes.append(spikeDict['goodSpikes'])
        somachannel.append(spikeDict['viSite_clu'])
        numTrials.append(stimDict['numRepeats'])


    numCells = np.array(numCells)
    troughPeak = np.array(troughPeak)
    numTrials = np.array(numTrials)

    return numCells, troughPeak, TPlist, samples, spikes, somachannel, numTrials

def get_all_psths(bin, numSec,pathlist,reshape = False):
    """"
    generate psths for all recordings and output aggregated data
    bin = interval to bin
    numsec = number of seconds per trial
    pathlist = where to find JRclust data 
    reshape = boolean - separates into trials (true) or keeps as full recording (false)
    """
    psths = []
    numCells, troughPeak, TPlist, samples, spikes, somachannel, numTrials = aggregate_recordings_wh(pathlist)

    for i,samp in enumerate(samples):
        duration = numTrials[i]*numSec
        psthDict = makeSweepPSTH_wh(bin, [samp], [spikes[i]],sample_rate=20000, units=None, duration=duration, verbose=False, rate=True, bs_window=[0, 0.25])
        
        temp_psth = np.array(np.transpose(psthDict['psths']))

        psths.append(temp_psth)

    if reshape:
        cells_psth = []
        for i,el in enumerate(psths):
            penReshape = np.reshape(el,(np.shape(el)[0],numTrials[i],-1))
            cells_psth.append(penReshape)
        psths = cells_psth

    return psths, numCells, troughPeak, TPlist, samples, spikes, numTrials 


def get_all_psths_period(bin, numSec,pathlist,timeVec,reshape = False):
    """"
    for population analysis

    timeVec = boolean for which timepoints in a signle recording to include in analysis
        # should be length samplerate * recordingtime (# seconds of whole penetration) 
    generate psths for all recordings and output aggregated data
    bin = interval to bin
    numsec = number of seconds per trial
    pathlist = where to find JRclust data 
    reshape = boolean - separates into trials (true) or keeps as full recording (false)
    """
    psths = []
    numCells, troughPeak, TPlist, samples, spikes, somachannel, numTrials = aggregate_recordings_wh(pathlist)


    for i,samp in enumerate(samples):
        duration = numTrials[i]*numSec

        timeVecAll = np.tile(timeVec,numTrials[i])
        time_inds = np.where(timeVecAll == True)
        downsample_ind = np.arange(0,len(timeVecAll),int(20000*bin))
        downsamp_vec = timeVecAll[downsample_ind]
 
        selectedSamps,_,spksInd = np.intersect1d(time_inds,samp,return_indices = True)
        selectedSpikes = spikes[i][spksInd]

        psthDict = makeSweepPSTH_wh(bin, [selectedSamps], [selectedSpikes],sample_rate=20000, units=None, duration=duration, verbose=False, rate=False, bs_window=[0, 0.25])
        
        temp_psth = np.array(np.transpose(psthDict['psths']))

        psths.append(temp_psth[:,downsamp_vec])

    if reshape:
        cells_psth = []
        for i,el in enumerate(psths):
            penReshape = np.reshape(el,(np.shape(el)[0],numTrials[i],-1))
            cells_psth.append(penReshape)
        psths = cells_psth

    return psths, numCells, troughPeak, TPlist, samples, spikes, numTrials, somachannel

def get_spikes(numSec,pathlist,timeVec):
    """"
    organize spike and samples from jrclust data
  """
    numCells, troughPeak, TPlist, samples, spikes, somachannel, numTrials = aggregate_recordings_wh(pathlist)

    samps = []
    spikes = []
    for i,samp in enumerate(samples):
        duration = numTrials[i]*numSec

        timeVecAll = np.tile(timeVec,numTrials[i])
        time_inds = np.where(timeVecAll == True)
        downsample_ind = np.arange(0,len(timeVecAll),20000)
        downsamp_vec = timeVecAll[downsample_ind]
 
        selectedSamps,_,spksInd = np.intersect1d(time_inds,samp,return_indices = True)
        selectedSpikes = spikes[i][spksInd]

    samps.append(np.ndarray(selectedSamps))
    spikes.append(np.ndarray(spikes))
    return samps,spikes



def calculate_responsive_evoked_psthmean_wh(baseWin,evokedWin,bin,data,numSec):
    """""
    baseWin = time window (in seconds) of each trial to call 'baseline'
    evokedWin = time window (in seconds) of each trial to call evoked period
        both of these are lists of 2 values
    bin = bin size of psth data 
    data = list of psths. elements in list are three dimmensional 
        arrays corresponding to a single recording. xyz = cell, trial , timepoint

    outputs: 
    base = the baseline of each cell, averaging across firing rates
        in baseWin and across all trials for one value per cell
    basestd = standard deviation of the baseline. averages across baseWin
        first then takes standard deviation of base values acriss trials for each cell
    
    evoked = evoked firing rate per cell. separated by recording, 
        each value is one cell's evoked rate. calculated by averaging rate
        in evokedWin, then taking the maximum firing rate across all trials
        for each cell, generating one max evoked firing rate value per cell.
    evoked_sub = evoked rate with baseline subtracted subtracted  
    responsive60p = boolean determining if a particular cell is 'responsive'
        at least 60 percent of the time. determined by taking the average firing
        rate in evokedWin and testing if that value is 3 times greater than the s
        standard deviation of the baseline value. 
    responsiveMax = boolaean determining if the maximum evoked firing rate 
        achieved across trials (equal to evoked above) is greater than 3-times
        the standard deviation of the baseline
    resp_trials_only = evoked firing rate of each cell, averaging across only the 
        trials where the firing rate is 3* the standard deviaton of baseline
    psth_mean = avereage psth per cell, taking only the trials that pass the 
        'responsive' criteria of evoked rate in that trial is 3* the standard deviation 
        of that cell's baseline

    """""
    base = []
    basestd = []
    evoked = []
    evoked_sub = []
    responsive60p = []
    responsiveMax = []
    resp_trials_only = []
    psth_mean = []

    baseInd = range(int(1/bin* baseWin[0]), int(1/bin*baseWin[1]),1)
    evokedInd = range(int(1/bin*evokedWin[0]),int(1/bin*evokedWin[1] ),1)
    for i,pen in enumerate(data):
        baseAll = np.mean(pen[:,:,baseInd],axis = 2) 
        b = np.mean(baseAll,axis = 1)
        base.append(b)
        st = np.std(baseAll,axis = 1)
        basestd.append(st)

        evokedAll = np.mean(pen[:,:,evokedInd],axis =2)
        evmean = np.mean(evokedAll,axis = 1)
        ev = np.max(evokedAll,axis = 1)
        evoked.append(ev)
        subbed = ev -b
        evoked_sub.append(subbed)

        # determine if responsive - respond at least 60% of the time
        respBool = (evokedAll.T - b  > b + 3*st).T
        respScore = np.sum(respBool,axis = 1) / np.shape(pen)[1]
        responsive60p.append(respScore > .60)

        resp_std =  ev - b > b + 3*st
        responsiveMax.append(resp_std)

        r_temp = []
        psth_mean_temp = []
        for i,cell in enumerate(evokedAll):
            # average across trials that are responsive
            r_temp.append(np.mean(cell[respBool[i]]))

        for i,resp_trials in enumerate(respBool):
            selected_trials = pen[i,resp_trials,:]
            psth_mean_temp.append(np.mean(selected_trials,axis = 0))

        resp_trials_only.append(r_temp)
        psth_mean.append(psth_mean_temp)

    psth_all_cells = np.zeros((1,int(numSec/bin)))
    for el in psth_mean:
        psth_all_cells = np.vstack((psth_all_cells,el))
    psth_all_cells = np.delete(psth_all_cells,0,0)

    outdict_alltrials = {}
    outdict_alltrials['base'] = base
    outdict_alltrials['basestd'] = basestd
    outdict_alltrials['evoked'] = evoked
    outdict_alltrials['evoked_sub'] = evoked_sub
    outdict_alltrials['responsive60p'] = responsive60p
    outdict_alltrials['responsiveMax'] = responsiveMax
    outdict_alltrials['resp_trials_only'] = resp_trials_only
    outdict_alltrials['psth_mean'] = psth_mean

    # calculate latency
    latency = []
    for cell in psth_all_cells:
        blurred = scipy.ndimage.gaussian_filter1d(cell,2)
        ttp = np.argmax(blurred[int(1/bin*evokedWin[0]):])
        latency.append(ttp)

    latency = np.array(latency)

    outdict_resptrials = {}
    outdict_resptrials['psth'] = psth_all_cells
    outdict_resptrials['base'] = np.mean(psth_all_cells[:,baseInd],axis = 1)
    outdict_resptrials['evoked'] = np.mean(psth_all_cells[:,evokedInd],axis = 1)
    outdict_resptrials['latency'] = latency 

    return outdict_resptrials,outdict_alltrials

def expand_cell_info_wh(troughPeak,regThresh, idlist,groups,genoNames,numCells):
    """""
    troughPeak = vector of cell spike widths from get_all_psths_period
    idlist = list of animal ids 
    regThresh = threshold for calling a spike width 'regular'
    groups = list of animal ids that belong to each group, only takes 2 groups right now
        order must match following parameter, genoNames
    genoNames = list of strings corresponding to names of groups
    numCells = list of number of cells sorter per recording, output from get_all_psths_period
    
    output is data describing cell identity 
    
    """

    # generate vectors summarizing cell features
    reg = []
    for i,el in enumerate(troughPeak):
        isreg = [1 if troughPeak[i] >= regThresh else 0]
        reg.extend(isreg)
    reg = np.array(reg)

    #genotype vector by recording
    genoList = [genoNames[0] if x in groups[0] else genoNames[1] for x in idlist]  
    
    # genotype and animal ID vectors by cell
    geno = []
    cellsbyid = []
    for i,el in enumerate(numCells):
        if idlist[i] in groups[0]:
            g = genoNames[0]
        else:
            g = genoNames[1]
        idnum = idlist[i]
        cellsbyid.extend(np.tile(idnum,el))
        geno.extend(np.tile(g,el))   
    geno = np.array(geno)
    cellsbyid = np.array(cellsbyid)   

    recNum = []
    for i,rec in enumerate(idlist):
        recNum.extend(np.tile(i,numCells[i]))
    recNum = np.array(recNum)

    return reg, genoList, geno, cellsbyid,recNum

def sort_psths(psths,filterVec,onset):
    responders = psths[filterVec,:]
    responders_order = np.argsort(np.sum(responders[:,onset:],axis = 1))
    responders_order = responders_order[::-1]
    responders_sorted = responders[responders_order,:]

    return responders_sorted, responders_order


def compute_sta(win, spikeTimes, vecToAverage):
    """
    returns spike triggered average of vecToAverage in window = win at 
        each spikeTimes index
    largest value in spikeTimes cannot be larger than length of vecToAverage
    if an element of spikeTimes - win[0] is < 0 or +win[0] > len(vecToAverage)
        it is ignored from average
    win = list, boundary window for triggered average
    spikeTimes = indices of spikes around which to avereage vecToAverage
    """
    spks = spikeTimes[spikeTimes > np.abs(win[0])+1]
    spks = spks[spks < len(vecToAverage)-(win[1]+1)]
    sta_mat = np.zeros((len(spks),np.diff(win)[0]))

    for spikeNum,time in enumerate(spks): 
        spike_win_1 = time + win[0]
        spike_win_2 = time + win[1]
            
        # pull out time points around spike from population rate
        sta_mat[spikeNum,:]= vecToAverage[spike_win_1:spike_win_2]

    sta = np.mean(sta_mat,axis = 0)
    return sta
        
def compute_population_coupling(win,psths):
    """"
    generates population coupling score for each recording and cell in 
    'psths'
    returns spike triggered average traces for each cell in window 'win'
    
    win = list of window boundaries for spike triggered population rate
        binned at same value as psths
    psths = list of psth recordings organized cell x bin 
    
    returns dict 
        norm - median of shuffled stPR per recording 
        stPR - spike triggered population rate per recording and cell in win
        stPR_shuff - shuffled stPR
        stPR_vals - normalized stPR peak values
        stPR_ind - index in window where peak stPR occurs
        stPR_norm - normalized spike triggered population rate in win
    """

    pop = []
    pop_shuff = []
    win = []
    stPR = []
    stPR_shuff = []
    shuff_normalizer = []
    stPR_vals = []
    stPR_ind = []
    stPR_norm = []

#    generate population psth and suffled psth
    for rec in psths:
        pop.append(np.sum(rec,axis = 0))
    
        shuff_mat = np.zeros((np.shape(rec)))
        for i,cell in enumerate(rec):
            shuff_mat[i,:] = shuffle(cell)
    
        pop_shuff.append(shuff_mat)

    # iterate through recordings 
    for i,rec in enumerate(psths):

        rec_sta = np.zeros((np.shape(rec)[0],np.diff(win)[0]))
        rec_sta_shuff = np.zeros((np.shape(rec)[0],np.diff(win)[0]))
    
        # iterate through cells and subtract its signal from population
        # rate & compute sta
        for j,cell in enumerate(rec): 
            pop_wo_cell = pop[i] - cell

            pop_wo_cell_shuff = np.delete(pop_shuff[i],j,0)
            pop_wo_cell_shuff = np.sum(pop_wo_cell_shuff,axis = 0)

            spks = np.where((cell != 0))[0]
 
        
            rec_sta[j,:] = compute_sta(win,spks,pop_wo_cell)
            rec_sta_shuff[j,:] = compute_sta(win,spks,pop_wo_cell_shuff)

        norm = np.median(rec_sta_shuff[:,500])
        shuff_normalizer.append(norm)
        stPR.append(rec_sta)
        stPR_shuff.append(rec_sta_shuff)
        stPR_norm.append(rec_sta - norm)

        stPR_vals.append(np.max(rec_sta,axis = 1) - norm)
        stPR_ind.append(np.argmax(rec_sta,axis = 1))

    outdict =  {'norm':shuff_normalizer,
                'stPR':stPR,
                'stPR_shuff':stPR_shuff,
                'stPR_vals':stPR_vals,
                'stPR_ind':stPR_ind,
                'stPR_norm':stPR_norm}  
    return outdict

def rate_measure(cells,samp_period,kernelsize,info):
    """
    info = list [number of trials, number of seconds, stim on in seconds,total cell count in dataset]
    kernelsize in miliseconds for convolution
    samp_period also in miliseconds - number of miliseconds represented by each data point in output
    """
   
    maxrate = np.zeros((info[3]))
    baserate = np.zeros((info[3]))
    stdbase = np.zeros((info[3]))
    ratevec = np.zeros((info[3],int(info[1]*(1/samp_period)*1000)))
    latency = np.zeros((info[3]))
    onset = np.zeros((info[3]))
    loopnum = 0

    samp_factor = 1/ (samp_period / 1000) 
    for rec in cells:
        for cell in rec: 
            trialBlock = np.zeros((info[0],int(info[1]*samp_factor)))
            for i,trial in enumerate(cell): 
                tr = trial / 20000
                train = SpikeTrain(tr*s,t_stop = info[1]*s)
            
                rate = instantaneous_rate(train, sampling_period=samp_period*ms,kernel=GaussianKernel(kernelsize*ms))
                trialBlock[i,:] = np.array(rate.T)
            avrate = np.mean(trialBlock,axis = 0)

            ratevec[loopnum,:] = avrate
            maxrate[loopnum] = np.max(avrate[int(samp_factor*info[2]):int(samp_factor* info[2]+3*samp_factor)])
            curbase = np.mean(avrate[int(10*samp_factor):int(samp_factor*info[2]-samp_factor)])

            baserate[loopnum] = curbase
            curstd = np.std(avrate[int(10*samp_factor):int(samp_factor*info[2]-samp_factor)])
            
            stdbase[loopnum] = curstd
            # take the index at which rate reaches half maximum
            #latency[loopnum] = np.where(avrate[int(samp_factor*info[2]):] > maxrate[loopnum] / 2)
            #latency[loopnum] = np.argmax(avrate[int(samp_factor*info[2]):int(samp_factor*info[2]+samp_factor*3)])
            
            postpeakvec = avrate[int(samp_factor*info[2]):]
            abovebase = np.where(postpeakvec> curbase + 3*curstd)
            halfMax = np.where(postpeakvec > maxrate[loopnum] / 2)
            latency[loopnum] = halfMax[0][0]
        
            try:
                onset[loopnum] = abovebase[0][0]
                
            except:
                onset[loopnum] = np.nan
                print(loopnum,' cell not responseive') 

            loopnum += 1


    max_subtracted = maxrate - baserate
    latency = samp_period*latency
    onset = samp_period*onset

    return ratevec,maxrate, baserate, stdbase, max_subtracted, latency,onset

def monoExp_plat(x, m, t,plat):
    return m * np.exp(-t * x) +plat

def monoExp(x, m, t):
    return m * np.exp(-t * x) 

def biExp_plat(x, m1, t1,m2,t2,plat):
    return m1 * np.exp(-t1 * x) + m2 * np.exp(-t2 * x) + plat

def biExp(x, m1, t1,m2,t2):
    return m1 * np.exp(-t1 * x) + m2 * np.exp(-t2 * x) 


def fit_bidecay_constant(ratevec,samplerate,norm = False):
    taus1 = []
    taus2 = []
    rsq = []
    model = []
    for i,cell in enumerate(ratevec):
        if norm == True:
            postPeak = cell[np.argmax(cell):-samplerate] / np.max(cell)
        else:
            postPeak = cell[np.argmax(cell):-samplerate]
        
        xs = range(0,len(postPeak),1)
        try:
            params, cv = scipy.optimize.curve_fit(biExp, xs, postPeak,bounds = (0,[np.inf,np.inf,np.inf,np.inf]))

            m1, t1, m2,t2 = params
            tauSec1 = (1/t1) / samplerate
            tauSec2 = (1/t2) / samplerate

            squaredDiffs = np.square(postPeak - biExp(xs, m1, t1, m2,t2))
            squaredDiffsFromMean = np.square(postPeak - np.mean(postPeak))
            rSquared = 1 - np.sum(squaredDiffs) / np.sum(squaredDiffsFromMean)

        except:
            tauSec1 = np.nan
            tauSec2 = np.nan
            rSquared = np.nan
            params = (np.nan,np.nan,np.nan,np.nan)
            print(i)

        rsq.append(rSquared)
        model.append(params)           
        taus1.append(tauSec1)
        taus2.append(tauSec2)

    taus1 = np.array(taus1)
    taus2 = np.array(taus2)
    rsq = np.array(rsq)
    model = np.array(model)

    return taus1,taus2,rsq,model

def fit_rate_data_bidecay(model,ratevec,fs,samples_to_fit):
    fit_curve = []
    cell_decay = []
    intercept = []
    for i,cell in enumerate(ratevec):
        maxcell = np.argmax(cell)
        excell_post = cell[maxcell:-10] / np.max(cell)

        xs = np.arange(0,samples_to_fit,1)

        exm1,ext1,exm2,ext2 = model[i]
        fit_curve.append(biExp(xs,exm1,ext1,exm2,ext2))
        intercept.append(biExp(samples_to_fit,exm1,ext1,exm2,ext2))
        cell_decay.append(excell_post)

    fit_curve = np.array(fit_curve)
    #
    # cell_decay = np.array(cell_decay)
    intercept = np.array(intercept)
    return fit_curve,cell_decay,intercept




def fit_bidecay_constant_plat(ratevec,samplerate,norm = False):
    taus1 = []
    taus2 = []
    rsq = []
    model = []
    plats = []
    for i,cell in enumerate(ratevec):
        if norm == True:
            postPeak = cell[np.argmax(cell):-samplerate] / np.max(cell)
        else:
            postPeak = cell[np.argmax(cell):-samplerate]
        
        xs = range(0,len(postPeak),1)
        try:
            params, cv = scipy.optimize.curve_fit(biExp_plat, xs, postPeak)

            m1, t1, m2,t2,plat = params
            tauSec1 = (1/t1) / samplerate
            tauSec2 = (1/t2) / samplerate

            squaredDiffs = np.square(postPeak - biExp_plat(xs, m1, t1, m2,t2,plat))
            squaredDiffsFromMean = np.square(postPeak - np.mean(postPeak))
            rSquared = 1 - np.sum(squaredDiffs) / np.sum(squaredDiffsFromMean)

        except:
            tauSec1 = np.nan
            tauSec2 = np.nan
            rSquared = np.nan
            params = (np.nan,np.nan,np.nan,np.nan,np.nan)
            plat = np.nan
            print(i)

        rsq.append(rSquared)
        model.append(params)           
        taus1.append(tauSec1)
        taus2.append(tauSec2)
        plats.append(plat)

    taus1 = np.array(taus1)
    taus2 = np.array(taus2)
    rsq = np.array(rsq)
    plats = np.array(plats)
    model = np.array(model)

    return taus1,taus2,rsq,model,plats

def fit_monodecay_constant(ratevec,samplerate,norm = False):
    taus = []
    rsq = []
    model = []
    for i,cell in enumerate(ratevec):
        if norm == True:
            postPeak = cell[np.argmax(cell):-samplerate] / np.max(cell)
        else:
            postPeak = cell[np.argmax(cell):-samplerate]
        xs = range(0,len(postPeak),1)
        try:
            params, cv = scipy.optimize.curve_fit(monoExp, xs, postPeak)

            m, t = params
            tauSec = (1/t) / samplerate
 

            squaredDiffs = np.square(postPeak - monoExp(xs, m, t))
            squaredDiffsFromMean = np.square(postPeak - np.mean(postPeak))
            rSquared = 1 - np.sum(squaredDiffs) / np.sum(squaredDiffsFromMean)

        except:
            tauSec = np.nan
            rSquared = np.nan
            params = (np.nan,np.nan,np.nan)
            print(i)

        rsq.append(rSquared)
        model.append(params)           
        taus.append(tauSec)

    rsq = np.array(rsq)
    taus = np.array(taus)


    return taus,rsq,model

def fit_monodecay_constant_plat(ratevec,samplerate,norm = False):
    taus = []
    rsq = []
    model = []
    plats = []
    for i,cell in enumerate(ratevec):
        if norm == True:
            postPeak = cell[np.argmax(cell):-samplerate] / np.max(cell)
        else:
            postPeak = cell[np.argmax(cell):-samplerate]
        xs = range(0,len(postPeak),1)
        try:
            params, cv = scipy.optimize.curve_fit(monoExp_plat, xs, postPeak)

            m, t,plat = params
            tauSec = (1/t) / samplerate
 

            squaredDiffs = np.square(postPeak - monoExp_plat(xs, m, t,plat))
            squaredDiffsFromMean = np.square(postPeak - np.mean(postPeak))
            rSquared = 1 - np.sum(squaredDiffs) / np.sum(squaredDiffsFromMean)

        except:
            tauSec = np.nan
            rSquared = np.nan
            params = (np.nan,np.nan,np.nan)
            plat = np.nan
            print(i)

        rsq.append(rSquared)
        model.append(params)           
        taus.append(tauSec)
        plats.append(plat)

    rsq = np.array(rsq)
    taus = np.array(taus)
    plats = np.array(plats)


    return taus,rsq,model,plats

def bootstrap_firing_curves(ratevec,numBoots, idlist,cellsbyid,genolist):

    # only works if genolist has control group called 'Cre-'
    bootstrapped1_mean = np.zeros((numBoots,np.shape(ratevec)[1]))
    bootstrapped2_mean = np.zeros((numBoots,np.shape(ratevec)[1]))


    for j in range(numBoots):

        resampledAnimals,resampledGenoList = sklearn.utils.resample(idlist,genolist)
        avBootstrappedRate = np.zeros((len(resampledAnimals),np.shape(ratevec)[1]))
        for i,animalResamp in enumerate(resampledAnimals): 
            selectedCells = ratevec[cellsbyid == animalResamp,:]

            resampledCell = sklearn.utils.resample(selectedCells)
            avBootstrappedRate[i,:] =np.mean(resampledCell,axis = 0)


        if sum(resampledGenoList == 'Cre-') == 0:
            bootstrapped1_mean[j,:] = np.nan

        else:
            bootstrapped1_mean[j,:] = np.mean(avBootstrappedRate[resampledGenoList == 'Cre-',:],axis = 0)

        if sum(resampledGenoList != 'Cre-') == 0:
            bootstrapped2_mean[j,:] = np.nan

        else:
            bootstrapped2_mean[j,:] = np.mean(avBootstrappedRate[resampledGenoList != 'Cre-',:],axis = 0)


    boot_mean1 = np.nanmean(bootstrapped1_mean,axis = 0)
    lower_ci_1 = np.nanquantile(bootstrapped1_mean, 0.025,axis = 0)
    upper_ci_1 = np.nanquantile(bootstrapped1_mean, 0.975,axis = 0)

    boot_mean2 = np.nanmean(bootstrapped2_mean,axis = 0)
    lower_ci_2 = np.nanquantile(bootstrapped2_mean, 0.025,axis = 0)
    upper_ci_2 = np.nanquantile(bootstrapped2_mean, 0.975,axis = 0)  

    group1 = ('Cre-',boot_mean1,lower_ci_1,upper_ci_1)
    group2 = ('mutant',boot_mean2,lower_ci_2,upper_ci_2)

    return group1,group2

def bootstrap_values(data,numBoots,idlist,cellsbyid,genolist):
    # this requires that one element in genolist is 'Cre-'
    # only works for two groups
    f = np.zeros((numBoots))
    p = np.zeros((numBoots))
    cohenD = np.zeros((numBoots))
    rankSerial = np.zeros((numBoots))
    cliffsD = np.zeros((numBoots))

    for j in range(numBoots):

        resampledAnimals,resampledGenoList = sklearn.utils.resample(idlist,genolist)
        avBootstrappedVal = np.zeros((len(resampledAnimals)))
        for i,animalResamp in enumerate(resampledAnimals): 
            selectedCells = data[cellsbyid == animalResamp]

            resampledCell = sklearn.utils.resample(selectedCells)
            avBootstrappedVal[i] =np.mean(resampledCell)

        gp1 = avBootstrappedVal[resampledGenoList=='Cre-']
        gp2 = avBootstrappedVal[resampledGenoList!='Cre-']

        try:
            curu,curp = mannwhitneyu(gp1,gp2,method = 'exact')
            f[j] = curu 
            p[j] = curp
            rankSerial[j] = 1 - ( 2*curu / (len(gp1)*len(gp2)) )
            cliffsD[j] = (2*curu / (len(gp1)*len(gp2)) ) - 1
            
            #print(np.std(np.concatenate((gp1,gp2))))

        except:
            f[j],p[j] = (np.nan,np.nan)
            rankSerial[j] = np.nan
            cliffsD[j] = np.nan

        # effect size 
        #cohenD[j] = (np.mean(gp1) - np.mean(gp2)) / np.std(gp1)

    return p,rankSerial,cliffsD

def aggcells(samples,spikes,numSec,idlist,wt,plot = True,fs = 20000):
    # look at all data separated by recording
    #generate event plot and 
    # use this code block to choose wich cell 
    # leaves trials empty if no cells spike, allows for trial-trial alignment 
    # between cells


    allcells= []
    for r,samp in enumerate(samples):
    
        if plot:
            fig,ax = plt.subplots()
        allcells_rec = []
        for k,cell in enumerate(np.unique(spikes[r])):
            rep_wt_bool = spikes[r] == cell
            rep_wt = samples[r][rep_wt_bool]

            trial = 0
            rep_cell_list = [[]] * 50
            curtrial = []
            for i,event in enumerate(rep_wt):
                effective_trial = int(np.floor(event / 900000))
                
                if event > 900000*(trial+1):
                    
                    curtrial = np.array(curtrial)
                    if trial == 0:
                        
                        rep_cell_list[trial] = curtrial
                        trial += 1
                    
                    else: 
                        
                        rep_cell_list[effective_trial-1] = curtrial-(900000*(trial))
                        trial += 1
                    curtrial = []
                    

                    
                else:
                    curtrial.append(event)

            if (len(curtrial) != 0) & (effective_trial != 50):
                #print(effective_trial)
                curtrial = np.array(curtrial)
                rep_cell_list[effective_trial] = curtrial-(900000*(trial))
            
            for i,el in enumerate(rep_cell_list):
                if type(el) == list:
                    rep_cell_list[i] = np.array(el)

            
            allcells_rec.append(rep_cell_list)

            num = len(np.unique(spikes[r]))
            rows = math.ceil(np.sqrt(num))
            cols = math.ceil(np.sqrt(num))

            if plot:
                plt.subplot(rows,cols,k+1)
                if idlist[r] in wt:
                    plt.eventplot(rep_cell_list,color = 'blue')
                    plt.axis('off')
                else:
                    plt.eventplot(rep_cell_list,color = 'orange')
                    plt.axis('off')
                plt.title(k)

        allcells.append(allcells_rec)

    return allcells

#generate noise correlations 

def noise_correlations(rec,sample_start,sample_stop):

    # generates a pearsonr correlation coeficient for every cell
    # against each other in a recording. 
    # smaple start and stop are the indices between which to count spikes 
    # for correlation

    numcells = len(rec)

    correlations = np.zeros((numcells,numcells))
    for el in range(numcells):
        rec_copy = np.array(np.copy(rec))
        refcell = rec[el]

        for j,target_cell in enumerate(rec_copy):

            refcount = []
            targetcount = []
            for k,trial in enumerate(refcell):
                refcount.append(np.sum(trial[sample_start:sample_stop]))
                targetcount.append(np.sum(target_cell[k][sample_start:sample_stop]))
            correlations[el,j] = pearsonr(refcount,targetcount)[0]
            

    iu = np.triu_indices(np.shape(correlations)[0])
    correlations[iu] = 1

    corr_map = correlations
    corr_vals = correlations[correlations < 1]
    return corr_map,corr_vals

def noise_correlations_norm(rec,sample_start,sample_stop):

    # generates a pearsonr correlation coeficient for every cell
    # against each other in a recording. 
    # smaple start and stop are the indices between which to count spikes 
    # for correlation

    numcells = len(rec)

    
    correlations = np.zeros((numcells,numcells))
    for el in range(numcells):
        rec_copy = np.array(np.copy(rec))
        refcell = rec[el]

        for j,target_cell in enumerate(rec_copy):

            refcount = []
            targetcount = []
            for k,trial in enumerate(refcell):
                refcount.append(np.sum(trial[sample_start:sample_stop]))
                targetcount.append(np.sum(target_cell[k][sample_start:sample_stop]))
            correlations[el,j] = pearsonr(refcount,targetcount)[0]
            

    iu = np.triu_indices(np.shape(correlations)[0])
    correlations[iu] = 1

    corr_map = correlations
    corr_vals = correlations[correlations < 1]
    return corr_map,corr_vals
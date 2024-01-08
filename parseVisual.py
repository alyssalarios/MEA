# collection of functions that will parse the visual data 
import scipy
import numpy as np
import os
import miscAnalysis_salt as ma
import glob 
import scipy.io
import skimage.measure
from scipy.stats import f_oneway
import math
import matplotlib.pyplot as plt
from elephant.statistics import time_histogram, instantaneous_rate
from elephant.kernels import GaussianKernel
from quantities import ms, s, Hz
from neo.core import SpikeTrain
from itertools import compress

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

def trialByTimeTransform(spikeDict,numTrials,numSamples):
    """
    takes spikes dictionary, total trials and samples in dataset
    makes output matrix of trial number x sample number. values 
    correspond to which unit spiked in that sample
    trial number is chronological 

    cellMatrix is data separated by cell in the third dimmension.
    cell x trial x sample  with 1 where that cell spiked and 0 otherwise
    """
    goodSamples = spikeDict['goodSamples']
    goodSpikes = spikeDict['goodSpikes']
    matrix = np.zeros((numTrials,numSamples))
    for i,element in enumerate(matrix):
        trialInd = (goodSamples <= (i+1)*numSamples) &  (goodSamples >= (i*numSamples))
        curTrialSamp = goodSamples[trialInd]
        trialSpikeTimes = (curTrialSamp - i*numSamples) -1

        trialSpikeID = goodSpikes[trialInd]
    
        for timeInd,time in enumerate(trialSpikeTimes):
            matrix[i,time] = trialSpikeID[timeInd]

    cellMatrix = np.zeros((np.max(goodSpikes),np.shape(matrix)[0],np.shape(matrix)[1]))
    for i,trial in enumerate(matrix):
        for j,sample in enumerate(trial):
            if sample != 0:
                cell = np.array(sample,dtype = int)
                cellMatrix[cell-1,i,j] = 1

    return cellMatrix,matrix


def processVisualData(path,binSize = 5):
    os.chdir(path)

    spikeDict = ma.importJRCLUST(os.getcwd()+'\\alldata\\S0.mat')

    # if buffer time is recorded, if not comment this out
    spikeDict = removeIntanBuffer(os.getcwd(),spikeDict)

    # import matlab metadata file and extract relevant parameters
    matfile = glob.glob('*Metadata.mat')
    mat = scipy.io.loadmat(matfile[0])
    stimDict = {}
    stimDict['stimIndex'] = mat['stimIndex'][0]
    stimDict['preStimTime'] = mat['preStimTime'][0][0]
    stimDict['stimDur'] = mat['stimDur'][0][0]
    stimDict['numRepeats'] = mat['numrepeats'][0][0]
    stimDict['orientationInfo'] = mat['orientationInfo'][0]

    numConditions = np.max(stimDict['stimIndex'])
    numTrials = int(numConditions) * int(stimDict['numRepeats'])
    totalSeconds = int(stimDict['preStimTime']) + int(stimDict['stimDur'])
    numSamples = totalSeconds * spikeDict['sampleRate']

    # organize spike matrix
    cell,matrix = trialByTimeTransform(spikeDict,numTrials,numSamples)

    # bin array and save
    if binSize > 0:
        block = int(spikeDict['sampleRate'] / 1000 * binSize)
        binnedCell = skimage.measure.block_reduce(cell, block_size=(1, 1, block), func=np.sum)
    else: 
        binnedCell = cell

    # remove cells that never spike
    blankCell = np.array(np.zeros((np.shape(binnedCell)[0])),dtype = bool)
    for i,el in enumerate(binnedCell): 
        blankCell[i] = np.sum(el) != 0

    binnedCell = binnedCell[blankCell,:,:]



    np.save('sortedSpikes_binned.npy',binnedCell)
    np.save('stimInfo.npy',stimDict)
    np.save('JRclustSpikes.npy',spikeDict)

    return binnedCell,stimDict,spikeDict

def findReponders(spikes):
    # determine if a cell is visuallly respopnsive or not
    base = np.zeros((np.shape(spikes)[0],np.shape(spikes)[1]))
    evoked = np.zeros((np.shape(spikes)[0],np.shape(spikes)[1])) 

    for i,cell in enumerate(spikes): 
        for j,trial in enumerate(cell):
            # hard coded for 6 second base and 2 sec stim s
            base[i,j] = np.sum(trial[1000:1200]) 
            evoked[i,j] = np.sum(trial[1200:1400]) 

    anova = f_oneway(base,evoked,axis = 1)

    # remove units that have no spikes (maybe spiked during buffer time)
    missing = np.isnan(anova[1])
    responders_ano = anova[1] < 0.01
    responders = responders_ano * ~missing 
    percent_responsive = np.sum(responders) / np.sum(~np.isnan(anova[1]))
    return percent_responsive,responders,evoked,base

def evokedFiring(evoked,base, stimIndex):
    # find maximum spike rate during stim for visually 
    # repsonsive units separated by noise or gratings
    
    noiseTrials= stimIndex == 1
    gratingTrials = stimIndex != 1
 
    evokedNoise = evoked[:,noiseTrials]
    maxNoise = np.max(evokedNoise,axis = 1)
    avNoise = np.mean(evokedNoise,axis = 1)

    evokedGratings = evoked[:,gratingTrials]
    maxGratings = np.max(evokedGratings,axis = 1)
    avGratings = np.mean(evokedGratings,axis = 1)
    
    avBase = np.mean(base,axis = 1)

    return maxNoise,avNoise,maxGratings,avGratings,avBase

def evokedFiring2(spikes, stimIndex):
    # find maximum spike rate during stim for visually 
    # repsonsive units separated by noise or gratings
    # 10 ms time bins 

    spikes200ms =  skimage.measure.block_reduce(spikes, block_size=(1, 1, 40), func=np.sum)

    noiseTrials= spikes200ms[:,stimIndex == 1,:]
    gratingTrials = spikes200ms[:,stimIndex != 1,:]
 
    base = np.mean(spikes200ms[:,:,1:30],axis =2)
    base = np.mean(base,axis = 1)

    maxNoise = np.max(noiseTrials[:,:,31:35],axis =2)
    maxNoise = np.max(maxNoise,axis = 1)
    avNoise = np.max(noiseTrials[:,:,31:35],axis = 2)
    avNoise = np.mean(avNoise,axis = 1)

    maxGratings = np.max(gratingTrials[:,:,31:35],axis = 2)
    maxGratings = np.max(maxGratings,axis = 1)
    avGratings = np.max(gratingTrials[:,:,31:35],axis = 2)
    avGratings = np.mean(avGratings,axis = 1)

    return maxNoise,avNoise,maxGratings,avGratings,base



def pool_parse_experiments(pathlist,idlist,penlist):

    for i,el in enumerate(pathlist):
        os.chdir(el)
        animalID = idlist[i]
        penetration = penlist[i]

        spikes = np.load('sortedSpikes_binned.npy')
        stimInfo = np.load('stimInfo.npy',allow_pickle = True)

        percent_responsive,responders,evoked,base = findReponders(spikes)

        stimIndex = stimInfo[()]['stimIndex']
        maxNoise,avNoise,maxGratings,avGratings,avBase = evokedFiring2(spikes,stimIndex)

        # concatanate
        id_list = [int(animalID) for el in maxNoise]
        pen = [int(penetration) for el in maxNoise]

        if i != 0:
            #spikesT = np.vstack((spikesT,spikes))
            respondersT = np.hstack((respondersT,responders))
            percent_responsiveT = np.hstack((percent_responsiveT,percent_responsive))
            animalID_percentT = np.hstack((animalID_percentT,animalID))
            penetration_percentT = np.hstack((penetration_percentT,penetration))
            maxNoiseT = np.concatenate((maxNoiseT,maxNoise))
            avNoiseT = np.concatenate((avNoiseT,avNoise))
            maxGratingsT = np.concatenate((maxGratingsT,maxGratings))
            avGratingsT = np.concatenate((avGratingsT,avGratings))
            avBaseT = np.concatenate((avBaseT,avBase))
            stimIndexT = np.hstack((stimIndexT,stimIndex))
            animalIDT = animalIDT + id_list
            penetrationT = penetrationT + pen
            print('concatenated')

        else:
            #spikesT = spikes
            respondersT = responders
            percent_responsiveT = percent_responsive
            animalID_percentT = animalID
            penetration_percentT = penetration
            maxNoiseT = maxNoise
            avNoiseT = avNoise
            maxGratingsT = maxGratings
            avGratingsT = avGratings
            avBaseT = avBase
            stimIndexT = stimIndex
            animalIDT = id_list
            penetrationT = pen  
            print('first')

        print(el)

    percent_df = {'percentResponsive': percent_responsiveT,
                  'animalID' : animalID_percentT,
                  'penetration': penetration_percentT}
    
    spikes_df = {#'spikesRaw': spikesT,
                 'responders': respondersT,
                 'maxNoise': maxNoiseT,
                 'avNoise':avNoiseT,
                 'maxGratings': maxGratingsT,
                 'avGratings': avGratingsT,
                 'avBase': avBaseT,
                 'animalID':animalIDT,
                 'penetration': penetrationT,
                 'stimIndex': stimIndexT}
    
    return percent_df, spikes_df

def filter_repeats_avspikes(pathlist,animallist,trials):
    # this will only keep the first n trials as input in second parameter
    
    for i,el in enumerate(pathlist):
        os.chdir(el)
        stimInfo = np.load('stimInfo.npy',allow_pickle = True)
        stimIndex = stimInfo[()]['stimIndex']

        spikes = np.load('sortedSpikes_binned.npy')
        spikes200 = skimage.measure.block_reduce(spikes, block_size=(1, 1, 40), func=np.sum)


        filt_stimIndex = stimIndex[0:trials]
        filt_spikes = spikes200[:,0:trials,:]
        

        avResp = np.zeros((np.shape(filt_spikes)[0],len(np.unique(filt_stimIndex)),np.shape(filt_spikes)[2]))
        for k,ind in enumerate(np.unique(filt_stimIndex)): 
            curTrial = filt_spikes[:,filt_stimIndex == ind,:]
            avResp[:,k,:] = np.mean(curTrial,axis = 1)

         # remove cells that never spike

        atleastonespike = np.sum(np.sum(spikes,axis = 2),axis =1) != 0
        avResp = avResp[atleastonespike,:,:]

        atleastonespike = np.sum(np.sum(filt_spikes,axis = 2),axis =1) != 0
        filt_spikes = filt_spikes[atleastonespike,:,:]

        filt_stimIndex = np.tile(filt_stimIndex,(np.shape(filt_spikes)[0],1))

        id = [animallist[i] for n in avResp]

        if i == 0:
            allResp = avResp
            animalID = id
            alltrials = filt_spikes
            allindices = filt_stimIndex
        else: 
            allResp = np.vstack((allResp,avResp))
            animalID = np.hstack((animalID,id))
            alltrials = np.vstack((alltrials,filt_spikes))
            allindices = np.vstack((allindices, filt_stimIndex))


    return allResp, animalID,alltrials,allindices

def processVisualData2(path):
    """"
    ** must run matlab function first to get bufferbool.mat in directory 
    imports JRclust data and removes spikes during intan buffer period

    loads associated matlab file and saves stim info 

    saves JRclust data and matlab data in given directory 

    difference between this and previous is that psth is not calculated
    """
    os.chdir(path)

    spikeDict = ma.importJRCLUST(os.getcwd()+'\\alldata\\S0.mat')

    # if buffer time is recorded, if not comment this out
    spikeDict = removeIntanBuffer(os.getcwd(),spikeDict)

    # import matlab metadata file and extract relevant parameters
    matfile = glob.glob('*Metadata.mat')
    mat = scipy.io.loadmat(matfile[0])
    stimDict = {}
    stimDict['stimIndex'] = mat['stimIndex'][0]
    stimDict['preStimTime'] = mat['preStimTime'][0][0]
    stimDict['stimDur'] = mat['stimDur'][0][0]
    stimDict['numRepeats'] = mat['numrepeats'][0][0]
    stimDict['orientationInfo'] = mat['orientationInfo'][0]

    numConditions = np.max(stimDict['stimIndex'])
    numTrials = int(numConditions) * int(stimDict['numRepeats'])
    totalSeconds = int(stimDict['preStimTime']) + int(stimDict['stimDur'])
    numSamples = totalSeconds * spikeDict['sampleRate']

    np.save('stimInfo.npy',stimDict)
    np.save('JRclustSpikes.npy',spikeDict)

    return stimDict,spikeDict


def aggregate_recordings_vis(pathlist):
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
    stimIndex = [] # index for randomly interleaved trials

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
        stimIndex.append(stimDict['stimIndex'])


    numCells = np.array(numCells)
    troughPeak = np.array(troughPeak)
    numTrials = np.array(numTrials)


    return numCells, troughPeak, TPlist, samples, spikes, somachannel, numTrials,stimIndex


def makeSweepPSTH_vis(bin_size, samples, spikes,sample_rate=20000, units=None, duration=None, verbose=False, rate=True, bs_window=[0, 0.25]):
    """
    written by GR - modified by AL 8/3/23
    Use this to convert spike time rasters into PSTHs with user-defined bin

    identical to function in parseWhisker, just named differently 
    to avoid conflict
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


def filter_by_index(psths,stimIndex,numTrials,trialsToKeep,indexList,numOris=13):
    ind_psth = []
    indFilt = []

    for i,el in enumerate(psths):
        selectedInds = np.isin(stimIndex[i],indexList)
        selectedInds[trialsToKeep:] = False

        indFilt.append(stimIndex[i][selectedInds])
        penReshape = np.reshape(el,(np.shape(el)[0],numTrials[i]*numOris,-1))

        ind_psth.append(penReshape[:,selectedInds,:])

    return ind_psth,indFilt

def get_all_psths_period(bin, numSec,pathlist,timeVec,stimIndexList,trialsToKeep = 200):
    """"
    for population analysis

    timeVec = boolean for which timepoints in a signle recording to include in analysis
        # should be length samplerate * length of one trial in seconds) 
    generate psths for all recordings and output aggregated data
    bin = interval to bin
    numsec = number of seconds per trial
    pathlist = where to find JRclust data 
    reshape = boolean - separates into trials (true) or keeps as full recording (false)
    stimIndexList =  which stim indices to keep when outputting psths 
    
    no option to NOT reshape output array because we need to separate by visual trial
    """
    psths = []
    numCells, troughPeak, TPlist, samples, spikes, somachannel, numTrials,stimIndex = aggregate_recordings_vis(pathlist)



    for i,samp in enumerate(samples):
        duration = numTrials[i]*numSec*13

        timeVecAll = np.tile(timeVec,numTrials[i]*13)
        time_inds = np.where(timeVecAll == True)
        downsample_ind = np.arange(0,len(timeVecAll),int(20000*bin))
        downsamp_vec = timeVecAll[downsample_ind]
 
        selectedSamps,_,spksInd = np.intersect1d(time_inds,samp,return_indices = True)
        selectedSpikes = spikes[i][spksInd]

        psthDict = makeSweepPSTH_vis(bin, [selectedSamps], [selectedSpikes],sample_rate=20000, units=None, duration=duration, verbose=False, rate=True, bs_window=[0, 0.25])
        
        temp_psth = np.array(np.transpose(psthDict['psths']))
        psths.append(temp_psth[:,downsamp_vec])

    selected_psth,indFilt = filter_by_index(psths,stimIndex,numTrials,trialsToKeep,stimIndexList)


    return psths, numCells, troughPeak, TPlist, samples, spikes, numTrials,selected_psth,indFilt,stimIndex


def sort_psths(psths,filterVec,onset):
    responders = psths[filterVec,:]
    responders_order = np.argsort(np.sum(responders[:,onset:],axis = 1))
    responders_order = responders_order[::-1]
    responders_sorted = responders[responders_order,:]

    return responders_sorted, responders_order

def calculate_responsive_evoked_psthmean_vis(baseWin,evokedWin,bin,data,numSec):
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

def expand_cell_info(troughPeak,regThresh, idlist,groups,genoNames,numCells):
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

def aggcells(samples,spikes,numrepeats,idlist,wt,plot = True,fs = 20000):
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
            rep_cell_list = [[]] * numrepeats * 13
            curtrial = []
            for i,event in enumerate(rep_wt):
                effective_trial = int(np.floor(event / 160000))
                
                if event > 160000*(trial+1):
                    
                    curtrial = np.array(curtrial)
                    if trial == 0:
                        
                        rep_cell_list[trial] = curtrial
                        trial += 1
                    
                    else: 
                        rep_cell_list[effective_trial-1] = curtrial-(160000*(trial))
                        
                        trial += 1
                    curtrial = []
                    

                    
                else:
                    curtrial.append(event)

            if (len(curtrial) != 0) & (effective_trial != 20):
                #print(effective_trial)
                curtrial = np.array(curtrial)
                rep_cell_list[effective_trial] = curtrial-(160000*(trial))
            
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

        plt.savefig('allcells_raster'+'_'+str(r), dpi=600, transparent=True,bbox_inches='tight')
    return allcells


def rate_measure_vis(allcells,stimIndex,fs,kernelsize,info):
    """
    info = list [number of seconds, stim on in seconds]
    
    """
    numSec = info[0]
    stimOnset = info[1]

    samp_factor = 1000/fs

    allrates = []
    baserate = []
    stdbase = []
    maxrate = []
    maxsub = []
    master_trial_block = []

    for rec,index in zip(allcells,stimIndex):
        trial_block_all_cells = np.zeros([len(rec),len(np.unique(index)),20,80])
        cellrates = []
        cellbase = []
        cellstd = []
        cellmax = []
        cellsub = []
        for k,cell in enumerate(rec):
            rate_matrix = np.zeros((len(np.unique(index)),int(numSec*samp_factor)))
            
            for i,el in enumerate(np.unique(index)):
            # for every unique stim type
                selection = index == el
                trial_data = list(compress(cell,selection))

                
                trial_block = np.zeros((len(trial_data),int(numSec*samp_factor)))
                for j,trial in enumerate(trial_data):
                # for every trial of the same stim
                    tr = trial / 20000
                    train = SpikeTrain(tr*s,t_stop = numSec*s)
            
                    rate = instantaneous_rate(train, sampling_period=fs*ms,kernel=GaussianKernel(kernelsize*ms))
                    trial_block[j,:] = np.array(rate.T)
                    trial_block_all_cells[k,i,j,:] = np.array(rate.T)

                rate_matrix[i,:] = np.mean(trial_block,axis = 0)
                

            cellrates.append(rate_matrix)
            
            basevec = np.mean(rate_matrix[:,int(stimOnset*samp_factor-3):int(stimOnset*samp_factor-1)],axis = 1)
            baserate.append(np.mean(basevec))
            stdvec = np.std(rate_matrix[:,int(stimOnset*samp_factor-3):int(stimOnset*samp_factor-1)],axis = 1)
            stdbase.append(np.mean(stdvec))
            maxvec = np.max(rate_matrix[:,int(stimOnset*samp_factor):],axis = 1)
            maxrate.append(maxvec)
            maxsub.append(maxvec - np.mean(basevec))

        
        allrates.append(cellrates)
        #print(np.shape(cellbase))
        #baserate.extend(cellbase)
        #stdbase.extend(cellstd)
        #maxrate.extend(cellmax)
        #maxsub.extend(cellsub)
    master_trial_block.append(trial_block_all_cells)
    if len(allrates) == 1:
        r = np.array(allrates[0])
    else: 
        r = allrates[0]
        for el in allrates[1:]:
            r = np.vstack([r,el])
    baserate = np.array(baserate)
    stdbase = np.array(stdbase)
    maxrate = np.array(maxrate)
    maxsub = np.array(maxsub)
    

    return r,maxrate, baserate, stdbase, maxsub,master_trial_block

def compute_responsivity(baserate,stdbase,max):
    """
    determine if cell is responsive, and if so inhibited or excited by stim
    responsive = 1 if excited by stim 
    responsive = -1 if inhibited 
               = 0 if non responsive

    separate responsive vectors for noise and gratings

    preferred orientation for gratings determined by orientation 
    evoking the highest maximum firing rate

    """
    mat = np.zeros((np.shape(max)))
    resp_mat = max.T > 3*stdbase + baserate
    resp_mat = resp_mat.T
    inhib_mat = max.T < 3*stdbase + baserate
    inhib_mat = inhib_mat.T

    mat[resp_mat] = 1
    mat[inhib_mat] = -1

    return mat

def get_spike_counts_vis(allcells,stimindex, info,countperiod,fs =20000):
    """
    info = list [number of seconds, stim on in seconds]
    count period is how long after stim onset to count spikes, in milliseconds
    makes psth with bins 1 ms then counts spikes in countperiod after stim
    
    """
    # everything below just generates a psth at 1 ms bin 
    # with correct experimental structure cell x trial x time
    numbins = int(fs * info[0] / 20)
    psths = []
    bin_samples = fs / 1000
    duration = fs * info[0]
    for i,rec in enumerate(allcells):
        psths_cell_trial = np.zeros([len(rec),len(rec[0]),numbins])
        for j,cell in enumerate(rec):
            for k,trial in enumerate(cell): 
                for l,spike in enumerate(trial):
                        if int(np.floor(spike/bin_samples)) == np.shape(psths_cell_trial)[2]:
                            psths_cell_trial[j,k,int(np.floor(spike/bin_samples))-1] += 1 ## for the rare instance when a spike is detected at the last sample of a sweep
                        else:
                            if spike/bin_samples > duration * (1/.001 ):
                                    print(spike)
                            else:
                                    psths_cell_trial[j,k,int(np.floor(spike/bin_samples))] += 1
        psths.append(psths_cell_trial)

    
    # now we are going to count how many spikes after stim onset
    # and organize that by stimIndex
        baseCount = []
        spikeCount = []
        trialSpikes = []
    for rec,ind in zip(psths,stimindex): 
        base = np.sum(rec[:,:,int(info[1]*1000 - countperiod):int(info[1]*1000)],axis = 2)
        baseCount.append(np.mean(base,axis = 1))

        spikes = np.sum(rec[:,:,int(info[1]*1000):int(info[1]*1000+countperiod)],axis = 2)
        spikeCount.append(spikes)
        spikeOrg = np.zeros([len(rec),len(np.unique(ind)),np.sum(ind==1)])

        for i in range(len(np.unique(ind))):
            trialSelect = ind == i+1
            spikeOrg[:,i,:] = spikes[:,trialSelect]
            
        trialSpikes.append(spikeOrg)

    return psths,baseCount,spikeCount,trialSpikes
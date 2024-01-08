# run analysis
import numpy as np
import os
import math
from parseWhisker import * 
from miscAnalysis_correlations_alyssa import *


# 11/6/23 - AL for old BMX:CreER cavf/f cohort whisker paradigm 
# anesthetized 

# before running this, every recording must have .mat file 
# named 'bufferBoolean' (output of matlab script concatBufferVec.m)
# run this manually in matlab for every recording before futher analysis 

## setup 
### only need to change variables in this cell for editing analysis / adding data

# M F F F F F M F F M M F F M F F M M 
pathlist = [#r'\\research.files.med.Harvard.edu\Neurobio\GU LAB\PERSONAL FILES\Alyssa\MEA\whiskerData\BMX CreER old\071123\p1',
            r'\\research.files.med.Harvard.edu\Neurobio\GU LAB\PERSONAL FILES\Alyssa\MEA\whiskerData\BMX CreER old\071323\p1',
            r'\\research.files.med.Harvard.edu\Neurobio\GU LAB\PERSONAL FILES\Alyssa\MEA\whiskerData\BMX CreER old\071323\p2',
            r'\\research.files.med.Harvard.edu\Neurobio\GU LAB\PERSONAL FILES\Alyssa\MEA\whiskerData\BMX CreER old\071423\p1',
            r'\\research.files.med.Harvard.edu\Neurobio\GU LAB\PERSONAL FILES\Alyssa\MEA\whiskerData\BMX CreER old\071423\p2',
            r'\\research.files.med.Harvard.edu\Neurobio\GU LAB\PERSONAL FILES\Alyssa\MEA\whiskerData\BMX CreER old\071823\p1',
            #r'\\research.files.med.Harvard.edu\Neurobio\GU LAB\PERSONAL FILES\Alyssa\MEA\whiskerData\BMX CreER old\071023\p1',

            r'\\research.files.med.Harvard.edu\Neurobio\GU LAB\PERSONAL FILES\Alyssa\MEA\whiskerData\BMX CreER old\073123\p1',
            r'\\research.files.med.Harvard.edu\Neurobio\GU LAB\PERSONAL FILES\Alyssa\MEA\whiskerData\BMX CreER old\073123\p2',
            #r'\\research.files.med.Harvard.edu\Neurobio\GU LAB\PERSONAL FILES\Alyssa\MEA\whiskerData\BMX CreER old\080123\p1',
           # r'\\research.files.med.Harvard.edu\Neurobio\GU LAB\PERSONAL FILES\Alyssa\MEA\whiskerData\BMX CreER old\080223\p1',
            r'\\research.files.med.Harvard.edu\Neurobio\GU LAB\PERSONAL FILES\Alyssa\MEA\whiskerData\BMX CreER old\080323\p1',
            r'\\research.files.med.Harvard.edu\Neurobio\GU LAB\PERSONAL FILES\Alyssa\MEA\whiskerData\BMX CreER old\080323\p2',

           # r'\\research.files.med.Harvard.edu\Neurobio\GU LAB\PERSONAL FILES\Alyssa\MEA\whiskerData\BMX CreER old\090823\p1',
            r'\\research.files.med.Harvard.edu\Neurobio\GU LAB\PERSONAL FILES\Alyssa\MEA\whiskerData\BMX CreER old\091223\p1',
            r'\\research.files.med.Harvard.edu\Neurobio\GU LAB\PERSONAL FILES\Alyssa\MEA\whiskerData\BMX CreER old\091223\p2',
           # r'\\research.files.med.Harvard.edu\Neurobio\GU LAB\PERSONAL FILES\Alyssa\MEA\whiskerData\BMX CreER old\091323\p1',
           # r'\\research.files.med.Harvard.edu\Neurobio\GU LAB\PERSONAL FILES\Alyssa\MEA\whiskerData\BMX CreER old\091423\p1'

         
]

# all recordings
#idlist = np.array([387,340,340,75,75,429,338,481,481,438,482,439,439,437,483,483,440,441])
#genolist = np.array(['wt','wt','wt','cko','cko','wt','cko','cko','cko','wt','cko','wt','wt','wt','cko','cko','wt','cko'])
#wt = [387,340,429,438,439,437,440]
#cko = [75,338,481,482,483,441]

# males only
#idlist = np.array([387,338,438,482,437,440,441])
#genolist = np.array(['wt','cko','wt','cko','wt','wt','cko'])
#wt = [387,438,437,440]
#cko = [338,482,441]

# females only
idlist = np.array([340,340,75,75,429,481,481,439,439,483,483])
genolist = np.array(['wt','wt','wt','cko','cko','wt','cko','cko','cko','wt','cko','wt','wt','wt','cko','cko','wt','cko'])
wt = [340,429,439]
cko = [75,481,483]



# experiment metadata 
numSec = 45 # total trial duration 
stimOnset = 30 # how many seconds into recording does the stim turn on
sr = 20000 #  sample rate
numtrials = 50 # number of trial repeats

## variables for analysis 
bin = 0.05 # bin size for psth
onset = int(stimOnset / bin) #for indexing purposes 
# which samples to keep for psths, for ONE trial 
timeVec = np.array(np.ones((sr*numSec)),dtype = bool)
regthresh = 0.65


## preprocess
#for el in pathlist:
   # preprocessWhiskerData(el)


## run it
# aggregate data and save 
psths, numCells, troughPeak, TPlist, samples, spikes, numTrials,somachannel = get_all_psths_period(bin, numSec,pathlist,timeVec,reshape = True)
reg,genoList,geno,cellsbyid,recNum = expand_cell_info_wh(troughPeak,regthresh, idlist,[wt,cko],['Cre-','BMX:CreER+'],numCells)
allcells = aggcells(samples,spikes,numSec,idlist,wt,plot = False)
info = [numtrials,numSec,stimOnset,int(np.sum(numCells))] # list [number of trials, number of seconds, stim on in seconds,total number of cells in dataset]

# for every sample = 100ms and convolution kernel = 250 ms
ratevec,maxrate,baserate,stdbase,maxsub,_,_ = rate_measure(allcells,100,250,info)
# for every sample = 1ms and convolution kernel = 10 ms
rates_smallkernel,_,_,_,_,latency,onset = rate_measure(allcells,1,10,info)

responsive = maxrate > baserate + 3*stdbase

# current source density for determiniting cortical layer
#csd = csd_maps(pathlist)

# model decay period
taus1_bi,taus2_bi,rsq_bi,model_bi = fit_bidecay_constant(ratevec,10,norm = True)
curve_bi, cell_decay,intercept = fit_rate_data_bidecay(model_bi,ratevec,10,150)

#save data path 
save_data_path = r'\\research.files.med.Harvard.edu\Neurobio\GU LAB\PERSONAL FILES\Alyssa\MEA\whiskerData\BMX CreER old\females'
os.chdir(save_data_path)

np.savez('allcells_data.npz',
                            reg = reg, 
                            troughPeak = troughPeak,
                            geno = geno,
                            cellsbyid = cellsbyid,
                            maxrate = maxrate,
                            baserate = baserate,
                            stdbase = stdbase,
                            maxsub = maxsub,
                            latency = latency,
                            onset = onset,
                            responsive = responsive,
                            intercept = intercept,
                            recNum = recNum,
                            exp_model_m1 = model_bi[:,0],
                            exp_model_t1 = model_bi[:,1],
                            exp_model_m2 = model_bi[:,2],
                            exp_model_t2 = model_bi[:,3])
np.savez('rate_data.npz',
                        ratevec = ratevec,
                        rates_smallkernel = rates_smallkernel,
                        curve_bi = curve_bi)
listDict = {}
listDict['samples'] = samples
listDict['spikes']=spikes
listDict['cell_decay'] = cell_decay
listDict['allcells'] = allcells
listDict['psths'] = psths
listDict['TPlist'] = TPlist

np.save('jrclustData_decay.npy',listDict)

np.savez('recording_info.npz',
                            numCells = numCells,
                            genoList = genoList,
                            numTrials = numTrials,
                            idlist = idlist,
                            numSec = numSec,
                            stimOnset = stimOnset,
                            Fs = sr)
                            
#np.save('csd.npy',csd)

# run analysis
import numpy as np
import os
import math
from parseVisual import * 
from miscAnalysis_correlations_alyssa import *


# 11/9/23 - AL for cav-/- visual experiment 1
#### work in progress, have not run this yet
#### need to figure out how to handle different number of trials for some recordings
## setup 
### only need to change variables in this cell for editing analysis / adding data


# data info 
pathlist = [r'\\research.files.med.harvard.edu\Neurobio\GU LAB\PERSONAL FILES\Alyssa\MEA\visualData\cav1_null_cohort1\041023\p1',
          r'\\research.files.med.harvard.edu\Neurobio\GU LAB\PERSONAL FILES\Alyssa\MEA\visualData\cav1_null_cohort1\041023\p2',
            r'\\research.files.med.harvard.edu\Neurobio\GU LAB\PERSONAL FILES\Alyssa\MEA\visualData\cav1_null_cohort1\041023\p3',
           # r'\\research.files.med.harvard.edu\Neurobio\GU LAB\PERSONAL FILES\Alyssa\MEA\visualData\cav1_null_cohort1\041323\p1',
           # r'\\research.files.med.harvard.edu\Neurobio\GU LAB\PERSONAL FILES\Alyssa\MEA\visualData\cav1_null_cohort1\041323\p2',
           # r'\\research.files.med.harvard.edu\Neurobio\GU LAB\PERSONAL FILES\Alyssa\MEA\visualData\cav1_null_cohort1\041323\p3',
            r'\\research.files.med.harvard.edu\Neurobio\GU LAB\PERSONAL FILES\Alyssa\MEA\visualData\cav1_null_cohort1\042623\p1',
           # r'\\research.files.med.harvard.edu\Neurobio\GU LAB\PERSONAL FILES\Alyssa\MEA\visualData\cav1_null_cohort1\050523\p1',
           # r'\\research.files.med.harvard.edu\Neurobio\GU LAB\PERSONAL FILES\Alyssa\MEA\visualData\cav1_null_cohort1\050523\p2',
           # r'\\research.files.med.harvard.edu\Neurobio\GU LAB\PERSONAL FILES\Alyssa\MEA\visualData\cav1_null_cohort1\050523\p3',
           # r'\\research.files.med.harvard.edu\Neurobio\GU LAB\PERSONAL FILES\Alyssa\MEA\visualData\cav1_null_cohort1\051223\p1',
           # r'\\research.files.med.harvard.edu\Neurobio\GU LAB\PERSONAL FILES\Alyssa\MEA\visualData\cav1_null_cohort1\051223\p2',

]

#save data path 
save_data_path = r'\\research.files.med.Harvard.edu\Neurobio\GU LAB\PERSONAL FILES\Alyssa\MEA\visualData\cav1_null_cohort1'

# wt only
idlist = np.array([212,212,212,214])

# all animals
#idlist = np.array([212,212,212,114,114,114,214,211,211,211,46,46])

#wtonly 
penlist = np.array([1,2,3,1])

#penlist = np.array([1,2,3,1,2,3,1,1,2,3,1,2])
wt = [212,214,417]
mut = [114,211,46]


# experiment metadata 
numSec = 8 # total trial duration 
stimOnset = 6 # how many seconds into recording does the stim turn on
sr = 20000 #  sample rate
numtrials = 34 ## or 24 I think, this is variable between some recordings

## variables for analysis 
bin = 0.05 # bin size for psth
onset = int(stimOnset / bin) #for indexing purposes 
stimIndexList = [1,2,3,4,5,6,7,8,9,10,11,12] # stim indices to include for analysis from matlab file

# which samples to keep for psths, for ONE trial 
timeVec = np.array(np.ones((sr*numSec)),dtype = bool)
regthresh = 0.55


## preprocess
#for el in pathlist:
 #   processVisualData2(el)


## run it
# aggregate data and save 
psths, numCells, troughPeak, TPlist, samples, spikes, numTrials,selected_psth,indFilt,stimIndex = get_all_psths_period(bin, numSec,pathlist,timeVec,stimIndexList)
reg,genoList,geno,cellsbyid,recNum = expand_cell_info(troughPeak,regthresh, idlist,[wt,mut],['Control','mut'],numCells)
allcells = aggcells(samples,spikes,numtrials,idlist,wt,plot = True)
info = [numSec,stimOnset] # list [number of trials, number of seconds, stim on in seconds,total number of cells in dataset]








# for every sample = 100ms and convolution kernel = 250 ms
ratevec,maxrate,baserate,stdbase,maxsub,_ = rate_measure_vis(allcells,stimIndex,100,250,info)
#noise_resp,gratings_resp = compute_responsivity(ratevec,baserate,stdbase,info,10)


# for every sample = 1ms and convolution kernel = 10 ms
#rates_smallkernel,_,_,_,_,latency,onset = rate_measure(allcells,1,10,info)

#responsive = maxrate > baserate + 3*stdbase

# current source density for determiniting cortical layer
#csd = csd_maps(pathlist)

# model decay period
#taus1_bi,taus2_bi,rsq_bi,model_bi = fit_bidecay_constant(ratevec,10,norm = True)
#curve_bi, cell_decay,intercept = fit_rate_data_bidecay(model_bi,ratevec,10,150)


os.chdir(save_data_path)
np.savez('allcells_data.npz',
                            reg = reg, 
                            troughPeak = troughPeak,
                            geno = geno,
                            cellsbyid = cellsbyid,

                         #   noise_resp = noise_resp,
                          #  gratings_resp = gratings_resp,
                           # latency = latency,
                           # onset = onset,
                           # responsive = responsive,
                          #  intercept = intercept,
                            recNum = recNum,
                        #    exp_model_m1 = model_bi[:,0],
                        #    exp_model_t1 = model_bi[:,1],
                         #   exp_model_m2 = model_bi[:,2],
                         #   exp_model_t2 = model_bi[:,3])
)

listDict = {}
listDict['samples'] = samples
listDict['spikes']=spikes
#listDict['cell_decay'] = cell_decay
listDict['allcells'] = allcells
listDict['psths'] = psths
listDict['TPlist'] = TPlist
listDict['stimIndex'] = stimIndex
listDict['maxrate'] = maxrate
listDict['baserate'] = baserate
listDict['stdbase'] = stdbase
listDict['maxsub'] = maxsub
listDict['allrates'] = ratevec

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


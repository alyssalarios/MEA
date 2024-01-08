# run analysis
import numpy as np
import os
import math
from parseWhisker import * 
from miscAnalysis_correlations_alyssa import *


# 10/3/23 - AL for young BMX:CreER cavf/f cohort whisker paragigm 
# anesthetized cohort2

## setup 
### only need to change variables in this cell for editing analysis / adding data

pathlist = [#r'\\research.files.med.Harvard.edu\Neurobio\GU LAB\PERSONAL FILES\Alyssa\MEA\whiskerData\BMX CreER 1month\082123\p1',
          #  r'\\research.files.med.Harvard.edu\Neurobio\GU LAB\PERSONAL FILES\Alyssa\MEA\whiskerData\BMX CreER 1month\082223\p1',
          #  r'\\research.files.med.Harvard.edu\Neurobio\GU LAB\PERSONAL FILES\Alyssa\MEA\whiskerData\BMX CreER 1month\082223\p2',
          #  r'\\research.files.med.Harvard.edu\Neurobio\GU LAB\PERSONAL FILES\Alyssa\MEA\whiskerData\BMX CreER 1month\083023\p1',
           # r'\\research.files.med.Harvard.edu\Neurobio\GU LAB\PERSONAL FILES\Alyssa\MEA\whiskerData\BMX CreER 1month\083023\p2',
           # r'\\research.files.med.Harvard.edu\Neurobio\GU LAB\PERSONAL FILES\Alyssa\MEA\whiskerData\BMX CreER 1month\083023\p3',
           # r'\\research.files.med.Harvard.edu\Neurobio\GU LAB\PERSONAL FILES\Alyssa\MEA\whiskerData\BMX CreER 1month\083123\p1',
           # r'\\research.files.med.Harvard.edu\Neurobio\GU LAB\PERSONAL FILES\Alyssa\MEA\whiskerData\BMX CreER 1month\083123\p2',
           # r'\\research.files.med.Harvard.edu\Neurobio\GU LAB\PERSONAL FILES\Alyssa\MEA\whiskerData\BMX CreER 1month\090523\p1',
           # r'\\research.files.med.Harvard.edu\Neurobio\GU LAB\PERSONAL FILES\Alyssa\MEA\whiskerData\BMX CreER 1month\090723\p1',
           # r'\\research.files.med.Harvard.edu\Neurobio\GU LAB\PERSONAL FILES\Alyssa\MEA\whiskerData\BMX CreER 1month\090723\p2',
          #  r'\\research.files.med.Harvard.edu\Neurobio\GU LAB\PERSONAL FILES\Alyssa\MEA\whiskerData\BMX CreER 1month\101123\p1',
           # r'\\research.files.med.Harvard.edu\Neurobio\GU LAB\PERSONAL FILES\Alyssa\MEA\whiskerData\BMX CreER 1month\101123\p2',

           r'\\research.files.med.Harvard.edu\Neurobio\GU LAB\PERSONAL FILES\Alyssa\MEA\whiskerData\BMX CreER 1month\110223\p2',
           r'\\research.files.med.Harvard.edu\Neurobio\GU LAB\PERSONAL FILES\Alyssa\MEA\whiskerData\BMX CreER 1month\110323\p1',
            r'\\research.files.med.Harvard.edu\Neurobio\GU LAB\PERSONAL FILES\Alyssa\MEA\whiskerData\BMX CreER 1month\110323\p2',
            r'\\research.files.med.Harvard.edu\Neurobio\GU LAB\PERSONAL FILES\Alyssa\MEA\whiskerData\BMX CreER 1month\110323\p3',


            r'\\research.files.med.Harvard.edu\Neurobio\GU LAB\PERSONAL FILES\Alyssa\MEA\whiskerData\BMX CreER 1month\110823\p1',
            r'\\research.files.med.Harvard.edu\Neurobio\GU LAB\PERSONAL FILES\Alyssa\MEA\whiskerData\BMX CreER 1month\111023\p1',
            r'\\research.files.med.Harvard.edu\Neurobio\GU LAB\PERSONAL FILES\Alyssa\MEA\whiskerData\BMX CreER 1month\111023\p2',

]



# females only
idlist = np.array([293,292,292,292,97,95,95])
genolist = np.array(['Mutant','Mutant','Mutant','Mutant','Control','Mutant','Mutant'])


#all recordings, unbalanced
#idlist = np.array([354,356,356,358,358,358,360,360,373,375,375,835,835,293,292,292,292,97,95,95])
#genolist = np.array(['Control','Control','Control','Mutant','Mutant','Mutant','Control','Control','Control','Mutant','Mutant','Mutant','Mutant',
               #      'Mutant','Mutant','Mutant','Mutant','Control','Mutant','Mutant'])

# all sexes, balanced
# idlist = np.array([354,356,356,358,358,360,360,373,375,835,293,292,97,95])
#genolist = np.array(['Control','Control','Control','Mutant','Mutant','Control','Control','Control','Mutant','Mutant','Mutant',
 #                    'Mutant','Control','Mutant'])

#males only
#idlist = np.array([354,356,356,358,358,358,360,360,373,375,375,835,835])
#genolist = np.array(['Control','Control','Control','Mutant','Mutant','Mutant','Control','Control','Control','Mutant','Mutant','Mutant','Mutant'])
                   

#alll mice 
#wt = [354,356,360,373,97]
#cko = [358,375,835,293,292,95]

#males
#wt = [354,356,360,373]
#cko = [358,375,835]

#females
wt = [97]
cko = [293,292,95]

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
regthresh = 0.55


## preprocess
#for el in pathlist:
 #   preprocessWhiskerData(el)


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
save_data_path = r'\\research.files.med.Harvard.edu\Neurobio\GU LAB\PERSONAL FILES\Alyssa\MEA\whiskerData\BMX CreER 1month\females'
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
listDict['somachannel'] = somachannel

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

#for noise correlations
cor_av = []
id_cors = []
geno_cors = []
epochs = []

noiseCor = False

if noiseCor:
   for pen,rec_id,rec_geno in zip(psths,idlist,genolist):

      times = range(0,900,20)
      for period in times:
         cor_map, cor_vals = noise_correlations(pen,period,period+20)

         cor_av.append(np.mean(cor_vals))
         id_cors.append(rec_id)
         geno_cors.append(rec_geno)
         epochs.append(period)

   noise_cor_data = {'cor':cor_av,
        'epochs':epochs,
        'geno':geno_cors,
        'id':id_cors}

   noise_cor_data = pd.DataFrame(noise_cor_data)
   noise_cor_data.to_pickle("noise_correlations.pkl")

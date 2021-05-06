"""
This is example code for running the Gibbs sampler. As it is different than
all the other samplers it gets its own example.
"""
import os
import time
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

from Samplers.gibbsSampler import GibbsSampler
from Utility.visualizations import video2d

#Inverts a Gaussian CDF
def quantCal(mu, sigma, p):
    val = tf.math.erfinv(2*p-1)
    val = val*tf.math.sqrt([2.0])
    val = val*sigma
    val = val+mu
    return(val)

#Two independent distributions
def quant(state, quantile, dim):
    newState = tf.where(dim==1,
                       quantCal(0,2,quantile),
                       quantCal(2,1,quantile))
    
    
    return(newState)

def main():    
    
    start = time.time()
    
    dimensions=2 #2d distribtuion
    samples = 100000 #Saved samples along each chain
    burnin = 0 #Number of samples along each chain before saving
    currentX = 100 #Starting xval
    currentY = 100 #Starting yval
    parallelSamples = 10 #Number of parallel chains
    
    startingVals = [[currentX]*parallelSamples,[currentY]*parallelSamples]
    
    #Declare the sampler
    mhSampler = GibbsSampler(dimensions, quant)
    
    #Run the sampler
    mhSampler.run_sampler(startingVals,burnin, samples, parallelSamples)
    
    #Extract the results
    sampledVals, acceptanceRate = mhSampler.extract_states()
    
    #Display basic results
    print("Acceptance rate", acceptanceRate)
    totalTime = time.time()-start
    print("Time elapsed:", totalTime, "seconds")
    totalSamples = (samples+burnin)*parallelSamples
    print("Time elapsed per sample:", (1000000)*totalTime/(totalSamples),
          "micro seconds")
    
    #Animate the sampling process for a real time video
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    video2d(sampledVals, "gibbs2", totalTime, fps=30, xrange=[-5,5], yrange=[-5,5])


if(__name__=="__main__"):
    main()

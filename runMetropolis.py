import os
import time
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

from math import pi

from metropolisHastings import MetropolisHastings
from visualizations import video2d


def gaussian(x, mu, sigma):
    return((1/(sigma*tf.sqrt(2*pi)))*tf.exp(-0.5*((x-mu)/sigma)**2))

def p(state):
    r=tf.sqrt(state[0]**2+state[1]**2)
    return(tf.math.log(gaussian(r,2,0.5)))

def main():    
    
    start = time.time()
    
    dimensions=2 #2d distribtuion
    alpha = 2.5 #max change in a dimension
    samples = 10000 #Saved samples along each chain
    burnin = 100 #Number of samples along each chain before saving
    currentX = 0 #Starting xval
    currentY = 0 #Starting yval
    parallelSamples = 10 #Number of parallel chains
    
    startingVals = [[currentX]*parallelSamples,[currentY]*parallelSamples]
    
    #Declare the sampler
    mhSampler = MetropolisHastings(dimensions, p, alpha)
    
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
    video2d(sampledVals, "donut", totalTime)


if(__name__=="__main__"):
    main()

"""
This is example code for running any of the samplers except the Gibbs sampler.
Note that different samplers require some different parameters, so be sure
to check that before using. The example used is the simple Metropolis-Hastings
sampler.
"""

import os
import time
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import matplotlib.pyplot as plt

from math import pi

#Import a sampler
from Samplers.metropolisHastings import MetropolisHastings

from Utility.visualizations import video2d

dtype = tf.float32

#Gaussian distributions
def gaussian(x, mu, sigma):
    a=tf.math.log(tf.cast((1/(sigma*tf.sqrt(2*pi))), dtype))
    b= tf.cast(-0.5*((x-mu)/sigma)**2, dtype)
    return(a+b)

#Donut distribution
def p(state):
    r=tf.sqrt((state[0])**2+(state[1]/10)**2)
    return(gaussian(r,2,0.3))

def main():    
    
    start = time.time()
    
    dimensions=2 #2d distribtuion
    alpha = 2.5 #max change in a dimension
    samples = 10000 #Saved samples along each chain
    burnin = 100 #Number of samples along each chain before saving
    currentX = 10 #Starting xval
    currentY = 1 #Starting yval
    parallelSamples = 32 #Number of parallel chains
    target=0.25 #Target acceptance rate
    
    startingVals = [[currentX]*parallelSamples,[currentY]*parallelSamples]
    
    #Declare the sampler
    mhSampler = MetropolisHastings(dimensions, p, alpha, 
                                   parallelSamples=parallelSamples,
                                   dtype=dtype, target=target)
    
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
    
    
    #Plot results
    xVals = sampledVals[:,0,:].reshape((-1))
    yVals = sampledVals[:,1,:].reshape((-1))
    plt.figure()
    plt.hist2d(xVals,yVals, bins=30)
    plt.show()
    
    #Animate the sampling process for a real time video
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    video2d(sampledVals, "donut_met", totalTime, fps=30, xrange=[-5,5], yrange=[-50,50])
    
    
    


if(__name__=="__main__"):
    main()

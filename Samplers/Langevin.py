import tensorflow as tf

from math import pi

from Samplers.sampler import Sampler

class Langevin(Sampler):
    """
    An implementation of the Langevin MCMC sampler. This sampler
    works by proposing a new state according to an approximation of Lanegvin
    dynamics. When simulated perfectly, Langevin dynamics will converge to the
    desired distribution. Due to simulation errors, however, an accept-reject
    step is required.
    """
    
    def __init__(self, dimensions, log_likelihood_fn, alpha,
                 parallelSamples = 1, dtype=tf.float32, mu = 1, gamma = 0.04,
                 t0 = 10, kappa = 0.75, target=0.6):
        """
        The constructor for the Langevin sampler

        Parameters
        ----------
        dimensions : the number of dimensions of the sampled distribution
        log_likelihood_fn : a function accepting a state from the distribution
            and returning the natural logarithm of the probability of that
            state. This probability need not be normalized.
        alpha : a parameter determining the maximum change in a state 
            component during a single sampling step. This value is updated
            using dual-averaging.
        parallelSamples : Number of parallel chains to run
        dtype : data type of data, tf.float32 is recomended
        mu : Scaling factor on initial epsilon for dual-averaging
        gamma : Controls shrinkage towards mu for dual-averaging
        t0 : Greater than 0, stabalizes initial iterations  for dual-averaging
        kappa : controls step size schedule for dual-averaging
        target : target acceptance probability for dual-averaging

        Returns
        -------
        None.

        """
        self.dtype = dtype
        self.dimensions = dimensions
        self.log_likelihood_fn = log_likelihood_fn
        self.sampledVals = []
        self.acceptance=tf.cast(0, self.dtype)  
        self.alpha = tf.cast(tf.ones((1,parallelSamples))*alpha, self.dtype)
        self.logAlphaBar=tf.cast(tf.zeros((1,parallelSamples)), self.dtype)
        self.mu = tf.cast(tf.math.log(mu*self.alpha), self.dtype)
        self.gamma=tf.cast(gamma, self.dtype)
        self.t0 = tf.cast(t0, self.dtype)
        self.kappa=tf.cast(kappa, self.dtype)
        self.h = tf.cast(0*self.alpha, self.dtype)
        self.target=tf.cast(target, self.dtype)


    @tf.function   
    def run_sampler_inner(self, initial_state):
        """
        The inner sampling step for the Langein sampler

        Parameters
        ----------
        initial_state : the starting point for the sampler. It should be
            structured as [[1st chain - 1st dim, 2nd chain - 1st dim , ...],
                           [1st chain - 2nd dim, 2nd chain - 2nd dim , ...],
                           ...]

        Returns
        -------
        sampledVals: the states sampled by the Metropolis-Hastings sampler
        acceptance: number of new states accepted accross all parallel
            iterations for each sampling step
        """
        
        acceptance = tf.cast(0, self.dtype)
        
        #Use float32 for ease of use with GPUs
        currentState = tf.cast(initial_state, self.dtype)
        with tf.GradientTape() as g:
          g.watch(currentState)
          currentProb = self.log_likelihood_fn(currentState)
          
        currentGrad = g.gradient(currentProb, currentState)
        
        
        #run sampler for the number of burn_in steps
        i = tf.constant(0)
        samples = self.burn_in
        condition = lambda i, currentState, currentGrad, currentProb, h, logAlphaBar, \
                                                alpha: tf.less(i, samples)
        
        
        h, logAlphaBar, alpha = self.h, self.logAlphaBar, self.alpha
        
        #Body of while loop for burn_in, no states or acceptance info kept
        def body(i, currentState, currentGrad, currentProb, h, logAlphaBar, alpha):
            m=tf.cast(i+1, self.dtype) #Step number
            
            #Progress one step
            currentState, currentGrad, currentProb, accept = self.one_step(currentState,
                                                              currentGrad,
                                                              currentProb,
                                                              alpha)
           
            #Update h val
            h = (1-1/(m+self.t0))*h+(1/(m+self.t0))*(self.target-accept)
            
            #Update log Alpha
            logAlpha = self.mu-h*(m**0.5)/self.gamma
            
            #Update log Alpha bar
            logAlphaBar = m**(-self.kappa)*logAlpha
            logAlphaBar += (1-m**(-self.kappa))*logAlphaBar
            
            #Set alpha
            alpha = tf.math.exp(logAlpha)
            
            return([tf.add(i, 1), currentState, currentGrad, currentProb, h, logAlphaBar,
                    alpha])
        
            

        #tf.while_loop to speed up sampling
        i, currentState, currentGrad, currentProb, h, logAlphaBar, alpha = tf.while_loop(
            condition, body, [i, currentState, currentGrad, currentProb, h, logAlphaBar,
                              alpha])
        
        alpha=tf.exp(logAlphaBar)
        tf.print("Final alpha", alpha)
        
        
        
        #tensorArray of set size to store samples
        sampledVals = tf.TensorArray(self.dtype, size=self.samples,
                                     dynamic_size=False)
        
        #run sampler for the number of sampling steps
        samples += self.samples
        def condition(i, currentState, currentGrad, currentProb, sampledVals, acceptance, alpha):
            return(tf.less(i, samples))
        
        #Body of while loop for sampling, states and acceptance info kept
        def body(i, currentState, currentGrad, currentProb, sampledVals, acceptance, alpha):
            
            currentState, currentGrad, currentProb, accept = self.one_step(currentState,
                                                                           currentGrad,
                                                                           currentProb,
                                                                           alpha)
            acceptance+=tf.reduce_sum(accept)
            sampledVals= sampledVals.write(i-self.burn_in, currentState)
            
            
            return([tf.add(i, 1), currentState, currentGrad, currentProb, sampledVals,
                    acceptance, alpha])

        #tf.while_loop to speed up sampling
        i, currentState,currentGrad,  currentProb, sampledVals,acceptance, alpha = \
            tf.while_loop(condition, body, [i, currentState, currentGrad, currentProb,
                                            sampledVals, acceptance, alpha])
        
        #trun sampledVals into a normal Tensor
        sampledVals = sampledVals.stack()                                                                                    
        
        return(sampledVals, acceptance)

        
        
    def one_step(self, currentState, currentGrad, currentProb, alpha):
        """
        A function which performs one step of the Metropolis-Hastings sampler

        Parameters
        ----------
        currentState : the current state for all the parallel chains
        currentGrad : current gradient
        currentProb : the current probabilities of the states
        alpha : current step size

        Returns
        -------
        updatedState : the updated state for all the parallel chains
        updatedGrad : the gradient at updatedState
        updatedProb : the updated probabilities of the states
        accepted : 1 if the new state was accepted, 0 otherwise for each chain
        
        """
        newState, newGrad, newProb = self.propose_states(currentState, currentGrad, alpha)
        updatedState, updatedProb, updatedGrad, accepted = self.accept_reject(
                                                                 currentState,
                                                                 currentProb,
                                                                 currentGrad,
                                                                 newState,
                                                                 newGrad,
                                                                 newProb)
        
        return(updatedState, updatedGrad, updatedProb, accepted)

        
        
    def propose_states(self, currentState, currentGrad, alpha):
        """
        

        Parameters
        ----------
        currentState : the current state of the sampler for each chain
        currentGrad : the gradeint at the current state of the sampler for 
            each chain
        alpha : current step size

        Returns
        -------
        newState : the proposed new state of the sampler for each chain
        newGrad : the gradient at the new state
        newProb : the probability at the new state

        """
        #Random step in direction of gradient
        step = alpha*tf.random.normal(currentState.shape, dtype=self.dtype)
        step = tf.add(step,0.5*self.alpha*currentGrad)
        
        #Get gradient of new state
        newState=tf.add(currentState, step)
        newProb = None
        with tf.GradientTape() as g:
          g.watch(newState)
          newProb = self.log_likelihood_fn(newState)
        
        newGrad = g.gradient(newProb, newState)
        
        return(newState, newGrad, newProb)

   
    def accept_reject(self, currentState,currentProb,currentGrad,newState,
                      newGrad, newProb):
        """
        

        Parameters
        ----------
        currentState : the current state for all the parallel chains
        currentProb : the current probabilities of the states
        currentGrad : the current gradient
        newState : the next proposed state for all the parallel chains
        newGrad : the gradient at the new state
        newProb : the probability at the new state

        Returns
        -------
        updatedProb : the updated probabilities of the states
        updatedState : the updated state for all the parallel chains
        updatedGrad : the updated gradient for all the parallel chains
        acceptProb : acceptance probability for each chain

        """
        
        #Prob of traveling between each of the two points
        transitionProb1 = self.gaussian_log_prob(newState, 
                                                 currentState+0.5*currentGrad*self.alpha,
                                                 self.alpha)
        transitionProb2 = self.gaussian_log_prob(currentState,
                                                 newState+0.5*newGrad*self.alpha,
                                                 self.alpha)
        
        randomAccept = tf.random.uniform(newProb.shape, dtype=self.dtype)
        acceptCriteria = randomAccept<tf.math.exp(newProb+transitionProb2-currentProb-transitionProb1)
        
        acceptProb=tf.math.exp(newProb+transitionProb2-currentProb-transitionProb1)
        acceptProb= tf.where(tf.math.is_nan(acceptProb),0*acceptProb,acceptProb)
        acceptProb = tf.where(0*acceptProb+1<acceptProb,0*acceptProb+1, acceptProb)
        
        updatedProb = tf.where(acceptCriteria, newProb, currentProb)
        updatedState = tf.where(acceptCriteria, newState, currentState)
        updatedGrad = tf.where(acceptCriteria, newGrad, currentGrad)
        
        return(updatedState, updatedProb, updatedGrad, acceptProb)
    
    def gaussian_log_prob(self, x, mu, sigma):
        """
        Caluclates the log probability of x given mean mu and standard
        deviation sigma.

        Parameters
        ----------
        x : value of interest
        mu : mean of distribution
        sigma : standard deviatoin of distribution

        Returns
        -------
        val : log probability of x

        """
        val = tf.math.log((1/(sigma*tf.cast(tf.sqrt(2*pi), self.dtype))))
        val = val - 0.5*((x-mu)/sigma)**2 
        val = tf.reduce_sum(val, axis=0)
        return(val)
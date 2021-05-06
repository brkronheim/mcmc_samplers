import tensorflow as tf

from Samplers.samplerV2 import Sampler

class MetropolisHastings(Sampler):
    """
    An implementation of the Metropolis-Hastings MCMC sampler. This sampler
    works by proposing a new state which differs in each dimension by a random
    amount normally distributed with mean 0, standard deviation alpha. The new
    state is always accepted if it has a higher probability than the previous
    state, and accepted with probability p_new/p_current if p_new<p_current.
    
    The value for alpha is adapted using the dual-averaging algorithm during
    burnin.
    
    This implementation differs from the other as it is built to handle
    lists of Tensors, as opposed to just a single vector.
    """
    
    def __init__(self, dimensions, log_likelihood_fn, alpha,
                 parallelSamples = 1, dtype=tf.float32, mu = 100, gamma = 0.4,
                 t0 = 10, kappa = 0.75, target=0.6):
        """
        The constructor for the Metropolis-Hastings sampler

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
        The inner sampling step for the Metropolis-Hastings sampler

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
        
        #currentState = tf.cast(initial_state, self.dtype)
        currentState = initial_state
        
        
        currentProb = self.log_likelihood_fn(currentState)
        
        
        #run sampler for the number of burn_in steps
        i = tf.constant(0)
        samples = self.burn_in
        condition = lambda i, currentState, currentProb, h, logAlphaBar, \
                                                alpha: tf.less(i, samples)
        
        
        h, logAlphaBar, alpha = self.h, self.logAlphaBar, self.alpha
        
        #Body of while loop for burn_in, no states or acceptance info kept
        def body(i, currentState, currentProb, h, logAlphaBar, alpha):
            m=tf.cast(i+1, self.dtype) #Step number
            
            #Progress one step
            currentState, currentProb, accept = self.one_step(currentState,
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
            
            return([tf.add(i, 1), currentState, currentProb, h, logAlphaBar,
                    alpha])
        
            

        #tf.while_loop to speed up sampling
        i, currentState, currentProb, h, logAlphaBar, alpha = tf.while_loop(
            condition, body, [i, currentState, currentProb, h, logAlphaBar,
                              alpha])
        
        alpha=tf.exp(logAlphaBar)
        tf.print("Final alpha", alpha)
        
        #tensorArray of set size to store samples
        sampledVals = [tf.TensorArray(self.dtype, size=self.samples,
                                     dynamic_size=False)for x in range(self.dimensions)]
        
        #run sampler for the number of sampling steps
        samples += self.samples
        def condition(i, currentState, currentProb, sampledVals, acceptance, alpha):
            return(tf.less(i, samples))
        
        #Body of while loop for sampling, states and acceptance info kept
        def body(i, currentState, currentProb, sampledVals, acceptance, alpha):
            
            currentState, currentProb, accept = self.one_step(currentState,
                                                              currentProb,
                                                              alpha)
            acceptance+=tf.reduce_sum(accept)
            sampledVals=[ vals.write(i-self.burn_in, state) for [vals, state] in zip(sampledVals, currentState)]
            
            
            return([tf.add(i, 1), currentState, currentProb, sampledVals,
                    acceptance, alpha])

        #tf.while_loop to speed up sampling
        i, currentState, currentProb, sampledVals,acceptance, alpha = \
            tf.while_loop(condition, body, [i, currentState, currentProb,
                                            sampledVals, acceptance, alpha])
        
        #trun sampledVals into a normal Tensor
        sampledVals = [vals.stack() for vals in sampledVals]                                                                                    
        
        return(sampledVals, acceptance)

        
        
    def one_step(self, currentState, currentProb, alpha):
        """
        A function which performs one step of the Metropolis-Hastings sampler

        Parameters
        ----------
        currentState : the current state for all the parallel chains
        currentProb : the current probabilities of the states

        Returns
        -------
        updatedState : the updated state for all the parallel chains
        updatedProb : the updated probabilities of the states
        accepted : 1 if the new state was accepted, 0 otherwise for each chain
        
        """
        newState = self.propose_states(currentState, alpha)
        updatedState, updatedProb, accepted = self.accept_reject(currentState,
                                                                 currentProb,
                                                                 newState)
        
        return(updatedState, updatedProb, accepted)

        
        
    def propose_states(self, currentState, alpha):
        """
        

        Parameters
        ----------
        currentState : the current state of the sampler for each chain
        alpha : current step size

        Returns
        -------
        newState : the proposed new state of the sampler for each chain

        """
        #step = alpha*tf.random.normal(currentState.shape, dtype=self.dtype)
        
        step = [tf.tensordot(alpha,tf.random.normal(state.shape, dtype=self.dtype),1) for state in currentState]
        
        newState = [tf.add(state, stepVal) for [state, stepVal] in zip(currentState, step)]
        
        #tf.add(currentState, step)
        return(newState)

   
    def accept_reject(self, currentState, currentProb, newState):
        """
        

        Parameters
        ----------
        currentState : the current state for all the parallel chains
        currentProb : the current probabilities of the states
        newState : the next proposed state for all the parallel chains

        Returns
        -------
        updatedState : the updated state for all the parallel chains
        updatedProb : the updated probabilities of the states
        acceptProb : acceptance probability for each chain

        """
        newProb = self.log_likelihood_fn(newState)
        
        randomAccept = tf.random.uniform(newProb.shape, dtype=self.dtype)
        acceptCriteria = randomAccept<tf.math.exp(newProb-currentProb)
        
        acceptProb=tf.math.exp(newProb-currentProb)
        acceptProb= tf.where(tf.math.is_nan(acceptProb),0*acceptProb,acceptProb)
        acceptProb = tf.where(0*acceptProb+1<acceptProb,0*acceptProb+1, acceptProb)
        
        updatedProb = tf.where(acceptCriteria, newProb, currentProb)
        
        updatedState = [tf.where(tf.reshape(tf.repeat(acceptCriteria,tf.size(stateN)//tf.size(acceptCriteria)), stateN.shape), stateN, stateO) for [stateN, stateO] in zip(newState, currentState)]
        return(updatedState, updatedProb, acceptProb)
    
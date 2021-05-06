import tensorflow as tf

from Samplers.sampler import Sampler

class MultiTryMH(Sampler):
    """
    An implementation of the Multi Try sampler. This sampler
    works by proposing a new series of new states using the normal MH mechanism,
    selecting one based on relative probabilites, then simulating back and also
    picking one based on relative probabilites. The acceptance depends on the
    relative probabilities of the two states.
    """

    def __init__(self, dimensions, log_likelihood_fn, alpha, mtSteps,
                 parallelSamples = 1, dtype=tf.float32, mu = 100, gamma = 0.4,
                 t0 = 10, kappa = 0.75, target=0.6):
        """
        The constructor for the Multi Try sampler

        Parameters
        ----------
        dimensions : the number of dimensions of the sampled distribution
        log_likelihood_fn : a function accepting a state from the distribution
            and returning the natural logarithm of the probability of that
            state. This probability need not be normalized.
        alpha : a parameter determining the maximum change in a state 
            component during a single sampling step. This value is updated
            using dual-averaging.
        mtSteps : Number of steps to try for each multi try proposal
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
        self.mtSteps = mtSteps
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
        
        currentState = tf.cast(initial_state, self.dtype) 
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
        sampledVals = tf.TensorArray(self.dtype, size=self.samples,
                                     dynamic_size=False)
        
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
            sampledVals= sampledVals.write(i-self.burn_in, currentState)
            
            
            return([tf.add(i, 1), currentState, currentProb, sampledVals,
                    acceptance, alpha])

        #tf.while_loop to speed up sampling
        i, currentState, currentProb, sampledVals,acceptance, alpha = \
            tf.while_loop(condition, body, [i, currentState, currentProb,
                                            sampledVals, acceptance, alpha])
        
        #trun sampledVals into a normal Tensor
        sampledVals = sampledVals.stack()                                                                                      
        
        return(sampledVals, acceptance)

        
        
    def one_step(self, currentState, currentProb, alpha):
        """
        A function which performs one step of the Multi-Try sampler

        Parameters
        ----------
        currentState : the current state for all the parallel chains
        currentProb : the current probabilities of the states
        alpha : current step size

        Returns
        -------
        updatedState : the updated state for all the parallel chains
        updatedProb : the updated probabilities of the states
        accepted : acceptance probability for each chain
        
        """
        newState, prob1, prob2 = self.propose_states(currentState, alpha)
        updatedState, accepted = self.accept_reject(currentState, newState, prob1, prob2)
        return(updatedState, currentProb, accepted)

        
        
    def propose_states(self, currentState, alpha):
        """
        

        Parameters
        ----------
        currentState : the current state of the sampler for each chain
        alpha : current step size

        Returns
        -------
        newState : the proposed new state of the sampler for each chain
        logProb1 : log probability of state 1
        logProb2 : log probabiilty of state 2

        """
        
        #Move in a random direction for each attempt
        matrix = tf.random.normal([currentState.shape[0],
                                             currentState.shape[1],
                                             self.mtSteps], dtype=self.dtype)
        step = tf.transpose(tf.tensordot(alpha,tf.transpose(matrix,(1,0,2)),
                                         axes=1),(1,0,2))
        
        #Copy initial state
        newState = tf.repeat(tf.expand_dims(currentState,-1), self.mtSteps,
                             axis=-1)
        
        #Update each initial state
        newState=tf.add(newState, step)
        
        #Get probs of new states
        probs = self.log_likelihood_fn(newState)
        
        expProbs1 = tf.reduce_sum(tf.math.exp(probs),1)
        selections = tf.squeeze(tf.random.categorical(probs, 1))
        if(selections.shape==()):
            selections=[selections]
        
        #Select a new state
        newState = tf.transpose(newState, perm=[1, 2,0])
        newState = tf.gather(newState,selections, batch_dims=1)
        newState = tf.transpose(newState)

        
        #Repeat same process, start from new state
        matrix = tf.random.normal([currentState.shape[0],
                                              currentState.shape[1],
                                              self.mtSteps-1], dtype=self.dtype)
        step2 = tf.transpose(tf.tensordot(alpha,tf.transpose(matrix,(1,0,2)),
                                          axes=1),(1,0,2))
        
        newState2 = tf.repeat(tf.expand_dims(newState,-1),self.mtSteps-1,
                              axis=-1)
        
        newState2=tf.add(newState2, step2)
        newState2=tf.concat([tf.expand_dims(currentState,-1), newState2], axis=2)
        probs2 = self.log_likelihood_fn(newState2)
        expProbs2 = tf.reduce_sum(tf.math.exp(probs2),1)

        logProb1 = tf.math.log(expProbs1)
        logProb2 = tf.math.log(expProbs2)
        
        return(newState, logProb1, logProb2)

   
    def accept_reject(self, currentState, newState, prob1, prob2):
        """
        

        Parameters
        ----------
        currentState : the current state for all the parallel chains
        newState : the next proposed state for all the parallel chains
        prob1 : log prob of state 1
        prob2 : log prob of state 2

        Returns
        -------
        updatedState : the updated state for all the parallel chains
        updatedProb : the updated probabilities of the states
        acceptProb : acceptance probability for each chain

        """
        randomAccept = tf.random.uniform(prob1.shape, dtype=self.dtype)
        acceptCriteria = randomAccept<tf.math.exp(prob1-prob2)
        
        acceptProb=tf.math.exp(prob1-prob2)
        acceptProb= tf.where(tf.math.is_nan(acceptProb),0*acceptProb,acceptProb)
        acceptProb = tf.where(0*acceptProb+1<acceptProb,0*acceptProb+1, acceptProb)
        
        updatedState = tf.where(acceptCriteria, newState, currentState)
        
        return(updatedState, acceptProb)
    
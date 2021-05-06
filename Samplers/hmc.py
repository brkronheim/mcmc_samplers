import tensorflow as tf

from Samplers.sampler import Sampler

class HMCLeapfrog(Sampler):
    """
    An implementation of the Hamiltonian Monte Carlo sampler. This sampler
    works by treating the probability distribution as a potential energy well.
    It moves around the well according to Hamilton's equations, allowing
    it to mix well spatially. This implementation uses the leapfrog
    differential equation solver for simulating the equations.
    """
    
    def __init__(self, dimensions, log_likelihood_fn, epsilon,  l, 
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
        epsilon : a parameter determining the step size for the differential
            equation solver
        l : number of leapfrog steps to perform for each sample
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
        self.dtype=dtype
        self.dimensions=dimensions
        self.log_likelihood_fn = log_likelihood_fn
        self.sampledVals = []
        self.acceptance=tf.cast(0, self.dtype)  
        self.epsilon = tf.cast(tf.ones((1,parallelSamples))*epsilon, self.dtype)
        self.logEpsilonBar=tf.cast(tf.zeros((1,parallelSamples)), self.dtype)
        self.mu = tf.cast(tf.math.log(mu*self.epsilon), self.dtype)
        self.gamma=tf.cast(gamma, self.dtype)
        self.t0 = tf.cast(t0, self.dtype)
        self.kappa=tf.cast(kappa, self.dtype)
        self.h = tf.cast(0*self.epsilon, self.dtype)
        self.target=tf.cast(target, self.dtype)
        self.L = l


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
        
        #Use float32 for ease of use with GPUs
        currentState = tf.cast(initial_state, self.dtype) 
        
        
        #run sampler for the number of burn_in steps
        i = tf.constant(0)
        samples = self.burn_in
        condition = lambda i, currentState, h, logEpsilonBar, epsilon: tf.less(i, samples)
        
        #Body of while loop for burn_in, no states or acceptance info kept
        def body(i, currentState, h, logEpsilonBar, epsilon):
            m=tf.cast(i+1, self.dtype) #Step number
            
            #Progress one step
            currentState, accept, epsilon = self.one_step(currentState, epsilon)
           
            #Update h val
            h = (1-1/(m+self.t0))*h+(1/(m+self.t0))*(self.target-accept)
            
            #Update log Epsilon
            logEpsilon = self.mu-h*(m**0.5)/self.gamma
            
            #Update log Epsilon bar
            logEpsilonBar = m**(-self.kappa)*logEpsilon+(1-m**(-self.kappa))*logEpsilonBar
            
            #Set epsilon
            epsilon = tf.math.exp(logEpsilon)
            
            return([tf.add(i, 1), currentState, h, logEpsilonBar, epsilon])
        
        
        h, logEpsilonBar, epsilon = self.h, self.logEpsilonBar, self.epsilon
        
        #tf.while_loop to speed up sampling
        i, currentState,h, logEpsilonBar, epsilon = tf.while_loop(condition, body, 
                                                     [i, currentState,
                                                      h, logEpsilonBar, epsilon])
        
        epsilon=tf.exp(logEpsilonBar)
        tf.print("Final step size", epsilon)
        
        #tensorArray of set size to store samples
        sampledVals = tf.TensorArray(self.dtype, size=self.samples,
                                     dynamic_size=False)
        
        
        
        #run sampler for the number of sampling steps
        samples += self.samples
        def condition(i, currentState, sampledVals, acceptance, h, logEpsilonBar, epsilon):
            return(tf.less(i, samples))
        
        #Body of while loop for sampling, states and acceptance info kept
        def body(i, currentState, sampledVals, acceptance, h, logEpsilonBar, epsilon):
            
            currentState, accept, epsilon = self.one_step(currentState, epsilon)
            
            acceptance+=tf.reduce_sum(accept)
            sampledVals= sampledVals.write(i-self.burn_in, currentState)
            
            
            return([tf.add(i, 1), currentState, sampledVals,
                    acceptance, h, logEpsilonBar, epsilon])

        #tf.while_loop to speed up sampling
        i, currentState, sampledVals,acceptance, h, logEpsilonBar, epsilon = \
            tf.while_loop(condition, body, [i, currentState, sampledVals, acceptance, h, logEpsilonBar, epsilon])
        
        #trun sampledVals into a normal Tensor
        sampledVals = sampledVals.stack()                                                                                    
        
        return(sampledVals, acceptance)

     
    def energy(self, position, momentum):
        """
        Calcualte the energy of a postion and momentum
        """
        
        potential = -self.log_likelihood_fn(position)
        kinetic = tf.cast(tf.reduce_sum(tf.square(momentum)/2, axis=0), self.dtype)
        total = potential + kinetic
        return(total)
    
    def getUGradient(self, position):
        """
        Calculate the energy gradient
        """
        energy = None
        with tf.GradientTape() as g:
          g.watch(position)
          energy = -self.log_likelihood_fn(position)
        grad = g.gradient(energy, position)
        return(grad, energy)
   
    
    def one_step(self, currentState, epsilon):
        """
        A function which performs one step of the HMC sampler

        Parameters
        ----------
        currentState : the current state for all the parallel chains
        epsilon : current step size

        Returns
        -------
        updatedState : the updated state for all the parallel chains
        accepted : acceptance probability for each chain
        epsilon : current step size
        
        """
        newState, initialE, newE = self.propose_states(currentState, epsilon)
        updatedState, updatedE, accepted = self.accept_reject(currentState, 
                                                              newState, 
                                                              initialE, newE)
        
        return(updatedState, accepted, epsilon)

        
        
    def propose_states(self, currentState, epsilon):
        """
        

        Parameters
        ----------
        currentState : the current state of the sampler for each chain
        epsilon : the current step size value

        Returns
        -------
        newState : the proposed new state of the sampler for each chain
        initialE : the energy at the starting point
        currentE : the current energy

        """
        i = tf.constant(0)
        currentMomentum = tf.random.normal(currentState.shape, dtype=self.dtype)
        initialE = self.energy(currentState, currentMomentum)

        
        def one_leapfrog_step(i, currentState, currentMomentum, currentEnergy, epsilon):
            """
            Run one leapfrog step
            """
            grad,_ = self.getUGradient(currentState)
            halfMomentum = currentMomentum - (epsilon/2)*grad
            newState = currentState + epsilon*halfMomentum
            
            grad, _ = self.getUGradient(newState)
            
            
            newMomentum = halfMomentum - (epsilon/2)*grad
            energy = self.energy(newState, newMomentum)
            
            return(tf.add(i,1),newState, newMomentum, energy, epsilon)
        
        def condition(i, currentState, currentMomentum, currentProb, epsilon):
            return(tf.less(i, self.L))
        
        i, newState, currentMomentum, currentE, epsilon =  \
            tf.while_loop(condition, one_leapfrog_step, [i, currentState, 
                                                         currentMomentum, 
                                                         initialE, epsilon])
        
        
        return(newState, initialE, currentE)

   
    def accept_reject(self, currentState, newState, initialE, newE):
        """
        The accept_reject function decides whether a proposed_state should be
        accepted or not.

        Parameters
        ----------
        currentState : the current state for all the parallel chains
        newState : the next proposed state for all the parallel chains
        initialE : the initial energy
        newE : the energy at the new state

        Returns
        -------
        updatedState : the updated state for all the parallel chains
        updatedProb : the updated probabilities of the states
        acceptProb : acceptance probability for each chain

        """
        
        acceptProb=tf.math.exp(initialE-newE)
        acceptProb= tf.where(tf.math.is_nan(acceptProb),0*acceptProb,acceptProb)
        acceptProb = tf.where(0*acceptProb+1<acceptProb,0*acceptProb+1, acceptProb)
        
        randomAccept = tf.random.uniform(acceptProb.shape, dtype=self.dtype)
        acceptCriteria = randomAccept<acceptProb
        
        updatedState = tf.where(acceptCriteria, newState, currentState)
        updatedE = tf.where(acceptCriteria, newE, initialE)
        
        return(updatedState, updatedE, acceptProb)
    
class HMCPERFL(Sampler):
    """
    An implementation of the Hamiltonian Monte Carlo sampler. This sampler
    works by treating the probability distribution as a potential energy well.
    It moves around the well according to Hamilton's equations, allowing
    it to mix well spatially. This implementation uses the PERFL
    differential equation solver for simulating the equations.
    """
    
    def __init__(self, dimensions, log_likelihood_fn, epsilon,  l, 
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
        epsilon : a parameter determining the step size for the differential
            equation solver
        l : number of leapfrog steps to perform for each sample
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
        self.dtype=dtype
        self.dimensions=dimensions
        self.log_likelihood_fn = log_likelihood_fn
        self.sampledVals = []
        self.acceptance=tf.cast(0, self.dtype)  
        self.epsilon = tf.cast(tf.ones((1,parallelSamples))*epsilon, self.dtype)
        self.logEpsilonBar=tf.cast(tf.zeros((1,parallelSamples)), self.dtype)
        self.mu = tf.cast(tf.math.log(mu*self.epsilon), self.dtype)
        self.gamma=tf.cast(gamma, self.dtype)
        self.t0 = tf.cast(t0, self.dtype)
        self.kappa=tf.cast(kappa, self.dtype)
        self.h = tf.cast(0*self.epsilon, self.dtype)
        self.target=tf.cast(target, self.dtype)
        self.L = l


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
        
        #Use float32 for ease of use with GPUs
        currentState = tf.cast(initial_state, self.dtype) 
        
        
        #run sampler for the number of burn_in steps
        i = tf.constant(0)
        samples = self.burn_in
        condition = lambda i, currentState, h, logEpsilonBar, epsilon: tf.less(i, samples)
        
        #Body of while loop for burn_in, no states or acceptance info kept
        def body(i, currentState, h, logEpsilonBar, epsilon):
            m=tf.cast(i+1, self.dtype) #Step number
            
            #Progress one step
            currentState, accept, epsilon = self.one_step(currentState, epsilon)
           
            #Update h val
            h = (1-1/(m+self.t0))*h+(1/(m+self.t0))*(self.target-accept)
            
            #Update log Epsilon
            logEpsilon = self.mu-h*(m**0.5)/self.gamma
            
            #Update log Epsilon bar
            logEpsilonBar = m**(-self.kappa)*logEpsilon+(1-m**(-self.kappa))*logEpsilonBar
            
            #Set epsilon
            epsilon = tf.math.exp(logEpsilon)
            
            return([tf.add(i, 1), currentState, h, logEpsilonBar, epsilon])
        
        
        h, logEpsilonBar, epsilon = self.h, self.logEpsilonBar, self.epsilon
        
        #tf.while_loop to speed up sampling
        i, currentState,h, logEpsilonBar, epsilon = tf.while_loop(condition, body, 
                                                     [i, currentState,
                                                      h, logEpsilonBar, epsilon])
        
        epsilon=tf.exp(logEpsilonBar)
        tf.print("Final step size", epsilon)
        
        #tensorArray of set size to store samples
        sampledVals = tf.TensorArray(self.dtype, size=self.samples,
                                     dynamic_size=False)
        
        
        
        #run sampler for the number of sampling steps
        samples += self.samples
        def condition(i, currentState, sampledVals, acceptance, h, logEpsilonBar, epsilon):
            return(tf.less(i, samples))
        
        #Body of while loop for sampling, states and acceptance info kept
        def body(i, currentState, sampledVals, acceptance, h, logEpsilonBar, epsilon):
            
            currentState, accept, epsilon = self.one_step(currentState, epsilon)
            
            acceptance+=tf.reduce_sum(accept)
            sampledVals= sampledVals.write(i-self.burn_in, currentState)
            
            
            return([tf.add(i, 1), currentState, sampledVals,
                    acceptance, h, logEpsilonBar, epsilon])

        #tf.while_loop to speed up sampling
        i, currentState, sampledVals,acceptance, h, logEpsilonBar, epsilon = \
            tf.while_loop(condition, body, [i, currentState, sampledVals, acceptance, h, logEpsilonBar, epsilon])
        
        #trun sampledVals into a normal Tensor
        sampledVals = sampledVals.stack()                                                                                    
        
        return(sampledVals, acceptance)

     
    def energy(self, position, momentum):
        """
        Calcualte the energy of a postion and momentum
        """
        
        potential = -self.log_likelihood_fn(position)
        kinetic = tf.cast(tf.reduce_sum(tf.square(momentum)/2, axis=0), self.dtype)
        total = potential + kinetic
        return(total)
    
    def getUGradient(self, position):
        """
        Calculate the energy gradient
        """
        energy = None
        with tf.GradientTape() as g:
          g.watch(position)
          energy = -self.log_likelihood_fn(position)
        grad = g.gradient(energy, position)
        return(grad, energy)
   
    
    def one_step(self, currentState, epsilon):
        """
        A function which performs one step of the HMC sampler

        Parameters
        ----------
        currentState : the current state for all the parallel chains
        epsilon : current step size

        Returns
        -------
        updatedState : the updated state for all the parallel chains
        accepted : acceptance probability for each chain
        epsilon : current step size
        
        """
        newState, initialE, newE = self.propose_states(currentState, epsilon)
        updatedState, updatedE, accepted = self.accept_reject(currentState, 
                                                              newState, 
                                                              initialE, newE)
        
        return(updatedState, accepted, epsilon)

        
        
    def propose_states(self, currentState, epsilon):
        """
        

        Parameters
        ----------
        currentState : the current state of the sampler for each chain
        epsilon : the current step size value

        Returns
        -------
        newState : the proposed new state of the sampler for each chain
        initialE : the energy at the starting point
        currentE : the current energy

        """
        i = tf.constant(0)
        currentMomentum = tf.random.normal(currentState.shape, dtype=self.dtype)
        initialE = self.energy(currentState, currentMomentum)
                
        def one_perfl_step(i, x, v, currentEnergy, epsilon):
            """
            Run one perfl step
            """
            xi = 0.1786178958448091
            lmbda = -0.2123418310626054
            chi = -0.06626458266981849
        
            c1 = xi
            c2 = chi
            c3 = 1-2*(chi+xi)
            c4 = chi
            c5 = xi
            
            d1 = (1-2*lmbda)/2
            d2 = lmbda
            d3 = lmbda
            d4 = (1-2*lmbda)/2
            
            x = x + c1*v*epsilon
        
            a,_ = self.getUGradient(x)
            v = v - d1*a*epsilon
    
            x = x + c2*v*epsilon
            a,_ = self.getUGradient(x)
            v = v - d2*a*epsilon
    
            x = x + c3*v*epsilon        
            a,_ = self.getUGradient(x)
            v = v - d3*a*epsilon
            
            x = x + c4*v*epsilon
            a,_ = self.getUGradient(x)
            v = v - d4*a*epsilon
            
            x = x + c5*v*epsilon
            
            
            
            energy = self.energy(x, v)
            
            return(tf.add(i,1), x, v, energy, epsilon)
        
        
        def condition(i, currentState, currentMomentum, currentProb, epsilon):
            return(tf.less(i, self.L))
        
        i, newState, currentMomentum, currentE, epsilon =  \
            tf.while_loop(condition, one_perfl_step, [i, currentState, 
                                                         currentMomentum, 
                                                         initialE, epsilon])
        
        
        return(newState, initialE, currentE)

   
    def accept_reject(self, currentState, newState, initialE, newE):
        """
        The accept_reject function decides whether a proposed_state should be
        accepted or not.

        Parameters
        ----------
        currentState : the current state for all the parallel chains
        newState : the next proposed state for all the parallel chains
        initialE : the initial energy
        newE : the energy at the new state

        Returns
        -------
        updatedState : the updated state for all the parallel chains
        updatedProb : the updated probabilities of the states
        acceptProb : acceptance probability for each chain

        """
        
        acceptProb=tf.math.exp(initialE-newE)
        acceptProb= tf.where(tf.math.is_nan(acceptProb),0*acceptProb,acceptProb)
        acceptProb = tf.where(0*acceptProb+1<acceptProb,0*acceptProb+1, acceptProb)
        
        randomAccept = tf.random.uniform(acceptProb.shape, dtype=self.dtype)
        acceptCriteria = randomAccept<acceptProb
        
        updatedState = tf.where(acceptCriteria, newState, currentState)
        updatedE = tf.where(acceptCriteria, newE, initialE)
        
        return(updatedState, updatedE, acceptProb)
    
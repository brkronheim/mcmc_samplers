import tensorflow as tf

from math import pi

from Samplers.sampler import Sampler

class RMHMC(Sampler):
    """
    An implementation of the Reimann Manifold HMC sampler. This sampler uses
    a metric tensor to move the sampling space to a Riemanninan manifold. This
    allows correlations between dimensions to be removed and axes to be evened
    out. This implementation uses either a modified Hessian as the metric or
    the diagonal of a modified Hessian
    """
    
    def __init__(self, dimensions, log_likelihood_fn, epsilon,  l, 
                 parallelSamples = 1, dtype=tf.float32, mu = 100, gamma = 0.4,
                 t0 = 10, kappa = 0.75, target=0.6, delta=1e-8, alpha=1000,
                 maxSteps=10):
        """
        The constructor for the RMHMC sampler

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
        delta : convergence criteria for implicit steps
        alpha : parameter for adjusting eigenvalues of Hessian matrix
        maxSteps : maximum number of steps in implicit lepafrog steps

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
        self.delta = delta
        self.alpha = alpha
        self.maxSteps = maxSteps

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
        i, currentState, h, logEpsilonBar, epsilon = tf.while_loop(condition, body, 
                                                     [i, currentState, h, logEpsilonBar, epsilon])
        
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

    def tau(self, position, momentum):
        """
        Kinetic term
        """
        _, m, _=self.getM(position)
        
        
        kinetic = tf.matmul(m, momentum)
        kinetic = tf.matmul(tf.transpose(momentum), kinetic)
        kinetic = tf.reduce_sum(kinetic, axis=0)/2
        return(kinetic)
        
    def phi(self, position, momentum):
        """
        Potential term
        """
        potential = -self.log_likelihood_fn(position)
        _, _, logDet = self.getM(position)
        logDet = self.dimensions*0.5*(logDet+tf.cast(self.dimensions*tf.math.log(2*pi), self.dtype))#tf.math.log(det)
        potential+=logDet        
        return(potential)

    def energy(self, position, momentum):
        """
        Total Energy
        """
        total = self.tau(position, momentum) + self.phi(position, momentum)
        return(total)
    
    def getM(self, position):
        """
        Get the metric tensor
        """
        energy = None
        firstDerivatives=None
        with tf.GradientTape() as g1:
            g1.watch(position)
            with tf.GradientTape() as g2:
                g2.watch(position)
                energy = -self.log_likelihood_fn(position)
            
            firstDerivatives=g2.gradient(energy, position)
        
        
        hessian = tf.squeeze(g1.jacobian(firstDerivatives, position))
        
        hessian = tf.clip_by_value(hessian, -1e10, 1e10)
        
        hessian = tf.where(tf.math.is_nan(hessian),0*hessian,hessian)
        hessian = hessian + tf.linalg.tensor_diag(tf.squeeze(tf.linalg.tensor_diag_part(hessian*0+0.000001)))
        
        l, Q = tf.linalg.eigh(hessian)
        l = l/tf.math.tanh(l*self.alpha)
        
        hessian = tf.matmul(tf.matmul(Q, tf.linalg.tensor_diag(l)), tf.transpose(Q))
        mInverse  = tf.matmul(tf.matmul(Q, tf.linalg.tensor_diag(1/l)), tf.transpose(Q))
        logDet = tf.reduce_sum(tf.math.log(l))
        
        
        #Uncomment below to just use diagonal
        """
        hessian = g1.gradient(firstDerivatives, position)
        
        hessian = tf.squeeze(hessian)
        hessian = hessian/tf.math.tanh(hessian*self.alpha)
        logDet = tf.reduce_sum(tf.math.log(hessian))
        mInverse = tf.linalg.tensor_diag((1/hessian))
        hessian = tf.linalg.tensor_diag(hessian)
        """
        return(hessian, mInverse, logDet)
    
    #The following four functions are just various derivatives
    def dPhidQ(self, position, momentum):
        energy = None
        with tf.GradientTape() as g:
          g.watch(position)
          energy = self.phi(position, momentum)
        
        grad = g.gradient(energy, position, unconnected_gradients=tf.UnconnectedGradients.ZERO)
        return(grad)
    
    def dPhidP(self, position, momentum):
        energy = None
        with tf.GradientTape() as g:
          g.watch(momentum)
          energy = self.phi(position, momentum)
        
        grad = g.gradient(energy, momentum, unconnected_gradients=tf.UnconnectedGradients.ZERO)
        
        return(grad)
    
    def dTaudQ(self, position, momentum):
        energy = None
        with tf.GradientTape() as g:
          g.watch(position)
          energy = self.tau(position, momentum)
        
        grad = g.gradient(energy, position, unconnected_gradients=tf.UnconnectedGradients.ZERO)
        
        return(grad)
    
    def dTaudP(self, position, momentum):
        energy = None
        with tf.GradientTape() as g:
          g.watch(momentum)
          energy = self.tau(position, momentum)
        
        grad = g.gradient(energy, momentum, unconnected_gradients=tf.UnconnectedGradients.ZERO)
        
        return(grad)

     
    def one_step(self, currentState, epsilon):
        """
        A function which performs one step of the RMHMC sampler

        Parameters
        ----------
        currentState : the current state for all the parallel chains
        epsilon : the current step size

        Returns
        -------
        updatedState : the updated state for all the parallel chains
        updatedProb : acceptance probability for each chain
        accepted : acceptance probability for each chain
        
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
        epsilon : current step size

        Returns
        -------
        newState : the proposed new state of the sampler for each chain
        initialE : starting energy
        currentE : ending energy

        """
        #Treat metric tensor as covariance matrix to sample momenta
        i = tf.constant(0)
        h,_,_ = self.getM(currentState)
        cholskey = tf.linalg.cholesky(h)
        cholskey = tf.linalg.band_part(cholskey, -1, 0)
        
        currentMomentum = tf.random.normal(currentState.shape, dtype = self.dtype)
        currentMomentum = tf.matmul(cholskey, currentMomentum)
        
        initialE = self.energy(currentState, currentMomentum)
        
        
        
        def one_leapfrog_step(i, currentState, currentMomentum,
                              currentEnergy, epsilon):
            
            """
            One implicit leapfrog step
            """
            
            #update momentum
            currentMomentum = currentMomentum - (epsilon/2)*self.dPhidQ(currentState, currentMomentum)
            
            
            def momentumStep(j, state, oldMomentum, momentum, delta, epsilon):
                #Implicit momentum update
                currentMomentum = oldMomentum - (epsilon/2)*self.dTaudQ(state, momentum)
                delta = tf.reduce_max(tf.math.abs(currentMomentum-momentum), axis=0)
                return(j+1, state, oldMomentum, currentMomentum, delta, epsilon)
                
                
            def momentumCondition(j, state, oldMomentum, momentum, delta, epsilon):
                return(j<self.maxSteps and delta>self.delta)
             
            j=tf.constant(0)
            
            delta=currentEnergy*0+self.delta*2
            
            
            _,_,_, currentMomentum,_,_ = tf.while_loop(momentumCondition,
                                                       momentumStep,
                                                       [j, currentState,
                                                        currentMomentum,
                                                        currentMomentum,
                                                        delta, epsilon])
            
            
            def stateStep(j, oldState, state, momentum, delta, epsilon):
                #Implicit location update
                currentState = oldState + (epsilon/2)*(self.dTaudP(oldState, momentum)+self.dTaudP(state, momentum))
                delta = tf.reduce_max(tf.math.abs(currentState-state), axis=0)
                
                return(j+1, oldState, currentState, momentum, delta, epsilon)
                
                
            def stateCondition(j, oldState, state, momentum, delta, epsilon):
                
                return(j<self.maxSteps and delta>self.delta)
             
            j=j*0
            
            delta=currentEnergy*0+self.delta*2
            
            
            _, _, newState, _,_,_ = tf.while_loop(stateCondition, stateStep,
                                                  [j, currentState,
                                                   currentState,
                                                   currentMomentum,
                                                   delta, epsilon])
            
            
            
            currentMomentum = currentMomentum - (epsilon/2)*self.dTaudQ(newState, currentMomentum)
            
            
            newMomentum = currentMomentum - (epsilon/2)*self.dPhidQ(newState, currentMomentum)
            
            
            energy = self.energy(newState, newMomentum)
            return(tf.add(i,1),newState, newMomentum, energy, epsilon)
        
        def condition(i, currentState, currentMomentum, currentProb, epsilon):
            return(tf.less(i, self.L))
        
        i, newState, currentMomentum, currentE, epsilon = \
            tf.while_loop(condition, one_leapfrog_step, [i, currentState,
                                                         currentMomentum,
                                                         initialE, epsilon])
        
        
        return(newState, initialE, currentE)

   
    def accept_reject(self, currentState, newState, initialE, newE):
        """
        

        Parameters
        ----------
        currentState : the current state for all the parallel chains
        newState : the next proposed state for all the parallel chains
        intiailE : staring energy
        newE : ending energy

        Returns
        -------
        updatedState : the updated state for all the parallel chains
        updatedE : the updated energies of the states
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
    
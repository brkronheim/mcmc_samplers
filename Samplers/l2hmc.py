import tensorflow as tf

from math import pi

from Samplers.sampler import Sampler

class L2HMC(Sampler):
    """
    An implementation of the Learning 2 HMC sampler. This sampler is a variant
    of HMC where an augmented Leapfrog Integrator is used with rescaling terms
    learned by a neural network. The goal is to be able to learn to sample
    from much more complex distributions that normal HMC sampler have
    trouble with. 
    """
    
    def __init__(self, dimensions, log_likelihood_fn, epsilon,  l, target=0.6,
                 width=100, parallelSamples=1, dtype=tf.float32,
                 temperature=1, mainLoss=1, burninLoss=0, lag2Loss=1):
        """
        The constructor for the Metropolis-Hastings sampler

        Parameters
        ----------
        dimensions : the number of dimensions of the sampled distribution
        log_likelihood_fn : a function accepting a state from the distribution
            and returning the natural logarithm of the probability of that
            state. This probability need not be normalized.
        epsilon : starting step size for differential equation solver
        l : number of leapfrog steps
        target : target acceptance rate
        width : perceptrons in each neural network layer
        parallelSamples : number of parallel sampling chains
        dtype : data type of data, tf.float32 is recomended
        temperature : starting temperature for annealing, will linearly
            decrease to 1 over burnin
        mainLoss : coeficient on main loss term
        burninLoss : coeficient on burnin loss term
        lag2loss : coeficient on lag2 loss term

        Returns
        -------
        None.

        """
        self.dtype=dtype
        self.dimensions=dimensions
        self.log_likelihood_fn = log_likelihood_fn
        self.sampledVals = []
        self.acceptance=tf.cast(0, self.dtype)  
        self.epsilon = tf.Variable(tf.cast(epsilon**0.5, self.dtype), trainable=True)
        self.target=tf.cast(target, self.dtype)
        self.L = l
        
        self.parallelSamples = parallelSamples
        
        
        self.mainLoss = mainLoss
        self.burninLoss = burninLoss
        self.lag2loss = lag2Loss
        
        #Neural network parameters
        
        vNetWeights1 = tf.random.normal((tf.cast(self.dimensions*2+2,
                                                 tf.int32),width),
                                        dtype=self.dtype)*tf.cast(tf.sqrt(2/tf.cast(self.dimensions*2+1, tf.int32)), self.dtype)
        self.vNetWeights1 = tf.Variable(vNetWeights1, trainable=True)
        vNetBiases1 = tf.ones((width), dtype=self.dtype)*0.01
        self.vNetBiases1 = tf.Variable(vNetBiases1, trainable=True)
        vNetWeights2 = tf.random.normal((width,width),
                                        dtype=self.dtype)*tf.cast(tf.sqrt(2/width),
                                                                  self.dtype)
        self.vNetWeights2 = tf.Variable(vNetWeights2, trainable=True)
        vNetBiases2 = tf.ones((width), dtype=self.dtype)*0.01
        self.vNetBiases2 = tf.Variable(vNetBiases2, trainable=True)
        vNetWeights3 = tf.random.normal((width,tf.cast(self.dimensions*3,
                                                       tf.int32)),
                                        dtype=self.dtype)*tf.cast(tf.sqrt(2/width),
                                                                  self.dtype)
        self.vNetWeights3 = tf.Variable(vNetWeights3, trainable=True)
        vNetBiases3 = tf.ones((tf.cast(self.dimensions*3, tf.int32)),
                              dtype=self.dtype)*0.01
        self.vNetBiases3 = tf.Variable(vNetBiases3, trainable=True)

        
        xNetWeights1 = tf.random.normal((tf.cast(self.dimensions*2+2,
                                                 tf.int32),width),
                                        dtype=self.dtype)*tf.cast(tf.sqrt(2/tf.cast(self.dimensions*1.5+1,
                                                                                    tf.int32)),
                                                                                    self.dtype)
        self.xNetWeights1 = tf.Variable(xNetWeights1, trainable=True)
        xNetBiases1 = tf.ones((width), dtype=self.dtype)*0.01
        self.xNetBiases1 = tf.Variable(xNetBiases1, trainable=True)
        xNetWeights2 = tf.random.normal((width,width),
                                        dtype=self.dtype)*tf.cast(tf.sqrt(2/width),
                                                                                 self.dtype)
        self.xNetWeights2 = tf.Variable(xNetWeights2, trainable=True)
        xNetBiases2 = tf.ones((width), dtype=self.dtype)*0.01
        self.xNetBiases2 = tf.Variable(xNetBiases2, trainable=True)
        xNetWeights3 = tf.random.normal((width,tf.cast(self.dimensions*3,
                                                       tf.int32)),
                                        dtype=self.dtype)*tf.cast(tf.sqrt(2/width),
                                                                  self.dtype)
        self.xNetWeights3 = tf.Variable(xNetWeights3, trainable=True)
        xNetBiases3 = tf.ones((tf.cast(self.dimensions*3, tf.int32)),
                              dtype=self.dtype)*0.01
        self.xNetBiases3 = tf.Variable(xNetBiases3, trainable=True)
 
        self.lambdaX1 = tf.Variable(tf.cast(0.001, self.dtype), trainable=True)
        self.lambdaX2 = tf.Variable(tf.cast(0.001, self.dtype), trainable=True)
        self.lambdaX3 = tf.Variable(tf.cast(0.001, self.dtype), trainable=True)
        self.lambdaW1 = tf.Variable(tf.cast(0.001, self.dtype), trainable=True)
        self.lambdaW2 = tf.Variable(tf.cast(0.001, self.dtype), trainable=True)
        self.lambdaW3 = tf.Variable(tf.cast(0.001, self.dtype), trainable=True)
 
    
        
        self.temperature = tf.cast(temperature, self.dtype)
 
        #Decaying learning rate
        initial_learning_rate = 0.001
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps=500,
            decay_rate=0.1,
            staircase=True)
        self.opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule,
                                            amsgrad=True)
        
        
    @tf.function   
    def run_sampler_inner(self, initial_state):
        """
        The inner sampling step for the l2HMC sampler

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
        condition = lambda i, currentState, previousState, temperature: tf.less(i, samples)
        
        #Body of while loop for burn_in, no states or acceptance info kept
        def body(i, currentState, previousState, temperature):
            loss=None
            loss1=None
            loss2=None
            loss3=None
            
            with tf.GradientTape(persistent=True) as tape:
                tape.watch([self.vNetWeights1,self.vNetWeights2, self.vNetWeights3,
                                         self.xNetWeights1,self.xNetWeights2, self.xNetWeights3,
                                         self.vNetBiases1,self.vNetBiases2, self.vNetBiases3,
                                         self.xNetBiases1,self.xNetBiases2, self.xNetBiases3, self.epsilon,
                                         self.lambdaX1,self.lambdaX2,self.lambdaX3,
                                         self.lambdaW1,self.lambdaW2,self.lambdaW3])
                temp = currentState
                currentState, accept, loss1, j, proposedState = self.one_step(currentState,i, temperature)
                
                randomState = tf.random.normal(currentState.shape, dtype=self.dtype)*1
                _, _, loss2, _, _ = self.one_step(randomState, i,tf.cast(1.0, self.dtype))
                loss3 = -accept*tf.math.pow((proposedState-previousState),2)
                loss3 = loss3
                loss = loss1*self.mainLoss+loss2*self.burninLoss+loss3*self.lag2loss
                previousState = temp
                
            #Check to make sure there are no nans, if so, skip
            grad = tf.math.reduce_sum(tape.gradient(loss, [self.xNetBiases3]))
            if(not tf.math.is_nan(grad)):
                self.opt.minimize(loss, [self.vNetWeights1,self.vNetWeights2, self.vNetWeights3,
                                         self.xNetWeights1,self.xNetWeights2, self.xNetWeights3,
                                         self.vNetBiases1,self.vNetBiases2, self.vNetBiases3,
                                         self.xNetBiases1,self.xNetBiases2, self.xNetBiases3, self.epsilon,
                                         self.lambdaX1,self.lambdaX2,self.lambdaX3,self.lambdaW1,
                                         self.lambdaW2,self.lambdaW3],tape=tape)
            #Print loss info every 10 epochs
            if(False or i%10==0):
                tf.print("newVals", self.lambdaX1,self.lambdaX2,self.lambdaX3,self.lambdaW1,self.lambdaW2,self.lambdaW3)
                tf.print("iter:",i,"loss:",tf.math.reduce_mean(loss1), tf.math.reduce_mean(loss2), tf.math.reduce_mean(loss3))
                tf.print("epsilon", self.epsilon**2)
            #Decrease temperature
            temperature = temperature - (self.temperature-1)/samples
            return([tf.add(i, 1), currentState, previousState, temperature])
        
        temperature = self.temperature
        #tf.while_loop to speed up sampling
        i, currentState,_,_= tf.while_loop(condition, body, 
                                                     [i, currentState, currentState, temperature])
        
        #epsilon=self.safe_exp(logEpsilonBar)
        tf.print("Final step size", self.epsilon**2)
        
        #tensorArray of set size to store samples
        sampledVals = tf.TensorArray(self.dtype, size=self.samples,
                                     dynamic_size=False)
        
        
        temperature = tf.cast(1, self.dtype)
        #run sampler for the number of sampling steps
        samples += self.samples
        def condition(i, currentState, sampledVals, acceptance, temperature):
            return(tf.less(i, samples))
        
        #Body of while loop for sampling, states and acceptance info kept
        def body(i, currentState, sampledVals, acceptance, temperature):
            
            currentState, accept, loss, j ,_= self.one_step(currentState,i, temperature)
            
            acceptance+=tf.reduce_sum(accept)
            sampledVals= sampledVals.write(i-self.burn_in, currentState)
            
            
            return([tf.add(i, 1), currentState, sampledVals,
                    acceptance, temperature])

        #tf.while_loop to speed up sampling
        i, currentState, sampledVals,acceptance,_ = \
            tf.while_loop(condition, body, [i, currentState, sampledVals, acceptance, temperature])
        
        #trun sampledVals into a normal Tensor
        sampledVals = sampledVals.stack()                                                                                    
        
        return(sampledVals, acceptance)

    def safe_exp(self, val):  
        """
        Clip exp between 0 and 1e20
        """
        return(tf.clip_by_value(tf.exp(val), tf.cast(0, self.dtype), tf.cast(1e20, self.dtype)))
        
    def energy(self, position, momentum):
        """
        Calculate total energy of position and momentum
        """
        potential = -self.log_likelihood_fn(position)
        kinetic = tf.reduce_sum(tf.square(momentum)/2, axis=0)
        total = potential + kinetic
        return(total)
    
    def getUGradient(self, position):
        """
        Energy gradient
        """
        energy = None
        with tf.GradientTape() as g:
          g.watch(position)
          energy = -self.log_likelihood_fn(position)
        grad = g.gradient(energy, position)
        return(grad, energy)
    
    def vNet(self, state, grad, time):
        """
        Neural network for momentum updates
        """
        time= tf.cast(time, self.dtype)
        t1 = tf.ones((1, self.parallelSamples), dtype=self.dtype)*tf.math.cos(2*tf.cast(pi, self.dtype)*time/tf.cast(self.L, self.dtype))
        t2 = tf.ones((1, self.parallelSamples), dtype=self.dtype)*tf.math.sin(2*tf.cast(pi, self.dtype)*time/tf.cast(self.L, self.dtype))
        networkInput = tf.transpose(tf.concat([state, grad, t1,t2], axis=0))
        
        layer1 = tf.matmul(networkInput, self.vNetWeights1)+self.vNetBiases1
        layer1 = tf.where(layer1<0, layer1*0.2, layer1)
        layer2 = tf.matmul(layer1, self.vNetWeights2)+self.vNetBiases2
        layer2 = tf.where(layer2<0, layer2*0.2, layer2)
        layerOut = tf.matmul(layer2, self.vNetWeights3)+self.vNetBiases3
        dims = tf.cast(self.dimensions, tf.int32)
        s = tf.transpose(tf.math.tanh(layerOut[:,:dims]))*self.lambdaW1
        q = tf.transpose(tf.math.tanh(layerOut[:,dims:2*dims]))*self.lambdaW2
        
        t = tf.transpose(tf.math.tanh(layerOut[:,2*dims:]))*self.lambdaW3
        
        return(s,q,t)
    
    def xNet(self, state, momentum, time):
        """
        Neural network for position updates
        """
        time= tf.cast(time, self.dtype)
        t1 = tf.ones((1, self.parallelSamples), dtype=self.dtype)*tf.math.cos(2*tf.cast(pi, self.dtype)*time/tf.cast(self.L, self.dtype))
        t2 = tf.ones((1, self.parallelSamples), dtype=self.dtype)*tf.math.sin(2*tf.cast(pi, self.dtype)*time/tf.cast(self.L, self.dtype))
        
        networkInput = tf.transpose(tf.concat([state, momentum, t1,t2], axis=0))
        layer1 = tf.matmul(networkInput, self.xNetWeights1)+self.xNetBiases1
        
        layer1 = tf.where(layer1<0, layer1*0.2, layer1)
        layer2 = tf.matmul(layer1, self.xNetWeights2)+self.xNetBiases2
        layer2 = tf.where(layer2<0, layer2*0.2, layer2)
        
        layerOut = tf.matmul(layer2, self.xNetWeights3)+self.xNetBiases3
        dims = tf.cast(self.dimensions, tf.int32)
        s = tf.transpose(tf.math.tanh(layerOut[:,:dims]))*self.lambdaX1
        q = tf.transpose(tf.math.tanh(layerOut[:,dims:2*dims]))*self.lambdaX2
        t = tf.transpose(tf.math.tanh(layerOut[:,2*dims:]))*self.lambdaX3
        
        return(s,q,t)

   
    def one_step(self, currentState, i, temperature):
        """
        A function which performs one step of the Metropolis-Hastings sampler

        Parameters
        ----------
        currentState : the current state for all the parallel chains
        i : the current step
        temperature : the current temperature

        Returns
        -------
        updatedState : the updated state for all the parallel chains
        accepted : acceptance probability for each chain
        loss : The loss of step
        i : the current step
        newState : the updated state for all the parallel chains
        
        
        """
        newState, initialE, newE, newJac = self.propose_states(currentState)
        updatedState, updatedE, accepted, loss = self.accept_reject(currentState, newState, initialE, newE, newJac, i, temperature)
        
        return(updatedState, accepted, loss, i, newState)

        
        
    def propose_states(self, currentState):
        """
        

        Parameters
        ----------
        currentState : the current state of the sampler for each chain

        Returns
        -------
        newState : the proposed new state of the sampler for each chain
        initialE : the intiial energy
        currentE : the current energy
        currentJac : the Jacobian of the transformation

        """
        i = tf.constant(0)
        
        #Get momentum and direction
        currentMomentum = tf.random.normal(currentState.shape, dtype = self.dtype)
        currentDirection = tf.random.uniform([tf.cast(self.parallelSamples, tf.int32)],0,2)
        currentDirection = tf.cast(tf.expand_dims(tf.where(currentDirection<1, -1, 1),0), self.dtype)
        currentDirection = tf.repeat(currentDirection, self.dimensions,axis=0)
        
        #Starting energy
        initialE = self.energy(currentState, currentMomentum)
        
        def one_leapfrog_step(i, currentState, currentMomentum, currentDirection, currentEnergy, currentJac):
            """
            Run one step of the augmented leapfrog algorithm
            """
            grad, _ = self.getUGradient(currentState)
            
            #First momentum update
            s, q, t = self.vNet(currentState, grad, i) #Network prediction
            currentJac += tf.math.reduce_sum(currentDirection*self.epsilon**2*s,axis=0)/2
            halfMomentum = tf.where(currentDirection == tf.cast(1, self.dtype),
                currentMomentum*self.safe_exp(s*self.epsilon**2/2) - (self.epsilon**2/2)*(grad*self.safe_exp(self.epsilon**2*q)+t),
                (currentMomentum + (self.epsilon**2/2)*(grad*self.safe_exp(self.epsilon**2*q)+t))*self.safe_exp(-s*self.epsilon**2/2))
            
            #Masks for the position netoworks
            state1 = tf.concat([tf.ones(self.dimensions//2, dtype=self.dtype),
                                tf.zeros(self.dimensions-self.dimensions//2,dtype=self.dtype)], axis=0)
            state1 = tf.expand_dims(state1, 1)
            state1 = tf.repeat(state1, self.parallelSamples, axis=1)
            state1 = tf.random.shuffle(state1) #Shuffling is the same for each chain
            state2 = 1-state1
            s1, q1, t1 = self.xNet(currentState*state2, halfMomentum, i)
            s2, q2, t2 = self.xNet(currentState*state1, halfMomentum, i)
            
            currentJac += tf.math.reduce_sum(currentDirection*self.epsilon**2*(state1*s1+state2*s2),axis=0)
            
            #Position updates
            newState = tf.where(state1 == tf.cast(1, self.dtype),
                                tf.where(currentDirection == tf.cast(1, self.dtype),
                                         currentState*self.safe_exp(self.epsilon**2*s1)+self.epsilon**2*(halfMomentum*self.safe_exp(self.epsilon**2*q1)+t1),
                                         (currentState-self.epsilon**2*(self.safe_exp(self.epsilon**2*q2)*halfMomentum+t2))*self.safe_exp(-self.epsilon**2*s2)),
                                tf.where(currentDirection == tf.cast(1, self.dtype),
                                         currentState*self.safe_exp(self.epsilon**2*s2)+self.epsilon**2*(halfMomentum*self.safe_exp(self.epsilon**2*q2)+t2),
                                         (currentState-self.epsilon**2*(self.safe_exp(self.epsilon**2*q1)*halfMomentum+t1))*self.safe_exp(-self.epsilon**2*s1)))
            
            #Finfal momentum update
            grad, _ = self.getUGradient(newState)
            
            
            s, q, t = self.vNet(currentState, grad, i)
        
            currentJac += tf.math.reduce_sum(currentDirection*self.epsilon**2*s,axis=0)/2
            
            
            newMomentum = tf.where(currentDirection == tf.cast(1, self.dtype),
                halfMomentum*self.safe_exp(s*self.epsilon**2/2) - (self.epsilon**2/2)*(grad*self.safe_exp(self.epsilon**2*q)+t),
                (halfMomentum + (self.epsilon**2/2)*(grad*self.safe_exp(self.epsilon**2*q)+t))*self.safe_exp(-s*self.epsilon**2/2))
            
            energy = self.energy(newState, newMomentum)
            
            return(tf.add(i,1),newState, newMomentum, currentDirection, energy, currentJac)
        
        #Check for end of sampling run
        def condition(i, currentState, currentMomentum, currentDirection, currentProb, currentJac):
            return(tf.less(i, self.L))
        
        initialJac = tf.cast(initialE*0, self.dtype)
        i, newState, currentMomentum, currentDirection, currentE, currentJac \
            = tf.while_loop(condition, one_leapfrog_step,
                            [i, currentState, currentMomentum,
                             currentDirection, initialE,initialJac])
        
        
        return(newState, initialE, currentE, currentJac)

   
    def accept_reject(self, currentState, newState, initialE, newE, newJac, i, temperature):
        """
        The accept_reject function decides whether a proposed_state should be
        accepted or not.

        Parameters
        ----------
        currentState : the current state for all the parallel chains
        newState : the next proposed state for all the parallel chains
        initialE : staring energy
        newE : ending energy
        newJac : Jacobian of transformation
        i : epoch
        temperature : current temperature
        
        
        
        Returns
        -------
        updatedState : the updated state for all the parallel chains
        updatedE : the updated energies of the states
        acceptProb : acceptance probability for each chain
        loss : loss of predicted states

        """
        
        eSJDist = tf.math.reduce_sum(tf.math.square(newState-currentState), axis=0)
        
        acceptProb=tf.math.exp(initialE/temperature+newJac-newE/temperature)
        
        acceptProb= tf.where(tf.math.is_nan(acceptProb),tf.cast(0*acceptProb, self.dtype),acceptProb)
        
        acceptProb = tf.where(0*acceptProb+1<acceptProb,0*acceptProb+1, acceptProb)
        
        eSJDist = acceptProb*eSJDist
        
        loss = -eSJDist
        randomAccept = tf.random.uniform(acceptProb.shape, dtype=self.dtype)
        acceptCriteria = randomAccept<acceptProb
        
        updatedState = tf.where(acceptCriteria, newState, currentState)
        updatedE = tf.where(acceptCriteria, newE, initialE)
        
        return(updatedState, updatedE, acceptProb, loss)
    
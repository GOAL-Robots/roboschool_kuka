import numpy as np
from dmp import DMP
import matplotlib.pyplot as plt

def softmax(x, lmb):
    e = np.exp(-x/lmb)
    return e/sum(e)


class BBO :
    "P^2BB: Policy Improvement through Black Vox Optimization"
    def __init__(self, rollout_func, num_params=10, num_rollouts=20, 
                 sigma=0.001, lmb=0.1, epochs=100, 
                 sigma_decay_amp=0, sigma_decay_period=0.1):
        '''
        :param num_params: Integer. Number of parameters to optimize 
        :param num_rollouts: Integer. number of rollouts per iteration
        :param sigma: Float. Amount of exploration around the mean of parameters
        :param lmb: Float. Temperature of the evaluation softmax
        :param epochs: Integer. Number of iterations
        :param rollout_func: Callable object to produce a rollout
            signature: (thetas - num_rollouts X num_params ) ->
                (errs -  num_rollouts X num_timesteps,
                 rollouts -  num_rollouts X num_timesteps)
        :param sigma_decay_amp: Initial additive amplitude of exploration
        :param sigma_decay_period: Decaying period of additive 
            amplitude of exploration
        '''
        
        self.sigma = sigma
        self.lmb = lmb
        self.num_rollouts = num_rollouts
        self.num_params = num_params
        self.theta = np.zeros(num_params)
        self.Cov = np.eye(num_params, num_params)
        self.rollout_func = rollout_func
        self.err = 1.0
        self.epochs = epochs
        self.decay_amp = sigma_decay_amp
        self.decay_period = sigma_decay_period
        self.epoch = 0
               
    def sample(self):
        """ Get num_rollouts samples from the current parameters mean
        """
        
        Sigma = self.sigma + self.decay_amp*np.exp(
            -self.epoch/(self.epochs * self.decay_period))
        # matrix of deviations from the parameters mean
        self.eps = np.random.multivariate_normal(
            np.zeros(self.num_params), 
            self.Cov * Sigma, self.num_rollouts)
    
    def update(self, Sk):
        ''' Update parameters
        
            :param Sk: array(Float), rollout costs in an iteration 
        '''
        # Cost-related probabilities of sampled parameters
        probs = softmax(Sk, self.lmb).reshape(self.num_rollouts, 1)
        # update with the weighted average of sampled parameters
        self.theta += np.sum(self.eps * probs, 0)
    
    def eval(self, errs):
        """ evaluate rollouts
            :param errs: array(float), Matrix containing errors 
                 at each time-step (columns) of each rollout (rows)
        """
        errs = errs**2
        self.err = np.mean(errs) # store the mean square error
        
        # comute costs
        Sk = np.zeros(self.num_rollouts)
        for k in range(self.num_rollouts):
            # final cost
            Sk[k] = errs[k, -1]
            # cost-to-go integral
            Sk[k] += np.sum( 
                errs[k, j:].sum() 
                for j in range(self.num_params)) 
            # regularization
            thetak = self.theta + self.eps[k]
            Sk[k] += 0.5 * self.sigma * (thetak).dot(thetak) 
        
        return Sk
        
    def iteration(self, explore = True):
        """ Run an iteration
            :param explore: Bool, If the iteration is for training (True) 
                or test (False)
        """
        self.sample()
        costs, rollouts = self.rollout_func(self.theta + explore*self.eps)    
        Sk = self.eval(costs)
        self.update(Sk)
        self.epoch += 1
        return rollouts, Sk
    
#------------------------------------------------------------------------------ 

if __name__ == "__main__":
    
    K = 50
    n = 20
    s = 0
    g = 1
    stime = 50
    dt = 0.01
    sigma = 0.1
    
    bbo_sigma = 0.0001
    bbo_lmb = 0.05
    epochs = 100

    # create dmps    
    dmps = [ DMP(n, s, g, stime, dt, sigma) 
                    for k in range(K) ]
    
    # target trajectory
    x = np.linspace(0, 3*np.pi, stime)
    target = np.sin(x) + x
    target /= target.max()
    #x = np.linspace(0, 1, stime)
    #target = x
    
    # function that call the rollouts within the bbo object
    def rollouts(thetas):
        
        rollouts = []
        errs = []
        for k, theta in enumerate(thetas):
            dmps[k].reset()
            dmps[k].theta = theta.copy()
            dmps[k].rollout()
            rollout = dmps[k].S["y"]
            err = (target - rollout)
            rollouts.append(rollout)
            errs.append(err)
        
        return np.vstack(errs), np.vstack(rollouts)
    
    # the BBO object
    bbo = BBO(rollouts, n, K, bbo_sigma, bbo_lmb, epochs, 0.0, 0.2)
    
    costs = np.zeros(epochs)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    line, = ax.plot(costs)
    for t in range(epochs):
        # iterate -------------
        rs,_ = bbo.iteration()
        # ---------------------
        costs[t] = bbo.err
        line.set_ydata(costs)
        ax.relim()
        ax.autoscale_view()
        plt.pause(0.001)
        
    # test ----------------------------------
    rollouts,_ = bbo.iteration(explore=False)
    # ---------------------------------------
    
    fig2 = plt.figure()
    plt.plot(target, lw=2, color="red")
    plt.plot(rs.T, lw=0.2, color="black")
    plt.plot(rollouts.T, color="green", lw=3)
    plt.show()

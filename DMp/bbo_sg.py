import sys
import numpy as np
from dmp import DMP
import matplotlib.pyplot as plt
np.set_printoptions(precision=3, suppress=True)

def cost_softmax(x, lmb):
    e = np.exp(-(x - np.min(x))/lmb)
    return e/sum(e)

def rew_softmax(x, lmb):
    e = np.exp((x - np.max(x))/lmb)
    return e/sum(e)

class BBO :
    "P^2BB: Policy Improvement through Black Vox Optimization"
    def __init__(self, num_params=10, 
                 dmp_stime=100, dmp_dt=0.1, dmp_sigma=0.1,
                 num_rollouts=20, num_dmps=1,
                 sigma=0.001, lmb=0.1, epochs=100, 
                 sigma_decay_amp=0, sigma_decay_period=0.1, 
                 softmax=cost_softmax):
        '''
        :param num_params: Integer. Number of parameters to optimize 
        :param n um_rollouts: Integer. number of rollouts per iteration
        :param num_dmps: Integer, number of dmps
        :param dmp_stime: Integer, length of the trajectories in timesteps
        :param dmp_dt: Float, integration step
        :param dmp_sigma: Float, standard deviation of dmp gaussian basis functions
        :param sigma: Float. Amount of exploration around the mean of parameters
        :param lmb: Float. Temperature of the evaluation softmax
        :param epochs: Integer. Number of iterations
        :param sigma_decay_amp: Initial additive amplitude of exploration
        :param sigma_decay_period: Decaying period of additive 
            amplitude of exploration
        '''
        
        self.dmp_stime = dmp_stime
        self.dmp_dt = dmp_dt
        self.dmp_sigma = dmp_sigma
        self.sigma = sigma
        self.lmb = lmb
        self.num_rollouts = num_rollouts
        self.num_dmps = num_dmps
        self.num_dmp_params = num_params
        self.num_params = int(self.num_dmps*(num_params + 1))
        self.theta = np.zeros(self.num_params)
        self.Cov = np.eye(self.num_params, self.num_params)
        self.epochs = epochs
        self.decay_amp = sigma_decay_amp
        self.decay_period = sigma_decay_period
        self.epoch = 0
        self.err = 1.0
        self.softmax = softmax
        self.target = None
        
        # create dmps  
        self.dmps = []
        for x in range(self.num_dmps):
            self.dmps.append([DMP(n=self.num_dmp_params, s=0,
                          g=1, stime=self.dmp_stime,
                          dt=self.dmp_dt, sigma=self.dmp_sigma) 
                     for k in range(self.num_rollouts)])
        
        # define the cost function 
        self.cost_func = self.supervised_cost_func
        
    def sample(self):
        """ Get num_rollouts samples from the current parameters mean
        """
        
        Sigma = self.sigma + self.decay_amp*np.exp(
            -self.epoch/(self.epochs * self.decay_period))
        # matrix of deviations from the parameters mean
        self.eps = np.random.multivariate_normal(
            np.zeros(self.num_params), 
            self.Cov * Sigma, self.num_rollouts)
    
    def set_target(self, target):
        self.target = target
    
    def update(self, Sk):
        ''' Update parameters
        
            :param Sk: array(Float), rollout costs in an iteration 
        '''
        # Cost-related probabilities of sampled parameters
        probs = self.softmax(Sk, self.lmb).reshape(self.num_rollouts, 1)
        # update with the weighted average of sampled parameters
        self.theta += np.sum(self.eps * probs, 0)
    
    def supervised_cost_func(self, rollouts):
        trgts = np.array(self.target)
        trgts = trgts.reshape(self.num_dmps, 1, self.dmp_stime)
        return (trgts - rollouts)**2
     
    def rollouts(self, thetas):
        """ Produce a rollout
            :param thetas: array(num_rollouts X num_params/num_dmps)
            :return (rollouts) 
                rollouts: list(array(num_rollouts, stime))
        """   
        rollouts = []
                
        rng = self.num_dmp_params + 1
        for idx, dmp  in enumerate(self.dmps): 
            dmp_rollouts = []
            for k, theta in enumerate(thetas):
                thetak = theta.copy()
                dmp_theta = thetak[(idx*rng):((idx+1)*rng -1)] 
                dmp[k].reset()
                dmp[k].theta = dmp_theta
                dmp[k].set_goal(dmp_theta[-1])
                dmp[k].rollout()
                rollout = dmp[k].S["y"]
                dmp_rollouts.append(rollout)
            rollouts.append(np.vstack(dmp_rollouts))
        return rollouts
    
    def outcomes(self, rollouts):
        """
        compute outcomes for a stack of rollouts
        :param rollouts: list(array(num_rollouts, stime)) 
                for each dmp a stak of k rollouts
        """              
        #rollouts = np.vstack(rollouts)
        rollouts = np.array(rollouts)        
        errs = self.cost_func(rollouts)           
        return errs    
    
    def eval(self, errs):
        """ evaluate rollouts
            :param errs: list(array(float)), Matrices containing DMPs' errors 
                 at each timestep (columns) of each rollout (rows) 
            return: array(float), overall cost of each rollout 
        """   
        self.err = np.mean(np.mean(errs,1)) # store the mean square error
   
        # comute costs
        Sk = np.zeros(self.num_rollouts)
        for k in range(self.num_rollouts):
            Sk[k] = 0
            # final costs
            for err in errs:
                Sk[k] += err[k,-1]
                for j in range(self.num_dmp_params) :
                    # cost-to-go integral
                    Sk[k] += err[k, j:-1].sum() 
            # regularization
            thetak = self.theta + self.eps[k]
            Sk[k] += 0.5 * np.mean(self.sigma) * (thetak).dot(thetak) 
    
        return Sk
        
    def iteration(self, explore = True):
        """ Run an iteration
            :param explore: Bool, If the iteration is for training (True) 
                or test (False)
            :return: (rollouts, total value of the iteration)
        """
        self.sample()
        rollouts = self.rollouts(self.theta + explore*self.eps) 
        costs = self.outcomes(rollouts)   
        Sk = self.eval(costs)
        self.update(Sk)
        self.epoch += 1
        return rollouts, self.err
    
#------------------------------------------------------------------------------ 

if __name__ == "__main__":
    
    dmp_num_theta = 10
    dmp_stime = 60
    dmp_dt = 0.3
    dmp_sigma = 0.05
    
    bbo_sigma = 1.0e-03
    bbo_lmb = 0.1
    bbo_epochs = 100
    bbo_K = 45
    bbo_num_dmps = 2

    # the BBO object
    bbo = BBO(num_params=dmp_num_theta, 
              dmp_stime=dmp_stime, dmp_dt=dmp_dt, dmp_sigma=dmp_sigma,
              num_rollouts=bbo_K, num_dmps=bbo_num_dmps,
               sigma=bbo_sigma, lmb=bbo_lmb, epochs=bbo_epochs,
              sigma_decay_amp=0.0, sigma_decay_period=0.1)
    

    # target trajectory
    partial_stime = int(dmp_stime*0.7)
    x = np.linspace(0.0, 1.0, partial_stime)
    a = x*2*np.pi
    targetx = x*np.cos(a)+x
    targetx /= max(targetx)
    targetx = np.hstack((targetx, np.ones(int(dmp_stime - partial_stime))))
    targety = x*np.sin(a)+x
    targety /= max(targety)
    targety = np.hstack((targety, np.ones(dmp_stime - partial_stime)))

 
    fig1 = plt.figure()
    ax01 = fig1.add_subplot(211)
    ax01.plot(targetx)
    ax01.plot(targety)
    ax02 = fig1.add_subplot(212)
    ax02.plot(targetx, targety)


    
    bbo.set_target([targetx, targety])

    costs = np.zeros(bbo_epochs)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    line, = ax.plot(costs)
    for t in range(bbo_epochs):
        # iterate -------------
        rs,_ = bbo.iteration()
        # ---------------------
        costs[t] = bbo.err
        line.set_ydata(costs)
        ax.relim()
        ax.autoscale_view()
        plt.pause(0.001)
    print
    
    # test ----------------------------------
    rollouts,_ = bbo.iteration(explore=False)
    # ---------------------------------------
    
    fig2 = plt.figure()
    ax11 = fig2.add_subplot(211)
    ax11.plot(rs[0].T, lw=0.2, color="#220000")
    ax11.plot(rs[1].T, lw=0.2, color="#002200")
    ax11.plot(rollouts[0].T, lw=0.2, color="#884444")
    ax11.plot(rollouts[1].T, lw=0.2, color="#448844")   
    ax12 = fig2.add_subplot(212)
    ax12.plot(targetx, targety, lw=2, color="red")
    ax12.plot(rs[0].T,rs[1].T, lw=0.2, color="black")
    ax12.plot(rollouts[0].T, rollouts[1].T, color="green", lw=3)
    plt.show()
    

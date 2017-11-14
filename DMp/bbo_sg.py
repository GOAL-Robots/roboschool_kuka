import numpy as np
from dmp import DMP
import matplotlib.pyplot as plt

def softmax(x, lmb):
    e = np.exp(-x/lmb)
    return e/sum(e)


class BBO :
    "P^2BB: Policy Improvement through Black Vox Optimization"
    def __init__(self, rollout_func, num_params=10, num_rollouts=20, num_dmps=1,
                 sigma=0.001, lmb=0.1, epochs=100, 
                 sigma_decay_amp=0, sigma_decay_period=0.1):
        '''
        :param num_params: Integer. Number of parameters to optimize 
        :param num_rollouts: Integer. number of rollouts per iteration
        :param num_dmps: number of dmps
        :param sigma: Float. Amount of exploration around the mean of parameters
        :param lmb: Float. Temperature of the evaluation softmax
        :param epochs: Integer. Number of iterations
        :param rollout_func: Callable object to produce a rollout
            signature: thetas (errs, rollouts)
                thetas := array(num_rollouts X num_params/num_dmps)
                errs := list(array(num_rollouts, stime))
                rollouts := list(array(num_rollouts, stime))
        :param sigma_decay_amp: Initial additive amplitude of exploration
        :param sigma_decay_period: Decaying period of additive 
            amplitude of exploration
        '''
        
        self.sigma = sigma
        self.lmb = lmb
        self.num_rollouts = num_rollouts
        self.num_dmps = num_dmps
        self.num_params = int(self.num_dmps*num_params + self.num_dmps)
        self.theta = np.zeros(self.num_params)
        self.theta[-self.num_dmps:] = 1
        self.Cov = np.eye(self.num_params, self.num_params)
        self.rollout_func = rollout_func
        self.epochs = epochs
        self.decay_amp = sigma_decay_amp
        self.decay_period = sigma_decay_period
        self.epoch = 0
        self.err = 1.0
          
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
            :param errs: list(array(float)), Matrices containing DMPs' errors 
                 at each timestep (columns) of each rollout (rows) 
            return: array(float), overall cost of each rollout 
        """
        errs = [err**2 for err in errs]
        self.err = np.mean(np.mean(errs,1)) # store the mean square error
   
        # comute costs
        Sk = np.zeros(self.num_rollouts)
        for k in range(self.num_rollouts):
            Sk[k] = 0
            # final costs
            for err in errs:
                Sk[k] + err[k,-1]
                for j in range(self.num_params - 2) :
                    # timestep cost 
                    Sk[k] += err[k, j]
                    # cost-to-go integral
                    Sk[k] += err[k, j:].sum() 
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
    
    dmp_num_theta = 10
    dmp_s = 0
    dmp_g = 1
    dmp_stime = 50
    dmp_dt = 0.1
    dmp_sigma = 0.01
    
    bbo_sigma = 3e-4
    bbo_lmb = 0.2
    bbo_epochs = 100
    bbo_K = 20
    bbo_num_dmps = 1

    # create dmps  
    dmps = []
    for x in range(bbo_num_dmps):
        dmps.append([DMP( n=dmp_num_theta, s=dmp_s, 
                          g=dmp_g, stime=dmp_stime, 
                          dt=dmp_dt, sigma=dmp_sigma) 
                     for k in range(bbo_K)])
      
    # target trajectory
    x = np.linspace(0, 3*np.pi, dmp_stime)
    target = np.sin(x) + x
    target /= target.max()

    
    # function that call the rollouts within the bbo object
    def rollouts(thetas):
        
        rollouts = []
        errs = []
                
        rng = dmp_num_theta + 1 
        for idx, dmp in enumerate(dmps): 
            dmp_rollouts = []
            dmp_errs = []
            for k, theta in enumerate(thetas):
                thetak = theta.copy()
                dmp_theta = thetak[(idx*rng):((idx+1)*rng)] 
                dmp[k].reset()
                dmp[k].theta = dmp_theta[:-1]
                dmp[k].set_goal(dmp_theta[-1])
                dmp[k].rollout()
                rollout = dmp[k].S["y"]
                err = (target - rollout)
                dmp_rollouts.append(rollout)
                dmp_errs.append(err)
            errs.append(np.vstack(dmp_errs))
            rollouts.append(np.vstack(dmp_rollouts))
        return errs, rollouts
    
    # the BBO object
    bbo = BBO(rollout_func=rollouts, num_params=dmp_num_theta, 
              num_rollouts=bbo_K, num_dmps=bbo_num_dmps,
               sigma=bbo_sigma, lmb=bbo_lmb, epochs=bbo_epochs,
              sigma_decay_amp=0.0, sigma_decay_period=0.01)
    

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
        
    # test ----------------------------------
    rollouts,_ = bbo.iteration(explore=False)
    # ---------------------------------------
    
    fig2 = plt.figure()
    plt.plot(target, lw=2, color="red")
    plt.plot(rs.T, lw=0.2, color="black")
    plt.plot(rollouts[0].T, color="green", lw=3)
    plt.show()

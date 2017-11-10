import numpy as np
from DMp.dmp import DMP
import matplotlib.pyplot as plt

def softmax(x, lmb):
    e = np.exp(-x/lmb)
    return e/sum(e)

class BBO :
    
    def __init__(self, num_params, sigma, lmb, K, epochs, rollout_func):
        self.sigma = sigma
        self.lmb = lmb
        self.K = K
        self.num_params = num_params
        self.theta = np.zeros(num_params)
        self.Cov = np.eye(num_params, num_params)
        self.rollout_func = rollout_func
        self.err = 1.0
        self.epochs = epochs
        self.epoch = 0
        
        
    def sample(self):
        decay = 0.01 + np.exp(-self.epoch/float(self.epochs*.1))
        self.eps = np.random.multivariate_normal(np.zeros(self.num_params), self.Cov*self.sigma*decay, self.K)
    
    def update(self, Sk):
        probs = softmax(Sk, self.lmb)
        self.theta += np.sum(self.eps * np.outer(probs, np.ones(self.num_params)), 0)
    
    def eval(self, rollouts, costs):
        
        Sk = np.mean(costs**2,1)
        self.err = Sk.min()
        return Sk
        
    def iteration(self, test = False):
        self.sample()
        rollouts, costs = self.rollout_func(self.eps*(1-test) + self.theta)
        Sk = self.eval(rollouts, costs)
        self.update(Sk)
        self.epoch += 1
        return rollouts, Sk
    
if __name__ == "__main__":
    
    K = 40
    n = 30
    s = 0
    g = 1
    stime = 40
    dt = 0.1
    sigma = 0.05
    
    bbo_sigma = .3
    bbo_lmb = 0.1
    epochs = 700
    
    dmps = [ DMP(n, s, g, stime, dt, sigma) 
                    for k in range(K) ]
    
    #x = np.linspace(0, 4*np.pi, stime)
    #target = np.sin(x) + x
    #target /= target.max()
    x = np.linspace(0, 1, stime)
    target = x
    
    def rollouts(thetas):
        
        rollouts = []
        errs = []
        for k, theta in enumerate(thetas):
            dmps[k].theta = theta.copy()
            dmps[k].rollout()
            rollout = dmps[k].S["y"]
            err = (target - rollout)
            rollouts.append(rollout)
            errs.append(err)
        
        return np.vstack(rollouts), np.vstack(errs)
                           
    bbo = BBO(n, bbo_sigma, bbo_lmb, K, epochs, rollouts)
    
    costs = np.zeros(epochs)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    line, = ax.plot(costs)
    ax.set_ylim([0,.05])
    for t in range(epochs):
        rs,_ = bbo.iteration()
        costs[t] = bbo.err
        line.set_ydata(costs)
        plt.pause(0.001)
    rs,_ = bbo.iteration(test=True)
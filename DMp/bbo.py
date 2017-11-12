import numpy as np
from dmp import DMP
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
        decay = 1.0 + 0*np.exp(-self.epoch/float(self.epochs*.3))
        self.eps = np.random.multivariate_normal(
            np.zeros(self.num_params), 
            self.Cov * self.sigma * decay, self.K)
    
    def update(self, Sk):
        probs = softmax(Sk, self.lmb).reshape(self.K, 1)
        self.theta += np.sum(self.eps * probs, 0)
    
    def eval(self, costs):    
         
        errs = costs**2
        self.err = np.min(np.mean(errs,1))
        
        Sk = np.zeros(self.K)
        
        for k in range(self.K):
            
            Sk[k] = errs[k, -1]
            
            Sk[k] += np.sum( 
                errs[k, j:].sum() 
                for j in range(self.num_params)) 
                    
            Sk[k] += 0.5 * self.sigma * (self.theta + self.eps[k]).dot(
                                self.theta + self.eps[k]) 
        return Sk
        
    def iteration(self, test = False):
        self.sample()
        if test == False:
            costs, rollouts = self.rollout_func(self.theta + self.eps)    
        elif test == True:
            costs, rollouts = self.rollout_func(self.theta + 0*self.eps)
        Sk = self.eval(costs)
        self.update(Sk)
        self.epoch += 1
        return rollouts, Sk
    
if __name__ == "__main__":
    
    K = 50
    n = 10
    s = 0
    g = 1
    stime = 200
    dt = 0.1
    sigma = 0.1
    
    bbo_sigma = 0.01
    bbo_lmb = 1.0
    epochs = 50
    
    dmps = [ DMP(n, s, g, stime, dt, sigma) 
                    for k in range(K) ]
    
    x = np.linspace(0, 3*np.pi, stime)
    target = np.sin(x) + x
    target /= target.max()
    #x = np.linspace(0, 1, stime)
    #target = x
    
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
                           
    bbo = BBO(n, bbo_sigma, bbo_lmb, K, epochs, rollouts)
    
    costs = np.zeros(epochs)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    line, = ax.plot(costs)
    for t in range(epochs):
        rs,_ = bbo.iteration()
        costs[t] = bbo.err
        line.set_ydata(costs)
        ax.relim()
        ax.autoscale_view()
        plt.pause(0.001)
    rollouts,_ = bbo.iteration(test=True)
    
    fig2 = plt.figure()
    plt.plot(target, lw=2, color="red")
    plt.plot(rs.T, lw=0.2, color="black")
    plt.plot(rollouts.T, color="green", lw=3)
    plt.show()

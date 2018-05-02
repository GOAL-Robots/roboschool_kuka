import sys
import numpy as np
from pdmp import DMP
import matplotlib.pyplot as plt
import bbo_pdmp as bbo
np.set_printoptions(precision=3, suppress=True)

class BBO_param(bbo.BBO):

    def rollouts(self, thetas):
        """ Produce a rollout
            :param thetas: array(num_rollouts X num_params/num_dmps)
            :return: list(array(num_rollouts, stime)) rollouts
        """ 
        assert type(self.hparams_samples) == np.ndarray
        assert self.hparams_samples.shape[1] > len(self.bins_hparams)
        
        samples_num = self.hparams_samples.shape[0]
        
        rollouts = []
       
        rng = self.num_dmp_params + 1
        for idx, dmp  in enumerate(self.dmps): 
            dmp_rollouts = []
            for k, theta in enumerate(thetas):
                sample = k%samples_num
                thetak = theta.copy()
                dmp_theta = thetak[(idx*rng):((idx+1)*rng -1)] 
                dmp[k].reset()
                dmp[k].theta = dmp_theta
                dmp[k].set_goal(dmp_theta[-1])
                dmp[k].rollout(self.hparams_samples[sample])
                rollout = dmp[k].S["y"]
                dmp_rollouts.append(rollout)
            rollouts.append(np.vstack(dmp_rollouts))
        return rollouts
    
if __name__ == "__main__":
        
    # consts
    dmp_num_theta = 10
    dmp_stime = 60
    dmp_dt = 0.3
    dmp_sigma = 0.05
    
    bbo_sigma = 1.0e-03
    bbo_lmb = 0.1
    bbo_epochs = 100
    bbo_K = 45
    bbo_num_dmps = 2

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

    # plot target
    fig1 = plt.figure()
    ax01 = fig1.add_subplot(211)
    ax01.plot(targetx)
    ax01.plot(targety)
    ax02 = fig1.add_subplot(212)
    ax02.plot(targetx, targety)

    # make a target list for the bbo object
    target = [targetx, targety]
    
    # make a cost function for the bbo object
    def supervised_cost_func(rollouts):
        trgts = trgts.reshape(bbo_num_dmps, 1, dmp_stime)
        return (trgts - rollouts)**2  
    
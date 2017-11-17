import sys
import numpy as np
from dmp import DMP
import matplotlib.pyplot as plt
np.set_printoptions(precision=3, suppress=True)

import bbo_sg as bbo

if __name__ == "__main__":

    dmp_num_theta = 20
    dmp_stime = 50
    dmp_dt = 0.3
    dmp_sigma = 0.1
    

    bbo_lmb = 0.3
    bbo_epochs = 60
    bbo_K = 20
    bbo_num_dmps = 2
    bbo_sigma = 5.0e-04
        
    # the BBO object
    bbo = bbo.BBO(num_params=dmp_num_theta, 
              dmp_stime=dmp_stime, dmp_dt=dmp_dt, dmp_sigma=dmp_sigma,
              num_rollouts=bbo_K, num_dmps=bbo_num_dmps,
              sigma=bbo_sigma, lmb=bbo_lmb, epochs=bbo_epochs,
              sigma_decay_amp=0.5, sigma_decay_period=0.02, 
              softmax=bbo.rew_softmax)
    
    
    coords = np.vstack([ np.random.uniform(-1.5, 1.5, (2,)), 
               np.random.uniform( 0.0, 1.5, (2,)) ]).T
    
    def rew_func(rollout, dmp_idx):
        rew = np.array(1*np.logical_and(rollout >= coords[dmp_idx][0], 
                                        rollout <=coords[dmp_idx][1]))
        return rew  
    
    bbo.cost_func = rew_func
    
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
    ax12 = fig2.add_subplot(212, aspect="equal")
    ax12.plot(rs[0].T,rs[1].T, lw=0.2, color="black")
    ax12.plot(rollouts[0].T, rollouts[1].T, color="green", lw=3)
    ax12.scatter(0.0, 0.0, color="blue", s=60)
    ax12.scatter(coords[0][0] + (coords[0][1] - coords[0][0])*0.5,
                 coords[1][0] + (coords[1][1] - coords[1][0])*0.5, 
                 color="red", s=60)
    ax12.set_xlim([-1.5,1.5])
    ax12.set_ylim([-1.5,1.5])
    plt.show()
         
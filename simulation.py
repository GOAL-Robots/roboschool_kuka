import matplotlib
matplotlib.use("Agg")

import gym, os, glob 
from OpenGL import GLU
import numpy as np
import matplotlib.pyplot as plt
import DMp
from DMp.bbo_pdmp import BBO, rew_softmax
import time, sys

from hashlib import sha1
from simulation_settings import *

#-----------------------------------------------------------------------------

if __name__ == "__main__":
    
    dirs = ["frames", "frames/lasts", "frames/bests", "frames/epochs"]
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)
        files = glob.glob(d + "/*")
        for f in files:
            if(os.path.isfile(f)):
                os.remove(f)

    env = gym.make("REALComp-v0")
    env.reward_func = GraspRewardFunc()
    env.robot.target = target
    env.robot.used_objects = ["table", target]
    env.robot.contact_threshold = contact_threshold
    env._render_width = 640
    env._render_height = 480
    env._cam_yaw = target_yaw[target]
    env.setCamera()
    
    # the BBO object
    bbo = BBO(num_params=dmp_num_theta, 
            dmp_stime=dmp_stime, dmp_dt=dmp_dt, dmp_sigma=dmp_sigma,
            num_rollouts=bbo_episodes, num_dmps=bbo_num_dmps,
            sigma=bbo_sigma, lmb=bbo_softmax_temp, epochs=bbo_epochs,
            sigma_decay_amp=bbo_sigma_decay_amp, 
            sigma_decay_period=bbo_sigma_decay_period, 
            softmax=rew_softmax, cost_func=Objective(env))
    
    def init_trj(ro):
        return np.hstack(( np.zeros([ro.shape[0], init_gap]), ro));

    # BBO learning iterations
    rew = np.zeros(bbo_epochs)
    best_rollout = None
    epoch_rollout = None
    max_Sk = 0
    max_Sks = None
    max_rews = None
    for k in range(bbo_epochs):
       
        # simulaton step
        start = time.time()
        rollouts, rews, Sk  = bbo.iteration()
        rew[k] = np.max(Sk)
        end = time.time()
        print("{:#4d} {:10.4f} -- {}".format(k, rew[k], end - start))
         
        # store epoch
        epoch_Sk = np.max(Sk) 
        epoch_Sk_idx = np.argmax(Sk)
        epoch_rollout = rollouts[epoch_Sk_idx]

        # store bests
        if max_Sk < epoch_Sk:
            best_rollout = epoch_rollout
            max_Sk = epoch_Sk

        if k%10 == 0 or k == bbo_epochs -1:

            if best_rollout is not None:
                curr_rollout = best_rollout
                curr_rollout = init_trj(best_rollout)
            
                # simulate for video storage 
                simulate = Simulator(curr_rollout, env, path="frames/bests", save=True)
                for t in range(curr_rollout.shape[1]): 
                    simulate.step()

                # save the plot with reward history
                fig = plt.figure(figsize=(800/100, 600/100), dpi=100)
                ax = fig.add_subplot(111)
                ax.plot(rew)
                ax.scatter(k,rew[k],color="red")
                fig.savefig("frames/rew.png",dpi=200)

        

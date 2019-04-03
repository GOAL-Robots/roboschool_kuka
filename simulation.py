import matplotlib
matplotlib.use("Agg")

import gym, os, glob 
from OpenGL import GLU
import numpy as np
import matplotlib.pyplot as plt
import DMp
from DMp.bbo_pdmp import BBO, rew_softmax
import roboschool
from PIL import Image
import time

#-----------------------------------------------------------------------------

def GraspRewardFunc(contact_dict, state):
    
    finger_reward = np.sum([ len([contact for contact in contacts 
        if "orange" in contact]) for part, contacts 
        in contact_dict.items() if "finger" in part ])

    fingers_reward = len(np.unique(contact_dict.keys()))
    
    table_reward = np.sum([ len([contact for contact in contacts 
        if "table" in contact]) for part, contacts 
        in contact_dict.items() if not "finger" in part ])
   
    obj_pose = state[-3:]

    distance = np.linalg.norm(obj_pose - GraspRewardFunc.initial_obj_pose)
    dist_sigma = 4.0*np.exp(-GraspRewardFunc.epoch)
    distance = np.exp(-(dist_sigma**-2)*distance**2)
    finger_amp = 1.0
    table_amp = 0.1


    return finger_amp*finger_reward*fingers_reward*distance -  table_amp*table_reward

GraspRewardFunc.epoch = 0
GraspRewardFunc.initial_obj_pose = [0.0, 0.0, 0.8]

#-----------------------------------------------------------------------------

class Simulator:
    def __init__(self, rollout, env, plot=False, save=False, path="frames/lasts" ):
        """
        :param rollout: A single rollout (n_joints x timesteps) from which joint commands are taken
        :param plot: if the simulation is rendered on a window
        :param save: if the simulation frames are saved on file
        :param path: path where jpegs are saved
        """
        self.t = 0
        self.rollout = rollout  
        self.plot = plot
        self.path = path
        self.save = save
        self.env = env
        self.env.reset()

        if save:
            np.savetxt(self.path+"/rollout",rollout)
        
    def step(self):    

        # we control only few joints
        ctrl_joints = self.rollout[:, self.t]
        action = np.zeros(9)
        
        action[0]   =  np.pi*0.0 + ctrl_joints[0]
        action[1]   =  np.pi*0.2 + ctrl_joints[1]
        action[2]   =  np.pi*0.2 + ctrl_joints[2] 
        action[3]   = -np.pi*0.2 + ctrl_joints[3]
        action[4:7] =  np.pi*0.0 + ctrl_joints[4:7] 
        action[7:] = ctrl_joints[7:]*np.pi
        
        # do the movement
        state, r, done, info_ = self.env.step(action)

        if self.plot:
            self.env.render("human")
        
        if self.save:
            rgb = self.env.render("rgb_array")
            im = Image.fromarray(rgb) 
            im.save(self.path + "/frame_{:04d}.jpeg".format(self.t))

        self.t += 1  
        return r


#-----------------------------------------------------------------------------

class Objective:
    """
    Objective for the BBO
    Given a simulation environment run  simulations with the 
    given joint trajectories at each call
    """
    curr_rew = 0
    best_rollout = None

    def __init__(self, env):
        self.env = env

    def __call__(self, rollouts):

        n_joints, n_episodes, timesteps = rollouts.shape 

        rews = np.zeros([n_episodes, timesteps])
        rew_means = np.zeros(n_episodes)
        for episode in range(n_episodes):
            
            # simulate with the current joint trajectory to read rewards
            simulate = Simulator(np.squeeze(rollouts[:,episode,:]), 
                    self.env, plot=False)
            for t in range(timesteps):
                rews[episode, t] = np.sum(simulate.step())
            rew_means[episode] = np.mean(rews[episode]) 
            
            # we save the best rollout till now
            if rew_means[episode] > Objective.curr_rew:
               Objective.best_rollout = np.squeeze(rollouts[:,episode,:]).copy()
            Objective.curr_rew = rew_means[episode];
        
        max_idx = np.argmax(rew_means)
        Objective.epoch_rollout = np.squeeze(rollouts[:,max_idx,:]).copy()

        return rews.reshape(1, *rews.shape)

#-----------------------------------------------------------------------------


if __name__ == "__main__":
    
    SIM_PLOT=False
       

    dirs = ["frames", "frames/lasts", "frames/bests", "frames/epochs"]

    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)
        files = glob.glob(d + "/*")
        for f in files:
            if(os.path.isfile(f)):
                os.remove(f)

    dmp_num_theta = 20
    dmp_stime = 100
    dmp_dt = 0.2
    dmp_sigma = 0.2

    bbo_lmb = 0.1
    bbo_epochs = 2000
    bbo_episodes = 20
    bbo_num_dmps = 9
    bbo_sigma = 1.0e-10
    bbo_sigma_decay_amp = 1.0
    bbo_sigma_decay_period = 1.0e100
    init_gap = 150
    
    env = gym.make("RoboschoolKuka-v1")
    env.unwrapped.set_eyeEnable(False)
    env.unwrapped.set_eyeShow(False)
    env.unwrapped.reward_func = GraspRewardFunc
    env.unwrapped.used_objects = ["ORANGE"]

    # the BBO object
    bbo = BBO(num_params=dmp_num_theta, 
            dmp_stime=dmp_stime, dmp_dt=dmp_dt, dmp_sigma=dmp_sigma,
            num_rollouts=bbo_episodes, num_dmps=bbo_num_dmps,
            sigma=bbo_sigma, lmb=bbo_lmb, epochs=bbo_epochs,
            sigma_decay_amp=bbo_sigma_decay_amp, 
            sigma_decay_period=bbo_sigma_decay_period, 
            softmax=rew_softmax, cost_func=Objective(env))
    
    def init_trj(ro):
        return np.hstack(( np.zeros([ro.shape[0], init_gap]), ro));


    # BBO learning iterations
    rew = np.zeros(bbo_epochs)
    for k in range(bbo_epochs):
       
        # simulaton step

        start = time.time()
        rollouts, rew[k] = bbo.iteration()
        end = time.time()
        
       
        # save and plot
        print("{:#4d} {:6.2f} -- {}".format(k, rew[k], end - start))

        if k%12 == 0 or k == bbo_epochs -1:
            rollouts = np.array(rollouts)
            rollout_0 = np.squeeze(rollouts[:,0,:])
        
            curr_rollout = rollout_0
            curr_rollout = init_trj(curr_rollout)
            # run the simulator on first episode of last iteration
            simulate = Simulator(curr_rollout, env,
                    path="frames/lasts", plot=SIM_PLOT, save=True)
            for t in range(curr_rollout.shape[1]): 
                simulate.step()
        
            # run the simulator on best rollout
            if Objective.best_rollout is not None:
                curr_rollout = Objective.best_rollout
            else:
                curr_rollout = rollout_0
            curr_rollout = init_trj(curr_rollout)
            simulate = Simulator(curr_rollout, env, 
                    path="frames/bests",  plot=SIM_PLOT, save=True)
            for t in range(curr_rollout.shape[1]): 
                simulate.step()
        
            # run the simulator on epoch rollout
            if Objective.best_rollout is not None:
                curr_rollout = Objective.epoch_rollout
            else:
                curr_rollout = rollout_0
            curr_rollout = init_trj(curr_rollout)
            simulate = Simulator(curr_rollout, env, 
                    path="frames/epochs",  plot=SIM_PLOT, save=True)
            for t in range(curr_rollout.shape[1]): 
                simulate.step()
        
            GraspRewardFunc.epoch = k/float(bbo_epochs)
        
            # save the plot with reward history
            fig = plt.figure(figsize=(800/100, 600/100), dpi=100)
            ax = fig.add_subplot(111)
            ax.plot(rew)
            fig.savefig("frames/rew.png",dpi=100)
        

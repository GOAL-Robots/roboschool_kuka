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




class Simulation:
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
        
    def __call__(self):    

        # we control only few joints
        ctrl_joints = self.rollout[:, self.t]
        action = np.zeros(9)
        
        action[1]   =  np.pi*0.2 + ctrl_joints[1]
        action[2]   =  np.pi*0.0 + ctrl_joints[2] 
        action[3]   =  np.pi*0.3 + ctrl_joints[3]
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


class rew_func:
    """
    Reward functor
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
        for episode in range(n_episodes):
            
            # simulate with the current joint trajectory to read rewards
            simulate_step = Simulation(np.squeeze(rollouts[:,episode,:]), 
                    self.env, plot=False)
            for t in range(timesteps):
                rews[episode, t] = np.sum(simulate_step())
            rew_mean = np.mean(rews[episode, t]) 
            
            # we save the best rollout till now
            if rew_mean > rew_func.curr_rew:
               rew_func.best_rollout = np.squeeze(rollouts[:,episode,:]).copy()
            rew_func.curr_rew = rew_mean;

        return rews.reshape(1, *rews.shape)


def GraspRewardFunc(contact_dict, state):
    
    finger_reward = np.sum([ len([contact for contact in contacts 
        if not "table" in contact]) for part, contacts 
        in contact_dict.items() if "finger" in part ])

    fingers_reward = len(np.unique(contact_dict.keys()))
   
    obj_pose = state[-3:]
    if GraspRewardFunc.initial_obj_pose is None:
        GraspRewardFunc.initial_obj_pose = obj_pose.copy()

    distance = np.linalg.norm(obj_pose - GraspRewardFunc.initial_obj_pose)

    return finger_reward*10*fingers_reward*distance

GraspRewardFunc.initial_obj_pose = None

if __name__ == "__main__":
    
    SIM_PLOT=True
        
    if not os.path.exists("frames"):
        os.makedirs("frames")

    if not os.path.exists("frames/lasts"):
        os.makedirs("frames/lasts")
    files = glob.glob('/frames/lasts/*')
    for f in files:
        os.remove(f)

    if not os.path.exists("frames/bests"):
        os.makedirs("frames/bests")
    files = glob.glob('/frames/bests/*')
    for f in files:
        os.remove(f)

    dmp_num_theta = 20
    dmp_stime = 50
    dmp_dt = 0.2
    dmp_sigma = 0.2

    bbo_lmb = 0.1
    bbo_epochs = 1000
    bbo_episodes = 20
    bbo_num_dmps = 9
    bbo_sigma = 1.0e-10
    bbo_sigma_decay_amp = 0.8
    bbo_sigma_decay_period = 1.0e100
    
    env = gym.make("RoboschoolKuka-v0")
    env.unwrapped.set_eyeEnable(False)
    env.unwrapped.set_eyeShow(False)
    env.unwrapped.reward_func = GraspRewardFunc

    # the BBO object
    bbo = BBO(num_params=dmp_num_theta, 
            dmp_stime=dmp_stime, dmp_dt=dmp_dt, dmp_sigma=dmp_sigma,
            num_rollouts=bbo_episodes, num_dmps=bbo_num_dmps,
            sigma=bbo_sigma, lmb=bbo_lmb, epochs=bbo_epochs,
            sigma_decay_amp=bbo_sigma_decay_amp, 
            sigma_decay_period=bbo_sigma_decay_period, 
            softmax=rew_softmax, cost_func=rew_func(env))

    # BBO learning iterations
    rew = np.zeros(bbo_epochs)
    for k in range(bbo_epochs):
        rollouts, rew[k] = bbo.iteration()
        print("{:#4d} {:6.2f}".format(k, rew[k]))
        
        if k%10 == 0 or k == bbo_epochs -1:
            rollouts = np.array(rollouts)
            rollout_0 = np.squeeze(rollouts[:,0,:])

            # run the simulator on first episode of last iteration
            simulate_step = Simulation(rollout_0, env,
                    path="frames/lasts", plot=SIM_PLOT, save=True)
            for t in range(dmp_stime): 
                simulate_step()

            # run the simulator on best rollout
            if rew_func.best_rollout is not None:
                curr_rollout = rew_func.best_rollout
            else:
                curr_rollout = rollout_0
            simulate_step = Simulation(curr_rollout, env, 
                    path="frames/bests",  plot=SIM_PLOT, save=True)
            for t in range(dmp_stime): 
                simulate_step()

            # save the plot with reward history
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111)
            ax.set_ylim([.1, 8.0])
            ax.plot(rew)
            fig.savefig("frames/rew.png")


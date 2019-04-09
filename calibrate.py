import matplotlib
matplotlib.use("Agg")

import gym, os, glob 
from OpenGL import GLU
import numpy as np
import matplotlib.pyplot as plt
import DMp
from DMp.bbo_pdmp import BBO, rew_softmax
import realcomp 
from PIL import Image
import time

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

        if save:
            np.savetxt(self.path+"/rollout",rollout)
        
    def __call__(self):    

        # we control only few joints
        ctrl_joints = self.rollout[:, self.t]
        action = np.zeros(9)
        
        action[0]   =  np.pi*0.0 + ctrl_joints[0]
        action[1]   =  np.pi*0.2 + ctrl_joints[1]
        action[2]   =  np.pi*0.0 + ctrl_joints[2] 
        action[3]   = -np.pi*0.2 + ctrl_joints[3]
        action[4:7] =  np.pi*0.0 + ctrl_joints[4:7] 
        action[7:] = ctrl_joints[7:]*np.pi
        
        # do the movement
        state, r, done, info_ = self.env.step(action)

        self.t += 1  
        return r

if __name__ == "__main__":
    
    env = gym.make("REALComp-v0")
    env.setCameraSize(640, 480)

    env.reward_func = lambda x, y: 0
    env.robot.used_objects = ["table", "tomato", "mustard", "orange"]
    env.robot.object_poses["mustard"][2] = 1 
    env.render("human")

    rollout = np.zeros([9, 30000])
    sim = Simulation(rollout, env, plot=False, save=True)

    for t in range(len(rollout.T)): 
        time.sleep(1/60)
        sim()

    

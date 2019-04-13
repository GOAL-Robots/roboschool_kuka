import matplotlib
matplotlib.use("Agg")

import gym, os, glob 
from OpenGL import GLU
import numpy as np
import matplotlib.pyplot as plt
import DMp
from DMp.bbo_pdmp import BBO, rew_softmax
from realcomp.envs import realcomp_robot
from PIL import Image
import time
import argparse, sys

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
        action[7:] = ctrl_joints[7:] 
        
        # do the movement
        state, r, done, info_ = self.env.step(action)
        if not self.save:  time.sleep(1/60)

        if self.plot:
            self.env.render("human")
        
        if self.save:
            rgb = self.env.render("rgb_array")
            im = Image.fromarray(rgb)
            im.save(self.path + "/frame_{:04d}.jpeg".format(self.t))

        self.t += 1  
        return r

if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument('-s','--save',
        help="Save frames",
        action="store_true", default=False)  
    args = parser.parse_args()
    
    if args.save :
        if not os.path.exists("frames"):
            os.makedirs("frames")

        if not os.path.exists("frames/lasts"):
            os.makedirs("frames/lasts")
        files = glob.glob('/frames/lasts/*')
        for f in files:
            os.remove(f)


    rollout = np.loadtxt("rollout")
    
    env = gym.make("REALComp-v0")
    env._render_width = 640
    env._render_height = 480
    env._cam_yaw = 180

    env.reward_func = lambda x, y: 0
    env.robot.used_objects = ["table", "tomato", "mustard", "orange"]
    env.render("human")

    def init_trj(ro, init=150):
        return np.hstack(( np.zeros([ro.shape[0],init]), ro));
    
    rollout = init_trj(rollout)
    sim = Simulation(rollout, env, plot=False, save=args.save)

    for t in range(len(rollout.T)): sim()

    

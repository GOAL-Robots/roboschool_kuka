import realcomp 
from PIL import Image
from realcomp.envs import realcomp_robot
import numpy as np
import time

target = "orange"
target_pitch = {"tomato": -30, "orange": -60, "mustard":-30}
target_yaw = {"tomato": 180, "orange": 90, "mustard":0}


dmp_num_theta = 20
dmp_stime = 100
dmp_dt = 0.2
dmp_sigma = 0.4

bbo_softmax_temp = 0.01
bbo_epochs = 300
bbo_episodes = 30
bbo_num_dmps = 9
bbo_sigma = 0.01
bbo_sigma_decay_amp = 2.0
bbo_sigma_decay_start = 0.5
bbo_sigma_decay_period = 0.05
init_gap = 50

dist_sigma = 1000.0
finger_amp = 1.0
dist_amp = .5
table_amp = 0.0

contact_threshold = 0.008
target_pos = 0.5

#-----------------------------------------------------------------------------

class GraspRewardFunc:

    epoch = 0
    initial_obj_pose = realcomp_robot.Kuka.object_poses["tomato"][:3]
    initial_obj_pose[-1] += 0.5 
   
    def __call__(self, contact_dict, state):

        finger_reward = len([contact for part, contacts 
            in contact_dict.items() for contact in contacts
            if "skin" in part and target in contact])

        fingers_reward = len(set([part for part,contacts in 
            contact_dict.items() for contact in contacts
            if "skin" in part and target in contact]))
        
        table_reward = len([contact for part, contacts
            in contact_dict.items() for contact in contacts 
            if "table" in contact])

        obj_pose = state[-3:]

        distance = np.linalg.norm(obj_pose - GraspRewardFunc.initial_obj_pose)
        distance = np.exp(-(dist_sigma**-2)*distance**2)
        
        return finger_amp*finger_reward + dist_amp*fingers_reward*distance - table_amp*table_reward

#-----------------------------------------------------------------------------

def init_trj(ro):
    return np.hstack(( np.zeros([ro.shape[0], init_gap]), ro));

class Objective:
    """
    Objective for the BBO
    Given a simulation environment run  simulations with the 
    given joint trajectories at each call
    """

    def __init__(self, env):
        self.env = env

    def __call__(self, rollouts):

        n_episodes, n_joints, timesteps = rollouts.shape 

        rews = np.zeros([n_episodes, timesteps])

        for episode in range(n_episodes):
            
            # simulate with the current joint trajectory to read rewards
            rollout = init_trj(np.squeeze(rollouts[episode,:,:]))
            simulate = Simulator(rollout, self.env, plot=False)
            for t in range(rollout.shape[1]):
                r = simulate.step()
                if t>=init_gap:
                    rews[episode, t-init_gap] = r

        return rews


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
        action[2]   =  np.pi*0.0 + ctrl_joints[2] 
        action[3]   = -np.pi*0.2 + ctrl_joints[3]
        action[4:7] =  np.pi*0.0 + ctrl_joints[4:7] 
        action[7:] = ctrl_joints[7:]
        
        # do the movement
        self.state = self.env.step(action)
        state, r, done, info_ = self.state 

        if self.plot == True:
            time.sleep(1/60)
        
        if self.save == True:
            rgb = self.env.render("rgb_array")
            im = Image.fromarray(rgb) 
            im.save(self.path + "/frame_{:04d}.jpeg".format(self.t))

        self.t += 1  
        return r


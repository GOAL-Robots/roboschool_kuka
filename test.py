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

class GraspRewardFunc:
    
    
    dist_sigma = 0.5
    finger_amp = 100.0
    dist_amp = 10.0
    table_amp = 0.1


    epoch = 0
    initial_obj_pose = realcomp_robot.Kuka.object_poses["tomato"][:3]
    initial_obj_pose[-1] += 0.3 
   
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
        
        return finger_amp*finger_reward + dist_amp*distance - table_amp*table_reward

import pyglet, pyglet.window as pw, pyglet.window.key as pwk
from pyglet import gl

class PygletInteractiveWindow(pw.Window):
    
    def __init__(self, env, width, height):

        pw.Window.__init__(self, width=width, height=height, vsync=False, resizable=True)
        self.theta = 0
        self.still_open = True

        @self.event
        def on_close():
            self.still_open = False

        @self.event
        def on_resize(width, height):
            self.win_w = width
            self.win_h = height

        self.keys = {}
        self.human_pause = False
        self.human_done = False

    def imshow(self, arr):

        H, W, C = arr.shape
        assert C==3
        image = pyglet.image.ImageData(W, H, 'RGB', arr.tobytes(), pitch=W*-3)
        self.clear()
        self.switch_to()
        self.dispatch_events()
        texture = image.get_texture()
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        texture.width  = W
        texture.height = H
        texture.blit(0, 0, width=self.win_w, height=self.win_h)
        self.flip()



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
        path = "frames/lasts"
    

    rollout = np.loadtxt("rollout")
    
    env = gym.make("REALComp-v0")
    env.setEye("eye0")
    env._render_width = 640
    env._render_height = 480
    env._cam_yaw = 180
    env.robot.used_objects = ["table", "tomato", "mustard", "orange"]
    env.robot.reward_func = GraspRewardFunc()

    p = PygletInteractiveWindow(env, 320, 240)

    env.render("human")
    env.reset()

    def init_trj(ro, init=20):
        return np.hstack(( np.zeros([ro.shape[0],init]), ro)); 
    rollout = init_trj(rollout)

    for t in range(len(rollout.T)): 
       
        # we control only few joints
        ctrl_joints = rollout[:, t]
        action = np.zeros(9)
        
        action[0]   =  np.pi*0.0 + ctrl_joints[0] 
        action[1]   =  np.pi*0.2 + ctrl_joints[1] 
        action[2]   =  np.pi*0.0 + ctrl_joints[2] 
        action[3]   = -np.pi*0.2 + ctrl_joints[3] 
        action[4:7] =  np.pi*0.0 + ctrl_joints[4:7] 
        action[7:] = ctrl_joints[7:] 
        
        # do the movement
        state, r, done, info_ = env.step(action)
        
        print("reward: %8.4f" % r)

        if len(info_["contacts"]) > 0:
            print(info_["contacts"])
       
        eye_rgb = env.eyes["eye0"].render(env.robot.parts["finger_01"].get_position())
        p.imshow(eye_rgb)
        
        if not args.save:  
            time.sleep(1/200)
        else:
            rgb = env.render("rgb_array")
            im = Image.fromarray(rgb)
            im.save(path + "/frame_{:04d}.jpeg".format(t))
            
            im = Image.fromarray(eye_rgb)
            im.save(path + "/eye_{:04d}.jpeg".format(t))
            t += 1


    

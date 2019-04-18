import gym, os, glob 
import time
import argparse
from simulation_settings import *

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
    env._render_width = 640
    env._render_height = 480
    env._cam_yaw = 180
    env.robot.used_objects = ["table", "tomato", "mustard", "orange"]
    env.reward_func = GraspRewardFunc()
    env.robot.contact_threshold = contact_threshold
    env.render("human")
    env.reset()

    rollout = init_trj(rollout)

    sim = Simulator(rollout, env, plot=not args.save, save=args.save)
    rews = []
    for t in range(len(rollout.T)): 
        sim.step()
        contacts = sim.state[3]["contacts"]
        reward = sim.state[1]
        if(len(contacts.items())>0):
            print(contacts)
        print("reward %6.3f:"%reward)



    

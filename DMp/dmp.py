# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import os

def init_rng():
    ''' Set a random number generator witha random seed
    '''
    seed = np.frombuffer(os.urandom(4), dtype=np.uint32)[0]
    rng = np.random.RandomState(seed)
    
    return rng, seed

def gauss(x, c, s) :
    return np.exp( -(1/(2*s**2))*(x - c)**2  )

class DMP(object) :
    """ Implements a 1D dynamical movememt primitive

    tau*ddy = alpha_ddy*(beta_ddy*(g - y) -dy) + f
    tau*dx = -alpha_x*x
    
    f =  (sum_i(psi_i(x)*theta_i) / sum_i(psi_i(x)) )*x*(g - y0)
    psi_i(k) = exp( - (1/(2*sigma_i**2))*(k - c_i)**2 )

    """

    def __init__(self, n = 30, s = 0, g = 1, stime = 200, dt = 0.01, 
            sigma = 0.01, rng = None, noise = None, n_sigma = 0.02) :
        """
        :param n: Number of parameters of the forcing component
        :param s: starting point
        :param g: end point
        :param stime: timesteps
        :param dt: integration time
        :param sigma: std dev of the gaussian bases
        :param noise: add noise to the output
        :param n_sigma: noise std dev
        :type n: int
        :type s: float
        :type g: float
        :type noise: bool    
        :type sigma: float    
        
        """
        self.n = n
        self.s = s
        self.g = g
        self.stime = stime
        self.noise = noise
        self.n_sigma = n_sigma
 
        # init random number generator 
        if rng is None:
            rng,_ = init_rng()
        self.rng = rng
        
        # forcing component params
        self.theta =  rng.randn(n)
        self.c = np.linspace(0,1, n)
        self.sigma = sigma
        
        self.dt = dt
        self.tau = 0.05*self.stime*self.dt
       
        # canonical system params
        self.x0 = 1
        self.alpha_x = 0.2

        # PD params
        self.y0 = s
        
        self.alpha_ddy = 3.0*self.alpha_x 
        self.beta_ddy = self.alpha_ddy/4.0

        self.reset()

    def set_start(self, start):
        # PD params
        self.y0 = start 
    
    def set_goal(self, goal):
        self.g = goal


    def get_bases(self, x):
        """ Computes the bases of a state x

        :param x: the current state of the canonical system
        :type x: float

        :return: an array of activations of the n bases
        :rtype: float
        """
        phi = np.array([ gauss(x, self.c[i], self.sigma) 
            for i in range(self.n) ])

        return phi

    def reset(self) :       
        
        # state storage
        self.S = { "ddy": np.zeros(self.stime),
              "dy": np.zeros(self.stime),
              "y": np.zeros(self.stime),
              "x": np.zeros(self.stime),
              "phi": np.zeros([self.n, self.stime]) 
              }
    
    def rollout(self) :
        """
        Performs a single episode of 'stime' timesteps

        :return: a dictionary with the timeseries of 
                    ddy (acceleration), 
                    dy (speed), -
                    y (position),
                    x (time-setting decay,
                    phi (vector of bases activations)
        :rtype: dict( str : np.array() )
        """
        
        # reset vars
        self.x = self.x0
        self.y = self.y0
        self.dy = 0
        self.ddy = 0

        for t in range(self.stime):

            # forcing component
            phi = self.get_bases(self.x)
            fc =  ( phi/phi.sum())
            fc *= self.x
            fc *= (self.g - self.y0)
        
            # PD acceleration
            pd =  self.alpha_ddy*(self.beta_ddy*(self.g - self.y) - self.dy)  
            
            # increment of the transformation system
            self.ddy = (self.dt/self.tau)*(pd + np.dot(fc, self.theta))

            if self.noise :
                self.ddy = self.ddy + self.rng.randn()*self.n_sigma

            # increment of the canonical system
            dx = -(self.dt/self.tau)*self.alpha_x*self.x

            # updates
            self.dy += self.ddy    # transformation system derivative 
            self.y += (self.dt/self.tau)*self.dy    # transformation system  
            self.x += dx   # canonical system

            # storage
            self.S["ddy"][t] = self.ddy
            self.S["dy"][t] = self.dy
            self.S["y"][t] = self.y
            self.S["x"][t] = self.x
            self.S["phi"][:,t] = phi
        
        return self.S

    def lwr(self, target) :
        ''' Locally weighted regression to learn the weights of 
            the forcing component
        '''

        s = self.S["x"]*(self.g -  self.y0)
        
        # target forrcing component
        ft = (self.tau**2)*target["ddy"] \
                - self.alpha_ddy*(self.beta_ddy*(self.g - target["y"]) \
                - self.tau*target["dy"])  

        # bases of the whole x timeseries
        phi =  self.S["phi"]

        # locally weighted regression
        for i,p in enumerate(phi):
            w = np.diag(p)
            self.theta[i] =  np.dot(s,np.dot(w, ft)) / np.dot(s, np.dot(w, s))



if __name__ == "__main__" :

    
    stime = 200
    dt = 0.01
    period = 10*np.pi
    x = np.linspace(0,1, stime+2)
    t = (0.5*np.sin(x*period-np.pi/2.0) + 0.5)*(1-x) + x
    t = t/t.max()

    target = { 
            "ddy": np.diff(np.diff(t))/(dt**2),
            "dy":np.diff(t)[1:]/dt,
            "y": t[2:] }

    plt.close("all")

    dmp = DMP(n=800, stime = stime, sigma = 0.0005, 
            dt = dt, noise = True,
            n_sigma = 0.005 )
    dmp.rollout()

    plt.figure(figsize=(8,16))    
    plt.subplot(211)
    plt.title("before regression")
    plt.plot(dmp.S["ddy"], c="red")
    plt.plot(dmp.S["dy"], c="blue")
    plt.plot(dmp.S["y"], c="black", lw=3)
    plt.plot(dmp.S["x"], c="gray",lw=2)
    plt.plot(target["y"], c="red", lw=3)
    plt.plot([0,stime],[1,1], c="gray")
    plt.plot([0,stime],[0,0], c="gray")
    plt.ylim([-0.5,1.5])
    
    dmp.lwr(target)
    dmp.rollout()
    
    plt.subplot(212)
    plt.title("after regression")
    plt.plot(dmp.S["ddy"], c="green")
    plt.plot(dmp.S["dy"], c="blue")
    plt.plot(dmp.S["y"], c="black", lw=3)
    plt.plot(target["y"], c="red", lw=3)
    plt.plot([0,stime],[1,1], c="gray")
    plt.plot([0,stime],[0,0], c="gray")
    plt.ylim([-0.5,1.5])
    
    plt.show()

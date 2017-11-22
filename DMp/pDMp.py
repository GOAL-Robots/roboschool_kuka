# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import os

def init_rng():
    '''
    Set a random number generator witha random seed
    '''
    seed = np.fromstring(os.urandom(4), dtype=np.uint32)[0]
    rng = np.random.RandomState(seed)
    
    return rng, seed

def gauss(x, c, s) :
    return np.exp( -(1/(2*s**2))*(x - c)**2  )

def ngrid(npts, mins, maxs):
    assert len(npts) == len(mins)
    assert len(mins) == len(maxs)
    grids = np.meshgrid(*[np.linspace(i,j,n) for n,i,j in zip(npts, mins, maxs)])
    sparse_grid =  np.vstack([grid.ravel() for grid in  grids]).T 
    return sparse_grid

class DMP :
    """
    implements a 1D parametrized dynamical movememt primitive


    """

    def __init__(self, n = 30, p = 0, pdim = None, s = 0, g = 1, stime = 200, dt = 0.01, sigma = 0.01, 
                 rng = None, noise = None, n_sigma = 0.02) :
        """
        :param  n       Number of parameters of the forcing component
        :param  p       Number of additional parameters
        :param  pdim    list of the bin number for each additional parameter
        :param  s       starting point
        :param  g       end point
        :param  stime   timesteps
        :param  dt      integration time
        :param  sigma   std dev of the gaussian bases
        :param  noise   add noise to the output
        :param  n_sigma noise std dev
        :type   n       int
        :type   s       float
        :type   g       float
        :type   noise   bool    
        :type   sigma   float    
        
        """
        self.n = n
        self.p = p
        self.s = s
        self.g = g
        self.stime = stime
        self.noise = noise
        self.n_sigma = n_sigma
 
        # init random number generator 
        if rng is None:
            rng,_ = init_rng()
        self.rng = rng
        

        # centroids
        if pdim is None:
            pdim = [10 for x in range(p)]
        self.c = ngrid(pdim +[n], 
                       np.zeros(p + 1), 
                       np.ones(p + 1))

        # forcing component params
        self.theta =  rng.randn(*self.c.shape)
        
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

        # state storage
        self.S = { "ddy": np.zeros(self.stime),
              "dy": np.zeros(self.stime),
              "y": np.zeros(self.stime),
              "x": np.zeros(self.stime),
              "phi": np.zeros([self.stime] + list(self.c.shape)) 
              }

    def set_start(self, start):
        # PD params
        self.y0 = start 
    
    def set_goal(self, goal):
        self.g = goal


    def get_bases(self, x, p=None):
        """
        Computes the bases of a state x

        :param x: the current state of the canonical system
        :type x: float
        :param x: the current state of the additional parameters
        :type p: np.array(float)

        :return an array of activations of the n bases
        :rtype float
        """
        if p is None:
            phi = gauss(np.hstack((np.zeros(len(c) - 1), x)), self.c, self.sigma) 
        else:
            phi = gauss(np.hstack((p, x)), self.c, self.sigma) 
            
        return phi

    def reset(self) :       
        
        # state storage
        self.S = { "ddy": np.zeros(self.stime),
              "dy": np.zeros(self.stime),
              "y": np.zeros(self.stime),
              "x": np.zeros(self.stime),
              "phi": np.zeros([self.stime] + list(self.c.shape)) 
              }
    
    def rollout(self, p) :
        """
        Performs a single episode of 'stime' timesteps
        
        :param p:   current additional parameters
        
        :return:    a dictionary with the timeseries of 
                    ddy (acceleration), 
                    dy (speed), -
                    y (position),
                    x (time-setting decay,
                    phi (vector of bases activations)
        :rtype:     dict( str : np.array() )
        """
        
        # reset vars
        self.x = self.x0
        self.y = self.y0
        self.dy = 0
        self.ddy = 0
 
        for t in xrange(self.stime):
 
            # forcing component
            phi = self.get_bases(self.x, p)
            fc =  ( phi/phi.sum())
            fc *= self.x
            fc *= (self.g - self.y0)            
         
            # PD acceleration
            pd =  self.alpha_ddy*(self.beta_ddy*(self.g - self.y) - self.dy)  
             
            # increment of the transformation system
            self.ddy = (self.dt/self.tau)*(pd + np.dot(fc.ravel(), self.theta.ravel()))
 
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
            self.S["phi"][t,:] = phi

if __name__ == "__main__" :

    
    stime = 20
    dt = 0.01

    dmp = DMP(n=3, p=1, stime = stime, sigma = 0.1, 
           dt = dt, noise = True,
            n_sigma = 0.005 )
    dmp.rollout(0)


    

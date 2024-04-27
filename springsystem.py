import numpy as np
from integration import Integrate
# import tensorflow as tf

class SpringSystem(Integrate):

    def __init__(self, start_state):
        self.state = start_state
        self.timeStep = 0.05
        self.trueWeights_1 = np.array([0.137, 0.2314, 0.06918, -0.6245, 0.0095, 0.0214])
        self.trueWeights_2 = np.array([0.123, 0.0952, 0.0001, 0.192])
        self.lDelta = 1
        self.substeps = 1
        self.recordSTATE = self.state
        self.recordTRUE_UNCERTAINTY = 0
        #Data Recording
        self.TRUE_DELTA_REC = []
        self.m1 = 5
        self.m2 = 1
        self.c0 = 1
        self.c1 = 1
        self.d = 1
        self.A = np.array([[0,1,0,0],
                           [-(self.c0+self.c1)/self.m1 , -(2*self.d)/self.m1, self.c1/self.m1, self.d/self.m1],
                           [0,0,0,1],
                           [self.c1/self.m2 , self.d/self.m2, -self.c1/self.m2, -2*self.d/self.m2]])
        self.B = np.array([[0,0],
                           [1/self.m1,0],
                           [0,0],
                           [0,1/self.m2]])

    def applyCntrl(self, action):
        self.state = self.simModel(action)

    
    def dynamicsSpringSystem(self, state, action):
        x1,x2,x3,x4 = state[0],state[1],state[2],state[3]
        delta_1 = self.trueWeights_1[0]+ self.trueWeights_1[1]*x1 + self.trueWeights_1[2]*x2 + self.trueWeights_1[3]*x1*(x2**2) + self.trueWeights_1[4]*x3*x4 + self.trueWeights_1[5]*(x3**2)*x4
        delta_2 = self.trueWeights_2[0]+ self.trueWeights_2[1]*x3 + self.trueWeights_2[2]*(x4**3) + self.trueWeights_2[3]*x1*x3 
        delta = np.array([[delta_1[0]],
                          [delta_2[0]]])
        self.TRUE_DELTA_REC.append(delta)
        xdot = self.A @ state + self.B @ (action + delta)
        return xdot
    
    def simModel(self, action):
        xstep = self.euler(self.dynamicsSpringSystem, self.state, action, self.timeStep, self.substeps)
        return xstep







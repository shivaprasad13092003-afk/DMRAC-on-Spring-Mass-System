import numpy as np
from integration import Integrate

class refModel(Integrate):

    def __init__(self, start_state):
        self.state = start_state
        self.timeStep = 0.05
        self.substeps = 1
        self.recordSTATE = self.state
        self.Am = np.array([[0,1,0,0],
                           [-25,-10,0,0],
                           [0,0,0,1],
                           [0,0,-25,-10]])
        self.Bm = np.array([[0,0],
                           [25,0],
                           [0,0],
                           [0,25]])

    def stepRefModel(self, ref_signal):
        self.state = self.simModel(ref_signal)

    def dynamicRefModel(self, state, ref_signal):
        xdot = self.Am @ state + self.Bm @ (ref_signal.reshape(2,1))
        return xdot

    def simModel(self, ref_signal):

        xstep = self.euler(self.dynamicRefModel, self.state, ref_signal, self.timeStep, self.substeps)
        
        return xstep

import numpy as np
import matplotlib.pyplot as plt
from condensate.core import gpcore

# constants
hbar = 1.05e-34
a0 = 5.3e-11

class Environment():
    """
    The Environment class contains everything that is not directly related to the wavefunction.
    This includes potentials, spoons, dimensions, numbers of atoms, and more
    Many of the parameters do not change over an experiment. Others are settings for various experiments.
    Fixed parameters:
        - DIM: dimension of the square grid
        - fov: field of view of the calculation, in meters
        - N: number of atoms

    Variable parameters:
        - V: external potential
        - reference frame: this sets whether the calculation is in the lab frame or in the rotating frame
        - edge: this adds an edge potential to the calculation
        - spoon: this adds a spoon, controlled by either the mouse or a Leap Motion controller

    """


    def __init__(self, DIM=512, fov=400e-6, N=1e6, omegaz=10, a_s=100*a0, mass=3.8e-26):
        self.DIM = DIM
        self.fov = fov
        self.dx = fov/DIM
        self.N = N
        self._omegaz = omegaz
        self.a_s = a_s
        self.mass = mass
        self.g = N* 4 * np.pi * (hbar**2) * (a_s  / mass)
        self.g *= np.sqrt(mass * omegaz / (2*np.pi*hbar))
        self.omega = 2*np.pi
        self.epsilon = 0
        self.lb = np.sqrt(hbar / (2*mass *self.omega))
        
        self.V = np.zeros((DIM,DIM))
        self.use_custom_V= False
        
        self.reference_frame = {'rotating': False, 'omegaR': [self.omega]}
        self.absorber = {'on': False, 'strength': 1, 'radius': self.fov/2}
        self.edge = {'on': False, 'strength': 5, 'radius': self.fov/2, 'width':self.fov/20}
        self.spoon = {
            'type': None, 'strength':1e5, 'radius': 20e-6,
            'leapx': -250, 'leapy': 500, 'leapxscale': 1, 'leapyscale': 1, 'zcontrol': False
        }
        
        print(f'''
            Running condensate on {DIM}x{DIM} grid with:
            atom number: {N:.0e} 
            mass:        {mass:.2e}
        ''')
        
    def show_potential(self, frame='auto'):
        DIM = self.DIM
        if (frame=='auto' and self.reference_frame['rotating']) or (frame=='rotating'):
            omega = self.reference_frame['omegaR'][-1]
            for i in range(DIM):
                for j in range(DIM):
                    x = (i-DIM//2)*fov / DIM
                    y = (j-DIM//2)*fov / DIM
                    rsq = x**2 + y**2
                    centrif = 0.5 * mass * (omega**2) * rsq
                    self.V[i,j] -= centrif/hbar
        a = plt.contour(self.V)
        plt.gca().set_aspect('equal', 'box')
        plt.show()
        
    def harmonic_potential(self, omega, epsilon=0):
        self.omega = omega
        self.lb = np.sqrt(hbar / (2* self.mass *omega))
        self.epsilon = epsilon
        self.omegaz(np.sqrt(8) * omega)

    def custom_potential(self, V):
        self.V = V
        self.use_custom_V = True
            
    def omegaz(self, omegaz):
        self._omegaz = omegaz
        self.g = self.N* 4 * np.pi * (hbar**2) * (self.a_s  / self.mass)
        self.g *= np.sqrt(self.mass * omegaz / (2*np.pi*hbar))
        return self._omegaz
    
    def rotating_frame(self, omegaR):
        if type(omegaR)==float: omegaR = [omegaR]
        self.reference_frame = {'rotating': True, 'omegaR': omegaR}
    
    def lab_frame(self):
        self.reference_frame['rotating'] = False
            
    def absorbing_boundaries(self, strength, radius):
        self.absorber = {'on': True, 'strength': strength, 'radius': radius}
    
    def no_absorbing_boundaries(self):
        self.absorber['on'] = False
        
    def edge(self, strength, radius, width):
        self.edge = {'on': False, 'strength': strength, 'radius': radius, 'width': width}
    
    def no_edge(self):
        self.edge['on'] = False

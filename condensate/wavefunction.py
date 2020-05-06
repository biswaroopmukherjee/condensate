
from copy import copy, deepcopy
import numpy as np
import matplotlib.pyplot as plt
from condensate.core import gpcore


class Wavefunction():
    """
    The Wavefunction class contains everything directly related to the wavefunction. This includes the density,
    the phase, the full wavefunction, and evolution functions. 
    Experimental parameters are set using the Environment class
    Evolution functions:
        - evolve: the main evolution function
        - relax: the same as evolve, but in imaginary time. This brings the condensate into the ground state.
    """

    def __init__(self, environment=None):
        
        self.env = environment if environment else Environment()
        self.Psi = (1+0.j)*np.zeros((self.env.DIM,self.env.DIM))
        self.initialize_Psi()
        
    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result    
        
    @property
    def density(self):
        self._density = self.env.N * np.abs(self.Psi) ** 2
        return self._density
    
    @property
    def phase(self):
        self._phase = np.angle(self.Psi) * (self.density >1)
        return self._phase
    
        
    def show_density(self,):
        a = plt.imshow(self.density)
        plt.colorbar()
        plt.show()
        
    def show_phase(self,):
        a = plt.imshow(self.phase, cmap='twilight')
        plt.colorbar()
        plt.show()
        

    def initialize_Psi(self, width=100, vortexnumber=0):
        DIM = self.env.DIM
        x = (1+0.j)*np.zeros((DIM,DIM))
        for i in range(DIM):
            for j in range(DIM):
                phase = 1
                if vortexnumber:
                    phi = vortexnumber * np.arctan2((i-DIM//2), (j-DIM//2))
                    phase = np.exp(1.j * np.mod(phi,2*np.pi))
                self.Psi[i,j] = np.exp(-( (i-DIM//2)/width)** 2.  -  ((j-DIM//2)/width)** 2. ) + 1.j
                self.Psi[i,j] *= phase
        
     
    def relax(self, **kwargs):
        kwargs['imaginary_time'] = True
        self.evolve(**kwargs)

    
    def evolve(self, dt=1e-4, steps=1000, imaginary_time=False, cooling=0.01,
               showevery=40, show=True, vmax='auto'):
        
        gpcore.Setup(self.env.DIM, self.env.fov, self.env.g, dt, imaginary_time, cooling)
        
        gpcore.SetHarmonicPotential(self.env.omega, self.env.epsilon)
        
        if self.env.edge['on']:
            gpcore.SetEdgePotential(self.env.edge['strength'], self.env.edge['radius'], self.env.edge['width'])
            
        gpcore.GetPotential(self.env.V)
        
        if self.env.absorber['on']:
            gpcore.AbsorbingBoundaryConditions(self.env.absorber['strength'], self.env.absorber['radius'])
        
        if self.env.reference_frame['rotating']:
            omegaR = self.env.reference_frame['omegaR']
            if (steps!=0) and (len(omegaR)!=steps): 
                raise ValueError('Check the length of omegaR (it should be steps)')
            gpcore.RotatingFrame(omegaR)
        
        if self.env.spoon['type']=='mouse':
            gpcore.SetupSpoon(self.env.spoon['strength'], self.env.spoon['radius'])
        elif self.env.spoon['type']=='leap':
            gpcore.SetupSpoon(self.env.spoon['strength'], self.env.spoon['radius'])
            gpcore.SetupLeapMotion(self.env.spoon['leapx'], 
                                   self.env.spoon['leapy'],
                                   self.env.spoon['leapxscale'], 
                                   self.env.spoon['leapyscale'],
                                   self.env.spoon['zcontrol'])
            
        if vmax=='auto': vmax=np.max(self.density/ self.env.N)
        
        gpcore.Evolve(self.Psi, int(steps), int(showevery), show, vmax)

import condensate 
from condensate import Environment, Potential, Streaming, Data
from condensate.evolution import Evolve, TimeDependent

system = Environment(size=[256,256], atom='Na', extent=7, interaction=0.1, datasave='/data_1', savemode='all')
initialPsidata = Data.import('/initialPsi')
initialPsi = initialPsidata.images[-1]

VHarmonic = Potential('Harmonic', omega=1)
VEllipse = Potential('Harmonic', omega=[1, TimeDependent('sin', omega=0.2, amp=0.1, offset=1)]) #epsilon is ellipticity, angle is the angle of the axis

stream = Streaming(system, initialPsi)
stream.add(Evolve(potential=VHarmonic, dt=1e3, idt=0, streaming=True, frame='lab'))
stream.run()

# stream defs

class Streaming:
    def __init__(self, system, initialPsi=None):
        self.system = system 
        self.evolution = []

    def add(self, evolvestep):
        self.evolution.append(evolvestep)

    def run(self):
        i = 0
        while True:
            [ax, ay] = get_accelerometer()
            theta = get_compass()
            self.evolve(ax, ay, theta)




import condensate 
from condensate import Environment, Potential, Sequential
from condensate.evolution import Evolve, TimeDependent

system = Environment(size=[256,256], atom='Na', extent=7, interaction=0.1, datasave='/data_1', savemode='all')

VHarmonic = Potential('Harmonic', omega=1)
VEllipse = Potential('Harmonic', omega=[1, TimeDependent('sin', omega=0.2, amp=0.1, offset=1)]) #epsilon is ellipticity, angle is the angle of the axis

sequence = Sequential(system)

sequence.add(Evolve(potential=VHarmonic, dt=0, idt=1e-3, steps=1e3, show=True, save=False, frame='lab'))
sequence.add(Evolve(potential=VEllipse, dt=1e3, idt=0, steps=1e5, show=True, save=True, frame='lab')

sequence.run(live=True, saveevery=200, saveframes=False, savemovie=True)




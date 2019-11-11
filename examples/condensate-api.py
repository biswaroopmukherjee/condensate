import condensate 
from condensate.chamber import Environment, Potential
from condensate.evolution import Evolve, Sequential, TimeDependent

system = Environment(size=[256,256], atom='Na', extent=7, interaction=0.1, datasave='/data_1', savemode='all')


VNoEllipse = Potential('Harmonic', omega=1, ellipticity=0)
VEllipse = Potential('Harmonic', omega=1, ellipticity=0.1, angle=0) #epsilon is ellipticity, angle is the angle of the axis
VCircular = Potential('CircularBox', radius=5, sharpness=18, realpart=0, imaginarypart=0.3) #epsilon, imaginary part for ABC
VSpinup = VEllipse + VCircular # should be able to add potentials



sequence = Sequential(system)
# should be able to load a wavefunction
sequence.add(Evolve(potential=VHarmonic, dt=0, idt=1e-3, steps=1e3, show=True, save=False, frame='lab'))
sequence.add(Evolve(potential=VSpinup, dt=1e3, idt=0, steps=1e5, show=True, save=True
                        frame='rotating', Omega=TimeDependent('tanh', max=0.97, ramptime=1e5) ))
sequence.add(Evolve(potential=VSpinup, dt=1e3, idt=0, steps=1e5, show=True, save=True, 
                        frame='rotating', Omega=1))

sequence.run(live=True, saveevery=200, saveframes=False, savemovie=True)




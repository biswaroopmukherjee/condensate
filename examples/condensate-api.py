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




# not sequential running: (do one then the other) 

system = Environment(size=[256,256], 
                     atom='Na',
                     extent=7, 
                     interaction=0.1, 
                     datasave='/data_1', 
                     savemode='all',
                     saveevery=200,
                     savemovie=True,
                     live=True)


wavefunction = Wavefunction(system)
wavefunction.Relax(potential=VHarmonic, dt=1e-3, steps=1e3, frame='lab') # This  should make imaginary time evolution more obvious
wavefunction.Evolve(potential=VHarmonic, dt=1e-3, cooling=0.1) # this should work regardless 
wavefunction.Evolve(potential=Vharmonic, frame='rotating', Omega=TimeDependent('tanh', max=0.97, ramptime=1e5) )

wavefunction.plot_density()
wavefunction.density # <- array of stuff 

# This seems a lot neater - less typing. But the exact sequence isn't 'saved' in an object? maybe Wavefunction.history?


# can also stream with this API:
wavefunction.Stream(source='accelerometer') #<--- this starts the streaming process, a continuous runtime loop



# ideally, timedependent should be replaced with python functions, defined in python and evaluated on the gpu
omega = TimeDependent('tanh', max=0.97, ramptime=1e5) 

# or :
def omega_tanh(t, ramptime=4):
    return np.tanh(t/ramptime)

omega = # anon function to t, 
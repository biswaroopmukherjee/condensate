import condensate 
from condensate import Environment, Potential, Sequential, Wavefunction
from condensate.evolution import Evolve, TimeDependent

system = Environment(size=[256,256], atom='Na', extent=7, interaction=0.1, datasave='/data_1', savemode='all')
initialPsi = Wavefunction('ThomasFermi', radius=0.13, L=7) # L = roughly number of vortices

VHarmonic = Potential('Harmonic', omega=1, ellipticity=0)

spinOmega=0.97

sequence = Sequential(system, initialPsi)

sequence.add(Evolve(potential=VHarmonic, dt=0, idt=1e-3, steps=1e3, show=True, save=False, frame='rotating', Omega=spinOmega))
sequence.add(Evolve(potential=VHarmonic, dt=1e-3, idt=0, steps=1e4, show=True, save=True, frame='rotating', Omega=spinOmega))

sequence.run(live=True, saveevery=200, saveframes=False, savemovie=True)

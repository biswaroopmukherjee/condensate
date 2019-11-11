import condensate 
from condensate import Environment, Potential, Sequential, Wavefunction
from condensate.evolution import Evolve, TimeDependent

system = Environment(size=[256,256], atom='Na', extent=7, interaction=0.1, datasave='/data_1', savemode='all')
initialPsi = Wavefunction('ThomasFermi') # Need to figure out how to initialize boxes

VBox = Potential('CircularBox', radius=9, sharpness=19, height=50)
VSpoon = Potential('Gaussian', sigma=0.2, height=5, 
                center=[TimeDependent('sin', omega=1, amp=1),
                        TimeDependent('cos', omega=1, amp=1) ]) #height can be negative, implying attractive
Vtotal = VBox + VSpoon

sequence = Sequential(system, initialPsi)

sequence.add(Evolve(potential=VBox, dt=0, idt=1e-3, steps=1e3, show=True, save=False, frame='lab'))
sequence.add(Evolve(potential=Vtotal, dt=1e-3, idt=0, steps=1e4, show=True, save=True, frame='lab')

sequence.run(live=True, saveevery=200, saveframes=False, savemovie=True)

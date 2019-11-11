import condensate 
from condensate import Environment, Potential, Sequential
from condensate.evolution import Evolve, TimeDependent

def experiment(omegas, epsilons):

    for Omega in Omegas:
        for epsilon in epsilons:

            system = Environment(size=[256,256], atom='Na', extent=7, interaction=0.1, 
                                datasave='/data_'+str(Omega)+'-'+str(epsilon), savemode='all')
            # should be able to save a dict into the data object          
            system.params={'omega':omega}

            VEllipse = Potential('Harmonic', omega=1, ellipticity=epsilon, angle=0) #epsilon is ellipticity, angle is the angle of the axis



            sequence = Sequential(system)
            # Evolve to ground state in imaginary time
            sequence.add(Evolve(potential=VEllipse, dt=0, idt=1e-3, steps=1e3, show=False, save=False, frame='lab'))
            # Spinup
            sequence.add(Evolve(potential=VEllipse, dt=1e3, idt=0, steps=1e5, show=True, save=False
                                    frame='rotating', Omega=TimeDependent('tanh', max=Omega, ramptime=1e5) ))
            # Hold
            sequence.add(Evolve(potential=VEllipse, dt=1e3, idt=0, steps=5e4, show=True, save=True, 
                                    frame='rotating', Omega=Omega))
            sequence.run(live=True, saveevery=200, saveframes=False, savemovie=True)

            




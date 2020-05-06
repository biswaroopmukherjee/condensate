import sys, os, time
sys.path.append(os.path.abspath('.'))

from condensate import Wavefunction, Environment

# setup the environment
env = Environment(DIM=512, fov=400e-6, N=2e6)
env.harmonic_potential(omega=10)
env.spoon['type'] = 'mouse'

# Relax to the groundstate and run in realtime
wf = Wavefunction(env)
wf.relax(vmax=3e7)

wf.evolve(steps=0, cooling=0.05)

time.sleep(10)
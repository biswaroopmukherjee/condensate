import sys, os
sys.path.append(os.path.abspath('..'))

from condensate.core import gpcore

print(gpcore.fft2d(6000, 10))
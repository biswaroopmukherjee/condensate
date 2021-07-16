import numpy as np
from IPython.display import HTML
from base64 import b64encode
import matplotlib.pyplot as plt
from skimage.filters import gaussian as gaussblur
import warnings
warnings.filterwarnings('ignore')

def set_plt_font_size(size=14):
    SMALL_SIZE = size
    MEDIUM_SIZE = size+1
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title


def linear(x,m,b): return m * x + b

def radial_profile(data, center=None):
  '''Azimuthal averaging
  '''
  y, x = np.indices((data.shape))
  if not center:
    center = [data.shape[0]//2, data.shape[1]//2]
  r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
  r = r.astype(np.int)
  tbin = np.bincount(r.ravel(), data.ravel())
  nr = np.bincount(r.ravel())
  radialprofile = tbin / nr
  return radialprofile 

def show_video(filename='output.mp4', size=300):
  # Allows us to view mp4 videos inline in the notebook
  mp4 = open(filename,'rb').read()
  data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
  return HTML(f"""
  <video width={size:d} controls>
        <source src="{data_url}" type="video/mp4">
  </video>
  """)

def phase_vortex_finder(inputPsi,
                        blursigma=10,
                        blurthreshold=0.5,
                        debug=False):
    '''
    Finds vortices using the phase. Steps:
    1) blur the density to make a blurmask
    2) find points where the curl of the velocity field (dphi) is nonzero, within the blurmask
    '''
    DIM=len(inputPsi)
    test = inputPsi.copy() / 1e3
    
    blurmask = gaussblur(np.abs(test), sigma=blursigma) > blurthreshold
    
    
        
    vorts = np.zeros((DIM,DIM))

    for i in range(DIM-1):
        for j in range(DIM-1):
            if blurmask[i,j]:
                g = np.zeros(4)
                dphi = np.zeros(4)
                g[0] = (test[i,j]     / test[i+1,j])   * (np.abs(test[i+1, j])   / np.abs(test[i,j]))
                g[1] = (test[i+1,j]   / test[i+1,j+1]) * (np.abs(test[i+1, j+1]) / np.abs(test[i+1,j]))
                g[2] = (test[i+1,j+1] / test[i,j+1])   * (np.abs(test[i, j+1])   / np.abs(test[i+1,j+1]))
                g[3] = (test[i,j+1]   / test[i,j])     * (np.abs(test[i, j])     / np.abs(test[i,j+1]))
                for k in range(4):
                    dphi[k] = np.angle(g[k])
                total = sum(dphi)
                vorts[i,j] = total / 2*np.pi
                
    if debug:
        image = np.abs(test) * blurmask
        image /= np.max(image)
        f,axarr = plt.subplots(figsize=(12,5), ncols=2)
        im = axarr[0].imshow(image, vmax=1)
        plt.colorbar(im, ax=axarr[0])
        axarr[0].imshow(vorts,cmap='gist_gray', alpha=vorts, vmax=.1)
        axarr[1].imshow(np.angle(test) * blurmask, cmap='bwr')
        plt.show()
        
    return vorts
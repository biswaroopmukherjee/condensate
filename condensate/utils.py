import numpy as np
from IPython.display import HTML
from base64 import b64encode
import matplotlib.pyplot as plt

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
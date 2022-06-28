import superimport

import numpy as np
import matplotlib.pyplot as plt

import pyprobml_utils as pml


from mpl_toolkits.mplot3d import proj3d
from scipy.stats import dirichlet

grain = 100 # 20 #how many points along each axis to plot
edgedist = 0.005 # 0.008 #How close to an extreme value of say [1,0,0] are we willing to plot.
weight = np.linspace(0, 1, grain)

#Most extreme corners of the sample space
Corner1 = np.array([1.0 - edgedist*2, edgedist, edgedist])
Corner2 = np.array([edgedist, 1.0 - edgedist*2, edgedist])
Corner3 = np.array([edgedist, edgedist, 1.0 - edgedist*2])

#Probability density function that accepts 2D coordiantes
def dpdf(v1,v2, alphavec):
    if (v1 + v2)>1:
        return np.nan
    vec = v1 * Corner1 + v2 * Corner2 + (1.0 - v1 - v2)*Corner3
    return dirichlet.pdf(vec, alphavec)
    
#Dirichlet parameter
alphas = [ [20,20,20], [3,3,20], [0.1,0.1,0.1] ]
azim = 20
for alpha_ in alphas:
    alphavec = np.array(alpha_)
    probs = np.array([dpdf(v1, v2, alphavec) for v1 in weight for v2 in weight]).reshape(-1,grain)

    #fig = plt.figure(figsize=(20,15))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X,Y = np.meshgrid(weight, weight)
    ax.plot_surface(Y, X, probs, cmap = 'jet', vmin=0, vmax=3,rstride=1,cstride=1, linewidth=0)
    ax.view_init(elev=25, azim=azim)
    ax.set_zlabel('p')
    ttl = ','.join(['{:0.2f}'.format(d) for d in alphavec])
    ax.set_title(ttl, fontsize=14)
    alpha = int(np.round(alphavec[0]*10))
    plt.tight_layout()
    pml.savefig(f'dirSimplexAlpha{alpha}Legible.png')
    plt.show()

###################################################################################
#                   Module imports (standard and custom modules)                  #
###################################################################################
import os, sys, gc, h5py, copy, pdb
import numpy as np, math
import matplotlib as mpl
#mpl.use('agg') # i native system has no gui interface
import matplotlib.pyplot as plt, matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from multiprocessing import Pool
import operator

#custom functions import 
visulaisation_code_folder = "/home/s4318421/git-cerberus/cerberus/vis/" #/home/kyriakos/Documents    /Code/000_cerberus_dev/githubRelease-cerberus/cerberus/vis/" # current cerberus visualisation and     outupt data access directory.                                                                   
derived_functions_code_folder ="./" #/media/H_drive/000_PhD/001_SimulationDiagnostics/000_BackupP    ostProcessingTools"


if visulaisation_code_folder not in sys.path:
  sys.path.insert(0, visulaisation_code_folder)
if derived_functions_code_folder not in sys.path:
  sys.path.insert(0, derived_functions_code_folder)

import PHM_MFP_Solver_Post_functions_v6 as phmmfp # running version 3
from get_boxlib import ReadBoxLib, get_files #from get_hdf5_data import ReadHDF5 # if hdf5 files used
from get_hdf5_data import ReadHDF5 # used for the HRMI 

###################################################################################
#                  do stuff 
###################################################################################
dataDir = "/media/kyriakos/Expansion/222_TINAROO_BACKUP/HLLC_Simulations_Production_Quality/Z-Correction-2048-FB-ANISO-Option-16"
processedDir = "/media/kyriakos/Expansion/222_TINAROO_BACKUP/HLLC_Simulations_Production_Quality/Z-Correction-2048-FB-ANISO-Option-16/SRMI-OP-16-Res-2048-FB-ANISO_level=-1.h5"

processedFiles = get_files(processedDir, include=[".h5"], get_all=False) 
useNprocs = 6

t_data, interface_ion = phmmfp.get_1D_time_series_data(processedFiles, species='ions', 
                          quantity='interface_location', nproc=useNprocs, cumsum=False)
t_data, interface_ele = phmmfp.get_1D_time_series_data(processedFiles, species='electrons', 
                          quantity='interface_location', nproc=useNprocs, cumsum=False)

nCellsY = len(interface_ele[-1].keys())
y_plot = np.linspace(0,1,nCellsY)
t_plot = np.linspace(0, 1, len(processedFiles))
interface_data = {}
jSamples = [0, int(nCellsY/2)] #, int(nCellsY/4), int(nCellsY/3)
for j in jSamples:
  interface_data[j, 'ion'] = np.linspace(0, 1, len(processedFiles))
  interface_data[j, 'electron'] = np.linspace(0, 1, len(processedFiles))
  interface_data[j, 'diffStart'] = np.linspace(0, 1, len(processedFiles))
  interface_data[j, 'diffEnd'] = np.linspace(0, 1, len(processedFiles))

for i in range(len(processedFiles)): # time index 
  for j in jSamples: # y index
    interface_data[j, 'electron'][i] = interface_ele[i][j][0][0] # start
    # end interface_ele[i][j][1][0] 

    interface_data[j, 'ion'][i] = interface_ion[i][j][0][0] # start
    # end interface_ele[i][j][1][0] 

    interface_data[j, 'diffStart'][i] = interface_ion[i][j][0][0] - interface_ele[i][j][0][0]

    interface_data[j, 'diffEnd'][i] = interface_ion[i][j][1][0] - interface_ele[i][j][1][0]

fig = plt.figure()
gs = gridspec.GridSpec(1, 2) 
ax1 = fig.add_subplot(gs[0,0]);
ax2 = fig.add_subplot(gs[0,1]);

nameAx = {'electron':ax1, 'ion':ax1, 'diffStart':ax2, 'diffEnd':ax2}

nameColours = {'electron':'r', 'ion':'k', 'diffStart':'b', 'diffEnd':'r'}

locLines = {0:'-', int(nCellsY/3):'--', int(nCellsY/4):'dashdot', int(nCellsY/2):'dotted'}


for name in ['electron', 'ion', 'diffStart', 'diffEnd']:
  for j in jSamples:
    nameAx[name].plot(t_plot, interface_data[j, name], label=name+' loc ' + str(j), linestyle=locLines[j], color=nameColours[name])

ax1.legend()
ax2.legend()
ax2.set_ylim(-0.003, 0.003)
fig.savefig("interface_offsets.png")



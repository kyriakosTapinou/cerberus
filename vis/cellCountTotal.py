import os, sys, gc, h5py, copy, pdb
import numpy as np, math
import matplotlib as mpl
#mpl.use('agg') # i native system has no gui interface
import matplotlib.pyplot as plt, matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from multiprocessing import Pool
import operator

visulaisation_code_folder = "/home/kyriakos/Documents/Code/000_cerberus_dev/githubRelease-cerberus/cerberus/vis"

if visulaisation_code_folder not in sys.path:
  sys.path.insert(0, visulaisation_code_folder)

import PHM_MFP_Solver_Post_functions_v6 as phmmfp # running version 3
from get_boxlib import ReadBoxLib, get_files #from get_hdf5_data import ReadHDF5 # if hdf5 files used
#from get_hdf5_data import ReadHDF5 # used for the HRMI 

import ast

dataFiles = [
  #"/home/kyriakos/Documents/Code/000_cerberus_dev/githubRelease-cerberus/cerberus/Exec/testing/ScalingStudy", 
  #"/media/kyriakos/Expansion/333_Alternative_REGIMES/OPTION_44_RES_512_FB_ANISO_CLEAN", 
  #"/media/kyriakos/Expansion/333_Alternative_REGIMES/OPTION_44_RES_512_FB_ISO_CLEAN"

  #"/media/kyriakos/Expansion/111_Magnetised_BRAGINSKII_RMI/44_X_beta_0p001_IDEAL_RES_2048_ref_4_4_2_2", 
  #"/media/kyriakos/Expansion/111_Magnetised_BRAGINSKII_RMI/37_X_beta_0p001_INTER_A_RES_2048_ref_4_4_2_2",  
  #"/media/kyriakos/Expansion/111_Magnetised_BRAGINSKII_RMI/44_X_beta_0p001_INTER_A_RES_2048_ref_4_4_2_2/"

  #"/media/kyriakos/Expansion/222_TINAROO_BACKUP/HLLC_Simulations_Production_Quality/Z-Correction-2048-FB-ANISO-Option-16/", 
  #"/media/kyriakos/Expansion/222_TINAROO_BACKUP/HLLC_Simulations_Production_Quality/Z-Correction-2048-INTRA-ANISO-Option-16/", 
  #"/media/kyriakos/Expansion/222_TINAROO_BACKUP/HLLC_Simulations_Production_Quality/Z-Correction-2048-INTER-ANISO-Option-16/", 
  #"/media/kyriakos/Expansion/222_TINAROO_BACKUP/HLLC_Simulations_Production_Quality/20220504-Op-16-Clean-Ideal-HLLC/", 

  #"/media/kyriakos/Expansion/111_Magnetised_BRAGINSKII_RMI/44_X_beta_0p001_INTER_A_RES_2048_ref_4_4_2_2/delete1", 
  #"/media/kyriakos/Expansion/111_Magnetised_BRAGINSKII_RMI/44_X_beta_0p001_INTER_A_RES_2048_ref_4_4_2_2/delete2"
  ]

fileExact = [
  #"/media/kyriakos/Expansion/111_Magnetised_BRAGINSKII_RMI/44_X_beta_0p001_IDEAL_RES_2048_ref_4_4_2_2/debugGridding/ref_0p95_debug_MLMG.plt00001", 
  #"/media/kyriakos/Expansion/111_Magnetised_BRAGINSKII_RMI/44_X_beta_0p001_IDEAL_RES_2048_ref_4_4_2_2/debugGridding/ref_0p99_debug_MLMG.plt00001", 
  "/media/kyriakos/Expansion/111_Magnetised_BRAGINSKII_RMI/44_X_beta_0p001_IDEAL_RES_2048_ref_4_4_2_2/dirty2048/gridChange_SRMI-Li3-option-44_xbeta_0p001_IDEAL_DIRTY.plt01281", 
  ]

def calcCellsEachLevel(dataFile):

  #useFile = get_files(dataFile, include=["plt"], exclude=["chk"])[-1]

  useFile = dataFile

  print(useFile)
  rc = ReadBoxLib(useFile)
  
  nLevels = rc.data['n_levels']
  
  cellTotal = 0
  for i in range(nLevels):
    subDir = f"/Level_{i}"
    headerFile = "/Cell_H"
    headerRead = open(useFile + subDir + headerFile, "r")
    blockList = []
    for line in headerRead:
      if line[0:2] == "((":
        #print(line)
        line = line.replace("\n", "")
        line = line.replace(") (", "), (")
        #print(line)

        #print("aa=" + line)
        #pdb.set_trace()
        #exec("aa=" + line)
        #pdb.set_trace()
        #print(aa)
        #pdb.set_trace()
        aa = ast.literal_eval(line)


        blockList.append(aa)
    cellCountLevel = 0
    for block in blockList: # assune 2D only 
  
      nx = block[1][0] - block[0][0] + 1
      ny = block[1][1] - block[0][1] + 1
  
      cellCountLevel += nx*ny

    cellTotal += cellCountLevel
      
    print(f"Level {i} has {cellCountLevel} cells")
  print(f"Total count:\t {cellTotal} cells")
  

#for dataFile in dataFiles:
for dataFile in fileExact:
  print(dataFile)
  calcCellsEachLevel(dataFile)




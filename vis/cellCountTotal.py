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
  ]

fileExact = [
  "/media/kyriakos/Expansion/TRMI_IMPLOSION_M2_RES512_NoMag/512_BOSE_IMPLOSION_TRMI_M_3_nonMag.plt29001"
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




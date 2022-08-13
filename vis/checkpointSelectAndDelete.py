"""
Written by Kyri (sorry lol)

Search checkpoint files for specified time interval, delete those that don't meet the interval to save space
"""

### Imports
import time, sys, os, pdb, numpy as np
visulaisation_code_folder = "/home/s4318421/git-cerberus/cerberus/vis/" 

if visulaisation_code_folder not in sys.path:
  sys.path.insert(0, visulaisation_code_folder)

from get_boxlib import ReadBoxLib, get_files #from get_hdf5_data import ReadHDF5 # if hdf5 files used

def searchDeleteChk(timeInterval, simStartTime, simEndTime, outputDir):
  os.system('pwd')
  os.chdir(outputDir)
  os.system('pwd')
  print(f"\n\nSearching and deleting in {outputDir}")
  outputFiles = get_files(outputDir, include=[".chk"], exclude=[".plt"], get_all=False) 

  #print("debug")
  #outputFiles = outputFiles[:23]
  outputFileTimes = {}
  intervalBin = {} # each interval has a list of fileNames associated 
  intervalFileSave = {}

  intervalDecimals = len(str(timeInterval).split('.')[-1])
  intervalRange = np.arange(simStartTime, simEndTime + timeInterval, timeInterval)

  deleteList = []
  corruptedList = []

  for i in range(intervalRange.shape[0]):
    intervalBin[i] = []
    intervalRange[i] = round(intervalRange[i], intervalDecimals) # floating point errors envountere
  
  for i in range(len(outputFiles)):  
    fileName = outputFiles[i]
    print("search\t", fileName)
    try:
      rc = ReadBoxLib(fileName)
    except: 
      print("Failed read...")
      deleteList.append(fileName)
      corruptedList.append(fileName)
      continue

    outputFileTimes[fileName] = rc.time
    if rc.time > simEndTime: continue # don't touch files outside the range

    print("\t", outputFileTimes[fileName])
    rc.close()

    timeRounded = round(outputFileTimes[fileName], intervalDecimals)
    print("\t", timeRounded)

    intervalAssigned = False
    champDelta = 999
    for j in range(intervalRange.shape[0]):

      deltaInterval = abs(outputFileTimes[fileName] - intervalRange[j])
      if timeRounded == intervalRange[j]: 
        intervalBin[j].append(fileName)
        intervalAssigned = True
        break
      elif deltaInterval < champDelta:
        champIndex = j
        champDelta = deltaInterval

    if not intervalAssigned:
      intervalBin[champIndex].append(fileName)

  print("Find champion")
  saveList = []
  for i in range(intervalRange.shape[0]):
    champName = ""; champDelta = 999
    if len(intervalBin[i]) != 0 :
      for fileName in intervalBin[i]:
        if abs(outputFileTimes[fileName] - intervalRange[i]) < champDelta:
          champDelta = abs(outputFileTimes[fileName] - intervalRange[i])
          champName = fileName
          print(champName)
      intervalFileSave[i] = champName
      saveList.append(champName)
  print("\nKeep these...")
  for i in saveList: 
    print(f"{outputFileTimes[i]}\t{i}")

  print("\nDelete these...")
  for fileName in outputFiles:
    if (fileName not in saveList) and (fileName not in corruptedList) and \
       (outputFileTimes[fileName] <= simEndTime): 
      print(f"{outputFileTimes[fileName]}\t{fileName}")
      deleteList.append(fileName)

  print("\nDelete these also (corrupted)...")
  for fileName in corruptedList:
      print(f"\t{fileName}")

  ### system commands to save and delete 
  print("\n\nUser satisfied with delete list? Y/N...")
  userConfirmation = input()
  if userConfirmation == 'Y': 
    # delete
    print('Deleting and moving stuff')
    os.system('pwd')
    os.system('mkdir savedChkFiles')
    for fileName in saveList:
      cmd = f"mv {fileName} ./savedChkFiles/"
      os.system(cmd)

    for fileName in deleteList:
      cmd = f"rm -r {fileName}"
      os.system(cmd)
  
  return 

### Parameters 

timeIntervalInput = 0.05 
simStartTimeInput = 0.0
simEndTimeInput = 1

outputDirInputs = [
### lo res magnus cleans
#"/media/kyriakos/Expansion/999_RES_512_RUNS/tinaroo_Ideal-Clean-HLLE/Ideal-Clean/",
#"/media/kyriakos/Expansion/999_RES_512_RUNS/magnus_HLLC_SRMI-Option-16-Res-512-INTRA-Isotropic/SRMI-Option-16-Res-512-INTRA-Isotropic/",
#"/media/kyriakos/Expansion/999_RES_512_RUNS/magnus_HLLC_SRMI-Option-16-Res-512-Inter-Anisotropic/SRMI-Option-16-Res-512-Inter-Anisotropic/",
#"/media/kyriakos/Expansion/999_RES_512_RUNS/magnus_HLLC_SRMI-Option-16-Res-512-Inter-Isotropic/SRMI-Option-16-Res-512-Inter-Isotropic/",
#"/media/kyriakos/Expansion/999_RES_512_RUNS/magnus_HLLC_SRMI-Option-16-Res-512-FB-Isotropic/SRMI-Option-16-Res-512-FB-Isotropic/",
#"/media/kyriakos/Expansion/999_RES_512_RUNS/magnus_SRMI-Option-16-Res-512-FB-Anisotropic/"

### HiRes magnus runs 
#"/media/kyriakos/Expansion/222_TINAROO_BACKUP/HLLC_Simulations_Production_Quality/SRMI-Option-16-Res-2048-FB-Anisotropic"
#"/media/kyriakos/Expansion/222_TINAROO_BACKUP/HLLC_Simulations_Production_Quality/SRMI-Option-16-Res-2048-INTRA-Anisotropic"
#"/media/kyriakos/Expansion/222_TINAROO_BACKUP/HLLC_Simulations_Production_Quality/20220427-Op-16-Clean-Intra-Isotropic-HLLC"
#"/media/kyriakos/Expansion/222_TINAROO_BACKUP/HLLC_Simulations_Production_Quality/Op-16-Clean-Inter-Isotropic"
#"/media/kyriakos/Expansion/222_TINAROO_BACKUP/HLLC_Simulations_Production_Quality/Op-16-Clean-Inter-Anisotropic"
"/media/kyriakos/Expansion/222_TINAROO_BACKUP/HLLC_Simulations_Production_Quality/Op-16-Res-2048-Clean-Full-Brag-Isotropic"
# clean the reconnection cases 
#"/media/kyriakos/Expansion/000_MAGNUS_SUPERCOMPUTER_BACKUP/ktapinou/FBR-collisional-anisotropic", 
#"/media/kyriakos/Expansion/000_MAGNUS_SUPERCOMPUTER_BACKUP/ktapinou/FBR-collisional-isotropic/",
]
# "/media/kyriakos/Expansion/999_RES_512_RUNS/magnus_HLLC_SRMI-Option-16-Res-512-INTRA-Anisotropic/SRMI-Option-16-Res-512-INTRA-Anisotropic"

for dirOutput in outputDirInputs:
  searchDeleteChk(timeIntervalInput, simStartTimeInput, simEndTimeInput, dirOutput)

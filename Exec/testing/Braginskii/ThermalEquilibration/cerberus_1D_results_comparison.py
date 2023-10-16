
import sys
cmd_folder = "/home/kyriakos/Documents/Code/000_cerberus_dev/githubRelease-cerberus/cerberus/vis"
if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)
    
from get_boxlib import ReadBoxLib, get_files

import numpy as np
import pylab as plt
import pdb   
# =============================================================================
# 
# =============================================================================
properties = ["rho-", "p-", "T-","x_vel-", "y_vel-", "z_vel-"]

directories = [
#### Braginskii style collisions
#"/home/kyriakos/Documents/Code/000_cerberus_dev/githubRelease-cerberus/cerberus/Exec/testing/Braginskii/ThermalEquilibration/Case_1_files_Jun28-1730",
#"/home/kyriakos/Documents/Code/000_cerberus_dev/githubRelease-cerberus/cerberus/Exec/testing/Braginskii/ThermalEquilibration/Case_1_2_files_Jun28-1811", 
#"/home/kyriakos/Documents/Code/000_cerberus_dev/githubRelease-cerberus/cerberus/Exec/testing/Braginskii/ThermalEquilibration/Case_2_2_files_Jun28-1843", 
#"/home/kyriakos/Documents/Code/000_cerberus_dev/githubRelease-cerberus/cerberus/Exec/testing/Braginskii/ThermalEquilibration/Case_2_1_filesAug08-1742",

### Qi regress 
#"/home/kyriakos/Documents/Code/000_cerberus_dev/githubRelease-cerberus/cerberus/Exec/testing/Braginskii/ThermalEquilibration/Case_2_1_QiRegressAug11-2007", 
#"/home/kyriakos/Documents/Code/000_cerberus_dev/githubRelease-cerberus/cerberus/Exec/testing/Braginskii/ThermalEquilibration/Case_2_2_QiRegressAug11-2008", 


#### Ghosh style collisions 
#"/home/kyriakos/Documents/Code/000_cerberus_dev/githubRelease-cerberus/cerberus/Exec/testing/Collisions/ghoshCollisionsBraginskiiComparison/Case_1_files_Jun28-1728"
#"/home/kyriakos/Documents/Code/000_cerberus_dev/githubRelease-cerberus/cerberus/Exec/testing/Collisions/ghoshCollisionsBraginskiiComparison/Case_1_2_files_Jun28-1816"
#"/home/kyriakos/Documents/Code/000_cerberus_dev/githubRelease-cerberus/cerberus/Exec/testing/Collisions/ghoshCollisionsBraginskiiComparison/Case_2_2_files_Jun28-1844", 

#"/home/kyriakos/Documents/Code/000_cerberus_dev/githubRelease-cerberus/cerberus/Exec/testing/Collisions/ghoshCollisionsBraginskiiComparison/Case_2_1_files_Aug08-1715", 
### altering the Ru component to the correct value i think 
#"/home/kyriakos/Documents/Code/000_cerberus_dev/githubRelease-cerberus/cerberus/Exec/testing/Collisions/ghoshCollisionsBraginskiiComparison/Case_Debug_files_2_1_ghoshAug10-1931", # ghosh collisions 
#"/home/kyriakos/Documents/Code/000_cerberus_dev/githubRelease-cerberus/cerberus/Exec/testing/Collisions/ghoshCollisionsBraginskiiComparison/Case_Debug_files_2_1_ghoshDUAug10-2024", # du style 
#"/home/kyriakos/Documents/Code/000_cerberus_dev/githubRelease-cerberus/cerberus/Exec/testing/Collisions/ghoshCollisionsBraginskiiComparison/Case_Debug_files_2_1_ghoshDU_noIonRUAug10-2037", # du style, no ion RU
#"/home/kyriakos/Documents/Code/000_cerberus_dev/githubRelease-cerberus/cerberus/Exec/testing/Collisions/ghoshCollisionsBraginskiiComparison/Case_Debug_files_2_2_ghoshDUAug10-2125", # du style 
#"/home/kyriakos/Documents/Code/000_cerberus_dev/githubRelease-cerberus/cerberus/Exec/testing/Collisions/ghoshCollisionsBraginskiiComparison/Case_Debug_files_2_2_ghoshDU_noIonRUAug10-2123", # du style, no ion RU


#              "/home/kyriakos/Documents/Code/000_cerberus_dev/githubRelease-cerberus/cerberus/Exec/testing/Collisions/ghoshThermalEquilbration",

#TODO corrected code both braginskii and ghosh
# braginskii  
#"/home/kyriakos/Documents/Code/000_cerberus_dev/githubRelease-cerberus/cerberus/Exec/testing/Braginskii/ThermalEquilibration/Correct_Case_1_1_filesAug12-1756", 
#"/home/kyriakos/Documents/Code/000_cerberus_dev/githubRelease-cerberus/cerberus/Exec/testing/Braginskii/ThermalEquilibration/Correct_Case_1_2_filesAug12-1805", 
#"/home/kyriakos/Documents/Code/000_cerberus_dev/githubRelease-cerberus/cerberus/Exec/testing/Braginskii/ThermalEquilibration/Correct_Case_2_1_files_Aug12-1756", 
#"/home/kyriakos/Documents/Code/000_cerberus_dev/githubRelease-cerberus/cerberus/Exec/testing/Braginskii/ThermalEquilibration/Correct_Case_2_2_files_Aug12-1805", 

#Ghosh
#"/home/kyriakos/Documents/Code/000_cerberus_dev/githubRelease-cerberus/cerberus/Exec/testing/Collisions/ghoshCollisionsBraginskiiComparison/Correct_Case_1_1_files_Aug12-1749", 
#"/home/kyriakos/Documents/Code/000_cerberus_dev/githubRelease-cerberus/cerberus/Exec/testing/Collisions/ghoshCollisionsBraginskiiComparison/Correct_Case_1_2_files_Aug12-1802", 
#"/home/kyriakos/Documents/Code/000_cerberus_dev/githubRelease-cerberus/cerberus/Exec/testing/Collisions/ghoshCollisionsBraginskiiComparison/Correct_Case_2_1_files_Aug12-1749", 
"/home/kyriakos/Documents/Code/000_cerberus_dev/githubRelease-cerberus/cerberus/Exec/testing/Collisions/ghoshCollisionsBraginskiiComparison/Correct_Case_2_2_files_Aug12-1802", 

"/home/kyriakos/Documents/Code/000_cerberus_dev/githubRelease-cerberus/cerberus/Exec/testing/Collisions/ghoshCollisionsBraginskiiComparison/NoEoption_Correct_Case_2_2_files_Aug16-1724" # Daryls original case
    ]

namesList = [["ion", "electron"], ["A", "B"], ["ion", "electron"], ["ion", "electron"]]
nameStyle = {"ion":"dashed", "electron":"dotted", "A":"dashed", "B":"dotted"}

namesList = [["A", "B"], ["A", "B"]]
nameStyle = {"ion":"dashed", "electron":"dotted", "A":"dashed", "B":"dotted"}

#namesList = [["ion", "electron"], ["ion", "electron"], ["A", "B"], ["A", "B"]]
#nameStyle = {"ion":"dashed", "electron":"dotted", "A":"dashed", "B":"dotted"}

#labelList = ["Brag-", "Brag-QiRegress", "Ghosh-", "Ghosh-noIonRu-"]
labelColour = ["r", "b"]
labelList = ["Brag-", "Ghosh-"]

labelList = ["Ghosh-", "Ghosh-D"]
#labelColour = ["r", 'm', "b", "k"]

data = {}
for i in range(len(directories)):
  # get a list of all the files in this directory
  direc = directories[i]
  files = get_files(direc, include=["plt"], exclude=["chk"], get_all=True)
  
  # get all the names of the different states
  #f = files[0]
  #ds = ReadBoxLib(f)
  names = namesList[i]; #["ions", "electrons"] #sorted(ds.names)
  label = labelList[i];
  n_names = len(names)
  n_times = len(files)
  for name in names:
      data[label + name] = {}
      for prop in properties:
          data[label + name][prop] = []
  data[label+"t"] = []
  print("Reading ", label+"data")
  print("\t", n_names, " names\t", n_times, " n_times")
  for f in files:
      ds = ReadBoxLib(f)
  
      data[label+"t"].append(ds.time)
      
      for name in names:
          for prop in properties:
              x, v = ds.get(prop + name)
              data[label+name][prop].append(v[0])

### total eneergy
gam = 5./3.;
for i in range(len(directories)):
  names = namesList[i]; 
  label = labelList[i];
  data[label] = {}
  aa = np.array(data[label + names[0]]['rho-'])
  data[label]["nrg-total-"] = np.zeros( aa.shape )

  for name in names:
      mv2 = (np.array(data[label + name]["rho-"])*np.array(data[label + name]["x_vel-"]))**2 + \
            (np.array(data[label + name]["rho-"])*np.array(data[label + name]["y_vel-"]))**2 + \
            (np.array(data[label + name]["rho-"])*np.array(data[label + name]["z_vel-"]))**2;
  
      data[label + name]["nrg-"] = np.array(data[label + name]["p-"])/(gam - 1.0) + \
                                  mv2/2/np.array(data[label + name]["rho-"]);
      data[label]["nrg-total-"] += data[label + name]["nrg-"]

fig = plt.figure(figsize=(10,10))
ax = {}

properties[0] = 'nrg-total-'
#properties[0] = 'nrg-'
properties = properties[0:4]
for i in range(len(properties)):
  ax[i] = fig.add_subplot(2,2,i+1)
  ax[i].set_ylabel(properties[i][:-1])
  ax[i].set_xlabel("time")

for i in range(len(directories)):
    #print(i)
    label = labelList[i]; names = namesList[i]; #["ions", "electrons"] #sorted(ds.names)
    useColour = labelColour[i]
    print(label)
    print(names)
    for name in names:
        useLine = nameStyle[name]
        #print "\t", name
        for j in range(len(properties)):
            if properties[j] == 'nrg-total-':
              ax[j].plot(data[label+"t"], data[label][properties[j]], color=useColour, \
                linestyle=useLine, label=label) # marker='x', markersize=2,              
            else:
              #print "\t\t",j 
              ax[j].plot(data[label+"t"], data[label+name][properties[j]], color=useColour, \
                linestyle=useLine, label=label+name) # marker='x', markersize=2, 

for i in range(len(properties)):
  ax[i].legend()

fig.tight_layout()
fig.savefig("20220816_deleteMe.png", dpi=300)
#fig.savefig("20220812_ThermalEquilibration-Case_2_2_mime_100_EnergyConsCorrected.png", dpi=300)
plt.close(fig)
    
    
print("DONE")

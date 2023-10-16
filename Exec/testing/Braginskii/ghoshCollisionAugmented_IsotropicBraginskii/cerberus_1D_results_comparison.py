
import sys
cmd_folder = "../../../../vis"
if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)
    
from get_boxlib import ReadBoxLib, get_files

import numpy as np
import pylab as plt
    
# =============================================================================
# 
# =============================================================================
properties = ["rho-", "p-", "T-","x_vel-"]

""" outdated from three releases ago 
directories = [ 
                "/home/kyriakos/Documents/Code/000_cerberus_dev/cerberus/Exec/testing/Braginskii/ghoshCollisionAugmented/Case_2", 
                "/home/kyriakos/Documents/Code/000_cerberus_dev/cerberus/Exec/testing/Braginskii/ghoshCollisionAugmented/Case_2_corrected", 
                "/home/kyriakos/Documents/Code/000_cerberus_dev/cerberus/Exec/testing/Collisions/ghoshCollisionsBraginskiiComparison/Case_2_corrected"
    ]
"""

directories = [
                "/home/kyriakos/Documents/Code/000_cerberus_dev/cerberus/Exec/testing/Collisions/ghoshCollisionsBraginskiiComparison/Case_2_corrected/",
                "/home/kyriakos/Documents/Code/000_cerberus_dev/cerberus/Exec/testing/Braginskii/ghoshCollisionAugmented/Case_2_corrected", 
                "/home/kyriakos/Documents/Code/000_cerberus_dev/githubRelease-cerberus/cerberus/Exec/testing/Braginskii/ghoshCollisionAugmented_IsotropicBraginskii/"

    ]

namesList =[["A", "B"], ["ions", "electrons"], ["ions", "electrons"], ]

labelList = ["Cerb-collisions-", "Cerb-Brag-aniso", "Cerb-Brag-iso-", ] 
          
data = {}
for i in range(len(directories)):
  # get a list of all the files in this directory
  direc = directories[i]
  files = get_files(direc, include=['plt'], get_all=True)
  
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
  print "Reading ", label+"t" 
  for f in files:
      ds = ReadBoxLib(f)
  
      data[label+"t"].append(ds.time)
      
      for name in names:
          for prop in properties:
              x, v = ds.get(prop + name)
              data[label+name][prop].append(v[0])

fig = plt.figure(figsize=(10,10))
ax = {}

for i in range(len(properties)):
  ax[i] = fig.add_subplot(2,2,i+1)
  ax[i].set_ylabel(properties[i][:-1])
  ax[i].set_xlabel("time")


for i in range(len(directories)):
    label = labelList[i]; names = namesList[i]; #["ions", "electrons"] #sorted(ds.names)
    for name in names:
        for j in range(len(properties)):
            ax[j].plot(data[label+"t"], data[label+name][properties[j]], linestyle="None", 
                       marker="x", markersize=2, label=label+name)

for i in range(len(properties)):
  ax[i].legend()

fig.tight_layout()
fig.savefig("GhoshCollisions-Cerb_collisions-Cerb_brag_aniso-Cerb_brag_iso-Comparison.png", dpi=300)
plt.close(fig)
 
print("DONE")

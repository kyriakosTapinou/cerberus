import os, sys, gc, copy, pdb, math, numpy as np # standard modules
import time 
visulaisation_code_folder ="/home/kyriakos/Documents/Code/000_cerberus_dev/githubRelease-cerberus/cerberus/vis"

if visulaisation_code_folder not in sys.path:
  sys.path.insert(0, visulaisation_code_folder)

import PHM_MFP_Solver_Post_functions_v6 as phmmfp # running version 3
from get_boxlib import ReadBoxLib, get_files

import matplotlib.gridspec as gridspec, matplotlib as mpl
import matplotlib.pyplot as plt 
plt.rcParams['axes.labelsize'] = 8
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8

#mpl.use('agg')
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from multiprocessing import Pool

def sortPltChk(inputString):
  stepKeyword = "step"
  if 'plt' in inputString and 'chk' in inputString:
    print("\nError in get_box_lib sortPltChk function\n")
    return inputString
  if 'plt' not in inputString and 'chk' not in inputString and 'step' not in inputString:
    if "Step" in inputString: 
      stepKeyword = "Step_"
    else:
      print(f"\nError in get_box_lib sortPltChk function: {inputString}\n")
      return inputString

  if stepKeyword in inputString: 
    return int(inputString.split(stepKeyword)[1].split("_level")[0])
  elif 'plt' in inputString: splitOn = 'plt'
  elif 'chk' in inputString: splitOn = 'chk'
  return int(inputString.split(splitOn)[1])

### shortened function for just the average interface
def interp_val(input_alpha, t_low, t_high): # interperate value 
  if input_alpha < 0.5:
    return t_low
  else:   
    return t_high

def get_interface(ch, name):
  #x, alpha = ch.get_flat("alpha-%s"%name) # problem when reading for the first time without intiialiing get of something first 
  x, alpha = ch.get("alpha-%s"%name)

  if alpha.max() > 1. and  alpha.min() > 0.5: pdb.set_trace()
  y= x[1]; x = x[0]; dx = x[1] - x[0]; dy = y[1] - y[0]
  n_x = np.shape(alpha)[0]; n_y = np.shape(alpha)[1]
  #get interface stats for search
  interface_x = 0.0; interface_amp = 0.1; interface_transition = 0.01;
  #averaged interface stats 
  alpha_y_collapsed = np.sum(alpha, 1)  # find more accurate integration, inbuilt funcitons maybe 
  alpha_y_collapsed = alpha_y_collapsed/n_y #p.shape(alpha)[1]  #max_alpha_y_collapsed
  tracer_tol = 0.45
  tracer_low = 0.5 - tracer_tol
  tracer_high = 0.5 + tracer_tol
  # ------------------------Volume of fluid interface
  grab3 = (alpha_y_collapsed > tracer_low) & (alpha_y_collapsed < tracer_high)
  
  avg_int_coords = []; start_switch = 0;
  for i in range( np.shape(grab3)[0]):
    if grab3[i] == True and start_switch ==0: #positive transition
      start_switch = 1.
      int_start = i
      interp_val_1 = interp_val(alpha_y_collapsed[i], tracer_low, tracer_high)

    if grab3[i] == False and start_switch == 1.:
      int_end = i
      start_switch = 0.
      interp_val_2 = interp_val(alpha_y_collapsed[i], tracer_low, tracer_high)
      x1 = np.interp([interp_val_1], alpha_y_collapsed[int_start-1:int_start+1], x[int_start-1:int_start+1])
      x2 = np.interp([interp_val_2], alpha_y_collapsed[int_end-1:int_end+1], x[int_end-1:int_end+1])

      avg_int_coords.append([x1,x2])
      del int_start, int_end, x1, x2 ; gc.collect() ;
      if len(avg_int_coords[-1])%2 != 0:
        print("Interface tracking is defective") 
        print("time\n", ch.time )

  return x, y, avg_int_coords


################################################ INPUTS ########################################
dataDir = "/media/kyriakos/Expansion/222_TINAROO_BACKUP/HLLC_Simulations_Production_Quality/Z-Correction-2048-FB-ANISO-Option-16"
labelAppend = "SRMI_16_FB_ANISO"

dataDir = "/media/kyriakos/Expansion/222_TINAROO_BACKUP/HLLC_Simulations_Production_Quality/Z-Correction-2048-INTER-ISO-Option-16"
labelAppend = "SRMI_16_INTER_ISO_PERTURBATION_EARLY_IONS"

level = -1
nameOne = "ions"
nameTwo = "electrons"

############################################ PARAMATERS ########################################
#### plot some time series of velocity 
timeSeriesSelect = [0.05, 0.075, 0.1, 0.125] # [0.2, 0.4, 0.8, 1.0] #[0.64, 0.72, 0.85, 1.0]
# [0.4, 0.45, 0.5, 0.525 ] #development times 
 
#### set the aspect ratio etc of the figure
fig_x = 6.7
wspacing = 0.02; hspacing = 0.

###search this box 
view_original =  [[-0.2, 0.0], [1.6, 1.0]]
view = copy.deepcopy(view_original)

### zoom settings
zoomHeight = 0.25 #0.25
zoomHack = True
if zoomHack:
  zoomHeightHack = 0.05
  zoomLabel = zoomHeightHack
else:
  zoomLabel = zoomHeight 

view[1][1] = zoomHeight # only the spike
view[0][0] = 0.0 
view[1][0] = zoomHeight 

### ensure the data are the same level etc. 
simFiles = get_files(dataDir,include=["plt"], exclude=["temp", "old", "chk"], get_all=False) 
simFiles = sorted(simFiles, key=sortPltChk)

rc = ReadBoxLib(simFiles[0], level, view)    
xy, rhoOne = rc.get("rho-%s"%nameOne); 
rc.close()

fig_x, fig_y = phmmfp.get_figure_size(fig_x, 2, len(timeSeriesSelect), 
                   1.*xy[1].shape[0]/xy[0].shape[0], wspacing, hspacing, 1)

fig = plt.figure(figsize=(fig_x,fig_y))

gs = gridspec.GridSpec(2, len(timeSeriesSelect), wspace=wspacing, hspace=hspacing)
ax = []

cc = 0
lastColumn = len(timeSeriesSelect) - 1 

forceLimits = True

simFiles = get_files(dataDir, include=["plt"], exclude=["temp", "old", "chk"], get_all=False) 
simFiles = sorted(simFiles, key=sortPltChk)

rc0 = ReadBoxLib(simFiles[0], 0, view); 
rc1 = ReadBoxLib(simFiles[1], 0, view); 
dt = round(rc1.time - rc0.time, 5)
rc0.close(); rc1.close()
print("Time step:\t", dt)
nFiles = []
for tDesired in timeSeriesSelect:
    nFiles.append( int(tDesired/dt ))

cc = 0 

for nFile in nFiles:
    rc = ReadBoxLib(simFiles[nFile], level, view_original)
    time_ = rc.time
    print("Time:\t", time_)
    ##find where the spike is 
    x, y, avg_int_coords = get_interface(rc, nameOne)
    if len(avg_int_coords[0]) == 0: pdb.set_trace()

    interfaceStart =  round(avg_int_coords[0][0][0], 3)
    interfaceEnd = avg_int_coords[0][1][0]
    print(f"Interface starts at {interfaceStart}")

    viewZoomUpper = copy.deepcopy(view_original)
    viewZoomUpper[0][0] = round(interfaceStart - zoomHeight/2, 3)
    viewZoomUpper[1][0] = round(interfaceStart + zoomHeight/2, 3)
    if False: #bias to the right side for 0.5 zoom height 
      viewZoomUpper[0][0] = round(interfaceStart - 0.05, 3) #zoomHeight/2, 3)
      viewZoomUpper[1][0] = round(interfaceStart + (zoomHeight - 0.05), 3) #zoomHeight/2, 3)
     
    viewZoomLower = copy.deepcopy(viewZoomUpper)
    viewZoomUpper[0][1] = 0.5; viewZoomUpper[1][1] = round(0.5 + zoomHeight, 3)
    viewZoomLower[0][1] = round(0.5 - zoomHeight, 3); viewZoomLower[1][1] = 0.5 
    print(f"views used:", viewZoomUpper, viewZoomLower)
    rc.close()
    rcUpper = ReadBoxLib(simFiles[nFile], level, viewZoomUpper)
    rcLower = ReadBoxLib(simFiles[nFile], level, viewZoomLower)

    xyUpper, rhoUpper = rcUpper.get("rho-%s"%nameOne); 
    xy_tracerUpper, alphaOneUpper = rcUpper.get('alpha-%s'%nameOne);
    xyLower, rhoLower = rcLower.get("rho-%s"%nameOne); 
    xy_tracerLower, alphaOneLower = rcLower.get('alpha-%s'%nameOne);

    xyUpper, rhoTwoUpper = rcUpper.get("rho-%s"%nameTwo); 

    xUpper = xyUpper[0]; yUpper = xyUpper[1]
    xLower = xyLower[0]; yLower = xyLower[1]

    #alphaOne = alphaOne/rhoOne
    #tol = 0.45; t_h = 0.5 + tol; t_l = 0.5 - tol
    #maskArrayOne = (t_l <  alphaOne) & (alphaOne < t_h) 

    if False: # get upper plane Jmagnitude
      xy, rhoCi, Jxi, Jyi, Jzi = phmmfp.get_charge_number_density(rcUpper,
                                  nameOne, returnMany = False)
      xy, rhoCe, Jxe, Jye, Jze = phmmfp.get_charge_number_density(rcUpper,
                                  nameTwo, returnMany = False)
      Jmag = np.sqrt( (Jxi+Jxe)**2 + (Jyi+Jye)**2 + (Jzi+Jze)**2)
      print(f"Jmag min max:\t{Jmag.min()}\t{Jmag.max()}")
      #pdb.set_trace()

      del rhoCi, Jxi, Jyi, Jzi, rhoCe, Jxe, Jye, Jze;

    if False: # get lower plane charge density      
      xy, rhoCi, Jxi, Jyi, Jzi = phmmfp.get_charge_number_density(rcLower, 
                                  nameOne, returnMany = False) 
      xy, rhoCe, Jxe, Jye, Jze = phmmfp.get_charge_number_density(rcLower, 
                                  nameTwo, returnMany = False) 
      rhoC = rhoCi + rhoCe
      del rhoCi, Jxi, Jyi, Jzi, rhoCe, Jxe, Jye, Jze;

    if True: # lower plan  yD
      xyLower, yD = rcLower.get("y_D-field"); 
      xyLower, yD = rcLower.get("y_D-field"); 

    yUpper, xUpper = np.meshgrid(yUpper, xUpper) 
    yLower, xLower = np.meshgrid(yLower, xLower) 

    forceJmag = 1.0 # 1.1
    forceRhoI = 9.4 #7.3 #7.4
    forceRhoE = 0.07
    forceRhoC = 1.3 #1.4 # 1.5  
    forceYD = 0.14
    for (x, y, value, rr, cmapValue, zeroMin, forceValue, cbLabel) in \
      [
       #(xUpper, yUpper, Jmag, 0, mpl.cm.Oranges, True, forceJmag, r"$| J |$"), # current den
       (xUpper, yUpper, rhoUpper, 0, mpl.cm.Greens, True, forceRhoI, r"$\rho_i$"), # current den
       #(xUpper, yUpper, rhoTwoUpper, 0, mpl.cm.Greens, True, forceRhoE, r"$\rho_e$"), # current den
       (xLower, yLower, yD, 1, mpl.cm.bwr, False, forceYD, r"$E_y$")
       #(xLower, yLower, rhoC, 1, mpl.cm.bwr, False, forceRhoC, r"$\rho_c$")
      ]:
      # plot the value for each time period
      ax.append(fig.add_subplot(gs[rr,cc]))
      vminValue = value.min(); vmaxValue = value.max()
      valueMaxAbs = max(abs(vminValue), abs(vmaxValue))
      valueMaxAbs = round(valueMaxAbs,3)
      if forceLimits:
        valueMaxAbs = forceValue

      if zeroMin:
        valueMin = 0
        if cmapValue == mpl.cm.Greens: valueMin = 2.0 # value.min(); #2.0   #2.3 #
      else:
        valueMin = -valueMaxAbs

      #hack for limits 
      if zoomHack:
        fac = zoomHeightHack/zoomHeight
        nx = x.shape[0]; ny = y.shape[0]
        nxnew = int(nx*fac); nynew = int(ny*fac)
        nx1 =  int( nx/2 - np.floor(nxnew/2)); nx2 = int( nx/2 + np.ceil(nxnew/2));
        if rr == 0:
          ny1 = 0 ; ny2 = int( nynew);
        elif rr == 1:
          ny1 = int(ny - nynew-1);  ny2 = ny
        x = x[nx1:nx2+1, nx1:nx2+1]
        y = y[ny1:ny2+1, ny1:ny2+1]
        value = value[nx1:nx2+1, ny1:ny2+1]
        #pdb.set_trace()

      ax[-1].pcolormesh(x, y, value, vmin=valueMin, vmax=valueMaxAbs, 
                        cmap=cmapValue, shading='auto')
      if forceLimits:
        cbLoc = "center right"
        cbAnchor = (0.125, 0., 1, 1)
        cbOrientation = "vertical"
      else:
        if rr == 0:
          cbAnchor = (0., 0.125, 1,1)
          cbLoc = "upper center"
        elif rr == 1:
          cbAnchor = (0., -0.125, 1,1)
          cbLoc = "lower center"
  
      if forceLimits and cc == lastColumn:
        div = make_axes_locatable(ax[-1])
        cax = inset_axes(ax[-1], width='5%', height='90%', loc=cbLoc,
                              bbox_to_anchor=cbAnchor, bbox_transform=ax[-1].transAxes, 
                              borderpad=0)
        norm = mpl.colors.Normalize(vmin=valueMin, vmax=valueMaxAbs)
        cb = mpl.colorbar.ColorbarBase(cax, orientation=cbOrientation, 
          cmap=cmapValue, norm=norm, 
          extend="both", ticks=[valueMin, valueMaxAbs], format='%.2f')
        cb.set_label(label=cbLabel, weight='bold', labelpad=-20)
        cb.ax.tick_params(labelsize=8) 
      elif not forceLimits:
        div = make_axes_locatable(ax[-1])
        cax = inset_axes(ax[-1], width='80%', height='10%', loc=cbLoc,
                         bbox_to_anchor=cbAnchor, bbox_transform=ax[-1].transAxes, 
                         borderpad=0)
        norm = mpl.colors.Normalize(vmin=valueMin, vmax=valueMaxAbs)
        cb = mpl.colorbar.ColorbarBase(cax, orientation="horizontal", 
          cmap=cmapValue, norm=norm, 
          extend="both", ticks=[valueMin, valueMaxAbs], format='%.2f')
        cb.set_label(label=cbLabel, weight='bold', labelpad=-40)
        cb.ax.tick_params(labelsize=8) 
  
      ax[-1].set_aspect(1)
      #ax[-1].ax[-1]is('off') 
      ax[-1].set_xticks([]); ax[-1].set_yticks([])
      #ax[-1].contour(x, y, maskArrayOne, levels=[0.5], colors='purple', linewidths=0.25)
      if rr == 0.: 
        ax[-1].title.set_text(f"t={round(time_, 2):.2f}")
        ax[-1].title.set_size(8)
    """
    if cc == 0: ax[-1].set_ylabel(handle)
    """
    cc += 1
  
#fig.subplots_adjust(wspace=0.01, hspace=0.0)
plt.show()
fig.savefig("20220728_" + labelAppend + "_zoom_" + str(zoomLabel).replace('.','p') +
            ".png", dpi=600)

"""
This code takes in the simulation output files (plot file with primitives) and calculates the trans
port influence of the transport coefficients on the conservation equations.

"""

import os, sys, gc, copy, pdb, math, numpy as np # standard modules
import time 
visulaisation_code_folder ="/home/kyriakos/Documents/Code/000_cerberus_dev/githubRelease-cerberus/cerberus/vis"

if visulaisation_code_folder not in sys.path:
  sys.path.insert(0, visulaisation_code_folder)

import PHM_MFP_Solver_Post_functions_v6 as phmmfp # running version 3
from get_boxlib import ReadBoxLib, get_files

import matplotlib.gridspec as gridspec, matplotlib as mpl
import matplotlib.pyplot as plt 
#mpl.use('agg')
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from multiprocessing import Pool

def graphData(dataFile, level, window, view, nameOutput, timeTitle, saturationFactor, plotVorticityCollisional=False):

  ch = ReadBoxLib(dataFile, level, window)
  
  xyRho, rho = ch.get("rho-ions")
  xRho = xyRho[0]; yRho = xyRho[1]; 

  #square up the figure dimensions 
  fig_x = 8.2  #6.7
  wspacing = 0.5 #0.01; 
  hspacing = 0.4 #0.01
  
  fig_x, fig_y = phmmfp.get_figure_size(fig_x, 4, 4, 
                   1.*xyRho[1].shape[0]/xyRho[0].shape[0], wspacing, hspacing, 1)

  #fig_x, fig_y = 6.7, 6.7

  fig = plt.figure(figsize=(fig_x,fig_y))

  #TODO note ignoring Zmom
  gs = gridspec.GridSpec(4, 4, wspace=0.05, hspace=0.05)
  ax = {}
  #ax1 = fig.add_subplot(gs[0,0])
  #ax2 = fig.add_subplot(gs[1,0])
  #ax3 = fig.add_subplot(gs[2,0])
  #ax4 = fig.add_subplot(gs[3,0])
  t1 = time.time()
  print("Calculating transport properties...")
  x, y, dudt_fluxes, srcDst = phmmfp.get_transportProperties(ch, ["ions", "electrons"], level,
    isoOveride=False, useNPROC = 6)
  print("...calculated")
  print(f"\nTook {time.time()-t1} to extract transport effects\n")
  ch = ReadBoxLib(dataFile, level, window)
  pdb.set_trace()
  constOC = (2/ch.data['beta']/ch.data['skin_depth']**2)
  ######=================
  if plotVorticityCollisional:
      fluxDict = {}; srcDict = {};
      for name in ['ions', 'electrons']: #delete useless data
        fluxDict[name] = {}
        srcDict[name] = {}
        # divide by density 
        deleteMe = ch.get("rho-%s"%name)[-1]

        for key in range(dudt_fluxes[name].shape[-1]):
          if key in [0, 1]: # 0:Xmom, 1:Ymom
            fluxDict[name][key] = np.copy(dudt_fluxes[name][:,:,key])/deleteMe*constOC
        for key in range(srcDst[name].shape[-1]):
          if key in [0, 1]: # 0:Xmom, 1:Ymom
            srcDict[name][key] = np.copy(srcDst[name][:,:,key])/deleteMe*constOC
      #del dudt_fluxes, srcDst, deleteMe;
      OC = {}# vorticity vontribution 
      #TODO Change the axis since the web page i was on seems to say axis=0 takes gradient acrosos rows???
      OC['R_i'] = np.gradient(srcDict['ions'][1], x, axis=0) - \
            np.gradient(srcDict['ions'][0], y, axis=1)
      OC['R_e'] = np.gradient(srcDict['electrons'][1], x, axis=0) - \
            np.gradient(srcDict['electrons'][0], y, axis=1)
      del srcDict; gc.collect()
      OC['PI_i'] = np.gradient(fluxDict['ions'][1], x, axis=0) - \
              np.gradient(fluxDict['ions'][0], y, axis=1)
      OC['PI_e'] = \
        np.gradient(fluxDict['electrons'][1], x, axis=0) - \
        np.gradient(fluxDict['electrons'][0], y, axis=1)
      del fluxDict, deleteMe; gc.collect();
      
      ## dirty plot 
      fig2 = plt.figure(figsize=(8, 8))
      gs2 = gridspec.GridSpec(2, 2, wspace=0.05, hspace=0.05)
      ax2 = fig2.add_subplot(gs2[0,0])
      #maxVal = max(abs(
      ax2.pcolormesh(x, y, OC['PI_e'], cmap='bwr')

      ax2 = fig2.add_subplot(gs2[0,1])
      #maxVal = max(abs(
      ax2.pcolormesh(x, y, OC['PI_i'], cmap='bwr')

      ax2 = fig2.add_subplot(gs2[1,0])
      #maxVal = max(abs(
      ax2.pcolormesh(x, y, OC['PI_e'], cmap='bwr')

      ax2 = fig2.add_subplot(gs2[1,1])
      #maxVal = max(abs(
      ax2.pcolormesh(x, y, OC['PI_i'], cmap='bwr')
      fig2.show()
      pdb.set_trace()
  ######=================

  if False:
    print("Calculating anisotropic transport properties...")
    xA, yA, dudt_fluxesA, srcDstA =\
       phmmfp.get_transportProperties(ch, ["ions", "electrons"], level,\
                                      isoOveride=False, useNPROC = 6)
    print("...calculated")

  print("\n\n########3Note Qfric and Qdelta are separate")
  y, x = np.meshgrid(y, x)

  Xmom = 0; Ymom = 1; Zmom = 2; EdenPi = 3; EdenQ = 4;
  maxVal = {}
  minVal = {}
  cmapValue = mpl.cm.PiYG
  """
  label = [[r"$\rho v_X$", r"$\rho v_Y$", r"$\rho v_Z$", r"$\varepsilon_\pi$", r"$\varepsilon_q$"], 
           [r"$\rho v_X$", r"$\rho v_Y$", r"$\rho v_Z$", r"$\varepsilon_\pi$", r"$\varepsilon_q$"], 
           [r"$\rho v_X$", r"$\rho v_Y$", r"$\rho v_Z$", r"$\varepsilon_{U}$", r"$\varepsilon_Q$"],
           [r"$\rho v_X$", r"$\rho v_Y$", r"$\rho v_Z$", r"$\varepsilon_{U}$", r"$\varepsilon_Q$"]]
  """
  """
  label = [[r"$\rho v_X$", r"$\rho v_Y$", r"$\varepsilon_\pi$", r"$\varepsilon_q$"], 
           [r"$\rho v_X$", r"$\rho v_Y$", r"$\varepsilon_\pi$", r"$\varepsilon_q$"], 
           [r"$\rho v_X$", r"$\rho v_Y$", r"$\varepsilon_{U}$", r"$\varepsilon_Q$"],
           #[r"$\rho v_X$", r"$\rho v_Y$", r"$\varepsilon_{U}$", r"$\varepsilon_Q$"]
          ]

  """

  labelIntra = [r"$\rho v_X$", r"$\rho v_Y$", r"$\varepsilon_\pi$", r"$\varepsilon_q$"]
  labelInter = [r"$R_{U,X}$", r"$R_{U,Y}$", r"$\varepsilon_{R}$", r"$\varepsilon_Q$"]
  labelVorticity = [r"$\nabla \times \partial_k \Pi_{k, e}$", r"$\nabla \times \partial_k \Pi_{k, i}$", r"$\nabla \times R_{i}$", r"$\nabla \times R_{e}$"]
  axes_map = {0:0, 1:1, 3:2, 4:3} #prop:axes
  axesVort_map = {"PI_e":0, "PI_i":1, "R_e":2, "R_i":3}  

  title = [r"$Intra_e$", r"$Intra_i$", r"$Inter_e$", r"$Inter_i$"]

  plotList = [
    (0, "ions", r"$Intra_i$", dudt_fluxes, labelIntra), 
    #(1, "ions", r"$Intra_i Aniso$", dudt_fluxesA), 
    (1, "electrons", r"$Intra_e$", dudt_fluxes, labelIntra), 
    #(1, "electrons", r"$Intra_e$", dudt_fluxesA), 
    (2, "ions", r"$Inter_i$", srcDst, labelInter),
    #(3, "ions", r"$Inter_i Aniso$", srcDstA)
    #(2, "electrons", r"$Inter_e Iso$", srcDst),
    #(3, "electrons", r"$Inter_e Aniso$", srcDstA)
    (3, "ions", r"$\dot\omega_z$", OC, labelVorticity),
    ]
    #maxVal[name]=0; minVal[name] = 0

  # find min max value for colourbars
  maxValProp = {}; minValueProp = {}
  #for (col, name, title, value) in plotList:
  for prop in range(EdenQ + 1):
    if prop == Zmom: continue 
    propRow = axes_map[prop]
    maxValProp[propRow] = 0.

  cc = 0
  for (col, name, title, value, label) in plotList:
    for prop in range(EdenQ + 1):
      if prop == Zmom: continue 
      propRow = axes_map[prop]
      maxVal = value[name][:,:,prop].max()
      minVal = value[name][:,:,prop].min()
      maxBound = max(abs(maxVal), abs(minVal))
      print(f"Row, col:\t {propRow}, {col}\tmaxBound:\t{maxBound}")
      maxValProp[propRow] = max(maxBound, maxValProp[propRow])

  singleColumnCB = False 
  bottom = True  
  print("Graphing...")
  cc = 0
  for (col, name, title, value, label) in plotList:
    for prop in range(EdenQ + 1):
      if prop == Zmom: continue 
      propRow = axes_map[prop]
      ax[propRow, col] = fig.add_subplot(gs[propRow, col])

      if not singleColumnCB:
        maxVal = value[name][:,:,prop].max()
        minVal = value[name][:,:,prop].min()
        maxBound = max(abs(maxVal), abs(minVal))
      else:
        maxBound = maxValProp[propRow]

      maxBound /= saturationFactor 

      ax[propRow, col].pcolormesh(x, y, value[name][:,:,prop], 
                   vmin = -maxBound, vmax = maxBound, cmap=cmapValue)

      if bottom:
        cbLoc = "lower center"
        cbAnchor = (0., -0.05, 1,1)
        cbOrientation = "horizontal"
      else:
        cbLoc = "center right"
        cbAnchor = (0.125, 0., 1, 1)
        cbOrientation = "vertical"
      
      if (not singleColumnCB) or (singleColumnCB and col == 2):
        div_ij = make_axes_locatable(ax[propRow, col])
        cax_ij  = inset_axes(ax[propRow, col], width='80%', height='5%', loc=cbLoc,
                        bbox_to_anchor=cbAnchor, 
                        bbox_transform=ax[propRow, col].transAxes, borderpad=0) 
        #div_ij.append_axes("bottom", size="3%", pad=0.05)
        norm_ij = mpl.colors.Normalize(vmin=-maxBound, vmax=maxBound)
        cb_ij = mpl.colorbar.ColorbarBase(cax_ij, orientation=cbOrientation, cmap=cmapValue, 
                norm=norm_ij, extend="both", ticks=[-maxBound, 0., maxBound], format='%.1e')
        cb_ij.ax.tick_params(labelsize=6) 
        
      ax[propRow,col].set_aspect(1)
      ax[propRow,col].set_xticks([]); ax[propRow,col].set_yticks([])
      ax[propRow,col].set_xlim(view[0]);  ax[propRow,col].set_ylim(view[1])
 
      if True: #col == 0:
        #ax[propRow,col].set_ylabel(label[propRow])
        ax[propRow,col].text(0.15, 0.95, label[propRow], fontsize=8, 
          horizontalalignment='right', verticalalignment='top',
          transform=ax[propRow,col].transAxes, color="k")

      if prop == 0: ax[propRow,col].set_title(title, fontsize=8, color="k")
      #fig.suptitle(f"t = {timeTitle}")

      #ax.text(0.95, 0.05,r'[%.3f, %.3f]'%(val.min(), val.max()), fontsize=7, 
      #  horizontalalignment='right', verticalalignment='bottom',
      #  transform=ax_i[name_ij].transAxes, color="k")

      #print(f"Species: {name} Property: {prop} maxBound: {maxBound:.3e}")

  saveName = nameOutput + f"_SATURATED_{saturationFactor}.png"
  fig.savefig(saveName , format='png', dpi=600, bbox_inches='tight')
  print("\tFigure saved:\t", saveName, "\n\n")

if __name__ == "__main__":
  dataDirs = [
  #["/media/kyriakos/Expansion/999_RES_512_RUNS/tinaroo_Ideal-Clean-HLLE/Ideal-Clean", "Ideal-HLLE"]
  #["/media/kyriakos/Expansion/999_RES_512_RUNS/magnus_HLLC_SRMI-Option-16-Res-512-FB-Isotropic/SRMI-Option-16-Res-512-FB-Isotropic", "FB-Iso"], 
  #["/media/kyriakos/Expansion/999_RES_512_RUNS/magnus_SRMI-Option-16-Res-512-FB-Anisotropic","FB-ANISO"],
  #["/media/kyriakos/Expansion/999_RES_512_RUNS/magnus_HLLC_SRMI-Option-16-Res-512-Inter-Isotropic/SRMI-Option-16-Res-512-Inter-Isotropic", "Inter-ISO"]
  #["/media/kyriakos/Expansion/999_RES_512_RUNS/magnus_HLLC_SRMI-Option-16-Res-512-Inter-Anisotropic/SRMI-Option-16-Res-512-Inter-Anisotropic", "Inter-ANISO"],
  #["/media/kyriakos/Expansion/999_RES_512_RUNS/magnus_HLLC_SRMI-Option-16-Res-512-INTRA-Isotropic/SRMI-Option-16-Res-512-INTRA-Isotropic", "INTRA-ISO"], 
  #["/media/kyriakos/Expansion/999_RES_512_RUNS/magnus_HLLC_SRMI-Option-16-Res-512-INTRA-Anisotropic/SRMI-Option-16-Res-512-INTRA-Anisotropic","INTRA-ANISO"], 
  #["/media/kyriakos/Expansion/222_TINAROO_BACKUP/HLLC_Simulations_Production_Quality/20220504-Op-16-Clean-Ideal-HLLC", "Ideal_HLLC"], 
  ["/media/kyriakos/Expansion/222_TINAROO_BACKUP/HLLC_Simulations_Production_Quality/Z_Correction_QiCorrection_2048_FB_ANISO-Option_16", "FB-ANISO"]
  ]

  for dataDir in dataDirs:
    name = dataDir[-1]
    dataDir = dataDir[0]
  
    dataFiles = get_files(dataDir, include=["plt"], exclude=["png"], get_all=False)
  
    FTF_inputs = {}
    FTF_inputs['times_wanted'] = [1] #[0.20, 0.5, 1.0]#0.1*i for i in range(1,2)]
    FTF_inputs['frameType'] = 2
    FTF_inputs['fileNames'] = dataFiles 
    FTF_inputs['level'] = 0

    # inter iso low res spike zoomed for t = 0.25
    if False:
      xl = -1.037569641038724 
      xh = 0.1707396064128992 
      yl = 0.0
      yh = 1.0
    elif False: # inter iso low res spike zoomed for t = 0.20
      xl = -0.3452644016713751 
      xh = 0.1309040413124145 
      yl = 0. ; yh = 1.0
    else:
      xl = 0.2 #-0.4
      xh = 1.2 # 1.4
      yl = 0.0
      yh = 1.0

    FTF_inputs['window'] = [[xl, yl], [xh, yh]]

    if False:
      xl = -0.0784
      xh = 0.14452
      yl = 0.17907
      yh = 0.87864

    view = [[xl, xh], [yl, yh]]

    level = -4
    print("Search function")
    if True:  
      data_index = [-1]
    else:
      data_index, n_time_slices, time_slices = phmmfp.find_time_frames(FTF_inputs)

    print("\t..done")
    saturationFactor = 1
    for (i, di) in enumerate(data_index):
      dataFile = dataFiles[di]
      nameOutput = "testDelete_20220913_TransportInfluence_anisotropic_" + name + "_t_" + str(FTF_inputs['times_wanted'][i]).replace(".","p")
      timeTitle = str(FTF_inputs['times_wanted'][i])
      graphData(dataFile, level, FTF_inputs['window'], view, nameOutput, timeTitle, 
                saturationFactor, plotVorticityCollisional=True)

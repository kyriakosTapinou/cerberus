#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-
"""
Sun 20200114
v1
  Adapted from CannyEdgeDetector_v3.py to find the pressure and lorentz force 
  contributions to the secondary RTI. Effectively results in the calculation of 
  the pressure gradient and Lorentz force accelerations that are then plotted 
  over a contour of the interface. 
v4 update for the default AMReX file format. 
"""
#======================================Standard modules==============================
import os, sys, gc, numpy as np # standard modules
import h5py, pylab as plt, matplotlib.gridspec as gridspec, matplotlib as mpl
#mpl.use('agg')
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from multiprocessing import Pool
import pdb, math
import copy 
import matplotlib.pyplot as plt
import matplotlib.colors as colors


#======================================Special modules===============================

#visulaisation_code_folder ="/home/s4318421/cerberus/vis"
visulaisation_code_folder ="home/kyriakos/Documents/Code/000_cerberus_dev/githubRelease-cerberus/cerberus/vis/" 
derived_functions_code_folder = "/media/H_drive/000_PhD/001_SimulationDiagnostics/000_BackupPostProcessingTools"

if visulaisation_code_folder not in sys.path:
  sys.path.insert(0, visulaisation_code_folder)
if derived_functions_code_folder not in sys.path:
  sys.path.insert(0, derived_functions_code_folder)

import PHM_MFP_Solver_Post_functions_v4 as phmmfp

from get_boxlib import ReadBoxLib, get_files

#======================================Functions=====================================

### Make function in a bad way because I am lazy 
def pack_er_in_boys(dataDir, name, x_l, x_h, label_append, time_points, a_scale, p_scale, o_scale, level, window):
  #####==================================Parameters====================================
  dataFiles = get_files(dataDir, include = ["plt"], get_all=False)
  properties = ["L_x", "L_y", "L_x_E", "L_x_B", "L_y_E", "L_y_B", 'msk']
  data_dict = {}  
  for i in properties:
    data_dict[i] = {}
  t_list = []
  tol = 0.45
  defaultOptions = {'name':name, 'level':level} 
  options_L_x_E = copy.deepcopy(defaultOptions); options_L_x_E['quantity'] = 'L_x_E'
  options_L_y_E = copy.deepcopy(defaultOptions); options_L_y_E['quantity'] = 'L_y_E'
  options_L_x_B = copy.deepcopy(defaultOptions); options_L_x_B['quantity'] = 'L_x_B'
  options_L_y_B = copy.deepcopy(defaultOptions); options_L_y_B['quantity'] = 'L_y_B'
  #options_ion_omega_dot_baro= {'name':'ion', 'quantity':'vort_dot_baro'}

  Vel_required = False
  if Vel_required == True: Velocity_dict = {};

  FTF_inputs = {}
  FTF_inputs['times_wanted'] = time_points  
  FTF_inputs['frameType'] = 2
  FTF_inputs['fileNames'] = dataFiles
  FTF_inputs['level'] = 0 
  FTF_inputs['window'] = window
  print("Find files corresponding to desired times...")
  data_indexes, n_time_slices, time_slices = phmmfp.find_time_frames(FTF_inputs)
  print("\t...times found.")

  for d_index in data_indexes:
    dFile = dataFiles[d_index]
    rc = ReadBoxLib(dFile, level, window)#, relative_offset=[-0.5,0.])
    t_list.append(rc.time)
    dS_val = rc.data["skin_depth"]
    dx = rc.data['levels'][level]['dx']; dy = dx
    ### ===== Properties =====  
    #x,prs = phmmfp.get_pressure(rc, 'ions')
    x, rho = rc.get("rho-%s"%name)
    #pdx_dict[d_index], pdy_dict[d_index] = get_grad_components( 
    #                                               prs_dict[d_index], dx, dy)
    
    # accelerations due to the Lorentz force
    x, y, data_dict['L_x_E'][d_index] = phmmfp.get_Lorentz(rc, options_L_x_E)
    x, y, data_dict['L_y_E'][d_index]= phmmfp.get_Lorentz(rc, options_L_y_E)
    x, y, data_dict['L_x_B'][d_index] = phmmfp.get_Lorentz(rc, options_L_x_B)
    x, y, data_dict['L_y_B'][d_index] = phmmfp.get_Lorentz(rc, options_L_y_B)
    data_dict['L_x'][d_index] = data_dict['L_x_E'][d_index] + data_dict['L_x_B'][d_index]
    data_dict['L_y'][d_index] = data_dict['L_y_E'][d_index] + data_dict['L_y_B'][d_index]
    ### ===== Interface =====
    IDI_contour = True
    if IDI_contour == True:
      cmap_gray_r2rgba = mpl.cm.gray_r
      x_tracer, tracer_array = rc.get("alpha-%s"%name); 
      y_tracer = x_tracer[1]; x_tracer = x_tracer[0];
      t_h = 0.5 + tol
      t_l = 0.5 - tol
      mask = (t_l <  tracer_array) & (tracer_array < t_h);
      mask = np.ma.masked_equal(mask, False)
      del tracer_array, #x_rho, y_rho, rho; 
      gc.collect()
      # convert msk to rgba style only for interface
      norm_gray_r2rgba = mpl.colors.Normalize(vmin=0., vmax=1.)
      gray_r2rgba = mpl.cm.ScalarMappable(norm=norm_gray_r2rgba, cmap=cmap_gray_r2rgba) 
      data_dict['msk'][d_index] = mask

    rc.close()
  ### ===== Guts ===== 
  dpi_use = 300; print(f"Use dpi:\t{dpi_use}")
  print("plot accelerations due to L_x, L_x_E, L_x_B, L_y, L_y_E, L_y_B")
  properties = ["L_x", "L_y", "L_x_E", "L_x_B", "L_y_E", "L_y_B"]
  titleList = [r"$\mathcal{L}_x$", r"$\mathcal{L}_y$", r"$\mathcal{L}_{x, E}$", r"$\mathcal{L}_{x, B}$", r"$\mathcal{L}_{y, E}$", r"$\mathcal{L}_{y, B}$"]
  wspacing = 0.1
  hspacing = 0.05
  gs = gridspec.GridSpec(len(data_indexes),len(properties), wspace=wspacing, hspace=hspacing)
  # default A4 page width is 6.7 inches, depending on the number of panels
  y_l =0; y_h = int( data_dict['L_x_E'][data_indexes[0]].shape[1])
  if True: y_h = int(y_h/2); 

  print('y_h', y_h)
  fig_x, fig_y = phmmfp.get_figure_size(6.7, len(data_indexes), len(properties), 
                  1.*data_dict['L_x_E'][data_indexes[0]][x_l:x_h,y_l:y_h].shape[0]/data_dict['L_x_E'][data_indexes[0]][x_l:x_h,y_l:y_h].shape[1], 
                  wspacing, hspacing, 1)
  #fig_x = fig_x*(len(properties) + spacing)
  #fig_y = fig_y*(len(data_indexes) + spacing)
  fig = plt.figure(dpi=dpi_use, figsize=(fig_x,fig_y))
  ax = [[] for i in range(len(data_indexes))]
  divider = [[] for i in range(len(data_indexes))]
  cax = [[] for i in range(len(data_indexes))]
  cb = [[] for i in range(len(data_indexes))]
  v_lim = [[] for i in range(len(data_indexes))]
  norm = [[] for i in range(len(data_indexes))]

  for i in range(len(data_indexes)):
    for j in range(len(properties)):
      ax[i].append( fig.add_subplot(gs[i,j]))
    divider[i] = [[] for j in range(len(properties))]
    cax[i] = [[] for j in range(len(properties))]
    cb[i] = [[] for j in range(len(properties))]
    v_lim[i] = [[] for j in range(len(properties))]
    norm[i] = [[] for j in range(len(properties))]

  # acceleration scales 
  use_global_scale = True; print(f"Use global scale for axis:\t {use_global_scale }") 
  default_lim = 0.
  if use_global_scale == True:
    for d_index in data_indexes:
      for prop in properties:
        default_lim = a_scale*max( abs(default_lim), 
                        abs(data_dict[prop][d_index][x_l:x_h,:].max()),
                        abs(data_dict[prop][d_index][x_l:x_h,:].min()))
    for i in range(len(data_indexes)):
      for j in range(len(properties)):
        v_lim[i][j] = default_lim
  
  else:# else individual acceleration limits for each property
    for j in range(len(properties)):
      default_lim = 0.
      prop = properties[j]
      for d_index in data_indexes:
        default_lim = max(abs(default_lim), abs(data_dict[prop][d_index][x_l:x_h,:].max()),
                        abs(data_dict[prop][d_index][x_l:x_h,:].min()))
      for i in range(len(data_indexes)):
        v_lim[i][j] = default_lim 

  cmap_use = mpl.cm.bwr
  for i in range(len(data_indexes)):

    #ax[i][0].set_ylabel(r't=%.3f'%(t_list[i]), fontsize=5, color="k") 
    #if i==0:
    #  ax[i][0].text(0., 0.95,r'%s, t=%.3f'%(name,t_list[i]), fontsize=5,transform=ax[i][0].transAxes, color="k") 
    #else:
    ax[i][0].text(0., 0.95,r't=%.3f'%(t_list[i]), fontsize=5, transform=ax[i][0].transAxes, color="k")
    for j in range(len(properties)):
      d_index = data_indexes[i]; prop = properties[j]
      ax[i][j].imshow( 
        #np.rot90(data_dict[properties[j]][data_indexes[i]][x_l:x_h, y_l:y_h], k=1, axes=(0,1)), 
        data_dict[properties[j]][data_indexes[i]][x_l:x_h, y_l:y_h],
        cmap=cmap_use, vmin=-v_lim[i][j], vmax=v_lim[i][j])
      if IDI_contour == True:
        #ax[i][j].imshow(data_dict['msk'][data_indexes[i]][x_l:x_h, y_l:y_h], 
        #  'gray_r', alpha=1., interpolation = 'none')

        ymesh, xmesh = np.meshgrid(y_tracer[y_l:y_h], x_tracer[x_l:x_h])
        #pdb.set_trace()
        ax[i][j].contourf(
          #np.rot90(data_dict['msk'][data_indexes[i]][x_l:x_h, y_l:y_h], k=1, axes=(0,1)), 
          data_dict['msk'][data_indexes[i]][x_l:x_h, y_l:y_h],
          colors='gray', corner_mask=False, alpha=0.6)

        #ax[i][j].contour(np.rot90(data_dict['msk'][data_indexes[i]][x_l:x_h, y_l:y_h],3), 
        #                 colors='gray', corner_mask=True, alpha=0.25)

      ax[i][j].text(0., 0.05, r'[%1.e, %1.e]'%(data_dict[prop][d_index].min(), data_dict[prop][d_index].max()), fontsize=6, transform=ax[i][j].transAxes, color="k")

      if i == 0:
        ax[i][j].set_title(titleList[j])
      if use_global_scale == True and j == len(properties)-1:
        divider[i][j] = make_axes_locatable(ax[i][j])
        cax[i][j] = inset_axes(ax[i][j], width='10%', height='90%', loc='center right',
                      bbox_to_anchor=(0.125, 0, 1,1), bbox_transform=ax[i][j].transAxes, 
                      borderpad=0)
        #cax[i][j]  = divider[i][j].append_axes("left", size="3%", pad=0.)
        norm[i][j] = mpl.colors.Normalize(vmin=-v_lim[i][j], vmax=v_lim[i][j])
        if "_B" in properties[j]:
          cb[i][j] = mpl.colorbar.ColorbarBase(cax[i][j],orientation="vertical",cmap=cmap_use, 
           norm=norm[i][j], extend="both", ticks=[-v_lim[i][j], 0., v_lim[i][j]], format='%.3f') 
        else:
          cb[i][j] = mpl.colorbar.ColorbarBase(cax[i][j],orientation="vertical",cmap=cmap_use, 
           norm=norm[i][j], extend="both", ticks=[-v_lim[i][j], 0., v_lim[i][j]], format='%.1f')
        cb[i][j].ax.tick_params(labelsize=4) 

      elif use_global_scale == False and i == len(time_points)-1:
        divider[i][j] = make_axes_locatable(ax[i][j])
        cax[i][j]  = inset_axes(ax[i][j], width='90%', height='10%', loc='lower center',
                      bbox_to_anchor=(0, -0.125, 1,1), bbox_transform=ax[i][j].transAxes, 
                      borderpad=0) # divider[i][j].append_axes("bottom", size="6%", pad=0.)
        norm[i][j] = mpl.colors.Normalize(vmin=-v_lim[i][j], vmax=v_lim[i][j])
        if "_B" in properties[j]:
          cb[i][j] = mpl.colorbar.ColorbarBase(cax[i][j],orientation="horizontal",cmap=cmap_use, 
           norm=norm[i][j], extend="both", ticks=[-v_lim[i][j]*0.9, v_lim[i][j]*0.9], \
           format='%1.3f')
        else:
          cb[i][j] = mpl.colorbar.ColorbarBase(cax[i][j],orientation="horizontal",cmap=cmap_use,\
           norm=norm[i][j], extend="both", ticks=[-v_lim[i][j]*0.9, v_lim[i][j]*0.9], \
           format='%1.1f')

        cb[i][j].ax.tick_params(labelsize=6) 

      ax[i][j].axis('on') 
      ax[i][j].axes.yaxis.set_ticklabels([]) 
      ax[i][j].axes.xaxis.set_ticklabels([]) 
      mpl.pyplot.tick_params(left=False) 
      mpl.pyplot.tick_params(bottom=False) 

  name = "Lorentz_components_%s_%s"%(name, label_append)
  name = name.replace(".","p")
  name += ".png"
  fig.savefig(name, format='png', dpi=dpi_use, bbox_inches='tight')
  print("saved ",name)
  plt.close(fig)
  return 

###========================================================run this shit 
a_scale = 1e-1
o_scale = 1e-3
p_scale = 1e-3

a_scale = 5e-1
o_scale = 5e-1
p_scale = 5e-1

h5dirs = [
#("SRMI-OP-16-Res-512-IDEAL-CLEAN", "/media/kyriakos/Expansion/999_RES_512_RUNS/tinaroo_Ideal-Clean-HLLE/Ideal-Clean/"),
#("SRMI-OP-16-Res-512-INTRA-ANISO", "/media/kyriakos/Expansion/999_RES_512_RUNS/magnus_HLLC_SRMI-Option-16-Res-512-INTRA-Anisotropic/SRMI-Option-16-Res-512-INTRA-Anisotropic/") ,
#("SRMI-OP-16-Res-512-INTRA-ISO-", "/media/kyriakos/Expansion/999_RES_512_RUNS/magnus_HLLC_SRMI-Option-16-Res-512-INTRA-Isotropic/SRMI-Option-16-Res-512-INTRA-Isotropic/") ,
#("SRMI-OP-16-Res-512-INTER-ANISO", "/media/kyriakos/Expansion/999_RES_512_RUNS/magnus_HLLC_SRMI-Option-16-Res-512-Inter-Anisotropic/SRMI-Option-16-Res-512-Inter-Anisotropic/") ,
#("SRMI-OP-16-Res-512-INTER-ISO", "/media/kyriakos/Expansion/999_RES_512_RUNS/magnus_HLLC_SRMI-Option-16-Res-512-Inter-Isotropic/SRMI-Option-16-Res-512-Inter-Isotropic/") ,
#("SRMI-OP-16-Res-512-FB-ISO", "/media/kyriakos/Expansion/999_RES_512_RUNS/magnus_HLLC_SRMI-Option-16-Res-512-FB-Isotropic/SRMI-Option-16-Res-512-FB-Isotropic/") ,
("SRMI-OP-16-Res-512-FB-ANISO-ISO", "/media/kyriakos/Expansion/999_RES_512_RUNS/magnus_SRMI-Option-16-Res-512-FB-Anisotropic/") ]

#("SRMI-Option-16-Res-2048-Intra-Anisotropic","/media/kyriakos/Expansion/000_MAGNUS_SUPERCOMPUTER_BACKUP/ktapinou/SRMI-Option-16-Res-2048-Intra-Anisotropic/")]

x_l = 0; x_h = -1
#label_append = h5dirs[0][0] + 'ELECTRONS_t_0p0_SATURATED_1en1'
time_points = [0.2, 0.5, 0.8, 1.0] 
#time_points = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05] #, 0.07, 0.08, 0.09, 0.1] 
window = [[-0.3, 0.], [1.1, 1.]]
level = -1;

#time_points = [0.06, 0.07, 0.08, 0.09]
for h5dir in h5dirs:
  label_append = h5dir[0] + '_ELECTRON_SATURATION_5en1'
  pack_er_in_boys(h5dir[1], 'electrons', x_l, x_h, label_append, time_points, a_scale, p_scale, o_scale, level, window)


"""
label_append = h5dirs[0][0] + 't_0p1_0p15'
time_points = [0.05, 0.06, 0.07, 0.08, 0.09, 0.1] 
pack_er_in_boys(h5dirs[0][1], 'ions', x_l, x_h, label_append, time_points, a_scale, p_scale, o_scale, level, window)
"""

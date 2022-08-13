#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-
"""
v4 
  20220310
  New script created, based off numericalPostProcessin_v19.py and incorporating the features from 
  comarisonStats_v4.py. 

"""
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
#               Functions for statistial agregation and visulisation              #
###################################################################################

def tilePcolormeshNormalise(i, j, i_last, x, y, val, name_ij, fig, gs, ax_i, div_ij, cax_ij, norm_ij, cb_ij, attr_gcl, view, label_ij, time_ij, column_heading):
  ax_i[name_ij] = fig.add_subplot(gs[i,j])
  if (i == i_last):
    #div_ij = make_axes_locatable(ax_i[name_ij])
    #cax_ij  = div_ij.append_axes("bottom", size="3%", pad=0.05)
    div_ij = make_axes_locatable(ax_i[name_ij])
    cax_ij  = inset_axes(ax_i[name_ij], width='100%', height='3%', loc='lower center',
                      bbox_to_anchor=(0., -0.125, 1,1), bbox_transform=ax_i[name_ij].transAxes, 
                      borderpad=0) #div_ij.append_axes("bottom", size="3%", pad=0.05)

  if ("_D-field" in name_ij) or ("_B-field" in name_ij) or  ('Lorentz' in name_ij) or \
  ("vorticity" in name_ij) or ('current' in name_ij) or ('charge' in name_ij):
    vminValue = -attr_gcl; vmidValue = 0.; cmapValue = mpl.cm.bwr # "bwr";
  else:
    vminValue = 0. #0.5*attr_gcl[name_ij]; 
    vmidValue = 0.5*attr_gcl; cmapValue = mpl.cm.Greens #"Greens" #magma_r";

  ax_i[name_ij].pcolormesh(x, y, val, vmin = vminValue, vmax = attr_gcl, cmap=cmapValue)
  if i == i_last:
    norm_ij = mpl.colors.Normalize(vmin=vminValue, vmax=attr_gcl)
    cb_ij = mpl.colorbar.ColorbarBase(cax_ij, orientation="horizontal",cmap=cmapValue, norm=norm_ij, 
      extend="both", ticks=[vminValue, vminValue, attr_gcl], format='%.3f')
    cb_ij.ax.tick_params(labelsize=7) 

  #ax_i[name_ij].set_aspect(1)
  #ax_i[name_ij].axis('off') 
  ax_i[name_ij].set_xticks([]); ax_i[name_ij].set_yticks([])
  ax_i[name_ij].set_xlim([view[0][0], view[1][0]]);  ax_i[name_ij].set_ylim([view[0][1], view[1][1]])
  ax_i[name_ij].text(0.05, 0.95, label_ij, fontsize=12, horizontalalignment='right',verticalalignment='top',transform=ax_i[name_ij].transAxes, color="k")
  if j == 0:
    ax_i[name_ij].set_ylabel('t=%.3f'%time_ij, fontsize=12, color="k")
  if i==0:
    ax_i[name_ij].set_title(column_heading, fontsize=12)
  ax_i[name_ij].text(0.95, 0.05,r'[%.3f, %.3f]'%(val.min(), val.max()), fontsize=10, 
    horizontalalignment='right', verticalalignment='bottom',transform=ax_i[name_ij].transAxes, 
    color="k")

  return 

def tilePcolormesh(i, j, i_last, x, y, val, name_ij, fig, gs, ax_i, div_ij, cax_ij, norm_ij, cb_ij, attr_gcl, view, label_ij, time_ij, column_heading):
  ax_i[name_ij] = fig.add_subplot(gs[i,j])
  div_ij = make_axes_locatable(ax_i[name_ij])
  cax_ij  = div_ij.append_axes("bottom", size="3%", pad=0.)
  delta_range = 1.5
  delta_range = val.max() - val.min()
  big = []
  big.append( delta_range*range_low + val.min() )
  big.append( delta_range*range_high+ val.min() )
  cmapValue = mpl.cm.magma_r
  ax_i[name_ij].pcolormesh(x,y,val,vmin=big[0],vmax=big[1],cmap=cmapValue ) #"magma_r")
  norm_ij = mpl.colors.Normalize(vmin=big[0], vmax=big[1])

  cb_ij = mpl.colorbar.ColorbarBase(cax_ij,
    orientation="horizontal",cmap=cmapValue, norm=norm_ij, extend="both", ticks=[big[0], 
    0.5*(big[0]+big[1]), big[1]], format='%.3f')
  cb_ij.ax.tick_params(labelsize=7) 

  ax_i[name_ij].set_aspect(1)
  #ax_i[name_ij].axis('off') 
  ax_i[name_ij].set_xticks([]); ax_i[name_ij].set_yticks([])
  ax_i[name_ij].set_xlim([view[0][0], view[1][0]]);  ax_i[name_ij].set_ylim([view[0][1], view[1][1]])
  ax_i[name_ij].text(0.95, 0.95, label_ij, fontsize=7, horizontalalignment='right',verticalalignment='top',transform=ax_i[name_ij].transAxes, color="k")
  if j == 0:
    ax_i[name_ij].set_ylabel('t=%.3f'%time_ij, fontsize=7, color="k")
  if i==0:
    ax_i[name_ij].set_title(column_heading, fontsize=7)
  ax_i[name_ij].text(0.95, 0.05,r'[%.3f, %.3f]'%(val.min(), val.max()), fontsize=7, 
    horizontalalignment='right', verticalalignment='bottom',transform=ax_i[name_ij].transAxes, color="k")

  return 

def plot_ScenariosPrimitive(dataFileList, outputType, raw_data_attr, levelList, label_prop, 
                            label_data, label_output, viewList,
                            range_low=0., range_high=1., t_low=0, t_high=0, t_dt = 0.0025 ):

  """
  Augmented plot_raw... to work for multiple input files given
  Function to output plots of single or multiple raw properties on contour plots. Extended to 
  include some specialist properties 
  key - nickname for dataset
  outputType - keyword in out files to search for e.g. ['plt'] or ['chk']
  raw_data_attr - desired variable , if list plot multiple collumns/rows dependeing on orietation
  dataDir - directory of raw data
  if the standard time slices are wanted
    data_index - specific times to show, if set to false, then calculate in hose
  range_low=0. - factor for refining contour 
  range_high=1 - factor for refining contour 
  if a specfic time range is desired 
    t_low - low point 
    t_high - high point 
    t_dt - time increment
  
  """
  columnLetter = ["a", "b", "c", "d", "e", "f", "g"] #default column lettering for creaitng figure
  plot_HorizontalStack = False ; # change orienation of property vs time to time vs property. 
  func_normalise_contours = True
  n_frames = len(dataFileList)
  #======================== Tracer contour overlay for plots =======================#
  IDI_contour = True; EDI_contour = False # overlay ion density interface?
  DI_contour = {}
  print(f"Overlay ion-density interface: {IDI_contour}\tOverlay electron-density interface: {EDI_contour}")
  contourNames= []
  if IDI_contour == True: 
    contourNames.append('ions'); DI_contour['ions'] = []
  if EDI_contour == True: 
    contourNames.append('electrons'); DI_contour['electrons'] = []

  attr = {}
  for attr_name in raw_data_attr:
    attr[attr_name]= {} 
  dS = [[] for i in range(n_frames)]; c = [[] for i in range(n_frames)]; 
  attr_t = [[] for i in range(n_frames)]; debye = [[] for i in range(n_frames)]

  x_attr = {}; y_attr = {}; x_tracer = {}; y_tracer = {};
  #=================================== prepare desired property ====================#
  for (i, dataFile) in enumerate(dataFileList):
    rc = ReadBoxLib(dataFile, level_list[i], viewList[i])
    dx = rc.data["levels"][levelList[i]]['dx']; dy = dx;
    x, dummy = rc.get("rho-ions");
    y_attr[i] = x[1]; x_attr[i] = x[0];
    print(f"time:\t{rc.data['time']}")
    for attr_name in raw_data_attr:
      if ('rho' in attr_name) and (('HRMI' in label_data[i]) or ('MHD' in label_data[i])):
        attr_name_access='rho-neutral'
      elif ('rho_E' in attr_name) and (('HRMI' in label_data[i]) or ('MHD' in label_data[i])):
        attr_name_access='rho_E-neutral'
      else:
        attr_name_access=attr_name  
      # harvest property 
      options = {}
      if "rho_E-EM" in attr_name_access: # special case of a derived property 
        options['level'] = levelList[i] 
        attr[attr_name][i], energySum[attr_name, i] = phmmfp.get_rho_E_EM(rc, options)
      elif "temperature-" in attr_name_access: # special case of a derived property 
        # temperature
        name = attr_name_access.split('temperature-')[1]
        try:
          x, attr[attr_name][i] = rc.get('T-%s'%name)
        except:
          options['name'] = name 
          attr[attr_name][i] = phmmfp.get_T_cons(rc, options)
      elif 'rho-charge' in attr_name_access:
        charge_density = phmmfp.get_charge_number_density(rc, "ions", returnMany=False)[1]
        charge_density += phmmfp.get_charge_number_density(rc, "electrons", returnMany=False)[1]
        attr[attr_name][i] = charge_density
      elif '-current' in attr_name_access: # current density 
        if 'x-' in attr_name_access: index = 2;
        elif 'y-' in attr_name_access: index = 3;
        elif 'z-' in attr_name_access: index = 4;
        attr[attr_name][i] = \
          phmmfp.get_charge_number_density(rc, "ions", returnMany=False)[index]
        attr[attr_name][i] += \
          phmmfp.get_charge_number_density(rc, "electrons", returnMany=False)[index]
      elif "Lorentz-" in attr_name_access:
        #Lorentz force
        name = attr_name_access.split('Lorentz-')[1].split('-')[1]
        direction = attr_name_access.split('Lorentz-')[1].split('-')[0]
        options = {'name':name, 'quantity':'L_%s_total'%direction, 'level':levelList[i]}
        attr[attr_name][i] = phmmfp.get_Lorentz(rc, options)[2]
      elif 'vorticity' in attr_name_access:
        name = attr_name_access.split('vorticity-')[1]
        options = {'name':name, 'quantity':'omega'}
        attr[attr_name][i] = phmmfp.get_vorticity(rc, options)[-1]
      elif attr_name[:6] == 'tracer':
        attr[attr_name][i] = rc.get_flat("alpha-%s"%sattr_name[7:])[1];
      else:
        try:
          x, attr[attr_name][i] = rc.get(attr_name_access);
        except:
          x, attr[attr_name][i] = rc.get_flat(attr_name_access);

        if "rho_E" in attr_name_access:
          energySum[attr_name, i] = np.sum(attr[attr_name][i])*dx*dy
      attr_t[i] = rc.data['time']; dS[i] = rc.data["skin_depth"]; c[i] = rc.data["lightspeed"]; 
      debye[i] = dS[i]/c[i];
    ### ===== Interface =====
    for name in contourNames:
      cmap_gray_r2rgba = mpl.cm.gray_r
      [x_tracer[i],y_tracer[i]], tracer_array = rc.get_flat('alpha-%s'%name);
      tol = 0.45#0.001 print(f"Interface tracer tolerance:\t{tol}")
      t_h = 0.5 + tol
      t_l = 0.5 - tol
      mask = (t_l <  tracer_array) & (tracer_array < t_h) 
      #mask = np.ma.masked_array(tracer_array, mask=((tracer_array > t_l) & (tracer_array < t_h)))
      del tracer_array; gc.collect()
      # convert msk to rgba style only for interface
      norm_gray_r2rgba = mpl.colors.Normalize(vmin=0., vmax=1.)
      gray_r2rgba = mpl.cm.ScalarMappable(norm=norm_gray_r2rgba, cmap=cmap_gray_r2rgba) 
      DI_contour[name].append(mask)
      del mask

    rc.close()
  # attribute contours
  wspacing = 0.1; hspacing = 0.05; 
  if plot_HorizontalStack:
    fig_x, fig_y = phmmfp.get_figure_size(6.7, len(raw_data_attr), n_frames, 
                   1.*y_attr[i].shape[0]/x_attr[i].shape[0], wspacing, hspacing, 1)
  else:
    fig_x, fig_y = phmmfp.get_figure_size(14, n_frames, len(raw_data_attr), 
                   1.*y_attr[i].shape[0]/x_attr[i].shape[0], wspacing, hspacing, 1)

  fig = plt.figure(figsize=(fig_x,fig_y))

  if not plot_HorizontalStack: 
    gs = gridspec.GridSpec(n_frames, len(raw_data_attr), wspace=wspacing, hspace=hspacing)
  else:
    gs = gridspec.GridSpec(len(raw_data_attr), n_frames, wspace=wspacing, hspace=hspacing)

  #axis
  ax = {}; divider = {}; cax = {}; norm = {}; cb = {}; big = {}; attr_gcl = {};

  for attr_name in raw_data_attr:
    if (('rho-electron' in attr_name) or ('rho_E-electron' in attr_name) or \
        ("rho_E-EM" in attr_name)) \
      and (('HRMI' in label_data[i]) or ('MHD' in label_data[i])):
      attr_gcl[attr_name] = False
    elif 'Lorentz-' in attr_name: # min to max value
      if False: print("HardCoded Lorentz limits"); attr_gcl[attr_name] = 5 #hardcoded value 
      else: attr_gcl[attr_name] = \
        max(abs( min([attr[attr_name][i].min() for i in range(n_frames)])), 
            abs( max([attr[attr_name][i].max() for i in range(n_frames)])))
    else: # find abs max value 
      attr_gcl[attr_name] = \
        max(abs( min([attr[attr_name][i].min() for i in range(n_frames)])), 
            abs( max([attr[attr_name][i].max() for i in range(n_frames)])))

  for i in range(n_frames):
    ax[i] = {}; divider[i] = {}; cax[i] = {}; norm[i] = {}; cb[i] = {}; big[i] = {}; 
    y_attr[i], x_attr[i] = np.meshgrid(y_attr[i], x_attr[i])
    if len(contourNames) > 0:
      y_tracer[i], x_tracer[i] = np.meshgrid(y_tracer[i], x_tracer[i])
    for j in range(len(raw_data_attr)):
      attr_name = raw_data_attr[j]; 
      #dcould hae pushed all this into the function but left it specific outside for more contorl
      ax[i][attr_name]  = 0; divider[i][attr_name] = 0; cax[i][attr_name] = 0; 
      norm[i][attr_name] = 0; cb[i][attr_name] = 0;

      if not plot_HorizontalStack: 
        grid_i = i; grid_j = j
      else: 
        grid_i = j; grid_j = i
      if func_normalise_contours: 
        tilePcolormeshNormalise(grid_i, grid_j, n_frames-1, x_attr[i], y_attr[i],attr[attr_name][i], 
          attr_name, fig, gs,  ax[i], divider[i][attr_name], cax[i][attr_name], 
          norm[i][attr_name], cb[i][attr_name], attr_gcl[attr_name], view, "", #label_prop[j], 
          attr_t[i], "("+columnLetter[j] + ") " + label_prop[j])
      else:
        tilePcolormesh(grid_i, grid_j, n_frames-1, x_attr[i],y_attr[i], attr[attr_name][i], 
          attr_name, fig, gs, ax[i], divider[i][attr_name], cax[i][attr_name], 
          norm[i][attr_name], cb[i][attr_name], attr_gcl[attr_name], view, label_prop[j], 
          attr_t[i], columnLetter[j])
      if IDI_contour and "Lorentz" in attr_name :
        #ax[i][attr_name].pcolormesh(
        #  x_tracer, y_tracer, DI_contour['ions'][i], cmap='gray', alpha=1., vmin=0, vmax=1.)
        ax[i][attr_name].contour(x_tracer[i], y_tracer[i], DI_contour['ions'][i], colors='gray', 
                                     corner_mask=True, alpha=0.1)
      elif EDI_contour:
        ax[i][attr_name].contour(x_tracer[i], y_tracer[i], DI_contour['electrons'][i],colors='gray', 
                                     corner_mask=True, alpha=0.1)

      # Label the y axis wth the data key 
      if (j == 0): ax[i][attr_name].set_ylabel(label_data[i], fontsize=12, color="k")
  #fig.suptitle(f"t = {attr_t[i]:.3f}", fontsize=12)

  
  name = "%s-lvl%i"%(label_output,level)
  name = name.replace('.','p')
  name += ".png"
  fig.savefig(name, format='png', dpi=600, bbox_inches='tight')
  print("{} contour saved.".format(name))
  plt.close(fig)
  
  del x_attr, y_attr, attr; gc.collect()
  return 

def compareInterfaceStatistics(fluid, key, date, reducedPlot, areaOnly, circulationComponentsOnly,
      cases, processedFiles, wspacing, hspacing, label_append, useNprocs):
  ### the line styles and colours associated with each property
  data_lines = {"circulation_interface_sum_half":'dashed',"tau_sc_interface_sum_half":'dashed', "baroclinic_interface_sum_half":'dashed', "curl_Lorentz_E_interface_sum_half":'dashed', "curl_Lorentz_B_interface_sum_half":'dashed', 'int_DGDT_sum_half':'dashed', "circulation_interface_sum_half":'dashed', "interface_area":'dashed', "y_avg_int_width":'dashed', "global_int_width":'solid', "growth_rate":'dashed', "global_growth_rate":'solid'}

  data_markers = {"circulation_interface_sum_half":'None',"tau_sc_interface_sum_half":'x', "baroclinic_interface_sum_half":'+', "curl_Lorentz_E_interface_sum_half":'x', "curl_Lorentz_B_interface_sum_half":'+', 'int_DGDT_sum_half':'None', "interface_area":'None', "y_avg_int_width":'None', "global_int_width":'None', "growth_rate":'None', "global_growth_rate":'None'}

  data_labels = {"circulation_interface_sum_half":'', "tau_sc_interface_sum_half":r"$\dot\Gamma_{comp.}-$", "baroclinic_interface_sum_half":r"$\dot\Gamma_{baro.}-$", 
                 "curl_Lorentz_E_interface_sum_half":r"$\dot\Gamma_{L,E}-$", "curl_Lorentz_B_interface_sum_half":r"$\dot\Gamma_{L,B}-$", "interface_area":'',
                 "y_avg_int_width":'', "global_int_width":'Gbl-',"growth_rate":'', "global_growth_rate":'Gbl-', 'int_DGDT_sum_half':''}

  plot_properties = ["circulation_interface_sum_half", "tau_sc_interface_sum_half", "baroclinic_interface_sum_half", "curl_Lorentz_E_interface_sum_half", "curl_Lorentz_B_interface_sum_half", "y_avg_int_width", "growth_rate", 'int_DGDT_sum_half']

  #### What figure tempalte you do you want to use 
  if reducedPlot: # for papers only pront total Gamma, Gamma dot half, eta, eta dot 
    oneD_properties = ["circulation_interface_sum_half", "tau_sc_interface_sum_half", "baroclinic_interface_sum_half", "curl_Lorentz_E_interface_sum_half", "curl_Lorentz_B_interface_sum_half", "y_avg_int_width", "growth_rate"]
    plot_properties = ["circulation_interface_sum_half", "y_avg_int_width", "growth_rate", 'int_DGDT_sum_half']
    plot_labels = [ r"$\Gamma_z$",r"$\.\Gamma_{z, total}$", r"$\eta$", r"$\.\eta$"]

    figure_nR = 2; figure_nC = 2;
    plot_limits = [[-0.01, 0.20], [-0.5, 3.0], [0.1, 0.45], [0., 0.65]] #plot_limits_dS_0p1_reduced 
  elif areaOnly:
    oneD_properties = ["interface_area"]
    plot_properties = ["interface_area"]
    plot_labels = [ r"$A_{int}$"]
    plot_limits = [ [0., 0.08] ]
    figure_nR = 1; figure_nC = 1;
  elif circulationComponentsOnly:
    oneD_properties = ["tau_sc_interface_sum_half", "baroclinic_interface_sum_half", "curl_Lorentz_E_interface_sum_half", "curl_Lorentz_B_interface_sum_half"]
    plot_properties = ["tau_sc_interface_sum_half", "baroclinic_interface_sum_half", "curl_Lorentz_E_interface_sum_half", "curl_Lorentz_B_interface_sum_half"]
    plot_labels = [r"$\dot \Gamma_{z, fluid}$", r"$\dot \Gamma_{z, EM}$"]
    figure_nR = 1; figure_nC = 2;
    plot_limits = [[-0.35, 1.5], [-0.6, 0.6]]
    data_markers = {"circulation_interface_sum_half":'None',"tau_sc_interface_sum_half":'None', "baroclinic_interface_sum_half":'x', "curl_Lorentz_E_interface_sum_half":'None', "curl_Lorentz_B_interface_sum_half":'x', 'int_DGDT_sum_half':'None', "interface_area":'None', "y_avg_int_width":'None', "global_int_width":'None', "growth_rate":'None', "global_growth_rate":'None'}
  else:
    plot_labels = [ r"$\Gamma_z$",r"$\dot \Gamma_{z, total}$",r"$\dot \Gamma_{z, fluid}$", 
                    r"$\dot \Gamma_{z, EM}$", r"$\eta$", r"$\dot \eta$"]
    figure_nR = 3; figure_nC = 2;
    plot_limits = [] #plot_limits_dS_0p1 
    data_markers = {"circulation_interface_sum_half":'None',"tau_sc_interface_sum_half":'None', "baroclinic_interface_sum_half":'x', "curl_Lorentz_E_interface_sum_half":'None', "curl_Lorentz_B_interface_sum_half":'x', 'int_DGDT_sum_half':'None', "interface_area":'None', "y_avg_int_width":'None', "global_int_width":'None', "growth_rate":'None', "global_growth_rate":'None'}

  ### Figure sizing and axis creation
  if areaOnly:
    fig_x, fig_y = phmmfp.get_figure_size(3.35, figure_nR, figure_nC, 1., wspacing, hspacing, 1)
  else:
    fig_x, fig_y = phmmfp.get_figure_size(6.7, figure_nR, figure_nC, 1., wspacing, hspacing, 1)
  fig = plt.figure(figsize=(fig_x,fig_y))
  gs = gridspec.GridSpec(figure_nR, figure_nC) # len(plot_labels),1)#,
  axes = {}; interface_data = {}; t_data = {}
  
  for i in range(len(plot_labels)):#setting for grid according to series order
    if i % 2 == 0:
      column = 0; row = int(i/2)
    else: 
      column = 1; row = int(i/2)
    axes[i] = fig.add_subplot(gs[row, column]); axes[i].set_ylabel(plot_labels[i])
    if len(plot_limits) > 0: axes[i].set_ylim(plot_limits[i]) 

  ### Axis setup
  if reducedPlot: # for papers only pront total Gamma, Gamma dot half, eta, eta dot 
    plot_axes = {"circulation_interface_sum_half":axes[0], 
                 "y_avg_int_width":axes[2],
                 "global_int_width":axes[2],"growth_rate":axes[3], 
                 "global_growth_rate":axes[3], 'int_DGDT_sum_half':axes[1]}
  elif areaOnly:
    plot_axes = {"interface_area":axes[0]}
  elif circulationComponentsOnly:
    plot_axes = {"tau_sc_interface_sum_half":axes[0], 
                 "baroclinic_interface_sum_half":axes[0], 
                 "curl_Lorentz_E_interface_sum_half":axes[1], 
                 "curl_Lorentz_B_interface_sum_half":axes[1]
                }
  else:
    plot_axes = {"circulation_interface_sum_half":axes[0], 
                 "tau_sc_interface_sum_half":axes[2], 
                 "baroclinic_interface_sum_half":axes[2], 
                 "curl_Lorentz_E_interface_sum_half":axes[3], 
                 "curl_Lorentz_B_interface_sum_half":axes[3], 
                 "y_avg_int_width":axes[4],
                 "global_int_width":axes[4],"growth_rate":axes[5], 
                 "global_growth_rate":axes[5], 'int_DGDT_sum_half':axes[1]}

  for i in oneD_properties:
    for (key, (dataDir, level)) in cases.items():
      if 'HYDRO' in key or 'HRMI' in key:
        case_species = 'neutral'
      else:
        case_species = fluid #'ions'
      if case_species == 'neutral' and 'Lorentz' in i:
        pass
      else:
        t_data[i, key], interface_data[i,key] = phmmfp.get_1D_time_series_data(processedFiles[key], 
          species=case_species, quantity=i, nproc=useNprocs, cumsum=False) 

  # Get the derived interface quantities 
  if not areaOnly:
    for (key, (dataDir,level)) in cases.items():
      t_data['int_DGDT_sum_half', key] = copy.deepcopy(t_data['tau_sc_interface_sum_half', key])
      interface_data['int_DGDT_sum_half', key] = copy.deepcopy(interface_data['tau_sc_interface_sum_half', key])
      if type(interface_data['tau_sc_interface_sum_half',key]) == list:
        for i in range(len(interface_data['int_DGDT_sum_half', key])):
          if 'HYDRO' in key or 'HRMI' in key:
            interface_data['int_DGDT_sum_half', key][i] = interface_data["tau_sc_interface_sum_half", key][i] + interface_data["baroclinic_interface_sum_half", key][i]
          else:
            interface_data['int_DGDT_sum_half', key][i] = interface_data["tau_sc_interface_sum_half", key][i] + interface_data["baroclinic_interface_sum_half", key][i] + \
                                                          interface_data["curl_Lorentz_E_interface_sum_half", key][i] + interface_data["curl_Lorentz_B_interface_sum_half", key][i] 
      else:
        if 'HYDRO' in key:  
          interface_data['int_DGDT_sum_half', key] = interface_data["tau_sc_interface_sum_half", key] + interface_data["baroclinic_interface_sum_half", key]
        else:
          interface_data['int_DGDT_sum_half', key] = interface_data["tau_sc_interface_sum_half", key] + interface_data["baroclinic_interface_sum_half", key] + interface_data["curl_Lorentz_E_interface_sum_half", key] + interface_data["curl_Lorentz_B_interface_sum_half", key]  

  # Plot this shit show
  use_lw = 0.75; use_ms = 2
  for prop in plot_properties: 
    for key in cases.keys():
      use_color = case_colors[key]
      if ('HYDRO' in key or 'HRMI' in key) and 'Lorentz' in prop:
        pass
      else:
        if 'HYDRO' in key or 'HRMI' in key:
          use_label = data_labels[prop]+'H-p=%s'%(key.split('_p')[1].split('_ny')[0])
        else:
          use_label = data_labels[prop]+case_prefix[key] #+r'$-d_S-%.1f$'%case_params[key][0]

        use_ls  = data_lines[prop]
        use_marker = dS_marker[key]
        """
        if False and 'circulation' in prop or 'growth' in prop  \
        or 'width' in prop or 'int_DGDT_sum_half' in prop:
          use_marker = ""
        elif 'circulation' in prop or 'growth' in prop or 'area' in prop or 'width' in prop or \
          'int_DGDT_sum_half' in prop:
          use_marker = dS_marker[key]
        else:
          use_marker = data_markers[prop]
        """
        print(f"{prop}\t{key}\t{use_marker}")
        plot_axes[prop].plot(t_data[prop, key], interface_data[prop,key], color=use_color, linestyle=use_ls, marker=use_marker, linewidth=use_lw, markersize=use_ms, label=use_label)
        #plot_axes[prop].set_aspect(1)
    ### sort legend according to labels 
    leg_handles, leg_labels = plot_axes[prop].get_legend_handles_labels()
    leg_hl = sorted(zip(leg_handles, leg_labels), key=operator.itemgetter(1))
    leg_handles2, leg_labels2 = zip(*leg_hl)
    plot_axes[prop].legend(leg_handles2, leg_labels2)
    plot_axes[prop].legend(frameon=False, ncol=2, prop={"size":6}, loc=0)

  for (i,ax) in axes.items():
    if (i != len(plot_labels)-1) and ( i != len(plot_labels)-2 ):
      #ax.axis('off') 
      ax.set_xticklabels([])
    else:
      ax.set_xlabel(r"$t$")
    
    for someName in t_data.keys(): break 
    ax.set_xlim(0, t_data[someName][-1]) #t_data[someName[0], cases.keys()[-1]][-1])
    ax.plot([0,1], [0,0], 'k--', lw=0.4, alpha=0.3) #set the dotted line on zero
  gs.tight_layout(fig, h_pad=0.05, w_pad=0.01)
  
  name = date + "_Interface_Statistics_" + label_append
 #"dS_0p1_T_S-Li_I_S-He_Interface_Statistics_" + label_append
  name = name.replace(".","p")
  name += ".tiff"
  fig.savefig(name, format='tiff', dpi=600, bbox_inches='tight')
  print("Saved ",name)
  plt.close(fig)
  return 

def compareInterfaceThickness(fluid, date, cases, processedFiles, label_append, useNprocs):
  ### generic motherfuckers 
  data_lines = {"interface_area":'dashed', "y_avg_int_width":'dashed', "global_int_width":'solid', 
                "growth_rate":'dashed', "global_growth_rate":'solid', 'interface_thickness':'dashed'}

  data_markers = {"interface_area":'None', "y_avg_int_width":'None', "global_int_width":'None', 
                  "growth_rate":'None', "global_growth_rate":'None', 'interface_thickness':'None'}

  data_labels = {"interface_area":'', "y_avg_int_width":'Avg-', "global_int_width":'Gbl-',
                 "growth_rate":'Avg-', "global_growth_rate":'Gbl-', 'interface_thickness':''}

  ### specifics 
  oneD_properties = ["interface_location", "y_avg_int_width", 
                     "global_int_width",  "growth_rate", "global_growth_rate", 
                     "interface_area"]

  plot_properties = ["interface_thickness", "y_avg_int_width", "global_int_width", 
                     "growth_rate", "global_growth_rate", "interface_area"]
  #plot_labels = [r"$\delta$", r"$\eta_{avg}$", r"$\eta_{global}$", r"$\dot \eta$", 
  #               r"$\dot \eta_{gbl}$", r"$A_{int}$"]
  plot_labels = [r"$\delta$", r"$\eta$", r"$\dot \eta$", r"$A_{int}$"]

  plot_limits = []
  ### set up plot figure amd gridspec ###
  figure_nR = 2; figure_nC = 2; wspacing = 0.3; hspacing = 0.1; 

  fig_x, fig_y = phmmfp.get_figure_size(6.7, figure_nR, figure_nC, 1., wspacing, hspacing, 1)

  fig = plt.figure(figsize=(fig_x,fig_y))
  gs = gridspec.GridSpec(figure_nR, figure_nC, figure=fig) 
  axes = {}; interface_data = {}; t_data = {}

  for i in range(len(plot_labels)):#setting for grid according to series order
    if i % 2 == 0:
      column = 0; row = int(i/2)
    else: 
      column = 1; row = int(i/2)
    axes[i] = fig.add_subplot(gs[row, column]); axes[i].set_ylabel(plot_labels[i])
    if len(plot_limits) > 0: axes[i].set_ylim(plot_limits[i]) 

  plot_axes = {"interface_thickness":axes[0], 
               "y_avg_int_width":axes[1], "global_int_width":axes[1],
               "growth_rate":axes[2], "global_growth_rate":axes[2],
               "interface_area":axes[3]}

  # extract data from h5 post process files 
  for i in oneD_properties:
    for (key, (dataDir, level)) in cases.items():
      if 'HYDRO' in key or 'HRMI' in key:
        case_species = 'neutral'
      else:
        case_species = fluid 
      if case_species == 'neutral' and 'Lorentz' in i:
        pass
      else:
        t_data[i, key], interface_data[i,key] = phmmfp.get_1D_time_series_data(processedFiles[key], 
          species=case_species, quantity=i, nproc=useNprocs, cumsum=False)

  # final time interface thickness 
  intThickness = {}
  for (key, (dataDir, level)) in cases.items():
    nCellsY = len(interface_data["interface_location", key][-1].keys())
    t_data["interface_thickness", key] = np.linspace(0,1,nCellsY)
    interface_data["interface_thickness", key]   = np.zeros((nCellsY))

    #intThickness[key] = np.zeros((nCellsY))
    for i in range(nCellsY):
      #intThickness[key][i] = interface_data['interface_location', key][-1][0][1][0] - \
      #           interface_data['interface_location', key][-1][0][0][0]
      interface_data["interface_thickness", key][i] =\
        interface_data['interface_location', key][-1][i][1][0] -\
                 interface_data['interface_location', key][-1][i][0][0]
  
  ### Plot this shit show ###
  use_lw = 0.25; use_ms = 2
  for prop in plot_properties: 
    for key in cases.keys():
      use_color = case_colors[key]

      if 'HYDRO' in key or 'HRMI' in key:
        use_label = data_labels[prop]+'H-p=%s'%(key.split('_p')[1].split('_ny')[0])
      else:
        use_label = data_labels[prop]+case_prefix[key] #+r'$-d_S-%.1f$'%case_params[key][0]

      use_ls  = data_lines[prop]
      use_marker = dS_marker[key]
      #if prop == 'interface_thickness':
      #  plot_axes[prop].plot(t_data[prop, key], intThickness[key], color=use_color, linestyle=use_ls, marker=use_marker, linewidth=use_lw, markersize=use_ms, label=use_label)

      plot_axes[prop].plot(t_data[prop, key], interface_data[prop,key], color=use_color, linestyle=use_ls, marker=use_marker, linewidth=use_lw, markersize=use_ms, label=use_label)

    ### sort legend according to labels 
    leg_handles, leg_labels = plot_axes[prop].get_legend_handles_labels()
    leg_hl = sorted(zip(leg_handles, leg_labels), key=operator.itemgetter(1))
    leg_handles2, leg_labels2 = zip(*leg_hl)
    plot_axes[prop].legend(leg_handles2, leg_labels2)
    plot_axes[prop].legend(frameon=False, ncol=2, prop={"size":6}, loc=1)

  for (i,ax) in axes.items():
    if (i != len(plot_labels)-1) and ( i != len(plot_labels)-2 ):
      #ax.axis('off') 
      ax.set_xticklabels([])
    else:
      ax.set_xlabel(r"$t$")
    
    for someName in t_data.keys(): break 
    ax.set_xlim(0, t_data[someName][-1]) #t_data[someName[0], cases.keys()[-1]][-1])
    ax.plot([0,1], [0,0], 'k--', lw=0.4, alpha=0.3) #set the dotted line on zero
  gs.tight_layout(fig, h_pad=0.05, w_pad=0.01)
  
  name = date + "_interface_geometry_" + label_append
  name = name.replace(".","p")
  name += ".tiff"
  fig.savefig(name, format='tiff', dpi=600, bbox_inches='tight')
  print("Saved ",name)
  plt.close(fig)

  return 

###################################################################################
#                               Parameter settings                                #
###################################################################################
prepare_data = False # look for existing data file (use dictionary name assigned 
                    # here), create new file or use the existing file.
plot = True # to plot or not to plot, that is the question...

plot_interface_stats = False # plot interface statistics 
plot_scenarios_primitive = True # not implemented 
plot_interface_thickness = False 
plot_scenarios_eden_series = False # not implemented 

plot_HRMI = True
plot_TRMI = False
plot_SRMI = True 
plot_IRMI = False
plot_16 = True
plot_18 = False
plot_21 = False 

useNprocs = 1

max_res = 2048 # not used just elgacy variable
view =  [[-0.4, 0.0], [1.4, 1.0]] # what windo of data to view 
view =  [[0.3, 0.0], [1.5, 1.0]] # view for t=1
window = view ; # no point having more data than needed window =[[-2.0, 0.0], [2.0, 1.0]] 
n_time_slices = 5 # number of time increments for contour plots 
time_slices = range(n_time_slices) #[0, 9] # which indexes in the data_index list (calculated later) to be used, range(n_time_slices) means use all

cwd = os.getcwd() 

if __name__ == '__main__':
###################################################################################
#                             Run postprocessing                                  #
###################################################################################

  # example of plotting raw data (or the specialist derived properties accounted for). 
  #format is {nickname:(directory abslute, max_level)}  
  simOutputDirec = {
#"SRMI-OP-16-Res-512-IDEAL-CLEAN":("/media/kyriakos/Expansion/999_RES_512_RUNS/tinaroo_Ideal-Clean-HLLE/Ideal-Clean/", -1), 
#"SRMI-OP-16-Res-512-INTRA-ANISO":("/media/kyriakos/Expansion/999_RES_512_RUNS/magnus_HLLC_SRMI-Option-16-Res-512-INTRA-Anisotropic/SRMI-Option-16-Res-512-INTRA-Anisotropic/", -1), 
#"SRMI-OP-16-Res-512-INTRA-ISO-":("/media/kyriakos/Expansion/999_RES_512_RUNS/magnus_HLLC_SRMI-Option-16-Res-512-INTRA-Isotropic/SRMI-Option-16-Res-512-INTRA-Isotropic/", -1), 
#"SRMI-OP-16-Res-512-INTER-ANISO":("/media/kyriakos/Expansion/999_RES_512_RUNS/magnus_HLLC_SRMI-Option-16-Res-512-Inter-Anisotropic/SRMI-Option-16-Res-512-Inter-Anisotropic/", -1), 
#"SRMI-OP-16-Res-512-INTER-ISO":("/media/kyriakos/Expansion/999_RES_512_RUNS/magnus_HLLC_SRMI-Option-16-Res-512-Inter-Isotropic/SRMI-Option-16-Res-512-Inter-Isotropic/", -1), 
#"SRMI-OP-16-Res-512-FB-ISO":("/media/kyriakos/Expansion/999_RES_512_RUNS/magnus_HLLC_SRMI-Option-16-Res-512-FB-Isotropic/SRMI-Option-16-Res-512-FB-Isotropic/", -1), 
#"SRMI-OP-16-Res-512-FB-ANISO-ISO":("/media/kyriakos/Expansion/999_RES_512_RUNS/magnus_SRMI-Option-16-Res-512-FB-Anisotropic/", -1), 

#
#"SRMI-OP-16-Res-2048-INTRA-ANISO":("/media/kyriakos/Expansion/222_TINAROO_BACKUP/HLLC_Simulations_Production_Quality/SRMI-Option-16-Res-2048-INTRA-Anisotropic/", -1), 
#"SRMI-OP-16-Res-2048-FB-ANISO-CLEAN":("/media/kyriakos/Expansion/222_TINAROO_BACKUP/HLLC_Simulations_Production_Quality/SRMI-Option-16-Res-2048-FB-Anisotropic/", -1), 
#"SRMI-OP-16-Res-2048-FB-ISO":("/media/kyriakos/Expansion/222_TINAROO_BACKUP/HLLC_Simulations_Production_Quality/Op-16-Res-2048-Clean-Full-Brag-Isotropic/", -1), ###suspect
#"SRMI-OP-16-Res-2048-INTER-ISO":("/media/kyriakos/Expansion/222_TINAROO_BACKUP/HLLC_Simulations_Production_Quality/Op-16-Clean-Inter-Isotropic/", -1), 
#"SRMI-OP-16-Res-2048-INTER-ANISO":("/media/kyriakos/Expansion/222_TINAROO_BACKUP/HLLC_Simulations_Production_Quality/Op-16-Clean-Inter-Anisotropic/", -1), 
#"SRMI-OP-16-Res-2048-INTRA-ISO":("/media/kyriakos/Expansion/222_TINAROO_BACKUP/HLLC_Simulations_Production_Quality/20220427-Op-16-Clean-Intra-Isotropic-HLLC/", -1), 

"SRMI-OP-16-Res-2048-IDEAL":("/media/kyriakos/Expansion/222_TINAROO_BACKUP/HLLC_Simulations_Production_Quality/20220504-Op-16-Clean-Ideal-HLLC/", -1), 
# z correction 
"SRMI-OP-16-Res-2048-INTRA-ANISO":("/media/kyriakos/Expansion/222_TINAROO_BACKUP/HLLC_Simulations_Production_Quality/Z-Correction-2048-INTRA-ANISO-Option-16/", -1), 
"SRMI-OP-16-Res-2048-FB-ANISO":("/media/kyriakos/Expansion/222_TINAROO_BACKUP/HLLC_Simulations_Production_Quality/Z-Correction-2048-FB-ANISO-Option-16/", -1), 
#"SRMI-OP-16-Res-2048-FB-ISO":("/media/kyriakos/Expansion/222_TINAROO_BACKUP/HLLC_Simulations_Production_Quality/Z-Correction-2048-FB-ISO-OPTION-16/", -1), ###suspect
#"SRMI-OP-16-Res-2048-INTER-ISO":("/media/kyriakos/Expansion/222_TINAROO_BACKUP/HLLC_Simulations_Production_Quality/Z-Correction-2048-INTER-ISO-Option-16/", -1), 
"SRMI-OP-16-Res-2048-INTER-ANISO":("/media/kyriakos/Expansion/222_TINAROO_BACKUP/HLLC_Simulations_Production_Quality/Z-Correction-2048-INTER-ANISO-Option-16/", -1), 
#"SRMI-OP-16-Res-2048-INTRA-ISO":("/media/kyriakos/Expansion/222_TINAROO_BACKUP/HLLC_Simulations_Production_Quality/Z-Correction-2048-INTRA-ISO-Option-16/", -1), 

#'PHM_HRMI_p0.5_ny2048':('/home/kyriakos/Documents/000_Species_RMI_Scenario_Results/000_R18_Scenario_Results/PHM_HRMI_p0.5_ny2048', -1), 
#'PHM_HRMI_p1_ny2048':('/home/kyriakos/Documents/000_Species_RMI_Scenario_Results/000_R18_Scenario_Results/PHM_HRMI_p1_ny2048', -1)
}
  
###################################################################################
#                                 Extract data                                    #
###################################################################################
  outputKeyword = 'plt'
  if prepare_data:
    for key, (simDir, level) in simOutputDirec.items():
      phmmfp.get_batch_data(key, simDir, level, max_res, window, n_time_slices, 
                            nproc=useNprocs, outputType=[outputKeyword]) #plt or chk files? default to plt
  
###################################################################################
#                                 Plot statistics                                 #
###################################################################################
  if plot:
    # which data to plot 
    processedFiles = {} # dictionary of the processed output files (h5 format)
    outputFiles = {} #  dictionary of the output files 
    cases = {} 
    case_params = {}
    
    colors = ['g', 'k', 'b', 'm', 'r', 'y']
    #### must match the keys used for the data 
    #dS_marker = {:'+', :'x', 
    """
    dS_marker = {
"SRMI-OP-16-Res-512-IDEAL-CLEAN":"None",
"SRMI-OP-16-Res-512-INTRA-ANISO":"x",
"SRMI-OP-16-Res-512-INTRA-ISO-":"None",
"SRMI-OP-16-Res-512-INTER-ANISO":"x",
"SRMI-OP-16-Res-512-INTER-ISO":"None",
"SRMI-OP-16-Res-512-FB-ISO":"None",
"SRMI-OP-16-Res-512-FB-ANISO-ISO":"x", 
'PHM_HRMI_p0.5_ny2048':'None',
'PHM_HRMI_p1_ny2048':'None',
}

    case_prefix = {"SRMI-OP-16-Res-512-IDEAL-CLEAN":"IDL",
"SRMI-OP-16-Res-512-INTRA-ANISO":"INTRA-A",
"SRMI-OP-16-Res-512-INTRA-ISO-":"INTRA-I",
"SRMI-OP-16-Res-512-INTER-ANISO":"INTER-A",
"SRMI-OP-16-Res-512-INTER-ISO":"INTER-I",
"SRMI-OP-16-Res-512-FB-ISO":"FB-I",
"SRMI-OP-16-Res-512-FB-ANISO-ISO":"FB-I", 
'PHM_HRMI_p0.5_ny2048':'P=0.5',
'PHM_HRMI_p1_ny2048':'P=1',
}

    case_colors = {"SRMI-OP-16-Res-512-IDEAL-CLEAN":"k",
"SRMI-OP-16-Res-512-INTRA-ANISO":"b",
"SRMI-OP-16-Res-512-INTRA-ISO-":"b",
"SRMI-OP-16-Res-512-INTER-ANISO":"g",
"SRMI-OP-16-Res-512-INTER-ISO":"g",
"SRMI-OP-16-Res-512-FB-ISO":"r",
"SRMI-OP-16-Res-512-FB-ANISO-ISO":"r",
'PHM_HRMI_p0.5_ny2048':'m',
'PHM_HRMI_p1_ny2048':'y'
}
    """
    dS_marker = {
"SRMI-OP-16-Res-512-IDEAL-CLEAN":"None",
"SRMI-OP-16-Res-2048-IDEAL":"None",
"SRMI-OP-16-Res-2048-INTRA-ANISO":"x",
"SRMI-OP-16-Res-2048-INTRA-ISO":"None",
"SRMI-OP-16-Res-2048-INTER-ANISO":"x",
"SRMI-OP-16-Res-2048-INTER-ISO":"None",
"SRMI-OP-16-Res-2048-FB-ISO":"None",
"SRMI-OP-16-Res-2048-FB-ANISO":"x", 
'PHM_HRMI_p0.5_ny2048':'None',
'PHM_HRMI_p1_ny2048':'None',
}

    case_prefix = {
"SRMI-OP-16-Res-512-IDEAL-CLEAN":"IDL-512",
"SRMI-OP-16-Res-2048-IDEAL":"IDL",
"SRMI-OP-16-Res-2048-INTRA-ANISO":"INTRA-A",
"SRMI-OP-16-Res-2048-INTRA-ISO":"INTRA-I",
"SRMI-OP-16-Res-2048-INTER-ANISO":"INTER-A",
"SRMI-OP-16-Res-2048-INTER-ISO":"INTER-I",
"SRMI-OP-16-Res-2048-FB-ISO":"FB-I",
"SRMI-OP-16-Res-2048-FB-ANISO":"FB-A", 
'PHM_HRMI_p0.5_ny2048':'P=0.5',
'PHM_HRMI_p1_ny2048':'P=1',
}

    case_colors = {
"SRMI-OP-16-Res-512-IDEAL-CLEAN":"k",
"SRMI-OP-16-Res-2048-IDEAL":"k",
"SRMI-OP-16-Res-2048-INTRA-ANISO":"b",
"SRMI-OP-16-Res-2048-INTRA-ISO":"b",
"SRMI-OP-16-Res-2048-INTER-ANISO":"g",
"SRMI-OP-16-Res-2048-INTER-ISO":"g",
"SRMI-OP-16-Res-2048-FB-ISO":"r",
"SRMI-OP-16-Res-2048-FB-ANISO":"r",
'PHM_HRMI_p0.5_ny2048':'m',
'PHM_HRMI_p1_ny2048':'y'
}


#'g', 'k', :'b', 'r', 'm', 'y', }
 
    label_append = 'SRMI_OPTION_16'
    for key, (simDir, useLevel) in simOutputDirec.items():
      if (plot_HRMI == False) and ('HRMI' in key):
        print('\tIgnore', key) 
        pass
      elif (plot_IRMI == False) and ('IRMI' in key):
        print('\tIgnore', key) 
        pass
      elif (plot_TRMI == False) and ('TRMI' in key): 
        print('\tIgnore', key) 
        pass
      elif (plot_SRMI == False) and ('SRMI' in key): 
        print('\tIgnore', key) 
        pass
      else: # it may be desired data check the parameters 
        if (plot_16 == False) and ( ('-16-' in key) or ('_16' in key) ):
          print('\tIgnore', key) 
          pass
        elif (plot_18 == False) and ( ('-18-' in key) or ('_18' in key) ):
         print('\tIgnore', key) 
         pass
        elif (plot_21 == False) and ( ('-21-' in key) or ('_21' in key) ):
         print('\tIgnore', key) 
         pass
        else:
          print("Use data:\t", key)
          cases[key] = (simDir, useLevel)
          level = useLevel
          dirName = phmmfp.get_save_name(key, simDir, level) # directry name for prcerssed files

          if "PHM_HRMI_p" in simDir: # the HRMI processed files are in the same diretory as the results 
            parentDir = simDir.split("/PHM_HRMI_p")[0]
            dirName = parentDir + "/" + dirName
            outputFiles[key] = ReadHDF5.get_files(simDir, include=[".h5"], exclude=["temp", "old"], get_all=False) 
            processedFiles[key] = get_files(dirName, include=[".h5"], get_all=False) 
          
          else:# standard boxlib files
            if True: # in the sim folder 
              if simDir[-1] == "/": dirName = simDir + dirName # where is the processed files in relation to the sim output
              else: dirName = simDir + "/" + dirName
  
            outputFiles[key] = get_files(simDir, include=["plt"], exclude=["temp", "old"], get_all=False) 
            processedFiles[key] = get_files(dirName, include=[".h5"], get_all=False) 
            rc = ReadBoxLib(outputFiles[key][0], 0, view)
            case_params[key] = (rc.data['skin_depth'],  rc.data['lightspeed'])

          rc.close() #close files
    
    print("\nData lists compiled --- we cool...\n\t...Begin plotting")

    # ============================= contour of conserved properties ============#
    if plot_scenarios_primitive:
      print('Plot species pimitive variables:')

      #EM properties 
      #label_prop = [r"$E_x$",  r"$E_y$", r"$B_z$", r"$J_x$", r"$J_y$"] #
      #raw_data_attr_names = ['x_D-field', 'y_D-field', 'z_B-field', 'x-current', 'y-current']

      #Fluid properties 
      #label_prop = [r"$\rho_i$", r"$\rho_{q}$", r"$\omega_{i}$"] #
      #raw_data_attr_names = ['rho-ions', 'rho-charge', 'vorticity-ions']
      label_prop = [r"$\rho_i$", r"$\rho_e$", r"$T_{i}$", r"$T_{e}$"] #
      raw_data_attr_names = ['rho-ions', 'rho-electrons','temperature-ions', 'temperature-electrons']

      # r"$\omega_{i}$", r"$\rho_{q}$", r"$T_e$", r"$T_i$", r"$\varrho_i$", r"$\omega_{i}$"
      #r"$\rho_{E,e}$", r"$\rho_{E,i}$", r"$\rho_{E, EM}$"]  r"$\rho_i$", r"$L_{x,i}$", 

      #EM properties 
      #raw_data_attr_names = ['rho_E-electrons', 'rho_E-ions', "rho_E-EM", 'vorticity-ion', 
      # 'rho-charge', 'temperature-electron', 'temperature-ion', 'tracer-ion']
      #'temperature-ions' 'vorticity-ions'
      levelList = []
      for timeInput in [1.0]: 
        #label_output = "20220621-Braginskii"+key+"t-%.1f-rho-EM-Jxy_"%(timeInput)
        label_output = "20220804_Braginskii-ANISO_Z-corrected_t-%.1f_rho-temp_"%(timeInput)
        input_files = []; level_list = []; ylabel_list = []; view_list = []
    
        cases_keys = [i for i in cases.keys()] #cases.keys()
        cases_keys.sort()
        cases_sorted = []
        ordered_cases_keys = []; ordered_cases_dict={}
        if False: 
            for key in cases_keys:
              cases_sorted.append( (key, cases[key]) )  
            cases_sorted.sort()
        else: 
          optionIndexes = {16:0, 18:10, 19:20, 21:30}                                          
          options = [16, 18, 19, 20, 21]
          scenarioIndexes = {'IDEAL':0, 'INTRA-ISO':1, 'INTRA-ANISO':2, 'INTER-ISO':3, 
                              'INTER-ANISO':4, 'FB-ISO':5, 'FB-ANISO':6}
                              #'INTER-DIRTY':5, 'FB-DIRTY':6, 'INTRA-ISO-MAG-Z':7, 
                              #'INTRA-ANISO-MAG-Z':8}
          scenarios = ['IDEAL', 'INTRA-ISO', 'INTRA-ANISO', 'INTER-ISO', 'INTER-ANISO', 'FB-ISO', 'FB-ANISO']
                      #'INTER-DIRTY', 'FB-DIRTY', 'INTRA-ISO-MAG-Z', 'INTRA-ANISO-MAG-Z']
          for key in cases:
            optionIndex = -1; scenarioIndex = -1
            for option in options:
              if ( "-" + str(option) in key): 
                optionIndex = optionIndexes[option]
                break 
            for scenario in scenarios:
              if (scenario in key): 
                scenarioIndex = scenarioIndexes[scenario]
                break 
            if optionIndex < 0 or scenarioIndex < 0: continue 
            
            ordered_cases_dict[optionIndex + scenarioIndex] = key;

          for i in range(max(ordered_cases_dict.keys()) + 1):
            if i in ordered_cases_dict.keys():
              cases_sorted.append( (ordered_cases_dict[i], cases[ordered_cases_dict[i]]) )  

        for (key, (outputFile,level)) in cases_sorted:
          #hard coded
          #print("\n###hard coded level overide")
          #level = -4
          FTF_inputs = {}
          FTF_inputs['times_wanted'] = [ timeInput ] 
          FTF_inputs['frameType'] = 2
          #FTF_inputs['dataIndex'] = data_index
          #FTF_inputs['n_time_slices'] = n_time_slices
          #FTF_inputs['time_slices'] = time_slices
          FTF_inputs['fileNames'] = outputFiles[key]
          FTF_inputs['level'] = 0 # set to zero to minimise memory usage - we only need time data 
          FTF_inputs['window'] = view
          if True: ### 
            print("\n\nhardocoded time search overide for last frame")
            input_files.append( outputFiles[key][-1])
          else:
            print("\tSearching for time")
            data_index, n_time_slices, time_slices = phmmfp.find_time_frames(FTF_inputs)
            print("\t\t...done")
            input_files.append( outputFiles[key][data_index[0]])
          level_list.append(level)
  
        view_list = [view]*len(cases_sorted); 
        if False:
          label_data = ["S16-IDEAL", "S16-INTRA-ISO", "S16-INTRA-ANISO", "S16-INTER-ISO", 
                        "S16-INTER-ISO", "S16-FB-ISO", "S16-FB-ANIISO"]
        elif False:
          label_data = ["S16-IDEAL", "S16-INTRA-ISO", "S16-INTER-ISO", "S16-FB-ISO"]
        elif True: 
          label_data = ["S16-IDEAL", "S16-INTRA-ANISO", "S16-INTER-ANISO", "S16-FB-ANISO"]

        plot_ScenariosPrimitive(input_files, [outputKeyword], raw_data_attr_names, level_list, 
          label_prop, label_data, label_output, view_list)

      # ======================= Interface statistics ===============================#
    if plot_interface_stats:
      print('\nPlotting comparison of interface statistics')
      date = "20220804"
      label_append = "IONS_comparison_16-Braginskii_RES_2048"
      ###   
      reducedPlot = True# only plotting the overall circ, circ gen, growth rate, and 
      areaOnly = False
      circulationComponentsOnly = False 

      wspacing = 0.3; hspacing = 0.1; 
      compareInterfaceStatistics("ions", key, date, 
        reducedPlot, areaOnly, circulationComponentsOnly,
        cases, processedFiles, wspacing, hspacing, label_append, useNprocs=useNprocs)

      # ======================= Interface statistics ===============================#
    if plot_interface_thickness:
      print('\nPlotting comparison of interface statistics')
      date = "20220607"
      label_append = "ELECTRONS_comparison_16-Braginskii"
    
      compareInterfaceThickness("electrons",date, cases, processedFiles, label_append, useNprocs)

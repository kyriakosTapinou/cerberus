#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-
"""
Sun 20190922 14:00
Adapted from Daryl Bond's code for calculating fluid properties such as circulation, 
vorticity, barclinic torque, Lorentz torque.i

NOTE HARD CODED INTERFACE CHARACTERISTICS in get_interface

Major change from v11 
  addition of time to the hdf5 plotting prep files 
  change of interface location algorithm
  generate a plot data file for each time step
Major changes from v12-2
  total interface area stat 
  derate plot height  
v13 
  post processing consolidated and compatible with Daryl's PHM 
  output naming convention. 

  Consolidated some plotting features

v18
  Now calls from centralised module PHM_MFP_Solver_Post_functions_v1 
  instead of defining in script the functions for the derived values from the
  checkpoint file. 

v19 
  20220307
  New script created, based off v18. I have rewritten the data extraction step (e.g. time rate of change data, vorticity calcs). and the visualisation/statistical agregation steps.

"""
###################################################################################
#                   Module imports (standard and custom modules)                  #
###################################################################################
import os, sys, gc, h5py, copy, pdb
import numpy as np, math
import matplotlib as mpl
mpl.use('agg') # i native system has no gui interface
import matplotlib.pyplot as plt, matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from multiprocessing import Pool

#custom functions import 
visulaisation_code_folder = "/home/kyriakos/Documents/Code/000_cerberus_dev/githubRelease-cerberus/cerberus/vis/" # current cerberus visualisation and outupt data access directory. "/home/s4318421/git-cerberus/cerberus/vis/" #
derived_functions_code_folder = "/media/H_drive/000_PhD/001_SimulationDiagnostics/000_BackupPostProcessingTools" # "./" #

if visulaisation_code_folder not in sys.path:
  sys.path.insert(0, visulaisation_code_folder)
if derived_functions_code_folder not in sys.path:
  sys.path.insert(0, derived_functions_code_folder)

import PHM_MFP_Solver_Post_functions_v5 as phmmfp # running version 3
from get_boxlib import ReadBoxLib, get_files #from get_hdf5_data import ReadHDF5 # if hdf5 files used



###################################################################################
#               Functions for statistial agregation and visulisation              #
###################################################################################

def tilePcolormeshNormalise(i, j, i_last, x, y, val, name_ij, fig, gs, ax_i, div_ij, cax_ij, norm_ij, cb_ij, attr_gcl, view, label_ij, time_ij, letter_ij):
  ax_i[name_ij] = fig.add_subplot(gs[i,j])
  if (i == i_last):
    div_ij = make_axes_locatable(ax_i[name_ij])
    cax_ij  = inset_axes(ax_i[name_ij], width='100%', height='3%', loc='lower center',
                      bbox_to_anchor=(0., -0.125, 1,1), bbox_transform=ax_i[name_ij].transAxes, 
                      borderpad=0) #div_ij.append_axes("bottom", size="3%", pad=0.05)

  saturateDensity = True; print("Staurating density plot")
  if "_D-field" in name_ij or "_B-field" in name_ij:
    vminValue = -attr_gcl; vmidValue = 0.; cmapValue = mpl.cm.bwr #"bwr";
  elif 'rho-electrons' in name_ij and saturateDensity : 
    attr_gcl = 0.035; vminValue = 0.025; vmidValue = 0.5* (0.035+0.025); cmapValue = mpl.cm.Greens #"Greens" #magma_r";
  elif 'rho-ions' in name_ij and saturateDensity : 
    attr_gcl = 3.5; vminValue = 2.5; vmidValue = 0.5*(2.5+3.5); cmapValue = mpl.cm.Greens # "Greens" #magma_r";
  elif '_vel-' in name_ij and saturateDensity : 
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

  ax_i[name_ij].set_aspect(1)
  #ax_i[name_ij].axis('off') 
  ax_i[name_ij].set_xticks([]); ax_i[name_ij].set_yticks([])
  ax_i[name_ij].set_xlim([view[0][0], view[1][0]]);  ax_i[name_ij].set_ylim([view[0][1], view[1][1]])
  ax_i[name_ij].text(0.05, 0.95, label_ij, fontsize=7, horizontalalignment='right',verticalalignment='top',transform=ax_i[name_ij].transAxes, color="k")
  if j == 0:
    ax_i[name_ij].set_ylabel('t=%.3f'%time_ij, fontsize=7, color="k")
  if i==0:
    ax_i[name_ij].set_title("("+letter_ij+")", fontsize=7)
  ax_i[name_ij].text(0.95, 0.05,r'[%.3f, %.3f]'%(val.min(), val.max()), fontsize=7, 
    horizontalalignment='right', verticalalignment='bottom',transform=ax_i[name_ij].transAxes, 
    color="k")

  return 

def tilePcolormesh(i, j, i_last, x, y, val, name_ij, fig, gs, ax_i, div_ij, cax_ij, norm_ij, cb_ij, attr_gcl, view, label_ij, time_ij, letter_ij):
  ax_i[name_ij] = fig.add_subplot(gs[i,j])
  div_ij = make_axes_locatable(ax_i[name_ij])
  cax_ij  = div_ij.append_axes("bottom", size="3%", pad=0.)
  delta_range = 1.5
  delta_range = val.max() - val.min()
  big = []
  big.append( delta_range*range_low + val.min() )
  big.append( delta_range*range_high+ val.min() )
  ax_i[name_ij].pcolormesh(x,y,val,vmin=big[0],vmax=big[1],cmap="magma_r")
  norm_ij = mpl.colors.Normalize(vmin=big[0], vmax=big[1])

  cb_ij = mpl.colorbar.ColorbarBase(cax_ij,
    orientation="horizontal",cmap="magma_r", norm=norm_ij, extend="both", ticks=[big[0], 
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
    ax_i[name_ij].set_title("("+letter_ij+")", fontsize=7)
  ax_i[name_ij].text(0.95, 0.05,r'[%.3f, %.3f]'%(val.min(), val.max()), fontsize=7, 
    horizontalalignment='right', verticalalignment='bottom',transform=ax_i[name_ij].transAxes, color="k")

  return 

def plot_raw_data_attribute(key, outputType, raw_data_attr, level, label_axis, label_output, 
                            dataDir, data_index, n_time_slices, time_slices, view,
                            range_low=0., range_high=1., t_low=0, t_high=0, t_dt = 0.0025 ):
  """
  Function to output plots of single or multiple raw properties on contour plots. Extended to 
  include some specialist properties 
  key - nickname for dataset
  outputType - keyword in out files to search for e.g. ['plt'] or ['chk']
  raw_data_attr - desired variable , if list plot multiple collumns/rows dependeing on orietation
  dataDir - directory of raw data
  if the standard time slices are wanted
    data_index - specific times to show, if set to false, then calculate in hose
    n_time_slices - number of 
    time_slices -    specific indexes of the contour prints
  range_low=0. - factor for refining contour 
  range_high=1 - factor for refining contour 
  if a specfic time range is desired 
    t_low - low point 
    t_high - high point 
    t_dt - time increment
  
  """
  IDI_contour = True; EDI_contour = True # overlay ion density interface?
  DI_contour = {}
  print(f"Overlay ion-density interface: {IDI_contour}\tOverlay electron-density interface: {EDI_contour}")
  columnLetter = ["a", "b", "c", "d", "e", "f", "g"] #default column lettering for creaitng figure
  contourNames= []; contourColors = []
  if IDI_contour == True: 
    contourNames.append('ions'); DI_contour['ions'] = []; contourColors.append(mpl.cm.gray_r)
  if EDI_contour == True: 
    contourNames.append('electrons'); DI_contour['electrons'] = []; 
    contourColors.append(mpl.cm.Purples)

  func_normalise_contours = True
  dataFiles = get_files(dataDir, include=[outputType], get_all=False)
  if data_index == False:      
    if t_high != 0:  
      n_time_slices = int(math.floor((t_high-t_low ) /t_dt) + 1)
      time_slices = range(n_time_slices) 
      t_low_index = math.floor(t_low/t_dt)
      data_index = []
      for i in range(n_time_slices):
        data_index.append( int(t_low_index + i) )
    else:
      contour_save_increment = math.floor(len(dataFiles)/(n_time_slices-1))
      data_index = []
      for i in range(n_time_slices-1):
        data_index.append( int(i*contour_save_increment) )
      data_index.append(int(len(dataFiles)-1))

  attr = {}; 
  dS = [[] for i in range(len(time_slices))]; c = [[] for i in range(len(time_slices))]; 
  attr_t = [[] for i in range(len(time_slices))]; debye = [[] for i in range(len(time_slices))]
  for attr_name in raw_data_attr:
    attr[attr_name]= {} 

  for i in range(len(time_slices)):
    for attr_name in raw_data_attr:
      rc = ReadBoxLib(dataFiles[data_index[time_slices[i]]], level, view)

      try:
        x_attr, attr[attr_name][i] = rc.get_flat(attr_name); 
      except: 
        x_attr, attr[attr_name][i] = rc.get(attr_name); 
      y_attr = x_attr[1]; x_attr = x_attr[0];

      if 'tracer' in attr_name:
        attr[attr_name][i] = rc.get_flat('alpha-%s'%attr_name[7:])[1]
      attr_t[i] = rc.data['time']; dS[i] = rc.data["skin_depth"]; c[i] = rc.data["lightspeed"]; 
      debye[i] = dS[i]/c[i];
  
    ### ===== Interface =====
    for (i,name) in enumerate(contourNames):
      cmap_gray_r2rgba = contourColors[i] #mpl.cm.gray_r
      [x_tracer,y_tracer], tracer_array = rc.get_flat('alpha-%s'%name);
      tol = 0.45#0.001
      print(f"Interface tracer tolerance:\t{tol}")
      t_h = 0.5 + tol
      t_l = 0.5 - tol
      mask = (t_l <  tracer_array) & (tracer_array < t_h) 
      mask = np.ma.masked_equal(mask, False)
      #mask = np.ma.masked_array(tracer_array, mask=((tracer_array > t_l) & (tracer_array < t_h)))
      del tracer_array; gc.collect()
      # convert msk to rgba style only for interface
      norm_gray_r2rgba = mpl.colors.Normalize(vmin=0., vmax=1.)
      gray_r2rgba = mpl.cm.ScalarMappable(norm=norm_gray_r2rgba, cmap=cmap_gray_r2rgba) 
      DI_contour[name].append(mask)
      del mask

    rc.close()
  # attribute contours
  wspacing = 0.1
  hspacing = 0.05
  fig_x = 20 # should be 6.7 for an a4 page 
  fig_x, fig_y = phmmfp.get_figure_size(fig_x, n_time_slices, len(raw_data_attr), 
                   1.*y_attr.shape[0]/x_attr.shape[0], wspacing, hspacing, 1)
  fig = plt.figure(figsize=(fig_x,fig_y))
  gs = gridspec.GridSpec(len(time_slices), len(raw_data_attr), wspace=wspacing, hspace=hspacing)#,width_ratios=[1], height_ratios=[1,1,1,1,1],)
  y_attr, x_attr = np.meshgrid(y_attr, x_attr) # y, x = np.meshgrid(y, x)
  if len(contourNames) > 0:
    y_tracer, x_tracer = np.meshgrid(y_tracer, x_tracer)
  #axis
  ax = {}; divider = {}; cax = {}; norm = {}; cb = {}; big = {}; attr_gcl = {};

  for attr_name in raw_data_attr:
    attr_gcl[attr_name] = max( abs( min([attr[attr_name][i].min() for i in range(len(time_slices))])), abs( max([attr[attr_name][i].max() for i in range(len(time_slices))])))
    if '_D-field' in attr_name or '_B-field' in attr_name: attr_gcl[attr_name] = 0.1*attr_gcl[attr_name]

  for i in range(len(time_slices)):
    ax[i] = {}; divider[i] = {}; cax[i] = {}; norm[i] = {}; cb[i] = {}; big[i] = {}; 
    for j in range(len(raw_data_attr)):
      attr_name = raw_data_attr[j]; 
      #dcould hae pushed all this into the function but left it specific outside for more contorl
      ax[i][attr_name]  = 0; divider[i][attr_name] = 0; cax[i][attr_name] = 0; 
      norm[i][attr_name] = 0; cb[i][attr_name] = 0;

      if func_normalise_contours: 
        tilePcolormeshNormalise(i, j, n_time_slices-1, x_attr, y_attr, attr[attr_name][i], 
          attr_name, fig, gs,  ax[i], divider[i][attr_name], cax[i][attr_name], 
          norm[i][attr_name], cb[i][attr_name], attr_gcl[attr_name], view, label_axis[j], 
          attr_t[i], columnLetter[j])
      else:
        tilePcolormesh(i, j, n_time_slices-1, x_attr, y_attr, attr[attr_name][i], 
          attr_name, fig, gs, ax[i], divider[i][attr_name], cax[i][attr_name], 
          norm[i][attr_name], cb[i][attr_name], attr_gcl[attr_name], view, label_axis[j], 
          attr_t[i], columnLetter[j])
      if IDI_contour:
        #ax[i][attr_name].pcolormesh(
        #  x_tracer, y_tracer, DI_contour['ions'][i], cmap='gray', alpha=1., vmin=0, vmax=1.)
        ax[i][attr_name].contourf(x_tracer, y_tracer, DI_contour['ions'][i], colors='gray', 
                                     corner_mask=False, alpha=0.6)
      elif EDI_contour:
        ax[i][attr_name].contourf(x_tracer, y_tracer, DI_contour['electrons'][i], colors='purples', 
                                     corner_mask=False, alpha=0.6)
 
  name = "%s-lvl%i"%(label_output,level)
  name = name.replace('.','p')
  name += ".png"
  fig.savefig(name, format='png', dpi=600, bbox_inches='tight')
  print("{} contour saved.".format(name))
  plt.close(fig)
  
  del x_attr, y_attr, attr; gc.collect()
  return 

def interfaceStatistics(fluid, key, date, simDir, level, grid_i, grid_j, 
                        #axesLabels, location, limits, props, propLabels, 
                        nproc=1):
    #interfaceStatistics("ions", key, date, simDir, level, 2, 2, [], [], nproc=6)
  """
  interfaceStatistics('electrons', key, date, simDir, level, grid_i=2, grid_j=2, 
    propLabels = [["Full interface", "1/2 interface"], 
                  [r"$\dot\Gamma_{comp.}$", r"$\dot\Gamma_{baro.}$",  r"$\dot\Gamma_{L,E}$", 
                   r"$\dot\Gamma_{L,B}$", r"$\Sigma\dot\Gamma_{z,half}$"], 
                  ["Avg", "Global"],
                  ["Avg", "Global"],
                  #[],
                     ], 
    axesLabels = [
                  r"$\Gamma_z$, $\lambda_D$=%g, c=%g $p=\frac{1}{2}$"%(debye_print,c_print), 
                  #r"$\dot \Gamma_{z,full}$",  
                  r"$\dot \Gamma_{z, half}$" , 
                  r"$\eta$", 
                  r"$\dot \eta$", 
                  #r"$A_{int}$", 
                  ]
    props = [["circulation_interface_sum", "circulation_interface_sum_half"], 
            #[],
             ["tau_sc_interface_sum_half", "baroclinic_interface_sum_half", 
             "curl_Lorentz_E_interface_sum_half", "curl_Lorentz_B_interface_sum_half"], 
             ["y_avg_int_width", ]
                 ], 
  , cumsum=False, nproc=useNproc)
  "global_int_width", cumsum=False, nproc=useNproc)
  "growth_rate", cumsum=False, nproc=useNproc)
  "global_growth_rate",cumsum=False, nproc=useNproc)


    location= [[0,0], 
               [1,0], 
               [0,1],
               [1,1]
              ],
    limits = [[-0.01, 0.25], 
              #[],
              [-0.5, 1.2], 
              [0.,0.5], 
              [-1.2,0.6]
              #[]
             ], 
      
               ):
  """
  """Function to find the interface statistics, the interface is defined by the 
    a volume of fluid method derived from the original tracer value on the interface. 
  Extract from the post processed files the interface statistics and plot a time 
  series of the desired properties. 
  fluid - string name of the fluid e.g. fluid, 'ions', 'electrons'
  key - the nickanme of the data set coresponding to the data extraction
  simDir - output data directory
  level - the level to extract 
  grid_i - thegrid rows 
  props - properties desired corresponding to the time series functions see below
  propLabels -  labels to be used in the visualisation
  """
  useNproc = nproc; # how many processes to spawn 
  processedSimDir = phmmfp.get_save_name(key, simDir, level)
  simFiles = get_files(simDir, include=['plt'], get_all=False)
  processed_files = get_files(processedSimDir, include=['.h5'], get_all=False)

  rc = ReadBoxLib(simFiles[0], 0, view); 
  debye_print = rc.data['Debye']; c_print = rc.data['lightspeed']; 
  rc.close()

  # attribute contours
  wspacing = 0.05
  hspacing = 0.05
  fig_x, fig_y = phmmfp.get_figure_size(6.7, grid_i, grid_j, 1., wspacing, hspacing, 1)
  fig = plt.figure(figsize=(fig_x,fig_y))
  # layout the grid for the figure 
  gs = gridspec.GridSpec(grid_i, grid_j)#,
                         #width_ratios=[1],
                         #height_ratios=[1,1,1,1,1,1],)
  # set up axes for subplots of circulation, vorticity terms(flow1, flow2, baro, tb, and tl)
  axes = []; 
  # when automating 
  #gi = 0; gj = 0; 
  #for i in range(grid_i*grid_j):
  #  
  #  axes.append(fig.add_subplot(gs[0,0]); axes.append(ax0); ax0.set_ylabel(r"$\Gamma_z$, $\lambda_D$=%g, c=%g $p=\frac{1}{2}$"%(debye_print,c_print));ax0.set_ylim([-0.01, 0.25]) 

  #  t, int_circ = phmmfp.get_1D_time_series_data(processed_files, species=fluid, quantity="circulation_interface_sum", cumsum=False, nproc=useNproc)  #circulation_int_sum

  ax0 = fig.add_subplot(gs[0,0]); axes.append(ax0); ax0.set_ylabel(r"$\Gamma_z$, $\lambda_D$=%g, c=%g $p=\frac{1}{2}$"%(debye_print,c_print));
  #ax1 = fig.add_subplot(gs[1,0]); axes.append(ax1); ax1.set_ylabel(r"$\dot \Gamma_{z,full}$"); 
  ax2 = fig.add_subplot(gs[1,0]); axes.append(ax2); ax2.set_ylabel(r"$\dot \Gamma_{z, half}$"); 
  #ax1.legend(loc="upper left")
  ax3 = fig.add_subplot(gs[0,1]); axes.append(ax3); ax3.set_ylabel(r"$\eta$"); 
  ax4 = fig.add_subplot(gs[1,1]); axes.append(ax4); ax4.set_ylabel(r"$\dot \eta$"); 
  #ax5 = fig.add_subplot(gs[2,1]); axes.append(ax5); ax5.set_ylabel(r"$A_{int}$"); 

  if False: # set limits 
    ax0.set_ylim([-0.01, 0.25]) 
    ax2.set_ylim([-0.5, 1.7]) 
    ax3.set_ylim([0.,0.5])
    ax4.set_ylim([-1.2,0.6])
    #ax5.set_ylim([0.,0.06])

  tp = []
  t, int_circ = phmmfp.get_1D_time_series_data(processed_files, species=fluid, quantity="circulation_interface_sum", cumsum=False, nproc=useNproc)  #circulation_int_sum
  tl, int_tsc  = phmmfp.get_1D_time_series_data(processed_files, species=fluid, quantity="tau_sc_interface_sum", cumsum=False, nproc=useNproc) # flow term 1
  #tl, int_tc  = get_1D_time_series_data(processed_files, species=fluid, quantity="tau_conv_interface_sum", cumsum=False, nproc=useNproc) # flow term 2
  tb, int_b   = phmmfp.get_1D_time_series_data(processed_files, species=fluid, quantity="baroclinic_interface_sum", cumsum=False, nproc=useNproc)   # baroclinic term 
  tl, int_L_E = phmmfp.get_1D_time_series_data(processed_files, species=fluid, quantity="curl_Lorentz_E_interface_sum", cumsum=False, nproc=useNproc) # Lorentz E component 
  tl, int_L_B = phmmfp.get_1D_time_series_data(processed_files, species=fluid, quantity="curl_Lorentz_B_interface_sum", cumsum=False, nproc=useNproc) # Lorentz B component
  tA, int_A   = phmmfp.get_1D_time_series_data(processed_files, species=fluid, quantity="interface_area", cumsum=False, nproc=useNproc) # interface area 

  int_DGDT_full = copy.deepcopy(int_tsc)
  if type(int_tsc) == list:
    for i in range(len(int_DGDT_full)):
      int_DGDT_full[i] = int_tsc[i] + int_b[i] + int_L_E[i] + int_L_B[i]
  else:
    int_DGDT_full = int_tsc + int_b + int_L_E + int_L_B
    #int_total_odot = int_tsc + int_tc + int_b + int_L_E + int_L_B

  t, int_circ_half = phmmfp.get_1D_time_series_data(processed_files, species=fluid, quantity="circulation_interface_sum_half", cumsum=False, nproc=useNproc)  #vorticity_int_sum
  tl, int_tsc_half  = phmmfp.get_1D_time_series_data(processed_files, species=fluid, quantity="tau_sc_interface_sum_half", cumsum=False, nproc=useNproc) # flow term 1
  #tl, int_tc_half  = phmmfp.get_1D_time_series_data(processed_files, species=fluid, quantity="tau_conv_interface_sum_half", cumsum=False, nproc=useNproc) # flow term 2
  tb, int_b_half   = phmmfp.get_1D_time_series_data(processed_files, species=fluid, quantity="baroclinic_interface_sum_half", cumsum=False, nproc=useNproc)   # baroclinic term 
  tl, int_L_E_half = phmmfp.get_1D_time_series_data(processed_files, species=fluid, quantity="curl_Lorentz_E_interface_sum_half", cumsum=False, nproc=useNproc) # Lorentz E component 
  tl, int_L_B_half= phmmfp.get_1D_time_series_data(processed_files, species=fluid, quantity="curl_Lorentz_B_interface_sum_half", cumsum=False, nproc=useNproc) # Lorentz B component

  int_DGDT_half = copy.deepcopy(int_tsc_half)
  if type(int_tsc_half) == list:
    for i in range(len(int_DGDT_half)):
      int_DGDT_half[i] = int_tsc_half[i] + int_b_half[i] + int_L_E_half[i] + int_L_B_half[i]
  else:
     int_total_odot_half = int_tsc_half + int_tc_half + int_b_half + int_L_E_half + int_L_B_half
     #int_DGDT_half = int_tsc_half + int_b_half + int_L_E_half + int_L_B_half

  teta, eta   = phmmfp.get_1D_time_series_data(processed_files, species=fluid, quantity="y_avg_int_width", cumsum=False, nproc=useNproc)
  teta, geta   = phmmfp.get_1D_time_series_data(processed_files, species=fluid, quantity="global_int_width", cumsum=False, nproc=useNproc)
  tetad, etad = phmmfp.get_1D_time_series_data(processed_files, species=fluid, quantity="growth_rate", cumsum=False, nproc=useNproc)
  tetad, getad = phmmfp.get_1D_time_series_data(processed_files, species=fluid, quantity="global_growth_rate",cumsum=False, nproc=useNproc)

  axes[-1].set_xlabel(r"$t$")
  for ax in axes[0:-1]:
      ax.set_xticklabels([])
  for ax in axes:
      ax.set_xlim(0, t[-1])
      ax.plot([0,1], [0,0], 'k--', lw=0.4, alpha=0.3) #set the dotted line on zero
 
  color = "k"; lw = 0.5
  tp.append([t, int_circ, {"color":color, "ls":"-", "lw":lw, "label":"Full interface"}, ax0])
  tp.append([t, int_circ_half, {"color":'g', "ls":"--", "lw":lw, "label":"1/2 interface"}, ax0])
  ###=====================================full interface properties 
  #tp.append([tl, int_tsc, {"color":'g', "ls":"-", "lw":lw, "label":r"$\dot\Gamma_{comp.}$"}, ax1])
  #tp.append([tl, int_tc, {"color":'b', "ls":"--", "lw":lw, "label":r"$\dot\Gamma_{conv.}$"}, ax1])
  #tp.append([tb, int_b, {"color":'c', "ls":"-","marker":"x", "ms":1., "lw":lw,"label":r"$\dot\Gamma_{baro.}$" }, ax1])
  #tp.append([tl, int_L_E, {"color":'r', "ls":"-.", "lw":lw, "label":r"$\dot\Gamma_{L,E}$"}, ax1])
  #tp.append([tl, int_L_B, {"color":'m', "ls":":", "lw":lw, "label":r"$\dot\Gamma_{L,B}$"}, ax1])
  #tp.append([tl, int_DGDT_full, {"color":color, "ls":"-", "lw":lw, "marker":"o", "ms":1., "label":r"$\Sigma\dot\Gamma_{z,full}$"}, ax1])
  ###=====================================Half interface properties 
  tp.append([tl, int_tsc_half, 
    {"color":'g', "ls":"-", "lw":lw, "label":r"$\dot\Gamma_{comp.}$"}, ax2])
  #tp.append([tl, int_tc_half, 
  #  {"color":'b', "ls":"--", "lw":lw, "label":r"$\dot\Gamma_{conv.}$"}, ax2])
  tp.append([tb, int_b_half, 
    {"color":'c', "ls":"-","marker":"x", "ms":1., "lw":lw,"label":r"$\dot\Gamma_{baro.}$" }, ax2])
  tp.append([tl, int_L_E_half, 
    {"color":'r', "ls":"-.", "lw":lw, "label":r"$\dot\Gamma_{L,E}$"}, ax2])
  tp.append([tl, int_L_B_half, 
    {"color":'m', "ls":":", "lw":lw, "label":r"$\dot\Gamma_{L,B}$"}, ax2])
  tp.append([tl, int_DGDT_half,{"color":color, "ls":"-", "lw":lw, 
    "marker":"o", "ms":1., "label":r"$\Sigma\dot\Gamma_{z,half}$"}, ax2])
  tp.append([teta, eta, {"color":color, "ls":"-", "lw":lw,"label":"Avg"}, ax3])
  tp.append([teta, geta, {"color":'g', "ls":"--", "lw":lw,"label":"Global"}, ax3])
  tp.append([tetad, etad, {"color":color, "ls":"-", "lw":lw,"label":"Avg"}, ax4]) 
  tp.append([tetad, getad, {"color":'g', "ls":"--", "lw":lw,"label":"Global"}, ax4]) 

  #tp.append([tA, int_A, {"color":'g', "ls":"-", "lw":lw,"label":"Full interface Area"}, ax5]) 

  for p in tp:
      #print(p[2])
      ax = p[-1]
      ax.plot(p[0], p[1], **p[2])
      ax.legend(frameon=False, ncol=2, prop={"size":8}, loc=1)
  gs.tight_layout(fig, h_pad=0.05, w_pad=0.01)
  
  name = date+"_Interface_Statistics_" + key + "lvl%i"%(level)
  name = name.replace(".","p")
  name += ".png"
  fig.savefig(name, format='png', dpi=600, bbox_inches='tight')
  print("saved ",name)
  plt.close(fig)
  return 

def ionElectronInterfaceStatistics(fluids, key, date, simDir, level, grid_i, grid_j, 
                        #axesLabels, location, limits, props, propLabels, 
                        nproc=1):
  """Function to comapre the interface statistics between the ion and electron fluids. 
  The interface is defined by the volume of fluid method derived from the original tracer value on the interface. 
  Extract from the post processed files the interface statistics and plot a time 
  series of the desired properties. 
  fluid - string name of the fluid e.g. fluid, 'ions', 'electrons'
  key - the nickanme of the data set coresponding to the data extraction
  simDir - output data directory
  level - the level to extract 
  grid_i - thegrid rows 
  props - properties desired corresponding to the time series functions see below
  propLabels -  labels to be used in the visualisation
  """
  useNproc = nproc; # how many processes to spawn 
  processedSimDir = phmmfp.get_save_name(key, simDir, level)
  simFiles = get_files(simDir, include=['plt'], get_all=False)
  processed_files = get_files(processedSimDir, include=['.h5'], get_all=False)

  rc = ReadBoxLib(simFiles[0], 0, view); 
  debye_print = rc.data['Debye']; c_print = rc.data['lightspeed']; 
  rc.close()

  # attribute contours
  wspacing = 0.05
  hspacing = 0.05
  fig_x, fig_y = phmmfp.get_figure_size(6.7, grid_i, grid_j, 1., wspacing, hspacing, 1)
  fig = plt.figure(figsize=(fig_x,fig_y))
  # layout the grid for the figure 
  gs = gridspec.GridSpec(grid_i, grid_j)#,
                         #width_ratios=[1],
                         #height_ratios=[1,1,1,1,1,1],)
  # set up axes for subplots of circulation, vorticity terms(flow1, flow2, baro, tb, and tl)
  axes = []; 
  # when automating 
  #gi = 0; gj = 0; 
  #for i in range(grid_i*grid_j):
  #  
  #  axes.append(fig.add_subplot(gs[0,0]); axes.append(ax0); ax0.set_ylabel(r"$\Gamma_z$, $\lambda_D$=%g, c=%g $p=\frac{1}{2}$"%(debye_print,c_print));ax0.set_ylim([-0.01, 0.25]) 

  #  t, int_circ = phmmfp.get_1D_time_series_data(processed_files, species=fluid, quantity="circulation_interface_sum", cumsum=False, nproc=useNproc)  #circulation_int_sum

  ax0 = fig.add_subplot(gs[0,0]); axes.append(ax0); ax0.set_ylabel(r"$\Gamma_z$, $\lambda_D$=%g, c=%g $p=\frac{1}{2}$"%(debye_print,c_print));
  #ax1 = fig.add_subplot(gs[1,0]); axes.append(ax1); ax1.set_ylabel(r"$\dot \Gamma_{z,full}$"); 
  ax2 = fig.add_subplot(gs[1,0]); axes.append(ax2); ax2.set_ylabel(r"$\dot \Gamma_{z, half}$"); 
  #ax1.legend(loc="upper left")
  ax3 = fig.add_subplot(gs[0,1]); axes.append(ax3); ax3.set_ylabel(r"$\eta$"); 
  ax4 = fig.add_subplot(gs[1,1]); axes.append(ax4); ax4.set_ylabel(r"$\dot \eta$"); 
  #ax5 = fig.add_subplot(gs[2,1]); axes.append(ax5); ax5.set_ylabel(r"$A_{int}$"); 

  if False: # set limits 
    ax0.set_ylim([-0.01, 0.25]) 
    ax2.set_ylim([-0.5, 1.7]) 
    ax3.set_ylim([0.,0.5])
    ax4.set_ylim([-1.2,0.6])
    #ax5.set_ylim([0.,0.06])

  tp = []; data = {}
  for fluid in fluids:
  t, int_circ = phmmfp.get_1D_time_series_data(processed_files, species=fluid, quantity="circulation_interface_sum", cumsum=False, nproc=useNproc)  #circulation_int_sum
  tl, int_tsc  = phmmfp.get_1D_time_series_data(processed_files, species=fluid, quantity="tau_sc_interface_sum", cumsum=False, nproc=useNproc) # flow term 1
  #tl, int_tc  = get_1D_time_series_data(processed_files, species=fluid, quantity="tau_conv_interface_sum", cumsum=False, nproc=useNproc) # flow term 2
  tb, int_b   = phmmfp.get_1D_time_series_data(processed_files, species=fluid, quantity="baroclinic_interface_sum", cumsum=False, nproc=useNproc)   # baroclinic term 
  tl, int_L_E = phmmfp.get_1D_time_series_data(processed_files, species=fluid, quantity="curl_Lorentz_E_interface_sum", cumsum=False, nproc=useNproc) # Lorentz E component 
  tl, int_L_B = phmmfp.get_1D_time_series_data(processed_files, species=fluid, quantity="curl_Lorentz_B_interface_sum", cumsum=False, nproc=useNproc) # Lorentz B component
  tA, int_A   = phmmfp.get_1D_time_series_data(processed_files, species=fluid, quantity="interface_area", cumsum=False, nproc=useNproc) # interface area 

  int_DGDT_full = copy.deepcopy(int_tsc)
  if type(int_tsc) == list:
    for i in range(len(int_DGDT_full)):
      int_DGDT_full[i] = int_tsc[i] + int_b[i] + int_L_E[i] + int_L_B[i]
  else:
    int_DGDT_full = int_tsc + int_b + int_L_E + int_L_B
    #int_total_odot = int_tsc + int_tc + int_b + int_L_E + int_L_B

  t, int_circ_half = phmmfp.get_1D_time_series_data(processed_files, species=fluid, quantity="circulation_interface_sum_half", cumsum=False, nproc=useNproc)  #vorticity_int_sum
  tl, int_tsc_half  = phmmfp.get_1D_time_series_data(processed_files, species=fluid, quantity="tau_sc_interface_sum_half", cumsum=False, nproc=useNproc) # flow term 1
  #tl, int_tc_half  = phmmfp.get_1D_time_series_data(processed_files, species=fluid, quantity="tau_conv_interface_sum_half", cumsum=False, nproc=useNproc) # flow term 2
  tb, int_b_half   = phmmfp.get_1D_time_series_data(processed_files, species=fluid, quantity="baroclinic_interface_sum_half", cumsum=False, nproc=useNproc)   # baroclinic term 
  tl, int_L_E_half = phmmfp.get_1D_time_series_data(processed_files, species=fluid, quantity="curl_Lorentz_E_interface_sum_half", cumsum=False, nproc=useNproc) # Lorentz E component 
  tl, int_L_B_half= phmmfp.get_1D_time_series_data(processed_files, species=fluid, quantity="curl_Lorentz_B_interface_sum_half", cumsum=False, nproc=useNproc) # Lorentz B component

  int_DGDT_half = copy.deepcopy(int_tsc_half)
  if type(int_tsc_half) == list:
    for i in range(len(int_DGDT_half)):
      int_DGDT_half[i] = int_tsc_half[i] + int_b_half[i] + int_L_E_half[i] + int_L_B_half[i]
  else:
     int_total_odot_half = int_tsc_half + int_tc_half + int_b_half + int_L_E_half + int_L_B_half
     #int_DGDT_half = int_tsc_half + int_b_half + int_L_E_half + int_L_B_half

  teta, eta   = phmmfp.get_1D_time_series_data(processed_files, species=fluid, quantity="y_avg_int_width", cumsum=False, nproc=useNproc)
  teta, geta   = phmmfp.get_1D_time_series_data(processed_files, species=fluid, quantity="global_int_width", cumsum=False, nproc=useNproc)
  tetad, etad = phmmfp.get_1D_time_series_data(processed_files, species=fluid, quantity="growth_rate", cumsum=False, nproc=useNproc)
  tetad, getad = phmmfp.get_1D_time_series_data(processed_files, species=fluid, quantity="global_growth_rate",cumsum=False, nproc=useNproc)

  axes[-1].set_xlabel(r"$t$")
  for ax in axes[0:-1]:
      ax.set_xticklabels([])
  for ax in axes:
      ax.set_xlim(0, t[-1])
      ax.plot([0,1], [0,0], 'k--', lw=0.4, alpha=0.3) #set the dotted line on zero
 
  color = "k"; lw = 0.5
  tp.append([t, int_circ, {"color":color, "ls":"-", "lw":lw, "label":"Full interface"}, ax0])
  tp.append([t, int_circ_half, {"color":'g', "ls":"--", "lw":lw, "label":"1/2 interface"}, ax0])
  ###=====================================full interface properties 
  #tp.append([tl, int_tsc, {"color":'g', "ls":"-", "lw":lw, "label":r"$\dot\Gamma_{comp.}$"}, ax1])
  #tp.append([tl, int_tc, {"color":'b', "ls":"--", "lw":lw, "label":r"$\dot\Gamma_{conv.}$"}, ax1])
  #tp.append([tb, int_b, {"color":'c', "ls":"-","marker":"x", "ms":1., "lw":lw,"label":r"$\dot\Gamma_{baro.}$" }, ax1])
  #tp.append([tl, int_L_E, {"color":'r', "ls":"-.", "lw":lw, "label":r"$\dot\Gamma_{L,E}$"}, ax1])
  #tp.append([tl, int_L_B, {"color":'m', "ls":":", "lw":lw, "label":r"$\dot\Gamma_{L,B}$"}, ax1])
  #tp.append([tl, int_DGDT_full, {"color":color, "ls":"-", "lw":lw, "marker":"o", "ms":1., "label":r"$\Sigma\dot\Gamma_{z,full}$"}, ax1])
  ###=====================================Half interface properties 
  tp.append([tl, int_tsc_half, 
    {"color":'g', "ls":"-", "lw":lw, "label":r"$\dot\Gamma_{comp.}$"}, ax2])
  #tp.append([tl, int_tc_half, 
  #  {"color":'b', "ls":"--", "lw":lw, "label":r"$\dot\Gamma_{conv.}$"}, ax2])
  tp.append([tb, int_b_half, 
    {"color":'c', "ls":"-","marker":"x", "ms":1., "lw":lw,"label":r"$\dot\Gamma_{baro.}$" }, ax2])
  tp.append([tl, int_L_E_half, 
    {"color":'r', "ls":"-.", "lw":lw, "label":r"$\dot\Gamma_{L,E}$"}, ax2])
  tp.append([tl, int_L_B_half, 
    {"color":'m', "ls":":", "lw":lw, "label":r"$\dot\Gamma_{L,B}$"}, ax2])
  tp.append([tl, int_DGDT_half,{"color":color, "ls":"-", "lw":lw, 
    "marker":"o", "ms":1., "label":r"$\Sigma\dot\Gamma_{z,half}$"}, ax2])
  tp.append([teta, eta, {"color":color, "ls":"-", "lw":lw,"label":"Avg"}, ax3])
  tp.append([teta, geta, {"color":'g', "ls":"--", "lw":lw,"label":"Global"}, ax3])
  tp.append([tetad, etad, {"color":color, "ls":"-", "lw":lw,"label":"Avg"}, ax4]) 
  tp.append([tetad, getad, {"color":'g', "ls":"--", "lw":lw,"label":"Global"}, ax4]) 

  #tp.append([tA, int_A, {"color":'g', "ls":"-", "lw":lw,"label":"Full interface Area"}, ax5]) 

  for p in tp:
      #print(p[2])
      ax = p[-1]
      ax.plot(p[0], p[1], **p[2])
      ax.legend(frameon=False, ncol=2, prop={"size":8}, loc=1)
  gs.tight_layout(fig, h_pad=0.05, w_pad=0.01)
  
  name = date+"_Interface_Statistics_" + key + "lvl%i"%(level)
  name = name.replace(".","p")
  name += ".png"
  fig.savefig(name, format='png', dpi=600, bbox_inches='tight')
  print("saved ",name)
  plt.close(fig)
  return 


###################################################################################
#                               Parameter settings                                #
###################################################################################
prepare_data = False # look for existing data file (use dictionary name assigned 
                    # here), create new file or use the existing file.
plot = True ; # to plot or not to plot, that is the question...

consVarComparison = False; 
plot_interface_stats = True # plot interface statistics 
plot_ion_electron_interface_comparison = True

max_res = 2048 # mas resolution used --- debug 
print("View changed from standard")
#view =  [[-0.4, 0.0], [1.4, 1.0]] # what windo of data to view 
view =  [[-0.2, 0.0], [1.6, 1.0]] # what windo of data to view 
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
                    #"SRMI-OP-16-Res-512-IDEAL-CLEAN":("/scratch/project/rmitfp/0000-2D-RMI-Collisions/000-Option-16-FullSuiteResults-MEMORYREDUCED/Ideal-Clean", -1), 
                    #"SRMI-OP-16-Res-512-INTRA-ANISO-MAG-Z":("/scratch/project/rmitfp/0000-2D-RMI-Collisions/000-Option-16-FullSuiteResults-MEMORYREDUCED/Intra-Aniso-Magnetised-Z", -1), 
                    #"SRMI-OP-16-Res-512-INTRA-ISO-MAG-Z":("/scratch/project/rmitfp/0000-2D-RMI-Collisions/000-Option-16-FullSuiteResults-MEMORYREDUCED/Intra-Iso-Magnetised-Z", -1), 
                    #"SRMI-OP-16-Res-512-INTRA-ANISO-CLEAN":("/scratch/project/rmitfp/0000-2D-RMI-Collisions/222-Option-16-Anisotropic-Intra", -1), 
                    #"SRMI-OP-16-Res-512-INTRA-ISO-CLEAN":("/scratch/project/rmitfp/0000-2D-RMI-Collisions/222-Option-16-isotropic-Intra", -1), 
                    #"TRMI-OP-16-Res-512-INTRA-ISO":("/scratch/project/rmitfp/0000-2D-RMI-Collisions/333-Option-16-TRMI/Intra-Iso-Clean", -1), 
                    #"SRMI-OP-16-Res-512-INTER-DIRTY":("/scratch/project/rmitfp/0000-2D-RMI-Collisions/000-Option-16-FullSuiteResults-MEMORYREDUCED/20210922-SRMI-option-16-Inter", -1), 
                    #"SRMI-OP-16-Res-512-FB-DIRTY":("/scratch/project/rmitfp/0000-2D-RMI-Collisions/000-Option-16-FullSuiteResults-MEMORYREDUCED/20210922-SRMI-option-16-FullBrag", -2), 

                    #"SRMI-OP-16-Res-2048-INTER-ANISO":("/media/kyriakos/Expansion/222_TINAROO_BACKUP/HLLC_Simulations_Production_Quality/Op-16-Clean-Inter-Anisotropic", -2), 
                    #"SRMI-OP-16-Res-2048-INTER-ISO":("/media/kyriakos/Expansion/222_TINAROO_BACKUP/HLLC_Simulations_Production_Quality/Op-16-Clean-Inter-Isotropic", -2), 

                    #"SRMI-Option-16-Res-2048-Ideal-dirty":("/media/kyriakos/Expansion/000_MAGNUS_SUPERCOMPUTER_BACKUP/ktapinou/SRMI-Option-16-Res-2048-Ideal-Dirty", -1)}
                    #"SRMI-Option-16-Res-2048-Intra-Anisotropic":("/media/kyriakos/Expansion/000_MAGNUS_SUPERCOMPUTER_BACKUP/ktapinou/SRMI-Option-16-Res-2048-Intra-Anisotropic/", -1), 
                    #"SRMI-Option-16-Res-2048-Intra-Isotropic":("/media/kyriakos/Expansion/222_TINAROO_BACKUP/20220125-Op-16-Clean-Intra",-1),

"SRMI-OP-16-Res-512-IDEAL-CLEAN":("/media/kyriakos/Expansion/999_RES_512_RUNS/tinaroo_Ideal-Clean-HLLE/Ideal-Clean/", -1), 
"SRMI-OP-16-Res-512-INTRA-ANISO":("/media/kyriakos/Expansion/999_RES_512_RUNS/magnus_HLLC_SRMI-Option-16-Res-512-INTRA-Anisotropic/SRMI-Option-16-Res-512-INTRA-Anisotropic/", -1), 
"SRMI-OP-16-Res-512-INTRA-ISO-":("/media/kyriakos/Expansion/999_RES_512_RUNS/magnus_HLLC_SRMI-Option-16-Res-512-INTRA-Isotropic/SRMI-Option-16-Res-512-INTRA-Isotropic/", -1), 
"SRMI-OP-16-Res-512-INTER-ANISO":("/media/kyriakos/Expansion/999_RES_512_RUNS/magnus_HLLC_SRMI-Option-16-Res-512-Inter-Anisotropic/SRMI-Option-16-Res-512-Inter-Anisotropic/", -1), 
"SRMI-OP-16-Res-512-INTER-ISO":("/media/kyriakos/Expansion/999_RES_512_RUNS/magnus_HLLC_SRMI-Option-16-Res-512-Inter-Isotropic/SRMI-Option-16-Res-512-Inter-Isotropic/", -1), 
"SRMI-OP-16-Res-512-FB-ISO":("/media/kyriakos/Expansion/999_RES_512_RUNS/magnus_HLLC_SRMI-Option-16-Res-512-FB-Isotropic/SRMI-Option-16-Res-512-FB-Isotropic/", -1), 
"SRMI-OP-16-Res-512-FB-ANISO-ISO":("/media/kyriakos/Expansion/999_RES_512_RUNS/magnus_SRMI-Option-16-Res-512-FB-Anisotropic/", -1)
                   }

###################################################################################
#                                 Extract data                                    #
###################################################################################
  outputKeyword = 'plt'
  if prepare_data:
    for key, (simDir, level) in simOutputDirec.items():
      phmmfp.get_batch_data(key, simDir, level, max_res, window, n_time_slices, 
                            nproc=6, outputType=[outputKeyword]) #plt or chk files? default to plt
  
###################################################################################
#                                 Plot statistics                                 #
###################################################################################
  if plot:
  
    print("Begin plotting")
    for key, (simDir, level) in simOutputDirec.items():
      # ========================= organise the time steps and increaments=========#
      dir_name = phmmfp.get_save_name(key, simDir, level)
      outputFiles = get_files(simDir, include=[outputKeyword], get_all=False)
  
      if n_time_slices == 1:
        contour_save_increment = 1
      else:
        contour_save_increment = math.floor(len(outputFiles)/(n_time_slices-1))
  
      data_index = []
      #===================Setting contour indexes
      for i in range(n_time_slices-1):
        data_index.append( int(i*contour_save_increment))
      data_index.append(int(len(outputFiles)-1))
      print("Search for times")
      if True: # Use non standard time frames for outputs overwritting settings above
        FTF_inputs = {}
        FTF_inputs['times_wanted'] = [0.1*i for i in range(1,6)]
        #[0.0, 0.01, 0.02, 0.03, 0.04, 0.05] #07, 0.08]  
        FTF_inputs['frameType'] = 2
        FTF_inputs['dataIndex'] = data_index
        FTF_inputs['n_time_slices'] = n_time_slices
        FTF_inputs['time_slices'] = time_slices
        FTF_inputs['fileNames'] = outputFiles
        FTF_inputs['level'] = level
        FTF_inputs['window'] = window
        print("Search function")
        data_index, n_time_slices, time_slices = phmmfp.find_time_frames(FTF_inputs)
        print("\t..done")
        if n_time_slices > 7:
          data_index = data_index[0:7]; n_time_slices = len(data_index); 
          time_slices = range(n_time_slices);
  
      data_index_EM = copy.deepcopy( data_index ); data_index_VEL = copy.deepcopy( data_index ) 
      n_time_slices_EM = n_time_slices; n_time_slices_VEL = n_time_slices;
      time_slices_EM = range(n_time_slices); time_slices_VEL = range(n_time_slices)
  
      print('Data_indexes used:', data_index)
   
      # ============================= contour of conserved properties ============#
      if  consVarComparison:
        print('\nPlotting conserved variables contour plot comparison')
    
        label_density_plots = '20220509-'+key+'density-EM-Early-Precursor-Saturated-2-velocity'
        label_axis = [r"$\rho_e$", r"$\rho_i$", r"$v_{x,e}$", r"$v_{x, e}$", r"$E_x$", r"$E_y$", r"$B_z$"] #r"$\varrho_i$"]
        raw_data_attr_names = ['rho-electrons','rho-ions', 'x_vel-electrons', 'y_vel-electrons', 'x_D-field', 'y_D-field', 'z_B-field']  
  
        plot_raw_data_attribute(key, outputKeyword, raw_data_attr_names, level, label_axis, 
          label_density_plots, simDir, data_index, n_time_slices, time_slices, view, 
          range_low=0., range_high=1., t_low=0, t_high=0, t_dt = 0.0025 )

      # ======================= Interface statistics ===============================#
      if plot_interface_stats:
        print('\nPlotting interface statistics')
        date = "20220608_IONS"
        interfaceStatistics("ions", key, date, simDir, level, 2, 2, nproc=8)

      if plot_ion_electron_interface_comparison:

        date = "20220610_ION_ELECTRON_COMPARISON"
        ionElectronInterfaceStatistics

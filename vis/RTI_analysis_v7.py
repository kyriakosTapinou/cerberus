#/usr/bin/env/ python3
# -*- coding: utf-8 -*-
"""
Sun 20200114
v1
  Adapted from CannyEdgeDetector_v3.py to find the pressure and lorentz force 
  contributions to the secondary RTI. Effectively results in the calculation of 
  the pressure gradient and Lorentz force accelerations that are then plotted 
  over a contour of the interface. 
v6
  Added much funtionality, also linked to centralised store if derived value 
  functions. 
  Added in scalar or list option for scales, allowing for scaling by a factor 
  or to set the scale. 
v7 updated for default AMReX file output format (cereberus went away from h5)

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
visulaisation_code_folder ="/home/kyriakos/Documents/Code/000_cerberus_dev/githubRelease-cerberus/cerberus/vis/" 
derived_functions_code_folder = "/media/H_drive/000_PhD/001_SimulationDiagnostics/000_BackupPostProcessingTools/"

if visulaisation_code_folder not in sys.path:
  sys.path.insert(0, visulaisation_code_folder)
if derived_functions_code_folder not in sys.path:
  sys.path.insert(0, derived_functions_code_folder)

#from get_hdf5_data_kyri_edit import ReadHDF5
from get_boxlib import ReadBoxLib, get_files
import PHM_MFP_Solver_Post_functions_v4 as phmmfp

#======================================Functions=====================================
def pack_er_in_boys(name, time_floats, x_l, x_h, a_scale, p_scale, o_scale, label_append):

  #======================================Example data and parameters===================
  dataDirs = [("SRMI-Option-16-Res-2048-Intra-Anisotropic","/media/kyriakos/Expansion/000_MAGNUS_SUPERCOMPUTER_BACKUP/ktapinou/SRMI-Option-16-Res-2048-Intra-Anisotropic/"), 
          ]
  dDir = dataDirs[0][1]
  label = dataDirs[0][0]

  #####==================================Parameters====================================
  dFiles = get_files(dDir, include=["plt"], get_all=False)
  #n = int(0.5*len(h5files))

  properties = ["a_x", "a_y", "pgrad_x", "pgrad_y", "L_x_total", "L_y_total",'o_dot_baro','msk']
  data_properties = ["a_x", "a_y", "pgrad_x", "pgrad_y", "L_x_total", "L_y_total", 'o_dot_baro']

  plot_properties = ["a_x", "a_y", "pgrad_x", "pgrad_y", "L_x_total", "L_y_total", 'o_dot_baro']
  cmap_lst = ['bwr', 'bwr','bwr','bwr','bwr','bwr', 'bwr']
  prop_labels = [r'$a_x$', r'$a_y$', r'$\frac{\partial P}{\partial x}$', 
                  r'$\frac{\partial P}{\partial y}$', r'$a_{L,x}$', r'$a_{L,y}$', 
                  r'$\dot{\omega}_{baro}$']
  data_dict = {}  
  for i in properties:
    data_dict[i] = {}
  t_list = []
  level = -1
  tol = 0.45
  view = [[-0.4, 0.0], [0.4,1.]]

  defaultOptions = {'name':name, 'level':level} 
  options_L_x_E = copy.deepcopy(defaultOptions); options_L_x_E['quantity'] = 'L_x_E'
  options_L_y_E = copy.deepcopy(defaultOptions); options_L_y_E['quantity'] = 'L_y_E'
  options_L_x_B = copy.deepcopy(defaultOptions); options_L_x_B['quantity'] = 'L_x_B'
  options_L_y_B = copy.deepcopy(defaultOptions); options_L_y_B['quantity'] = 'L_y_B'

  options_L_x_total = copy.deepcopy(defaultOptions); options_L_x_total['quantity'] = 'L_x_total'
  options_L_y_total = copy.deepcopy(defaultOptions); options_L_y_total['quantity'] = 'L_y_total'
  options_ion_omega_dot_baro = {'name':'ion', 'quantity':'vort_dot_baro', 'level':level}

  Vel_required = False
  if Vel_required == True: vel_dict = {};

  cmap_gray_r2rgba = mpl.cm.gray_r

  # find times for files 
  FTF_inputs = {}
  FTF_inputs['times_wanted'] = time_floats
  FTF_inputs['frameType'] = 2
  FTF_inputs['fileNames'] = dFiles
  FTF_inputs['level'] = 0 
  FTF_inputs['window'] = view
  print("Find files corresponding to desired times...")
  data_indexes, n_time_slices, time_slices = phmmfp.find_time_frames(FTF_inputs)
  print("\t...times found.")

  for (i_t, time_float) in enumerate(time_floats):
    n = data_indexes[i_t] #int(time_float*len(dFiles))
    #if n >= len(dFiles): n = len(dFiles)-1
    dFile = dFiles[n]
    rc = ReadBoxLib(dFile, level, view)
    dx = rc.data['levels'][level]['dx']; dy = dx[1]; dx = dx[0]
   
    print('\nTime:\t',rc.time, 'n=', n)
    ### ===== Properties =====  
    prs_inputs = {}
    prs_inputs['name'] = name
    x_prs, y_prs, prs = phmmfp.get_pressure(rc, prs_inputs) #'%s'%name)
    x_rho, rho = rc.get("rho-%s"%name); y_rho = x_rho[1]; x_rho = x_rho[0]
    data_dict['pgrad_x'][time_float], data_dict['pgrad_y'][time_float] = \
      phmmfp.get_grad_components( prs, dx, dy)

    # accelerations due to the Lorentz force
    """
    x, y, L_x_E = get_Lorentz(rc, options_L_x_E)
    x, y, L_y_E = get_Lorentz(rc, options_L_y_E)
    x, y, L_x_B = get_Lorentz(rc, options_L_x_B)
    x, y, L_y_B = get_Lorentz(rc, options_L_y_B)
    # sums in x and y 
    L_x = L_x_E + L_x_B
    L_y = L_y_E + L_y_B
    """
    x, y, data_dict['L_x_total'][time_float] = phmmfp.get_Lorentz(rc, options_L_x_total)
    x, y, data_dict['L_y_total'][time_float] = phmmfp.get_Lorentz(rc, options_L_y_total)
   
    # accelerations due to the pressure gradients 
    a_dpdx = -data_dict['pgrad_x'][time_float]/rho
    a_dpdy = -data_dict['pgrad_y'][time_float]/rho
    # total acceleration contributions 
    data_dict['a_x'][time_float] = a_dpdx + data_dict['L_x_total'][time_float]
    data_dict['a_y'][time_float] = a_dpdy + data_dict['L_y_total'][time_float]

    """
    # find which acceleration dominates
    a_dpdx_dom = (np.absolute(a_dpdx/a_total_x) > np.absolute(L_x_E/a_total_x) ) & ( np.absolute(a_dpdx/a_total_x) > np.absolute(L_x_B/a_total_x))
    L_x_E_dom =( np.absolute(L_x_E/a_total_x) > np.absolute(a_dpdx/a_total_x) ) & (np.absolute(L_x_E/a_total_x) > np.absolute(L_x_B/a_total_x))
    L_x_B_dom = (np.absolute(L_x_B/a_total_x) > np.absolute(a_dpdx/a_total_x)) & (np.absolute(L_x_B/a_total_x) > np.absolute(L_x_E/a_total_x))
    
    a_dpdy_dom = (np.absolute(a_dpdy/a_total_y) > np.absolute(L_y_E/a_total_y)) & (np.absolute(a_dpdy/a_total_y) > np.absolute(L_y_B/a_total_y))
    L_y_E_dom =( np.absolute(L_y_E/a_total_y) > np.absolute(a_dpdy/a_total_y)) & (np.absolute(L_y_E/a_total_y) > np.absolute(L_y_B/a_total_y))
    L_y_B_dom = (np.absolute(L_y_B/a_total_y) > np.absolute(a_dpdy/a_total_y)) & (np.absolute(L_y_B/a_total_y) > np.absolute(L_y_E/a_total_y))
    
    dom_mask_x = 1.*a_dpdx_dom + 2.*L_x_E_dom + 3.*L_x_B_dom
    dom_mask_y = 1.*a_dpdy_dom + 2.*L_y_E_dom + 3.*L_y_B_dom
    """
    
    # baroclinic vorticity contributions 
    options_omega_dot_baro= {'name':'%s'%name, 'quantity':'vort_dot_baro', 'level':level}
    x, y, data_dict['o_dot_baro'][time_float] = phmmfp.get_vorticity_dot(rc, options_omega_dot_baro )

    """    
    # bwr to rgba for plotting 
    o_b_dot = {}
    limit_bwr2rgba = {}
    for time_float in time_points:
      limit_bwr2rgba[time_float] = 0.1*max(abs( data_dict['o_b_dot'][time_float][x_l:x_h,:]*1.*data_dict['msk'][time_float][x_l:x_h,:]).max(), abs( data_dict['o_b_dot'][time_float][x_l:x_h,:]*1.*data_dict['msk'][time_float][x_l:x_h,:]).min())
  
      norm_bwr2rgba = mpl.colors.Normalize(vmin=-limit_bwr2rgba[time_float], vmax=limit_bwr2rgba[time_float])
      cmap_bwr2rgba = mpl.cm.bwr
      bwr2rgba = mpl.cm.ScalarMappable(norm=norm_bwr2rgba, cmap=cmap_bwr2rgba) 
      o_b_dot[time_float] = bwr2rgba.to_rgba( data_dict['o_b_dot'][time_float][x_l:x_h,:] )
      o_b_dot[time_float][:,:,3] = o_b_dot[time_float][:,:,3]*1.*data_dict['msk'][time_float][x_l:x_h,:]
    """
    
    Vel_required = False
    if Vel_required:
      try:
        x_vel, y_vel, vel_dict['x'][time_float] = rc.get_flat("x-vel-%s"%name);
        x_vel, y_vel, vel_dict['y'][time_float] = rc.geT_flat("y-vel-%s"%name);
      except:
        x, y, mx = rc.expression("{x-mom-%s}"%name)
        x, y, my = rc.expression("{y-mom-%s}"%name)
        vel_dict['x'][time_float] = mx/rho; vel_dict['y'][time_float] = my/rho; 

    x_gamma, gamma = rc.get("gamma-%s"%name)

    if 'ion' in name or 'electron' in name: ### ===== Interface =====
      x, tracer_array = rc.get("alpha-%s"%name);
      x_tracer = x[0]; y_tracer = x[1]; 

      t_h = 0.5 + tol
      t_l = 0.5 - tol
      mask = (t_l <  tracer_array) & (tracer_array < t_h); mask = np.ma.masked_equal(mask, False);
      del tracer_array; gc.collect()
    """
    elif name =='electron': 
      print("Ghetto electron interface")
      rho_avg = 0.5*(0.02+0.01) #abs(rho[0,0] + rho[-1, -1])/2.
      rho_tol = 0.25*(0.02-0.01)*0.5 #0.4*abs(rho[0,0] - rho[-1, -1])/2.
      #pdb.set_trace()
      t_h = rho_avg + rho_tol
      t_l = rho_avg - rho_tol
      mask = (t_l <  rho) & (rho < t_h) 
      del rho_tol, rho_avg; gc.collect()
    """
    # convert msk to rgba style only for interface

    norm_gray_r2rgba = mpl.colors.Normalize(vmin=0., vmax=1.)
    gray_r2rgba = mpl.cm.ScalarMappable(norm=norm_gray_r2rgba, cmap=cmap_gray_r2rgba) 
    data_dict['msk'][time_float] = gray_r2rgba.to_rgba( 1.*mask, alpha = 0.4)
    data_dict['msk'][time_float][:,:,3] = data_dict['msk'][time_float][:,:,3]*1.*mask
    
    #data_dict['msk'][time_float] = mask*1.
   
    del x_prs, y_prs, x, y, prs, rho, mask; gc.collect()

  ### ===== plotting =====
  print("plot total a_x, a_y, dp/dx, dp_dy, omega")
  #x_l = 3500; #x_h = 5000 #x_rho.shape[0]
  print('x_h = ', x_h)  
  y_l = 0; y_h = int(y_rho.shape[0]/2)
  y_tracer, x_tracer = np.meshgrid(y_tracer[y_l:y_h], x_tracer[x_l:x_h])  
  #define axis 
  wspacing = 0.1; hspacing = 0.05
  # should use 6.7 for figures :
  fig_x, fig_y = phmmfp.get_figure_size(10., len(time_floats), len(plot_properties), 
                   1.*x_rho[x_l:x_h].shape[0]/y_rho.shape[0]*2., wspacing, hspacing, 1)
  #(fig_x, fig_y) = (10, 20)

  #fig_x, fig_y = get_figure_size('thesis', 1)
  #fig_y = fig_y*len(time_slices_EM)*plot_height_factor
  #fig_x = fig_x*5*plot_width_factor
  fig = plt.figure(figsize=(fig_x,fig_y))
  gs = gridspec.GridSpec(len(time_floats), len(plot_properties), wspace=wspacing, 
        hspace=hspacing) #,width_ratios=[1], height_ratios=[1,1,1,1,1],)
  
  #fig, ax = plt.subplots(1,7)
  ax = {}; divider={}; norm={}; cb={}; big={}; cax={}; val_gcl = {}; v_lim = {}
  scales = [a_scale, a_scale, p_scale, p_scale, a_scale, a_scale, o_scale]

  for (j, key) in enumerate(data_properties):
    print("MaxMin scale\t", key, scales[j])
    val_gcl[key] = []
    val_gcl[key].append( min([data_dict[key][i].min() for i in time_floats]))
    val_gcl[key].append( max([data_dict[key][i].max() for i in time_floats]))
    val_gcl[key] = scales[j]*max( abs(val_gcl[key][0]), abs(val_gcl[key][1]))
    print("val_gcl\t", val_gcl[key])
    for i in time_floats:
      v_lim[key, i] = scales[j]*max( abs(data_dict[key][i].min()), abs(data_dict[key][i].max()))
      print("v_lim\t", v_lim[key, i])

  """
  # acceleration scales 
  v_lim[0] = a_scale*max(abs(a_total_x[x_l:x_h, :].min()), abs(a_total_x[x_l:x_h, :].max()))
  v_lim[1] = a_scale*max(abs(a_total_y[x_l:x_h, :].min()), abs(a_total_y[x_l:x_h, :].max()))
  v_lim[4] = max(abs(L_x[x_l:x_h, :].min()), abs(L_x[x_l:x_h, :].max())) 
  v_lim[5] = max(abs(L_y[x_l:x_h, :].min()), abs(L_y[x_l:x_h, :].max()))
  """  
  if True: # normalise across time 
    if True: #normalise across all accelerations, normalise across all pressures 
      a_lim = max( [ val_gcl[i] for i in ["a_x", "a_y", "L_x_total", "L_y_total"] ] )
      p_lim = max( [ val_gcl[i] for i in ["pgrad_x", "pgrad_y"] ] )

  for i in range(len(time_floats)):
    for j in range(len(plot_properties)):
      time_float = time_floats[i]
      key          = plot_properties[j]
      ax[i,j]      = fig.add_subplot(gs[i,j])
      print(time_float, key)
      if True: # normalise contours across time for accelerations, pressures, and odbaro
        if key in ["a_x", "a_y", "L_x_total", "L_y_total"]:
          use_lim = a_lim 
        elif key in ["pgrad_x", "pgrad_y"]:
          use_lim = p_lim
        else:
          use_lim = val_gcl['o_dot_baro']

        use_cmap = cmap_lst[j]

        ax[i,j].imshow(data_dict[key][time_float][x_l:x_h, y_l:y_h], vmin=-use_lim, vmax=use_lim, cmap=use_cmap)
        
        if key != 'o_dot_baro':
          ax[i,j].imshow(data_dict['msk'][time_float][x_l:x_h,y_l:y_h,:], 
            'gray_r', interpolation = 'none')
          #print(x_tracer.shape, y_tracer.shape, data_dict['msk'][time_float][x_l:x_h,y_l:y_h].shape)
          """
          ax[i,j].contourf(x_tracer, y_tracer, data_dict['msk'][time_float][x_l:x_h,y_l:y_h], 
            #corner_mask=False, alpha=0.6)
            colors='gray', corner_mask=False, alpha=0.6)
          """
        if i == len(time_floats) - 1:
          norm[i,j]  = mpl.colors.Normalize(vmin=-use_lim, vmax=use_lim)
          divider[i,j] = make_axes_locatable(ax[i,j])
          cax[i,j] = inset_axes(ax[i,j], width='80%', height='5%', loc='lower center',
                      bbox_to_anchor=(0., -0.125, 1,1), bbox_transform=ax[i,j].transAxes, 
                      borderpad=0)
          cb[i,j] = mpl.colorbar.ColorbarBase(cax[i,j], orientation="horizontal",
                    cmap="bwr", norm=norm[i,j], extend="both",   
                    ticks=[-use_lim, use_lim], format='%.1f')#format='%1.1E')
          cb[i,j].ax.tick_params(labelsize=5)
      """
      else:
        big[i,j,0] = -max( abs(em_fields[key][i].min()), abs(em_fields[key][i].max()))
        big[i,j,1] = max( abs(em_fields[key][i].min()), abs(em_fields[key][i].max()))
        ax[i,j].pcolormesh(x_em, y_em, em_fields[key][i], vmin=big[i,j,0], 
                           vmax=big[i,j,1], cmap="bwr")
        divider[i,j] = make_axes_locatable(ax[i,j])
        cax[i,j] = inset_axes(ax[i,j], width='80%', height='10%', loc='lower center', 
                    bbox_to_anchor=(0., -0.05, 1,1), bbox_transform=ax[i,j].transAxes, 
                    borderpad=0)
        #cax[i,j]     = divider[i,j].append_axes("bottom", size="10%", pad=0.)
        norm[i,j] = mpl.colors.Normalize(vmin=big[i,j,0], vmax=big[i,j,1])
        cb[i,j] = mpl.colorbar.ColorbarBase(cax[i,j], orientation="horizontal",
                  cmap="bwr", norm=norm[i,j], extend="both", 
                  ticks=[big[i,j,0], 0.5*(big[i,j,1] + big[i,j,0]), big[i,j,1]],
                  format='%1.2E')
      """
      ax[i,j].text(0.95, 0.05, 
        '[%1.E, %1.E]'%(data_dict[key][time_float].min(), data_dict[key][time_float].max()), 
        fontsize=7, horizontalalignment='right', verticalalignment='bottom',
        transform=ax[i,j].transAxes, color="k")
      #ax[i,j].set_aspect(1)
      ax[i,j].axis('on') #
      ax[i,j].set_xticks([]); ax[i,j].set_yticks([])
      mpl.pyplot.tick_params(left=False); mpl.pyplot.tick_params(bottom=False) 

      #ax[i,j].set_xlim(view[0])
      #ax[i,j].set_ylim(view[1])
      if key == plot_properties[0]: #proptime_float ==time_floats[-1]:
        ax[i,j].text(0.05, 0.95,'t=%.3f'%time_floats[i], fontsize=7,horizontalalignment='left'
          ,verticalalignment='top',transform=ax[i,j].transAxes, color="k")
      if i == 0:
        ax[i,j].set_title( prop_labels[j], fontsize=9 )
        #ax[i,j].text(0.95, 0.95, prop_labels[j], fontsize=9, horizontalalignment='right',
        #             verticalalignment='top',transform=ax[i,j].transAxes, color="k")

  #plt.show()
  dpi_use = 600
  name = "20220322_SRMI-He-RTI_accelerations_%s"%(label_append)
  name = name.replace(".","p")
  name += ".png"
  fig.savefig(name, format='png', dpi=dpi_use) # bbox_inches='tight')
  print("saved ",name)
  plt.close(fig)

    
###run this shit 
#times = [0.16, 0.21, 0.23]; label_append = 'Ion-Post_shock_kink'; name = 'ion' #post shock kink
#x_l = 3500; x_h = 4500
#times = [0.02, 0.03, 0.04, 0.07, 0.120, 0.16, 0.18]; label_append = 'electron_ERTI_oscillations_No_Interface'; name = 'electron' # electron early 

#times = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]; label_append = 'Shock_acceleration'; name = 'ion' #post shock kink
#x_l = 3500; x_h = 7500

### Investigating dS=1.0 shock interaction
#times = [ .04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2]
#label_append = 'interface_refraction_t_0p045_0p25'; #x_l = 3000; x_h = 5000
#name = 'ion'
### Inevestigate narrowing of the spike 
#times = [0.6, 0.625, 0.65, 0.775, 0.8, 0.825, 0.85, 0.875, 0.9]
#label_append = 'spike_narrowing_t_0p6_0p9'; x_l = 4000; x_h = 7000
#name = 'ion'

### Comparing to Lorentz plots in paper 2 for SMRI-He
times = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05] # [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07] #[0.5, 0.7, 0.8, 1.]

label_append = 'ELECTRONS_Early_Precursor_SATURTED_t_0'; 
x_l = 0; x_h = -1
name = 'electrons'


#times = [0.03] 
a_scale = 1e-3# ele 1e-3 #ele 1e-4
o_scale = 1e-3# ele 1e-2 #ele 1e-3 
p_scale = 1e-3# ele 1e-2 #ele 1e-
pack_er_in_boys(name, times, x_l, x_h , a_scale, p_scale, o_scale, label_append)
  

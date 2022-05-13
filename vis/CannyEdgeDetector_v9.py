#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-
"""
Sun 20200114
v1
  Implementing the Canny Edege Detector after attempting a simple shock detector based
  on gradient in pressure and density. 

  Canny edge detector method uses the Canny edge detector for image processing 
  combined with a Rankine-Hugoniot condition. Assuming RH condition holds for 2FP
  however this should be verified. The algorithm is as follows:
    1 - Compute gradient mangnitudes and directions of edges 
    2 - Non-Maximum suppression i.e. only proceed with local maximum values 
    3 - Theoretical estimation based on RH condition. 

  Note that for the now the domain boundaries are not considered in the analyis as
  candidate shock points. i.e. only the indexs from i = [1,n-1] and j=[1,n-1] are 
  possible shock candidates. 

  Note convention according to Fujimoto Canny edge paper:
    1 denotes upstream/shock unaffected region
    2 denotes downstream/shock affected region

  Note assumption of constant gamma in entire domain for all fluids. 

  Note that, as per Fujimoto, weak shocks of M2/M1 < 1.01 are exluded for 
  consideration and an error of 0.5 is allowed for the comparison of theoretical
  to cfd M2 states. 

  Note the hack in the defiintion of array_dir in the get_grad_mag_dir function, to handle 
  the exceptions of divide by zero, was tryiing to vatch the signs but wouldn't work sp here we are 
  Could use np.div( input_array_dy, abs(input_array_dy), out=np.ones(input_array_dy.shape), where = input_array_dy!=0)
  
  Note check the v_s hack for the math.tan(0) divide by condition

  Note the authors errors in the paper

v2
  introduced variabl spacing, improved the continuation of shock capturing 

v3
  introduced check for oblique shocks
    involved solving for two direction arrays, one is the previously used 
    prs_mag_dir, the direction of the net pressure gradeint according to 
    arctan function, and the absolute angle of the pressure gradient for usage
    in calculating normal components etc. through the arctan2 function  

v9 - adjusted for boxlib file format from Cerberus (lost rrack of changes up tp v8)
   - Could we possibly increase the distance or chain of cells making up the shock thickness 

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
#======================================Special modules===============================

visulaisation_code_folder ="/home/kyriakos/Documents/Code/000_cerberus_dev/githubRelease-cerberus/cerberus/vis/"
if visulaisation_code_folder not in sys.path:
  sys.path.insert(0, visulaisation_code_folder)

from get_boxlib import ReadBoxLib, get_files

derived_functions_code_folder ="/media/H_drive/000_PhD/001_SimulationDiagnostics/000_BackupPostProcessingTools"
if derived_functions_code_folder not in sys.path:
  sys.path.insert(0, derived_functions_code_folder)

import PHM_MFP_Solver_Post_functions_v4 as phmmfp 

#======================================Functions=====================================

def get_grad_mag_dir(input_array, input_dx, input_dy=False):
  if input_dy == False: input_dy = input_dx
  """
  Find the magnitude and direction of the gradient of an input array with supplied 
  discretisation. Return the grad mag array and the direction array. 
  """
  # is it best to use a variable for the gradient arrays? or is the speed up from a single line of c backend better?
  input_array_dx = np.gradient(input_array, input_dx, axis=0)
  input_array_dy = np.gradient(input_array, input_dy, axis=1)

  array_grad = ( input_array_dx**2. + input_array_dy**2. )**0.5
#  array_dir  = np.arctan( np.divide(input_array_dy, input_array_dx, out=np.ones(input_array_dy.shape)*np.inf, where=input_array_dx!=0.))
  array_dir_1  = np.arctan( np.divide(input_array_dy, input_array_dx, out=input_array_dy*1e-18, where=input_array_dx!=0.))
  array_dir_2  = np.arctan2( input_array_dy, input_array_dx, out=np.arctan(input_array_dy*1e-18), where=input_array_dx!=0.)

  ##### Note the hack above pdb.set_trace()
  return array_grad, array_dir_1, array_dir_2 

def get_ref_cells(input_i, input_j, input_angle, input_spacing = 1):
  """
  Evaluates the indexes of the reference cells to use for the input cell. Return the
  two reference points. 

  # testing ref cells further away
  """
  
  if input_angle >= -math.pi/8. and input_angle < math.pi/8.:
    return [(input_i+input_spacing, input_j), (input_i-input_spacing, input_j)]

  elif input_angle >= math.pi/8. and input_angle < 3.*math.pi/8.:
    return [(input_i+input_spacing, input_j+input_spacing), (input_i-input_spacing, input_j-input_spacing)]

  elif (input_angle >= 3.*math.pi/8. and input_angle < math.pi/2.) or (input_angle >= -math.pi/2. and input_angle < -3.*math.pi/8.):
    return [(input_i, input_j+input_spacing), (input_i, input_j-input_spacing)]

  elif input_angle >= -3.*math.pi/8. and input_angle < -math.pi/8.:
    return [(input_i+input_spacing, input_j-input_spacing), (input_i-input_spacing, input_j+input_spacing)]

  elif int(input_angle) == int(math.pi/2.) or int(input_angle) == -int(math.pi/2.):
    pdb.set_trace()
    return [(input_i, input_j+input_spacing), (input_i, input_j-input_spacing)]
      
  else:
    print("ref points error")
    pdb.set_trace()

def get_row_non_max_suppression(d_in): # may wish to change this to a column search 
                                   # because it lends itself better to 
                                   # multiprocessing (more chunks)
  """
  note the columns are eventually offset by one to handle the no domain boundary 
  handling, this is already handled for the rows by the input row number. Note the first
  and last column are the domain boundaries. 
  """
  input_row     = d_in['input_row'] # global index of the row 
  prs_grad_mag_rows = d_in['prs_grad_mag_rows'] # 3xj slice of the magnitude array
  dir_row_single = d_in['dir_row_single'] # single row slice of the direction array 
  try:
    input_spacing = d_in['input_spacing'] 
  except:
    pdb.set_trace()#input_spacing = 1
  candidate_j = []

  for j in range(input_spacing, len(dir_row_single)-input_spacing): #ignore the spacing columns for the left and right boundary 
    [ref_1, ref_2] = get_ref_cells(input_spacing, j, dir_row_single[j], input_spacing )
    if ref_1[0] < 0 or ref_2[0]<0: pdb.set_trace()
      
    #if ref_1[1] > 2047 or ref_2[1]>2047: pdb.set_trace()
      
    if (prs_grad_mag_rows[input_spacing,j] <=  prs_grad_mag_rows[ref_1[0], ref_1[1]]) \
       or (prs_grad_mag_rows[input_spacing,j] <=  prs_grad_mag_rows[ref_2[0], ref_2[1]]):
      pass # reject the candidate 
    else: # proceed with candidate 
      candidate_j.append( (input_row, j) )

  return candidate_j # could there be an issue with returning an empty cell later?

def get_non_max_suppression(input_d_in_1_master, nproc = 1):
  """
  Function for handling the single or multiprocessing for evaluating the candidate 
  points for a shock in each row. 
  """
  if nproc == 1:
    data = []
    for d in input_d_in_1_master: 
      #data.append(get_single_data(d))
      data.append(get_row_non_max_suppression(d))
  else:
    p = Pool(nproc)
    data = p.map(get_row_non_max_suppression, input_d_in_1_master)
    #data = p.imap(get_single_data, din, )
    p.close()
    p.join()
 
  return data 

def get_single_RankineHugoniotEstimate(d_in):
  """
  Estimate if the candidate point is a shock by checking the theoretical error in
  the conditions across the shock. 
  """
  u_1 = d_in['u_1']; v_1 = d_in['v_1']; h_1 = d_in['h_1']; c_1 = d_in['c_1'] ;
  u_2 = d_in['u_2']; v_2 = d_in['v_2']; h_2 = d_in['h_2']; c_2 = d_in['c_2'] ;
  M_error = d_in['M_error']
 
  gamma_constant = d_in['gamma'] 
  prs_grad_dir_val = d_in['prs_grad_dir_abs']
   
  input_i = d_in['input_i']; input_j = d_in['input_j'] ;

  ### Check for oblqiue shock if angle b/w velocity and pressure grad are nonzero
  try:
    if math.acos(round((u_1*math.cos(prs_grad_dir_val) + v_1*math.sin(prs_grad_dir_val))/(u_1**2+v_1**2)**0.5, 4)) != 0: # angle bw is not zero therefore oblique shockyboi
      #print("oblique")
      U_1n_mag = (u_1*math.cos(prs_grad_dir_val) + v_1*math.sin(prs_grad_dir_val))
      u_1n = U_1n_mag*math.cos(prs_grad_dir_val) 
      v_1n = U_1n_mag*math.sin(prs_grad_dir_val)
      U_2n_mag = (u_2*math.cos(prs_grad_dir_val) + v_2*math.sin(prs_grad_dir_val))
      u_2n = U_2n_mag*math.cos(prs_grad_dir_val) 
      v_2n = U_2n_mag*math.sin(prs_grad_dir_val)
    else:
      #print("normal")
      u_1n = u_1; v_1n = v_1
      u_2n = u_2; v_2n = v_2 
  except:
    print('matherror')
    pdb.set_trace()
  ### calculate reference frame speed 
  u_s = (2.*h_1 + u_1n**2. + v_1n**2. -2.*h_2 - u_2n**2. - v_2n**2.)/2./(u_1n - u_2n + math.tan(prs_grad_dir_val)*(v_1n - v_2n))
#  try:
#    v_s = (2.*h_1 + u_1n**2. + v_1n**2. -2.*h_2 - u_2n**2. - v_2n**2.)/2./((1./math.tan(prs_grad_dir_val))*(u_1n - u_2n) + v_1n-v_2n)
#  except:
#    if math.tan(prs_grad_dir_val) == 0.:
#      v_s = 0. 
#    else:
#      pdb.set_trace()
  v_s = u_s*math.tan(prs_grad_dir_val)
#  if (u_1n - u_s)**2 - ( v_1n - v_s)**2 < 0.:
#    print(input_i, input_j)    
#    pdb.set_trace()
  
  M_1 = ( (u_1n - u_s)**2. + ( v_1n - v_s)**2. )**0.5/c_1
  M_2 = ( (u_2n - u_s)**2. + ( v_2n - v_s)**2. )**0.5/c_2

  #if M_1 < 1.: #and M_2 < 1.:
#    pdb.set_trace()
   # return (False, (False,False), False, False, False) 

#  if  (2.+(gamma_constant-1.)*(M_1**2.)) / (2.*gamma_constant*(M_1**2.)-(gamma_constant-1.)) < 0. :
#    pdb.set_trace()
  M_2t = ( (2.+(gamma_constant-1.)*(M_1**2.)) / (2.*gamma_constant*(M_1**2.)-(gamma_constant-1.)) )**0.5
     
#  if M_1/M_2 - 1 <= 0.00001:# exclude 
#    return (False, (False,False), False, False, False)
#  print("M_1/M_2 - 1" ,M_1/M_2 - 1)
#  print("M_2 error", abs(M_2-M_2t)/M_2)
  if abs(M_2-M_2t)/M_2 <= M_error: #0.5 # record as shock point 
    return (True, (input_i, input_j), M_1, M_2, M_2t)
  else: # to erroneous 
    return (False, (False,False), False, False, False) 

def get_RankineHugoniotEstimate(input_d_in_2_master, nproc = 1):

  if nproc == 1:
      data = []
      for d in input_d_in_2_master: 
        #data.append(get_single_data(d))
        data.append(get_single_RankineHugoniotEstimate(d))
  else:
      p = Pool(nproc)
      data = p.map(get_single_RankineHugoniotEstimate, input_d_in_2_master)
      #data = p.imap(get_single_data, din, )
      p.close()
      p.join()
 
  return data 

def CannyShockDetector(dataDir, name, level, label_prefix, label_suffix, time_points, 
                       view, scale_lst, filter_values=False, filter_levels=False, 
                       y_half=False, plot_properties=["shock", "rho_grad", "prs_grad", "prs", "Lx", "Ly"]):

  titleDict = {"shock":'Shock detector', "prs_grad":r"$|\frac{\partial P}{\partial \vec{x}} |$", "prs":"P", "Lx":r"$L_x$", "Ly":r"$L_y$", "o_b_dot":r"$\dot{\omega}_b$", "rho_grad":r"$|\frac{\partial \rho}{\partial \vec{x}} |$"}
  colourDict = {"shock":"magma", "prs_grad":"magma", "prs":"magma", "Lx":"bwr", "Ly":"bwr", "o_b_dot":"bwr", "rho_grad":"magma"}

  data_properties = []
  #data_properties =  ["shock", "prs_grad", "prs", "Lx", "Ly", "o_b_dot"]
  cmap_lst = []; title_lst = []
  for prop in plot_properties:
    cmap_lst.append(colourDict[prop])
    title_lst.append(titleDict[prop])
    if prop not in data_properties: data_properties.append(prop)
  if "prs_grad" in  data_properties: data_properties = data_properties + ["o_b_dot"];

  print(f"Directory: {dataDir}\tTimes: {time_points}\tSpecies: {name}")

  #####==================================Parameters====================================
  dataFiles = get_files(dataDir, include=['plt'], get_all=True)
  properties = ['rho_grad', "shock", "prs_grad", "prs", "o_b_dot", 'msk', "Lorentz"]
  debugFlag = False 

  if debugFlag == True:
    print("Debugging on ln: 400")
    properties = ["prs_grad", "prs", "o_b_dot", 'msk', "Lorentz"]

  data_dict = {}
  for i in properties:
    data_dict[i] = {}
  if "Lorentz" in properties:
    data_dict['Lx'] = {}; data_dict['Ly'] = {};

  t_list = []; dS_list = [] 
  print("\nGenerally set to 0.25 mach error, reduced for electron breakdown investigation")
  Mach_error = 0.05; tol_msk = 0.45; spacing = 5; 
  #Mach_error = 0.025
  print('\nMach error:', Mach_error, '\tSpacing:', spacing, '\tInterface tolerance:', tol_msk, 'level:', level)
  ### Harvest data
  for i in range(len(time_points)):
    print( "Time: ", time_points[i])
    time_float = time_points[i]
    if filter_values==False and filter_levels==False: 
      filter_levels = 3; filter_value = False
    elif filter_values == False:
      filter_level = filter_levels[i]; filter_value = False
    else:
      filter_value = filter_values[i]; filter_level = 3;

    n = int(time_float*len(dataFiles))
    print( "n = ", n)
    if time_float > 0.995:
      dataFile = dataFiles[-1]
    else:
      dataFile = dataFiles[n]
    
    rc = ReadBoxLib(dataFile, level, view)
    t_list.append(rc.time); dS_list.append(rc.data["skin_depth"])
    dx = rc.data["levels"][level]['dx'][0]; dy = rc.data["levels"][level]['dx'][1]
    print('\nTime:\t', rc.time)   
    ### ===== Properties =====  
    x, y, data_dict['prs'][time_float] = phmmfp.get_pressure(rc, {'name':name})
    x, rho = rc.get("rho-%s"%name);
    try:
      u_vel = rc.get("x_vel-%s"%name)[-1];
      v_vel = rc.get("y_vel-%s"%name)[-1];
    except:
      mx = rc.get("x_mom-%s"%name)[-1]; my = rc.get("y_mom-%s"%name)[-1]
      mz = rc.get("z_mom-%s"%name)[-1]
      u_vel = mx/rho; v_vel = my/rho; z_vel = mz/rho; 
    
      del mx, my, mz; gc.collect()
    x, tracer_array = rc.get("alpha-%s"%name);
    x, gamma = rc.get("gamma-%s"%name);

    print( "\nfinished primitive grab")
    # baroclinic vorticity contributions 
    options_omega_dot_baro= {'name':name, 'quantity':'vort_dot_baro', 'level':level}
    if "o_b_dot" in properties:
      x, y, data_dict['o_b_dot'][time_float] = phmmfp.get_vorticity_dot(rc, options_omega_dot_baro) 
      print("\nFinished vorticity")

    if "Lorentz" in properties:
      options_Lorentz= {'name':name, 'quantity':'Lorentz', 'level':level}
      x, y, data_dict['Lx'][time_float], data_dict['Ly'][time_float] = phmmfp.get_vorticity_dot(rc, options_Lorentz) 
      print("\nFinished Lorentz grab")
    y = x[1]; x = [0]; 
    rc.close() 

    ### ===== Interface =====
    if "msk" in properties:
      t_h = 0.5 + tol_msk
      t_l = 0.5 - tol_msk
      mask = (t_l <  tracer_array) & (tracer_array < t_h) 
      data_dict['msk'][time_float] = mask # np.ma.masked_equal(mask, False)
      del mask 
      print("\nFinished mask grab")


    ### ===== Guts: prs grad and shock detector =====
    if "rho_grad" in properties:
      data_dict['rho_grad'][time_float], rho_grad_dir, rho_grad_dir_abs = get_grad_mag_dir(rho, dx, dy)
      print("finished density gradient grab")

    if "prs_grad" in properties:
      data_dict['prs_grad'][time_float], prs_grad_dir, prs_grad_dir_abs = get_grad_mag_dir(data_dict['prs'][time_float], dx, dy)
      print("\nfinished pressure gradient grab")

    if "shock" in properties:
      if False in np.equal( prs_grad_dir, prs_grad_dir):
        pdb.set_trace()
      
      h_partial = gamma/(gamma-1.)*(data_dict['prs'][time_float]/rho)
      c_sound   = (gamma*data_dict['prs'][time_float]/rho)**0.5 
      
      # generate input data for candidates according to local maxima
      d_in_1_master = []
      for i in range(spacing, data_dict['prs_grad'][time_float].shape[0]-spacing): # for each row except the spacing rows of top and bottom boundary
        d_in_1_master.append({'input_row':i, 'prs_grad_mag_rows':data_dict['prs_grad'][time_float][i-spacing:i+spacing+1, :], 'dir_row_single':prs_grad_dir[i,:], 'input_spacing':spacing})
      
      # extract candidate points
      candidate_points = get_non_max_suppression(d_in_1_master, nproc = 1)
      
      # Filter noise according to repeated averaging 
      if True:
        print("Filtering noise based on avgs of pressure gradients")
        first_pass = True
        i = 0
        tol_f = 0.4# 0.6    
        loopVar = True
        while i < filter_level:
          if first_pass:
            print("#####=====Iteration:", i)
            first_pass = False
            avg_prs_grad = data_dict['prs_grad'][time_float].sum()/data_dict['prs_grad'][time_float].size
            if filter_value == False:
              tol = avg_prs_grad*0.5
            else: 
              tol = filter_value
            print("Original avg pressure gradient", avg_prs_grad )
            print("Pressure gradient absolute tol:", tol )
      
          else:
            ## update
            avg_prs_grad = data_dict['prs_grad'][time_float][mask_prs].sum()/data_dict['prs_grad'][time_float][mask_prs].size # update the tolerance 
            print( "#####=====Iteration:", i  )
            print( "new prs_grad avg", avg_prs_grad  )
            print( "max, min", data_dict['prs_grad'][time_float][mask_prs].max(), data_dict['prs_grad'][time_float][mask_prs].min() )
          
            while avg_prs_grad*tol_f <  data_dict['prs_grad'][time_float][mask_prs].min():
              tol_f += 0.05
            tol = avg_prs_grad*tol_f
            print( "Pressure gradient absolute tol:", tol )
      
          t_h =  tol
          mask_prs =  data_dict['prs_grad'][time_float] > t_h
    
          plot = False 
          if plot:
            pdb.set_trace()
            fig = plt.figure(frameon=False)
            plt.imshow(mask_prs, 'magma', interpolation = 'none')
            plt.imshow(data_dict['msk'][time_float], 'gray', alpha=0.75, interpolation = 'none')    
            plt.show()
    
            pdb.set_trace()
          if filter_value != False:
            break 
          #pdb.set_trace()
          i += 1 
      
      del d_in_1_master, avg_prs_grad; gc.collect()
      
      #return 
    
      test_candidates = []
      for list_1 in candidate_points:
        for candidate in list_1:
          if mask_prs[candidate[0], candidate[1]] == True: #candidate in filter_values:    
            test_candidates.append(candidate)
      
      del candidate_points, mask_prs; gc.collect()
      
      candidate_shock_mask = np.zeros(data_dict['prs'][time_float].shape)
      for (i,j)  in test_candidates:
        candidate_shock_mask[i, j] = 1

      #pdb.set_trace() 
      #fig = plt.figure(frameon=False)
      #plt.imshow(data_dict['msk'][time_float], 'magma', interpolation = 'none')
      #plt.imshow(candidate_shock_mask, 'winter', alpha=0.5, interpolation = 'none')
      #plt.show()
       
      ### evaluate candidate points according to Ranking-Hugoniot normal shock relations
      # for a single point we have 
      d_in_2_master = []
      for (i,j) in test_candidates:
        # evaluate upstream and down stream references according to total pressure 
        [ref_1, ref_2] = get_ref_cells(i, j, prs_grad_dir[i, j], spacing)
        # if ref_1 is the upstream
        if (data_dict['prs'][time_float][ref_1[0], ref_1[1]] + 0.5*rho[ref_1[0], ref_1[1]]*(u_vel[ref_1[0], ref_1[1]]**2. + v_vel[ref_1[0], ref_1[1]]**2.)) \
           > (data_dict['prs'][time_float][ref_2[0], ref_2[1]] + 0.5*rho[ref_2[0], ref_2[1]]*(u_vel[ref_2[0], ref_2[1]]**2. + v_vel[ref_2[0], ref_2[1]]**2.)):
          ref_u = ref_1
          ref_d = ref_2
        elif (data_dict['prs'][time_float][ref_1[0], ref_1[1]] + 0.5*rho[ref_1[0], ref_1[1]]*(u_vel[ref_1[0], ref_1[1]]**2. + v_vel[ref_1[0], ref_1[1]]**2.)) \
           < (data_dict['prs'][time_float][ref_2[0], ref_2[1]] + 0.5*rho[ref_2[0], ref_2[1]]*(u_vel[ref_2[0], ref_2[1]]**2. + v_vel[ref_2[0], ref_2[1]]**2.)):
          ref_u = ref_2
          ref_d = ref_1
        else:
          # check this is not a shocki
          print( "Equal total pressures detected in candidate" )
          pdb.set_trace()
    
        c_1 = c_sound[ref_u[0], ref_u[1]] 
        c_2 = c_sound[ref_d[0], ref_d[1]]  
        d_in_2_master.append({'input_i':i, 'input_j':j, 'u_1':u_vel[ref_u[0], ref_u[1]], 'v_1':v_vel[ref_u[0], ref_u[1]], 
                              'h_1':h_partial[ref_u[0], ref_u[1]], 'u_2':u_vel[ref_d[0], ref_d[1]], 
                              'v_2':v_vel[ref_d[0], ref_d[1]], 'h_2':h_partial[ref_d[0], ref_d[1]], 
                              'gamma':gamma[i,j], 'prs_grad_dir_abs':prs_grad_dir_abs[i,j], 'c_1':c_1, 'c_2':c_2, 'M_error':Mach_error})
      #pdb.set_trace()
      shock_points = get_RankineHugoniotEstimate(d_in_2_master, nproc = 1)
      
      del d_in_2_master, rho, gamma, prs_grad_dir, prs_grad_dir_abs; gc.collect()
      
      data_dict['shock'][time_float] = np.zeros(data_dict['prs'][time_float].shape)
      for (isShock, (i, j), M_1, M_2, M_2t) in shock_points:
        if isShock:
          data_dict['shock'][time_float][i, j] = 1
    
  ### Plotting 
  print( "Begin plotting" )
  """
  if time_float < 0.5:
    x_l = 3000; x_h = 5500; #, y_l, y_h = 
  elif time_float < 0.8:
    x_l = 3000; x_h = 6500; #, y_l, y_h = 
  elif time_float < 0.95:
    x_l = 3000; x_h = 7000; #, y_l, y_h = 
  else:
    x_l = 3000; x_h = 7500; #, y_l, y_h = 
  """
  #x_l = 3000; x_h = 7500 
  dpi_use = 600
  wspacing = 0.01; hspacing = 0.025 #0.025
  x_l = 0; x_h = data_dict['prs'][time_float].shape[0]
  print( "\n\tOriginal shape for window\t", data_dict['prs'][time_float].shape)
  print( "\n\tReduced shape \t", data_dict['prs'][time_float][x_l:x_h, :].shape)
  y_l = 0; 
  if y_half: y_h = data_dict['prs'][time_float][x_l:x_h, :].shape[1]/2
  else: y_h = data_dict['prs'][time_float][x_l:x_h, :].shape[1]

  gs = gridspec.GridSpec(len(time_points), len(plot_properties), wspace=wspacing, hspace=hspacing)
  # default A4 page width is 6.7 inches, depending on the number of panels
  ratioy = data_dict['prs'][time_float][x_l:x_h, y_l:y_h].shape[1]
  ratiox = data_dict['prs'][time_float][x_l:x_h, y_l:y_h].shape[0]

  fig_x, fig_y = phmmfp.get_figure_size(6.7, len(time_points), len(plot_properties), \
    1.*ratioy/ratiox, wspacing, hspacing, 1)

  print('fig_x, fig_y', fig_x, fig_y, 'x_l', x_l, 'x_h', x_h, 'y_l', y_l, 'y_h', y_h, \
        "yLength", data_dict['prs'][time_float][x_l:x_h, y_l:y_h].shape[1], "xLength",\
         data_dict['prs'][time_float][x_l:x_h, y_l:y_h].shape[0])

  obdot_cmap = 'bwr'
  obdotColourScheme = obdot_cmap

  fig = plt.figure(dpi=dpi_use, figsize=(fig_x,fig_y))

  ax = [[] for i in range(len(time_points))]
  divider = [[] for i in range(len(time_points))]
  cax = [[] for i in range(len(time_points))]
  cb = [[] for i in range(len(time_points))]
  v_lim_min = [[] for i in range(len(time_points))]
  v_lim_max = [[] for i in range(len(time_points))]
  norm = [[] for i in range(len(time_points))]
  for i in range(len(time_points)):
    for j in range(len(plot_properties)):
      ax[i].append( fig.add_subplot(gs[i,j]))
    divider[i] = [[] for j in range(len(data_properties))]
    cax[i] = [[] for j in range(len(data_properties))]
    cb[i] = [[] for j in range(len(data_properties))]
    v_lim_min[i] = {} #[[] for j in range(len(data_properties))]
    v_lim_max[i] = {} #[[] for j in range(len(data_properties))]
    norm[i] = [[] for j in range(len(data_properties))]

  ### contour scales 
  use_global_scale = True
  useSetScales = True  

  print( 'Begin plotting loop' )
  #scale_lst = [1, 1e-2, 1, 1e-2] 
  if True: 
    LMag = 40  
    OMag = 1
    dpdxMag = 80 
    drhodxMag = 80 
  if use_global_scale == True:
    for j in range(len(data_properties)):
      default_lim = 0.
      prop = data_properties[j]; print("scale setting: ", prop)
      for time_float in time_points:
        print( "\nData property", j, "\tTime: ", time_float)
        default_lim = max( abs(default_lim), 
                        abs(data_dict[prop][time_float][x_l:x_h,:].max()),
                        abs(data_dict[prop][time_float][x_l:x_h,:].min()))

      for i in range(len(time_points)):
        if prop == 'prs_grad' or prop == 'shock' or prop == 'rho_grad':
          v_lim_min[i][prop] = 0.
        elif prop == 'prs':
          v_lim_min[i][prop] = 0. #0.5*default_lim#for d_s = 10
        elif prop == 'o_b_dot':
          v_lim_min[i][prop] = -OMag # -30 #-default_lim*scale_lst[j]
        elif prop == 'Lx' or prop == 'Ly':
          v_lim_min[i][prop] = -LMag #-1 #-40. #-default_lim*scale_lst[j]

        if useSetScales:
          if prop == 'prs_grad': 
            v_lim_max[i][prop] = dpdxMag #50 #15#. #default_lim*scale_lst[j] #25#100.0
          elif prop == 'rho_grad': 
            v_lim_max[i][prop] = drhodxMag #50 #15#. #default_lim*scale_lst[j] #25#100.0
          elif prop == 'shock':
            v_lim_max[i][prop] = 1. 
          elif prop == 'prs':
            v_lim_max[i][prop] = 4 #default_lim #2#7 #0.5*default_lim#for d_s = 10
          elif prop == 'o_b_dot':
            v_lim_max[i][prop] = OMag #30 #default_lim*scale_lst[j]
          elif prop == 'Lx' or prop == 'Ly':
            v_lim_max[i][prop] = LMag
        else:
          v_lim_max[i][prop] = default_lim*scale_lst[j]

  """
  else:# else individual limits for each property
    for j in range(len(properties)):
      for i in range(len(time_points)):  
        default_lim = 0.
        prop = properties[j]
        time_float = time_points[i]
        v_lim[i][j] = max(abs(data_dict[prop][time_float].max()),
                        abs(data_dict[prop][time_float].min()))
  """
  #### RGBA for omega dot baro and mask in gray

  o_b_dot = {}
  limit_bwr2rgba = {}
  cmapDict = {'plasma':mpl.cm.plasma, 'bwr':mpl.cm.bwr, 'magma':mpl.cm.magma, 'seismic':mpl.cm.seismic}

  for time_float in time_points:
    limit_bwr2rgba[time_float] = 0.1*max(abs( data_dict['o_b_dot'][time_float][x_l:x_h,y_l:y_h]*1.*data_dict['msk'][time_float][x_l:x_h,y_l:y_h]).max(), abs( data_dict['o_b_dot'][time_float][x_l:x_h,y_l:y_h]*1.*data_dict['msk'][time_float][x_l:x_h,y_l:y_h]).min())
    #normalise
    norm_obdot = mpl.colors.Normalize(vmin=-limit_bwr2rgba[time_float], 
                      vmax=limit_bwr2rgba[time_float])
    norm_gray2rgba = mpl.colors.Normalize(vmin=0., vmax=1.)
    #assign colour map 
    obdot_use_cmap = cmapDict[obdotColourScheme] 
    cmap_gray2rgba = mpl.cm.gray

    obdot2rgba = mpl.cm.ScalarMappable(norm=norm_obdot, cmap=obdot_use_cmap) 
    gray2rgba = mpl.cm.ScalarMappable(norm=norm_gray2rgba, cmap=cmap_gray2rgba) 

    o_b_dot[time_float] =obdot2rgba.to_rgba(data_dict['o_b_dot'][time_float][x_l:x_h,y_l:y_h])
    o_b_dot[time_float][:,:,3] = o_b_dot[time_float][:,:,3]*\
                                    1.*data_dict['msk'][time_float][x_l:x_h,y_l:y_h]

    mask_del = data_dict['msk'][time_float]
    data_dict['msk'][time_float] = gray2rgba.to_rgba( 1.*mask_del )
    data_dict['msk'][time_float][:,:,3] = data_dict['msk'][time_float][:,:,3]*1.*mask_del
    del mask_del; gc.collect()

  if name == 'ion':
    name_label = 'Ion'
  else:
    name_label = 'Electron'
  #### Plotting 
  for i in range(len(time_points)):
    fontsize_val = 8 #10
    if True: # print time stamp or not
      if i==0:
        ax[i][0].text(0.05, 0.05,r't=%.3f'%(t_list[i]), fontsize=fontsize_val, transform=ax[i][0].transAxes, color="w")
      else:
        ax[i][0].text(0.05, 0.05,r't=%.3f'%(t_list[i]), fontsize=fontsize_val, transform=ax[i][0].transAxes, color="w")
    
    for j in range(len(plot_properties)):
      time_float = time_points[i]; prop = plot_properties[j]; cmap_use = cmap_lst[j];

      #if prop == "shock": pdb.set_trace()
      ax[i][j].imshow(np.rot90(data_dict[prop][time_float][x_l:x_h,y_l:y_h], k=1, axes=(0,1)), 
        cmap=cmap_use, vmin=v_lim_min[i][prop], vmax=v_lim_max[i][prop])
      #ax[i][j].pcolormesh(data_dict[prop][time_float][x_l:x_h,y_l:y_h], 
      #  cmap=cmap_use, vmin=v_lim_min[i][prop], vmax=v_lim_max[i][prop])
      if True and (prop == 'prs_grad' or prop == 'rho_grad'):
        ax[i][j].imshow(np.rot90(o_b_dot[time_float], k=1, axes=(0,1)), cmap=obdot_cmap, vmin=v_lim_min[i]['o_b_dot'], vmax=v_lim_max[i]['o_b_dot'])
        #ax[i][j].imshow(np.rot90(o_b_dot[time_float], k=1, axes=(0,1)), cmap=obdot_cmap, 
        #                alpha=0.7, vmin=v_lim_min[i]['o_b_dot'], vmax=v_lim_max[i]['o_b_dot'])

      elif False and prop != "o_b_dot":
        ax[i][j].imshow(np.rot90(data_dict['msk'][time_float][x_l:x_h,y_l:y_h], k=1, 
          axes=(0,1)), 'gray_r', interpolation = 'none')
      
      if False and j > 0:
        ax[i][j].text(0.05, 0.05, r'[%3.1f, %3.1f]'%(data_dict[prop][time_float].min(), data_dict[prop][time_float].max()), fontsize=10, transform=ax[i][j].transAxes, color="w")
      
      ax[i][j].axis('off') 
      
      if True and i == 0:
        ax[i][j].set_title(title_lst[j])
      
      if use_global_scale == True and i == len(time_points)-1 and j>0:
        divider[i][j] = make_axes_locatable(ax[i][j])
        #cax[i][j]  = divider[i][j].append_axes("bottom", size="2.5%", pad=0.)
        cax[i][j] = inset_axes(ax[i][j], width='80%', height='7%', loc='lower center',
                      bbox_to_anchor=(0., -0.125, 1,1), bbox_transform=ax[i][j].transAxes, 
                      borderpad=0)
        norm[i][j] = mpl.colors.Normalize(vmin=v_lim_min[i][prop], vmax=v_lim_max[i][prop])
        cb[i][j] = mpl.colorbar.ColorbarBase(cax[i][j],orientation="horizontal",cmap=cmap_use, norm=norm[i][j], extend="both", ticks=[v_lim_min[i][prop], v_lim_max[i][prop]], format='%.1f') 
        cb[i][j].ax.tick_params(labelsize=10) 
        cb[i][j].ax.set_label(title_lst[j])
        if False and prop == 'prs_grad':
          o_b_dot_div = make_axes_locatable(ax[i][j])
          o_b_dot_cax  = o_b_dot_div.append_axes("top", size="2.5%", pad=0.)
          o_b_dot_norm = mpl.colors.Normalize(vmin=v_lim_min[i]['o_b_dot'], vmax=v_lim_max[i]['o_b_dot'])
          o_b_dot_cb = mpl.colorbar.ColorbarBase(o_b_dot_cax,orientation="horizontal",cmap='bwr', norm=o_b_dot_norm, extend="both", ticks=[v_lim_min[i]['o_b_dot'], v_lim_max[i]['o_b_dot']]) 
          o_b_dot_cb.ax.tick_params(labelsize=10,top=True, labeltop=True ) 

      elif use_global_scale == False and i == len(time_points)-1 and j>0:
        divider[i][j] = make_axes_locatable(ax[i][j])
        cax[i][j] = inset_axes(ax[i][j], width='90%', height='10%', loc='lower center',
                      bbox_to_anchor=(0., -0.125, 1,1), bbox_transform=ax[i][j].transAxes, 
                      borderpad=0)
        norm[i][j] = mpl.colors.Normalize(vmin=v_lim_min[i][prop], vmax=v_lim_max[i][prop])
        cb[i][j] = mpl.colorbar.ColorbarBase(cax[i][j],orientation="horizontal",cmap=cmap_use, norm=norm[i][j], extend="both", ticks=[v_lim_min[i][prop], v_lim_max[i][prop]], format='%1.E')
        cb[i][j].ax.tick_params(labelsize=10) 
      
  print( 'Saving' )
  name = "%s_%s_shock_ME_%.2f_dS_%1.e_%s"%(label_prefix, name, Mach_error, dS_list[0], label_suffix)
  name = name.replace(".","p")
  name += ".jpeg"
  fig.savefig(name, format='jpeg', dpi=dpi_use, bbox_inches='tight')
  print( "saved ",name )
  plt.close(fig)

################# run this  
if __name__ == "__main__":
  dataDirs = [
    "/media/kyriakos/Expansion/000_MAGNUS_SUPERCOMPUTER_BACKUP/ktapinou/SRMI-Option-16-Res-2048-Intra-Anisotropic",
      ]
  view =  [[-0.4, 0.0], [1.4, 1.0]] #
  # for ds = 10
  scale_lst_10 = [1, 1.25e-2, 1, 1e-2, 1, 1] 
  # for ds = 1
  scale_lst_1 = [1, 1e-2, 1, 1e-6] 
  # for ds = 0.1
  scale_lst_0p1 = [1, 1e-2, 1, 1e-6, 0.1, 0.1]

  filter_levels_list = [1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
  filter_values = [10., 10., 10., 10., 10., 10., 10., 10., 8., 7., 6.]
  
  species_name = 'ions'
  filter_values = [6,6,6,6,6,6]
  filter_values = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,3 ]
  filter_levels = [3,3,3,3,3,3]

  if True:
    # Paper TRS analysis section # Paper two 
    time_points = [1] #0.1, 0.3, 0.5, 0.7, 0.9, 1]; 
    label_append = '_Sumarry'; x_l = 0; x_h = -1
    label_prefix = '20220504_SRMI_OPTION-16_LxLy'
    filter_values = [12]*len(time_points)
    CannyShockDetector(dataDirs[0], species_name, -2, label_prefix, label_append, time_points,\
                       view, scale_lst_10, filter_values, 3, y_half=False, \
                       plot_properties=["shock", "rho_grad", "prs_grad", "prs"])


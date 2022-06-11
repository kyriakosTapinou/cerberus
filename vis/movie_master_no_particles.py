import sys
cmd_folder ="/home/kyriakos/Documents/Code/000_cerberus_dev/githubRelease-cerberus/cerberus/vis/" #"/home/s4318421/git-cerberus/cerberus/vis"  # nopep8
import pdb 

if cmd_folder not in sys.path:  # nopep8
    sys.path.insert(0, cmd_folder)

import PHM_MFP_Solver_Post_functions_v6 as phmmfp # running version 3

from tile_mov import tile_movie
from make_mov import make_all, get_particle_trajectories

import matplotlib 

import matplotlib.pyplot as plt # changed this biz
import matplotlib.gridspec as gridspec
from matplotlib import ticker

import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

#matplotlib.use('Agg')
plt.rcParams.update({
#    'figure.figsize': [12, 8],
    "text.usetex": False,
    #"font.family": "sans-serif",
    #"font.sans-serif": ["Helvetica"]
    })

# ==============================================================================
# MAKE MOVIES
# ==============================================================================
def get_interface_mask(ds, c):
    name = c["component"]
    cmap_gray_r2rgba = matplotlib.cm.gray_r

    [x_tracer,y_tracer], tracer_array = ds.get('alpha-%s'%name);
    tol = 0.45#0.001
    t_h = 0.5 + tol
    t_l = 0.5 - tol
    #mask = (t_l <  tracer_array) & (tracer_array < t_h) 
    #mask = np.ma.masked_equal(mask, False)
    mask = np.ma.array(tracer_array, mask=(t_l <  tracer_array) & (tracer_array < t_h) )
    norm_gray_r2rgba = matplotlib.colors.Normalize(vmin=0., vmax=1.)
    gray_r2rgba = matplotlib.cm.ScalarMappable(norm=norm_gray_r2rgba, cmap=cmap_gray_r2rgba) 

    del tracer_array;
    return {"x":x_tracer, "y":y_tracer, "value":mask}

def get_mass_density(ds, c):
    x, r = ds.get("rho-%s"%c["component"])
    time = np.array([ds.time])
    return {"x":x[0], "y":x[1], "value":r, "time":time}

def get_vorticity_wrapper(ds, c):
  inputs = {"name":c["name"], "quantity":c["quantity"]}
  x, y, omega = phmmfp.get_vorticity(ds, inputs) 
  time = np.array([ds.time])
  return {"x":x, "y":y, "value":omega, "time":time}

def get_Lorentz_wrapper(ds, c):
  inputs = {"name":c["name"], "quantity":c["quantity"], "level":c["level"]}
  x, y, L = phmmfp.get_Lorentz(ds, inputs) 
  return {"x":x, "y":y, "value":L}

def get_charge_density(ds, c):
    x, re = ds.get("rho-electrons")
    x, me = ds.get("mass-electrons", grid='node')
    x, qe = ds.get("charge-electrons", grid='node')
    x, ri = ds.get("rho-ions")
    x, mi = ds.get("mass-ions", grid='node')
    x, qi = ds.get("charge-ions", grid='node')

    time = np.array([ds.time])
    return {"x":x[0], "y":x[1], "value":re/me*qe + ri/mi*qi, "time":time}

def get_mom_density(ds, c):
    x, r = ds.get("rho-%s"%c["component"])
    x, xv = ds.get("x_vel-%s"%c["component"])
    x, yv = ds.get("y_vel-%s"%c["component"])
    x, zv = ds.get("z_vel-%s"%c["component"])

    xm = r*xv
    ym = r*yv
    zm = r*zv
    mom = np.sqrt(xm*xm + ym*ym + zm*zm)
    return {"x":x[0], "y":x[1], "value":mom}

def get_Jz(ds, c):
    x, ri = ds.get("rho-ions")
    x, mi = ds.get("mass-ions", grid='node')
    x, re = ds.get("rho-electrons")
    x, me = ds.get("mass-electrons", grid='node')

    x, zvi = ds.get("z_vel-ions")
    x, zve = ds.get("z_vel-electrons")

    x, qi = ds.get("charge-ions", grid='node')
    x, qe = ds.get("charge-electrons", grid='node')


    return {"x":x[0], "y":x[1], "value":ri/mi*qi*zvi + re/me*qe*zve}

def get_Jy(ds, c):
    x, ri = ds.get("rho-ions")
    x, mi = ds.get("mass-ions", grid='node')
    x, re = ds.get("rho-electrons")
    x, me = ds.get("mass-electrons", grid='node')

    x, yvi = ds.get("y_vel-ions")
    x, yve = ds.get("y_vel-electrons")

    x, qi = ds.get("charge-ions", grid='node')
    x, qe = ds.get("charge-electrons", grid='node')

    return {"x":x[0], "y":x[1], "value":ri/mi*qi*yvi + re/me*qe*yve}

def get_Jx(ds, c):
    x, ri = ds.get("rho-ions")
    x, mi = ds.get("mass-ions", grid='node')
    x, re = ds.get("rho-electrons")
    x, me = ds.get("mass-electrons", grid='node')

    x, xvi = ds.get("x_vel-ions")
    x, xve = ds.get("x_vel-electrons")

    x, qi = ds.get("charge-ions", grid='node')
    x, qe = ds.get("charge-electrons", grid='node')

    return {"x":x[0], "y":x[1], "value":ri/mi*qi*xvi + re/me*qe*xve}

def get_number_density(ds, c):
    x, r = ds.get("rho-%s"%c["component"])
    x, m = ds.get("mass-%s"%c["component"], grid='node')

    time = np.array([ds.time])
    return {"x":x[0], "y":x[1], "value":r/m, "time":time}

def get_EM(ds, c):
    x, EM = ds.get("%s-field"%c["component"])

    time = np.array([ds.time])
    return {"x":x[0], "y":x[1], "value":EM, "time":time}

def get_rhoE_EM(ds, c):
    sumEM2 = 0.
    for EM in ["x_D", "y_D", "z_D", "x_B", "y_B", "z_B"]:
      x, EM = ds.get("%s-field"%EM)
      sumEM2 += EM*EM
    
    beta = ds.data["beta"]
    sumEM2 = sumEM2 / beta 
    time = np.array([ds.time])
    return {"x":x[0], "y":x[1], "value":sumEM2, "time":time}

def get_E_error(ds, c):
    x, re = ds.get("rho-electrons")
    x, me = ds.get("mass-electrons", grid='node')
    x, qe = ds.get("charge-electrons", grid='node')
    x, ri = ds.get("rho-ions")
    x, mi = ds.get("mass-ions", grid='node')
    x, qi = ds.get("charge-ions", grid='node')
    Larmor = ds.data['Larmor']
    Debye  = ds.data['Debye']
    
    divE = 0.
    i = 0
    for EM in ["x_D", "y_D"]:
      x, EM = ds.get("%s-field"%EM)
      dx = x[0][1] - x[0][0]
      divE += np.gradient(EM, x[i], axis=i)
      i += 1
    divError = divE - Larmor/Debye/Debye*(re/me*qe + ri/mi*qi)

    time = np.array([ds.time])
    return {"x":x[0], "y":x[1], "value":divError, "time":time}

def get_B_error(ds, c):
    divB = 0.
    i = 0
    for EM in ["x_B", "y_B"]:
      x, EM = ds.get("%s-field"%EM)
      dx = x[0][1] - x[0][0]
      divB += np.gradient(EM, x[i], axis=i)
      i += 1
    time = np.array([ds.time])
    return {"x":x[0], "y":x[1], "value":divB, "time":time}

def get_particles(ds, c):
    idat, rdat =  ds.get_particles(c["component"])
    return {"i":idat, "r":rdat}

def extractPlotData(frame, data, prop, label, getTime=True, signedMaxMin=False):
    if getTime: time = data[prop]["time"][()]
    else: time = None

    xn = data[prop]["x"]
    yn = data[prop]["y"]
    val = data[prop]["value"]
    if signedMaxMin:
      val_min = -max(abs(frame[prop]["min"]), abs(frame[prop]["max"]))

      val_max = - val_min
      useCmap = matplotlib.cm.bwr
    else:
      val_min = 0.
      val_max = frame[prop]["max"]
      useCmap = matplotlib.cm.Greens
    return xn, yn, val, val_min, val_max, label, time, useCmap

def subPlot(axes, fig, gs, xn, yn, nr, nc, val, val_min, val_max, val_label, useCmap):
    #print("subPlot")
    ax = fig.add_subplot(gs[nr, nc]); axes.append(ax)
    #print("Pre pcolormesh")
    pcm = ax.pcolormesh(xn, yn, val, shading='nearest', cmap=useCmap, vmin=val_min, vmax=val_max) #, shading='auto' )
    #print("post pcolomesh")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='1.5%', pad=0.05)
    #fig.colorbar(pcm, cax=cax) #, label=val_label)
    cb = fig.colorbar(pcm, cax=cax, cmap=useCmap) #, label=val_label)
    tick_locator = ticker.MaxNLocator(nbins=5)
    cb.locator = tick_locator
    cb.update_ticks()
    ax.set_title(val_label)
    return 

def plot2(frame, data, output_name):
    #print("Plot start")
    #xn, yn, rhoi, rhoi_min, rhoi_max, label_rhoi, time, rhoi_useCmap =\
    #  extractPlotData(frame, data, "rho-ions", r"$\rho_i$", getTime=True)
    #xn_rhoe, yn_rhoe, rhoe, rhoe_min, rhoe_max, label_rhoe, dummy, rhoe_useCmap = \
    #  extractPlotData(frame, data, "rho-electrons", r"$\rho_e$", getTime=False)
    xn, yn, omegai, omegai_min, omegai_max, label_omegai, time, omegai_useCmap = \
      extractPlotData(frame, data, "omega-ions", r"$\omega_i$", getTime=True, signedMaxMin=True)
    xn_omegae, yn_omegae, omegae, omegae_min, omegae_max, label_omegae, dummy, omegae_useCmap = \
      extractPlotData(frame,data, "omega-electrons", r"$\omega_e$", getTime=False, signedMaxMin=True)
    xn_momi, yn_momi, momi, momi_min, momi_max, label_momi, dummy, momi_useCmap = \
      extractPlotData(frame, data, "mom-ions", r"$\rho v_i$", getTime=False, signedMaxMin=False)
    xn_mome, yn_mome, mome, mome_min, mome_max, label_mome, dummy, mome_useCmap = \
     extractPlotData(frame, data, "mom-electrons", r"$\rho v_e$", getTime=False, signedMaxMin=False)

    #xn, yn, Ex, Ex_min, Ex_max, label_Ex, time, Ex_useCmap = \
    #  extractPlotData(frame, data, "Ex", r"$E_x$", getTime=True, signedMaxMin=True)
    #xn_Ey, yn_Ey, Ey, Ey_min, Ey_max, label_Ey, dummy, Ey_useCmap = \
    #  extractPlotData(frame, data, "Ey", r"$E_y$", getTime=False, signedMaxMin=True)
    #xn_Bz, yn_Bz, Bz, Bz_min, Bz_max, label_Bz, dummy, Bz_useCmap = \
    #  extractPlotData(frame, data, "Bz", r"$B_z$", getTime=False, signedMaxMin=True)
    #xn_Jz, yn_Jz, Jz, Jz_min, Jz_max, label_Jz, dummy, Jz_useCmap = extractPlotData(frame, data, "Jz", r"$J_z$", getTime=False, signedMaxMin=True)
    #xn_Jy, yn_Jy, Jy, Jy_min, Jy_max, label_Jy, dummy, Jy_useCmap = extractPlotData(frame, data, "Jy", r"$J_y$", getTime=False, signedMaxMin=True)
    #xn_Jx, yn_Jx, Jx, Jx_min, Jx_max, label_Jx, dummy, Jx_useCmap = extractPlotData(frame, data, "Jx", r"$J_x$", getTime=False, signedMaxMin=True)
    #xn_dive, yn_dive, dive, dive_min, dive_max, label_dive, dummy, useCmap = extractPlotData(frame, data, "divE_error", r"$\div{E}$", getTime=False)
    #xn_rhoEM, yn_rhoEM, rhoEM, rhoEM_min, rhoEM_max, label_rhoEM, dummy, useCmap = extractPlotData(frame, data, "rhoE_EM", r"$\rho_{E,EM}$", getTime=False)
    xn_rc, yn_rc, rc, rc_min, rc_max, label_rc, dummy, rc_useCmap = extractPlotData(frame, data, "rc", r"$\rho_c$", getTime=False, signedMaxMin=True)

    xn_Lix, yn_Lix, Lix, Lix_min, Lix_max, label_Lix, dummy, Lix_useCmap = extractPlotData(frame, data, "Lorentz-ion-x", r"$\mathcal{L}_{i,x}$", getTime=False, signedMaxMin=True)
    xn_Liy, yn_Liy, Liy, Liy_min, Liy_max, label_Liy, dummy, Liy_useCmap = extractPlotData(frame, data, "Lorentz-ion-y", r"$\mathcal{L}_{i,y}$", getTime=False, signedMaxMin=True)
    xn_Lex, yn_Lex, Lex, Lex_min, Lex_max, label_Lex, dummy, Lex_useCmap = extractPlotData(frame, data, "Lorentz-ele-x", r"$\mathcal{L}_{e,x}$", getTime=False, signedMaxMin=True)
    xn_Ley, yn_Ley, Ley, Ley_min, Ley_max, label_Ley, dummy, Ley_useCmap = extractPlotData(frame, data, "Lorentz-ele-y", r"$\mathcal{L}_{e,y}$", getTime=False, signedMaxMin=True)

    limits = frame["q"]["xy_limits"]
    yn, xn = np.meshgrid(yn, xn)
    axes = []

    l = 0 #streak length
    fig = plt.figure(figsize=[16, 8], constrained_layout=True) #figsize=(8,9)) # note it is y, x, measurement 
    # fig = plt.figure(constrained_layout=True) # if on tinaroo or other

    gs = gridspec.GridSpec(ncols=3, nrows=3, figure=fig) #, hspace=0.1)

    #print("Pre plot")
    #subPlot(axes, fig, gs, xn, yn, 0, 0, rhoi, rhoi_min, rhoi_max, label_rhoi, rhoi_useCmap)
    #subPlot(axes, fig, gs, xn, yn, 0, 1, rhoe, rhoe_min, rhoe_max, label_rhoe, rhoe_useCmap)
    subPlot(axes,fig, gs, xn, yn, 0, 0, omegai, omegai_min, omegai_max, label_omegai, omegai_useCmap)
    subPlot(axes,fig, gs, xn, yn, 0, 1, omegae, omegae_min, omegae_max, label_omegae, omegae_useCmap)

    if False:
      print("Hard coded limits on rho")
      subPlot(axes, fig, gs, xn, yn, 0, 0, rhoi, rhoi_min, 12, label_rhoi, useCmap)
      subPlot(axes, fig, gs, xn, yn, 0, 1, rhoe, rhoe_min, 0.12, label_rhoe, useCmap)

    subPlot(axes, fig, gs, xn, yn, 0, 2, momi, momi_min, momi_max, label_momi, momi_useCmap)
    subPlot(axes, fig, gs, xn, yn, 1, 2, mome, mome_min, mome_max, label_mome, mome_useCmap)

    #subPlot(axes, fig, gs, xn, yn, 1, 0, Ex, Ex_min, Ex_max, label_Ex, Ex_useCmap)
    #subPlot(axes, fig, gs, xn, yn, 1, 1, Ey, Ey_min, Ey_max, label_Ey, Ey_useCmap)
    #subPlot(axes, fig, gs, xn, yn, 2, 0, Bz, Bz_min, Bz_max, label_Bz, Bz_useCmap)
    #subPlot(axes, fig, gs, xn, yn, 0, 1, Jx, Jx_min, Jx_max, label_Jx, useCmap)
    #subPlot(axes, fig, gs, xn, yn, 1, 1, Jy, Jy_min, Jy_max, label_Jy, useCmap)
    #subPlot(axes, fig, gs, xn, yn, 2, 1, Jz, Jz_min, Jz_max, label_Jz, useCmap)
    #subPlot(axes, fig, gs, xn, yn, 2, 0, dive, dive_min, dive_max, label_dive, useCmap)
    #subPlot(axes, fig, gs, xn, yn, 1, 1, rhoEM , rhoEM_min, rhoEM_max, label_rhoEM, useCmap)
    subPlot(axes, fig, gs, xn, yn, 2, 2, rc, rc_min, rc_max, label_rc, rc_useCmap)
    subPlot(axes, fig, gs, xn, yn, 1, 0, Lix, Lix_min, Lix_max, label_Lix, Lix_useCmap)
    subPlot(axes, fig, gs, xn, yn, 2, 0, Liy, Liy_min, Liy_max, label_Liy, Liy_useCmap)
    subPlot(axes, fig, gs, xn, yn, 1, 1, Lex, Lex_min, Lex_max, label_Lex, Lex_useCmap)
    subPlot(axes, fig, gs, xn, yn, 2, 1, Ley, Ley_min, Ley_max, label_Ley, Ley_useCmap)

    
    if True: # overlay mask 
      # get mask and overlay 
      xn_MskI, yn_MskI, MskI, MskI_min, MskI_max, label_MskI, dummy, MskI_useCmap = extractPlotData(frame, data, "mask-ion", r"$interface_i$", getTime=False, signedMaxMin=False)
      xn_MskE, yn_MskE, MskE, MskE_min, MskE_max, label_MskE, dummy, MskE_useCmap = extractPlotData(frame, data, "mask-ele", r"$interface_e$", getTime=False, signedMaxMin=False)
      pdb.set_trace()
      yn_MskI, xn_MskI = np.meshgrid(yn_MskI, xn_MskI)
      yn_MskE, xn_MskE = np.meshgrid(yn_MskE, xn_MskE)
      axes[0].contourf(xn_MskE, yn_MskE, MskE, colors='purple', 
        corner_mask=False, alpha=0.6)
      axes[1].contourf(xn_MskI, yn_MskI, MskI, colors='gray', 
        corner_mask=False, alpha=0.6)
      axes[4].contourf(xn_MskE, yn_MskE, MskE, colors='purple', 
        corner_mask=False, alpha=0.6)
      axes[4].contourf(xn_MskI, yn_MskI, MskI, colors='gray', 
        corner_mask=False, alpha=0.6)

    fig.suptitle(f"t = {time:.3f}")
    for (i, ax) in enumerate(axes):
        #if i == 0:
        #  ax.set_title(f"t = {time:.3f}")
        ax.set_xlim(limits[0][0], limits[1][0])
        ax.set_ylim(limits[0][1], limits[1][1])

        ax.set_aspect(1)
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
    #fig.tight_layout()a
    #print("Pre save")
    fig.savefig(output_name, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return

if 1:
    dt = 0.01 # 0.005

    Q = []

    q = {}
    q["files_dir"] = "."
    q["level"] = -1


    q["get"] = [
        #{"func":get_mass_density, "tag":"rho-ions", "component":"ions"},
        #{"func":get_mass_density, "tag":"rho-electrons", "component":"electrons"},
        {"func":get_vorticity_wrapper, "tag":"omega-electrons",   
                                       "name":"electrons", "quantity":"omega"},
        {"func":get_vorticity_wrapper, "tag":"omega-ions",   
                                       "name":"ions", "quantity":"omega"},
        {"func":get_Lorentz_wrapper, "tag":"Lorentz-ion-x", "level":q["level"],
                                       "name":"ions", "quantity":"L_x_total"},
        {"func":get_Lorentz_wrapper, "tag":"Lorentz-ion-y",  "level":q["level"], 
                                       "name":"ions", "quantity":"L_y_total"},
        {"func":get_Lorentz_wrapper, "tag":"Lorentz-ele-x",   "level":q["level"],
                                       "name":"electrons", "quantity":"L_x_total"},
        {"func":get_Lorentz_wrapper, "tag":"Lorentz-ele-y",   "level":q["level"],
                                       "name":"electrons", "quantity":"L_y_total"},
        {"func":get_mom_density, "tag":"mom-ions", "component":"ions"},
        {"func":get_mom_density, "tag":"mom-electrons", "component":"electrons"},
        
        #{"func":get_Jz, "tag":"Jz"},
        {"func":get_charge_density, "tag":"rc"},
        {"func":get_interface_mask, "tag":"mask-ion", "component":"ions"},
        {"func":get_interface_mask, "tag":"mask-ele", "component":"electrons"},

        #{"func":get_Jx, "tag":"Jx"},
        #{"func":get_Jy, "tag":"Jy"},
        #{"func":get_EM, "tag":"Ex", "component":"x_D"},
        #{"func":get_EM, "tag":"Ey", "component":"y_D"},
        #{"func":get_EM, "tag":"Bz", "component":"z_B"},
        #{"func":get_number_density, "tag":"nd-ions", "component":"ions"},
        #{"func":get_number_density, "tag":"nd-electrons", "component":"electrons"},
        #{"func":get_particles, "tag":"particles-ion", "get_streak":True, "component":"ion"},
        #{"func":get_particles, "tag":"particles-electron", "get_streak":True, "component":"electron"},
        #{"func":get_E_error, "tag":"divE_error", "component":""},
        #{"func":get_B_error, "tag":"divB_error", "component":""},
        #{"func":get_rhoE_EM, "tag":"rhoE_EM", "component":""},
    ]

    q["plot"] = plot2
    q["name"] = "MOVIE_POST_ION_ELECTRON_INTERFACE_ANALYSIS_MASKED" #rhoi-rhoe-omegai-omegae-momi-mome" #SRMI-Li3-option-16-Intra-Iso-By-Clean"

    ##
    q["framerate"] = 10 # 10 # 20
    q["mov_save"] = q["files_dir"] + "/mov"
    q["offset"] = [0.0, 0.0]
    q["xy_limits"] = [[-0.75,0], [1.25,1]]
    q["file_include"] = [".plt00000"]
    q["file_exclude"] = ["chk"]
    q["cores"] = 1
    q["time_span"] = [] # np.arange(0,10+dt, dt)
    q["force_data"] = False
    q["force_frames"] = False
    q["only_frames"] = False 
    q["redo_streaks"] = False
    q["dpi"] = 200

    q["normalize"] = "all"

    Q.append(q)
    #print("make_all function")
    make_all(Q)

print("DONE")

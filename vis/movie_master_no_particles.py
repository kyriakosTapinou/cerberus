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

import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

#matplotlib.use('Agg')

plt.rcParams.update({
    "text.usetex": False,
    #"font.family": "sans-serif",
    #"font.sans-serif": ["Helvetica"]
    })

# ==============================================================================
# MAKE MOVIES
# ==============================================================================

def get_mass_density(ds, c):
    x, r = ds.get("rho-%s"%c["component"])
    time = np.array([ds.time])
    return {"x":x[0], "y":x[1], "value":r, "time":time}

def get_vorticity_wrapper(ds, c):
  inputs = {"name":c["name"], "quantity":c["quantity"]}
  x, y, omega = phmmfp.get_vorticity(ds, inputs) 
  return {"x":x, "y":x, "value":omega}

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

def extractPlotData(frame, data, prop, label, getTime=True):
    if getTime: time = data[prop]["time"][()]
    else: time = None

    xn = data[prop]["x"]
    yn = data[prop]["y"]
    val = data[prop]["value"]
    val_min = frame[prop]["min"]
    val_max = frame[prop]["max"]
    return xn, yn, val, val_min, val_max, label, time

def subPlot(axes, fig, gs, xn, yn, nr, nc, val, val_min, val_max, val_label):
    #print("subPlot")
    ax = fig.add_subplot(gs[nr, nc]); axes.append(ax)
    #print("Pre pcolormesh")
    pcm = ax.pcolormesh(xn, yn, val, vmin=val_min, vmax=val_max)
    #print("post pcolomesh")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='1.5%', pad=0.05)
    fig.colorbar(pcm, cax=cax) #, label=val_label)
    ax.set_title(val_label)
    return 

def plot2(frame, data, output_name):
    #print("Plot start")
    xn, yn, rhoi, rhoi_min, rhoi_max, label_rhoi, time = extractPlotData(frame, data, "rho-ions", r"$\rho_i$", getTime=True)
    xn_rhoe, yn_rhoe, rhoe, rhoe_min, rhoe_max, label_rhoe, dummy = extractPlotData(frame, data, "rho-electrons", r"$\rho_e$", getTime=False)
    xn_omegai, yn_omegai, omegai, omegai_min, omegai_max, label_omegai, dummy = \
      extractPlotData(frame, data, "omega-ions", r"$\omega_i$", getTime=False)
    xn_omegae, yn_omegae, omegae, omegae_min, omegae_max, label_omegae, dummy = \
      extractPlotData(frame, data, "omega-electrons", r"$\omega_e$", getTime=False)

    #xn_Ex, yn_Ex, Ex, Ex_min, Ex_max, label_Ex, time = extractPlotData(frame, data, "Ex", r"$E_x$", getTime=True)
    #xn_Ey, yn_Ey, Ey, Ey_min, Ey_max, label_Ey, dummy = extractPlotData(frame, data, "Ey", r"$E_y$", getTime=False)
    #xn_Bz, yn_Bz, Bz, Bz_min, Bz_max, label_Bz, dummy = extractPlotData(frame, data, "Bz", r"$B_z$", getTime=False)
    #xn_Jz, yn_Jz, Jz, Jz_min, Jz_max, label_Jz, dummy = extractPlotData(frame, data, "Jz", r"$J_z$", getTime=False)
    #xn_Jy, yn_Jy, Jy, Jy_min, Jy_max, label_Jy, dummy = extractPlotData(frame, data, "Jy", r"$J_y$", getTime=False)
    #xn_Jx, yn_Jx, Jx, Jx_min, Jx_max, label_Jx, dummy = extractPlotData(frame, data, "Jx", r"$J_x$", getTime=False)
    #xn_dive, yn_dive, dive, dive_min, dive_max, label_dive, dummy = extractPlotData(frame, data, "divE_error", r"$\div{E}$", getTime=False)
    #xn_rhoEM, yn_rhoEM, rhoEM, rhoEM_min, rhoEM_max, label_rhoEM, dummy = extractPlotData(frame, data, "rhoE_EM", r"$\rho_{E,EM}$", getTime=False)
    #xn_rc, yn_rc, rc, rc_min, rc_max, label_rc, dummy = extractPlotData(frame, data, "rc", r"$\rho_c$", getTime=False)

    limits = frame["q"]["xy_limits"]
    yn, xn = np.meshgrid(yn, xn)
    axes = []

    l = 0 #streak length

    fig = plt.figure(figsize=(16,16))

    # fig = plt.figure(constrained_layout=True) # if on tinaroo or other
    fig = plt.figure() # if on local 

    #gs = gridspec.GridSpec(ncols=2, nrows=2, figure=fig)#, hspace=0.1)
    gs = gridspec.GridSpec(2, 2)

    #print("Pre plot")
    subPlot(axes, fig, gs, xn, yn, 0, 0, rhoi, rhoi_min, rhoi_max, label_rhoi)
    subPlot(axes, fig, gs, xn, yn, 0, 1, rhoe, rhoe_min, rhoe_max, label_rhoe)

    subPlot(axes, fig, gs, xn, yn, 1, 0, omegai, omegai_min, omegai_max, label_omegai)
    subPlot(axes, fig, gs, xn, yn, 1, 1, omegae, omegae_min, omegae_max, label_omegae)

    if False:
      print("Hard coded limits on rho")
      subPlot(axes, fig, gs, xn, yn, 0, 0, rhoi, rhoi_min, 12, label_rhoi)
      subPlot(axes, fig, gs, xn, yn, 0, 1, rhoe, rhoe_min, 0.12, label_rhoe)

    #subPlot(axes, fig, gs, xn, yn, 1, 0, Ex, Ex_min, Ex_max, label_Ex)
    #subPlot(axes, fig, gs, xn, yn, 1, 1, Ey, Ey_min, Ey_max, label_Ey)
    #subPlot(axes, fig, gs, xn, yn, 2, 0, Bz, Bz_min, Bz_max, label_Bz)
    #subPlot(axes, fig, gs, xn, yn, 0, 1, Jx, Jx_min, Jx_max, label_Jx)
    #subPlot(axes, fig, gs, xn, yn, 1, 1, Jy, Jy_min, Jy_max, label_Jy)
    #subPlot(axes, fig, gs, xn, yn, 2, 1, Jz, Jz_min, Jz_max, label_Jz)
    #subPlot(axes, fig, gs, xn, yn, 2, 0, dive, dive_min, dive_max, label_dive)
    #subPlot(axes, fig, gs, xn, yn, 1, 1, rhoEM , rhoEM_min, rhoEM_max, label_rhoEM)
    #subPlot(axes, fig, gs, xn, yn, 2, 1, rc, rc_min, rc_max, label_rc)

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
    q["level"] = 2


    q["get"] = [
        {"func":get_mass_density, "tag":"rho-ions", "component":"ions"},
        {"func":get_mass_density, "tag":"rho-electrons", "component":"electrons"},

        {"func":get_vorticity_wrapper, "tag":"omega-electrons",   
                                       "name":"electrons", "quantity":"omega"},
        {"func":get_vorticity_wrapper, "tag":"omega-ions",   
                                       "name":"ions", "quantity":"omega"},

        #{"func":get_mom_density, "tag":"mom-ions", "component":"ions"},
        #{"func":get_mom_density, "tag":"mom-electrons", "component":"electrons"},
        #{"func":get_Jz, "tag":"Jz"},
        #{"func":get_charge_density, "tag":"rc"},
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
    q["name"] = "MOVIE_POST" #SRMI-Li3-option-16-Intra-Iso-By-Clean"

    ##
    q["framerate"] = 10 # 10 # 20
    q["mov_save"] = q["files_dir"] + "/mov_test_delete_me"
    q["offset"] = [0.0, 0.0]
    q["xy_limits"] = [[-0.75,0], [1.25,1]]
    q["file_include"] = [".plt"]
    q["file_exclude"] = ["chk"]
    q["cores"] = 8
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

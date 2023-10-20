-- ======== PROBLEM ==========

shock_r = 1.2 -- same stand off as our RMI tests in planar 
shock_transition = 0.001

interface_r = 1.0
interface_transition = 0.01
interface_wavenumber = 8
interface_amplitude = 0.04*2*math.pi/interface_wavenumber

shock_mach = 2.0
density_L = 1.0
density_R = 3.0

mass_ion = 1.0
mass_electron = 0.01

gam = 5/3
pressure_nd = 0.5
axis_beta = {8.94938194246656,0,0}

-- computations

mid_r = (shock_r + interface_r)/2

p1 = pressure_nd
p2 = pressure_nd
p0 = p1*(1 + ((2*gam)/(gam+1))*(shock_mach^2 - 1))

rho1 = density_L
rho2 = density_R
rho0 = rho1/(1 - (2/(gam+1))*(1 - 1/shock_mach^2))

a0 = math.sqrt(gam*p0/rho0)
a1 = math.sqrt(gam*p1/rho1)
a2 = math.sqrt(gam*p2/rho2)

u1 = 0.0
u2 = 0.0
u0 = shock_mach*a1 - a0*math.sqrt(((gam-1)*shock_mach^2 + 2)/(2*gam*shock_mach^2 - (gam-1)))

-- functions

local function islarge (x) return x > 2 ^ 28 end
local function issmall (x) return x < 2 ^ (-28) end

local function log1p (x) -- not very precise, but works well
  local u = 1 + x
  if u == 1 then return x end -- x < eps?
  return math.log(u) * x / (u - 1)
end

function atanh (x)
    local a, y
  y = math.abs(x)
  if y < .5 then
    if issmall(y) then return x end
    a = 2 * y
    a = .5 * log1p(a + a * y / (1 - y))
  else
    if y < 1 then
      a = .5 * log1p(2 * y / (1 - y))
    elseif y > 1 then
      return (x - x) / (x - x) -- nan
    else -- y == 1
      return x / 0 -- inf with sign
    end
  end
  return x < 0 and -a or a -- transfer sign
end

function smooth_interface(r, theta, L, R, t, f)
    if math.abs(L - R)  < 1e-14 then
        return L
    end

    local at

    if (L < R) then
        at = (10.0*L - 9.0*R)/(10.0*(R-L))
    else
        at = (10.0*R - 9.0*L)/(10.0*(L-R))
    end
    
    local slope = (2.0/t)*atanh(at)
    return -((math.tanh(-slope*(r - f(theta)))-1.0)/2.0)*(L-R)+R
end

function RMI_shock_interface(theta)
    return shock_r
end

function RMI_density_interface(theta)
    return interface_r - interface_amplitude*math.cos(interface_wavenumber*theta)
end



-- fluids

function number_density(dat)

    local x = dat['x']
    local y = dat['y']
    local r = math.sqrt(x^2 + y^2)
    local theta = math.atan2(y/x)

    local n

    if (r >= mid_r) then
        n = smooth_interface(r, theta, rho1, rho0, shock_transition, RMI_shock_interface)
    else
        n = smooth_interface(r, theta, rho2, rho1, interface_transition, RMI_density_interface)
    end
    
    return n
end

function ion_density(dat)
    return mass_ion*number_density(dat)
end

function electron_density(dat)
    return mass_electron*number_density(dat)
end

function tracer(dat)
    local x = dat['x']
    local y = dat['y']
    local r = math.sqrt(x^2 + y^2)
    local theta = math.atan2(y/x)

    local t = smooth_interface(r, theta, 1, 0, interface_transition, RMI_density_interface)
    
    return 0 --t
end

function pressure(dat)
    local x = dat['x']
    local y = dat['y']
    local r = math.sqrt(x^2 + y^2)
    local theta = math.atan2(y/x)

    local p = smooth_interface(r, theta, p1, p0, shock_transition, RMI_shock_interface)

    return p
end

function radial_velocity(r, theta)
    return smooth_interface(r, theta, u1, u0, shock_transition, RMI_shock_interface)
end

function velocity_x(dat)
    local x = dat['x']
    local y = dat['y']
    local r = math.sqrt(x^2 + y^2)
    local theta = math.atan2(y/x)
    local vel = radial_velocity(r, theta)
    return -vel*x/r
end

function velocity_y(dat)
    local x = dat['x']
    local y = dat['y']
    local r = math.sqrt(x^2 + y^2)
    local theta = math.atan2(y/x)
    local vel = radial_velocity(r, theta)
    return -vel*y/r
end

-- === SETTINGS ===
verbosity = 1
cfl = 0.5
cfl_viscous = 1.0
do_CTU = 1 -- turn of corner transport upwind 
do_face_sources = 1
srin_switch = 0;
source_collision_frequency_constraint = 0;
braginskii_anisotropic = 1;
viewFluxSrcContributions = 0;

force_dt = 0

-- REFERENCE VALUES 
mu_0_dim = 1.25663706e-6 
ep_0_dim = 8.85418782e-12
kb = 1.38064852e-23

ref_lightspeed = 299792458.0; --dimensional speed of light 
q_ref = 1.60217662e-19 -- Coulombs
m_ref = 1.6726219000e-27

--specified
ref_length = 1e-4 -- m 
ref_mass = m_ref -- kg
n_ref = 1e27 -- m^-3
ref_density = n_ref*m_ref
--ref_density = 1.0e2
lightspeed = 6.8489145143E+02 -- 2500; 3.0629276852E+02 -- 
lightspeed_nd = lightspeed
beta = 200 --1

ref_velocity = ref_lightspeed/lightspeed;
--n_ref = ref_density/ref_mass
B_ref = math.sqrt(2*mu_0_dim*n_ref*ref_mass*ref_velocity*ref_velocity/beta)
ref_omega_c = q_ref*B_ref/ref_mass; 
ref_omega_p = math.sqrt(n_ref*q_ref*q_ref/ref_mass/ep_0_dim); 
ref_larmor_dim = ref_velocity/ref_omega_c;
ref_skin_dim = ref_mass/(q_ref*math.sqrt(mu_0_dim*ref_density)) 
ref_skin_nd = ref_skin_dim/ref_length;

Larmor = ref_larmor_dim/ref_length 
Debye = ref_skin_nd/lightspeed_nd 

--  magnetic field strength
B = {0,0,0}
p = 2*pressure_nd -- use total pressure
for i, b in ipairs(axis_beta) do
    if b > 0 then
        B[i] = math.sqrt(p*beta/b)
    end
end

-- === DEFINE STATES ===

states = {

    ion = {
        type='hydro',
        mass=mass_ion,  
        charge= 1.0, 
        gamma=5/3, 
        reconstruction='MC', 
        flux='HLLC',
        refine_grad_threshold = {rho=0.1},
        value = {
            rho   = ion_density,
            x_vel = velocity_x,
            y_vel = velocity_y,
            p     = pressure,
            alpha = tracer,
        },
        viscosity = {cfl=cfl_viscous , type='BraginskiiIon', forceViscosity=false, 
                     forceViscosityValue=0., DebyeReference=Debye, LarmorReference=Larmor},
        particles='particle_file',
        bc = {
            x={
                lo={fill_hydro_bc = 'symmetry'},
            },
        },
    },

    electron = {
        type='hydro',
        mass=mass_electron, 
        charge=-1.0, 
        gamma=5/3, 
        reconstruction='MC',  
        flux='HLLC',
        refine_grad_threshold = {rho=0.1},
        value = {
            rho   = electron_density,
            x_vel = velocity_x,
            y_vel = velocity_y,
            p     = pressure,
            alpha = tracer,
        },
        viscosity = {cfl=cfl_viscous, type='BraginskiiEle', forceViscosity=false, 
                     forceViscosityValue=0., DebyeReference=Debye, LarmorReference=Larmor},
        particles='particle_file',
        bc = {
            x={
                lo={fill_hydro_bc = 'symmetry'},
            },
        },
    },


    field = {
        type='field',
        reconstruction='O6', 
        flux='RankineHugoniot',
        -- D_clean = 1.05,
        project_D_divergence=1,
        value = {
            x_B = B[1],
            y_B = B[2],
            z_B = B[3],
        },
        bc = {
            x={
                lo={
                    fill_D_bc='symmetry', fill_B_bc='symmetry',
                    -- fill_D_bc='inflow', x_D=0.0, y_D=0.0, z_D=0.0, 
                    -- fill_B_bc='inflow', x_B=0.0, y_B=0.0, z_B=0.0
                },
                hi={
                    --fill_D_bc='inflow', x_D=0.0, y_D=0.0, z_D=0.0, 
                    --fill_B_bc='inflow', x_B=0.0, y_B=0.0, z_B=0.0
                    fill_D_bc='outflow', 
                    fill_B_bc='outflow', 

                },
            },
            y={
                lo={
                    --fill_D_bc='inflow', x_D=0.0, y_D=0.0, z_D=0.0, 
                    --fill_B_bc='inflow', x_B=0.0, y_B=0.0, z_B=0.0
                    fill_D_bc='outflow', 
                    fill_B_bc='outflow', 
                },
                --lo={
                --    fill_D_bc='symmetry', fill_B_bc='symmetry',
                    -- fill_D_bc='inflow', x_D=0.0, y_D=0.0, z_D=0.0, 
                    -- fill_B_bc='inflow', x_B=0.0, y_B=0.0, z_B=0.0
                --},
                hi={
                    --fill_D_bc='inflow', x_D=0.0, y_D=0.0, z_D=0.0, 
                    --fill_B_bc='inflow', x_B=0.0, y_B=0.0, z_B=0.0
                    fill_D_bc='outflow', 
                    fill_B_bc='outflow', 
                },
            },
        },
    }

}

sources = {

    plasma={
        solver = 'implicit',
        sources = {
            plasma={'ion', 'electron', 'field',
                type='plasma5',
            },
            -- div_clean={'field', ratio_D=0.1, type='damp_divergence'},
        },
    },
    collide={
        solver='explicit',
        sources={
            braginskii={'ion', 'electron', 'field', type='braginskii',
                        reconstruction='vanLeer', DebyeReference=Debye, LarmorReference=Larmor,},
        },
    }, 

}

function make_particles(N)
    n = 0
    file = io.open('particle_file', "w")
    file:write(N,"\n")

    for theta=0,math.pi/2,math.pi/(2*N) do
        r = interface_r - interface_amplitude*math.cos(interface_wavenumber*theta)

        x = r*math.cos(theta)
        y = r*math.sin(theta)
  
        file:write(x," ",y,"\n")
    end
  
    io.close(file)
  end

  
  make_particles(1000)

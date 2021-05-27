#include "MFP_hydro.H"

#include <functional>

#include "sol.hpp"
#include "MFP_utility.H"
#include "MFP_global.H"
#include "MFP_lua.H"
#include "MFP_Riemann_solvers.H"
#include "MFP_hydro_bc.H"
#include "Eigen"

#ifdef PYTHON
#include "MFP_diagnostics.H"
#endif

using GD = GlobalData;

//------------
// Hydro State

Vector<std::string> HydroState::cons_names = {
    "rho",
    "x_mom",
    "y_mom",
    "z_mom",
    "nrg",
    "tracer"
};

Vector<std::string> HydroState::prim_names = {
    "rho",
    "x_vel",
    "y_vel",
    "z_vel",
    "p",
    "T",
    "alpha"
};

Vector<set_bc> HydroState::bc_set = {
    &set_scalar_bc,
    &set_x_vel_bc,
    &set_y_vel_bc,
    &set_z_vel_bc,
    &set_scalar_bc,
    &set_scalar_bc,
    &set_scalar_bc
};

Vector<int> HydroState::flux_vector_idx = {+HydroState::FluxIdx::Xvel};
Vector<int> HydroState::cons_vector_idx = {+HydroState::ConsIdx::Xmom};
Vector<int> HydroState::prim_vector_idx = {+HydroState::PrimIdx::Xvel};

//-----------------------------------------------------------------------------

std::string HydroState::tag = "hydro";
bool HydroState::registered = GetStateFactory().Register(HydroState::tag, StateBuilder<HydroState>);

HydroState::HydroState() : State(){}
HydroState::HydroState(const sol::table& def)
{
    name = def.get<std::string>("name");
    global_idx = def.get<int>("global_idx");
    sum_cons.resize(+ConsIdx::NUM);
}
HydroState::~HydroState(){}

void HydroState::init_from_lua()
{
    BL_PROFILE("HydroState::init_from_lua");

    State::init_from_lua();

    sol::state& lua = GD::lua;

    const sol::table state_def = lua["states"][name];

    GD::num_fluid += 1;


    //
    // get mass, charge, and density
    //

    expand(state_def["mass"], mass);
    expand(state_def["charge"], charge);
    expand(state_def["gamma"], gamma);

    mass_const = mass[0] == mass[1];
    charge_const = charge[0] == charge[1];
    gamma_const = gamma[0] == gamma[1];

    //
    // viscous terms coefficients
    //
    set_viscosity();

    //
    // domain boundary conditions
    //

    const Vector<std::string> dir_name = {"x", "y", "z"};
    const Vector<std::string> side_name = {"lo", "hi"};
    const Vector<std::string>& hydro_var = get_prim_names();
    const int N = hydro_var.size();

    BoundaryState &bs = boundary_conditions;
    bs.phys_fill_bc.resize(+HydroState::PrimIdx::NUM);

    for (int ax = 0; ax < AMREX_SPACEDIM; ++ax) {
        for (int lh=0; lh<2; ++lh) {

            std::string side_bc = state_def["bc"][dir_name[ax]][side_name[lh]]["fill_hydro_bc"].get_or<std::string>("outflow");
            int i_side_bc = bc_names.at(side_bc);

            // get any custom values/functions
            for (int j=0; j<N; ++j) {

                if (lh==0) {
                    bs.phys_fill_bc[j].setLo(ax,i_side_bc);
                } else {
                    bs.phys_fill_bc[j].setHi(ax,i_side_bc);
                }

                const sol::object v = state_def["bc"][dir_name[ax]][side_name[lh]][hydro_var[j]].get_or(sol::object());
                Optional3D1VFunction f = get_udf(v);
                bs.set(ax,hydro_var[j],lh,f);

                // special case for inflow condition
                if (i_side_bc == PhysBCType::inflow && !f.is_valid()) {
                    Abort("Setting 'fill_hydro_bc = inflow' requires all primitive variables to be defined, '" + hydro_var[j] + "' is not defined");
                }
            }
#ifdef AMREX_USE_EB
            bool is_symmetry = (i_side_bc == PhysBCType::symmetry) || (i_side_bc == PhysBCType::slipwall) || (i_side_bc == PhysBCType::noslipwall);
            if (lh==0) {
                bs.eb_bc.setLo(ax,is_symmetry ? BCType::reflect_even : BCType::foextrap);
            } else {
                bs.eb_bc.setHi(ax,is_symmetry ? BCType::reflect_even : BCType::foextrap);
            }
#endif
        }
    }

    // check validity of inflow bc
    boundary_conditions.post_init();


    //
    // shock detector threshold
    //
    set_shock_detector();

    //
    // particles
    //

    particle_init = state_def["particles"].get_or<std::string>("");


    //
    // positivity
    //
    enforce_positivity = state_def["enforce_positivity"].get_or(0);

    extra_slope_limits = state_def["extra_slope_limits"].get_or(1);

}



#ifdef AMREX_USE_EB
void HydroState::set_eb_bc(const sol::table &bc_def)
{

    std::string bc_type = bc_def.get<std::string>("type");

    if (bc_type == HydroSlipWall::tag) {
        eb_bcs.push_back(std::unique_ptr<BoundaryEB>(new HydroSlipWall(flux_solver.get())));
    } else if (bc_type == HydroNoSlipWall::tag) {
        if (!viscous) {
            Abort("Requested EB bc of type '" + bc_type + "' without defining 'viscosity' for state '" + name + "'");
        }
        eb_bcs.push_back(std::unique_ptr<BoundaryEB>(new HydroNoSlipWall(flux_solver.get(), viscous.get(), bc_def)));
    } else if (bc_type == DirichletWall::tag) {
        eb_bcs.push_back(std::unique_ptr<BoundaryEB>(new DirichletWall(flux_solver.get(), get_prim_names(), get_prim_vector_idx(), bc_def)));
    } else {
        Abort("Requested EB bc of type '" + bc_type + "' which is not compatible with state '" + name + "'");
    }
}
#endif

Real HydroState::init_from_number_density(std::map<std::string, Real> data)
{
    Real nd = functions["nd"](data);
    Real alpha = functions["alpha"](data);
    Real m = get_mass(alpha);

    return nd*m;
}

void HydroState::set_udf()
{
    using namespace std::placeholders;

    sol::state& lua = GD::lua;

    sol::table state_def = lua["states"][name];

    // check if we have 'value' defined
    const sol::table value = state_def["value"].get_or(sol::table());


    if (!value.valid())
        Abort("State "+name+" does not have 'value' defined for initial conditions");

    // get a list of any initialisation functions that need to be called during the run
    sol::table dynamic = state_def["dynamic"].get_or(sol::table());

    const auto ignore = init_with_value();

    bool check, success;

    const Vector<std::string> &prim_names = get_prim_names();
    for (int i = 0; i<prim_names.size(); ++i) {

        std::string comp = prim_names[i];

        // is there a variable with this name?
        success = false;
        check = value[comp].valid();

        // doesn't exist, is there an alternative?
        if (!check) {

            // use number density instead of density
            if (comp.compare("rho") == 0) {

                check = value["nd"].valid();

                if (!check)
                    Abort("State "+name+" does not have 'rho' or 'nd' defined for initial conditions");

                Optional3D1VFunction nd;
                success = get_udf(value["nd"], nd, 0.0);

                functions["nd"] = nd;

                Optional3D1VFunction rho;

                rho.set_func(std::bind(&HydroState::init_from_number_density, this, _1));

                functions[comp] = rho;
            }

        }

        if (!success) {

            Optional3D1VFunction v;

            success = get_udf(value[comp], v, 0.0);

            if (!success) {
                for (const auto &j : ignore) {
                    if (i == j.first) {
                        v.set_value(j.second);
                        success = true;
                        break;
                    }
                }
            }

            functions[comp] = v;
        }

        if (dynamic.valid()) {
            for (const auto &d : dynamic) {
                if (d.second.as<std::string>().compare(comp) == 0) {
                    dynamic_functions[i] = &functions[comp];
                }
            }
        }

    }

    return;
}

AssociatedType HydroState::get_association_type() const
{
    if (charge[0] < 0.0 && charge[1] < 0.0) {
        return AssociatedType::Electron;
    } else if (charge[0] > 0.0 && charge[1] > 0.0) {
        return AssociatedType::Ion;
    } else if (charge[0] == 0.0 && charge[1] == 0.0) {
        return AssociatedType::Neutral;
    } else {
        Abort("Charge of state '"+name+"' is not uniformly positive, negative or neutral");
        return AssociatedType::isNull; //  keep the syntax checker happy by returning something here
    }
}

const Vector<std::string>& HydroState::get_cons_names() const
{
    return cons_names;
}

const Vector<std::string>& HydroState::get_prim_names() const
{
    return prim_names;
}

const Vector<set_bc>& HydroState::get_bc_set() const
{
    return bc_set;
}


Real HydroState::get_mass(Real alpha) const
{
    BL_PROFILE("HydroState::get_mass");

    if (mass_const) return mass[0];

    // clamp alpha
    alpha = clamp(alpha, 0.0, 1.0);

    Real m0 = mass[0];
    Real m1 = mass[1];

    return (m0*m1)/(m0*alpha + m1*(1-alpha));
}

Real HydroState::get_mass(const Vector<Real> &U) const
{
    BL_PROFILE("HydroState::get_mass");

    if (mass_const) return mass[0];

    // clamp alpha
    Real alpha = get_alpha_from_cons(U);
    return get_mass(alpha);
}

Real HydroState::get_charge(Real alpha) const
{
    BL_PROFILE("HydroState::get_charge");

    if (charge_const) return charge[0];

    // clamp alpha
    alpha = clamp(alpha, 0.0, 1.0);

    Real m0 = mass[0];
    Real m1 = mass[1];

    Real q0 = charge[0];
    Real q1 = charge[1];

    return (alpha*m0*q1 + (1-alpha)*m1*q0)/(m0*alpha + m1*(1-alpha));
}

Real HydroState::get_charge(const Vector<Real> &U) const
{
    BL_PROFILE("HydroState::get_charge");

    if (charge_const) return charge[0];

    // clamp alpha
    Real alpha = get_alpha_from_cons(U);
    return get_charge(alpha);
}

Real HydroState::get_gamma(Real alpha) const
{
    BL_PROFILE("HydroState::get_gamma");

    if (gamma_const) return gamma[0];

    // clamp alpha
    alpha = clamp(alpha, 0.0, 1.0);

    Real m0 = mass[0];
    Real m1 = mass[1];

    Real g0 = gamma[0];
    Real g1 = gamma[1];

    Real cp0 = g0/(m0*(g0-1));
    Real cp1 = g1/(m1*(g1-1));

    Real cv0 = 1/(m0*(g0-1));
    Real cv1 = 1/(m1*(g1-1));

    return ((1-alpha)*cp0 + alpha*cp1)/((1-alpha)*cv0 + alpha*cv1);
}

Real HydroState::get_gamma(const Vector<Real> &U) const
{
    BL_PROFILE("HydroState::get_gamma_U");

    if (gamma_const) return gamma[0];

    // clamp alpha
    Real alpha = get_alpha_from_cons(U);
    return get_gamma(alpha);
}

Real HydroState::get_cp(Real alpha) const
{
    BL_PROFILE("HydroState::get_cp");

    if (gamma_const && mass_const) return gamma[0]/(mass[0]*(gamma[0]-1));


    // clamp alpha
    alpha = clamp(alpha, 0.0, 1.0);

    Real m0 = mass[0];
    Real m1 = mass[1];

    Real g0 = gamma[0];
    Real g1 = gamma[1];

    Real cp0 = g0/(m0*(g0-1));
    Real cp1 = g1/(m1*(g1-1));

    return (1-alpha)*cp0 + alpha*cp1;
}

Real HydroState::get_cp(const Vector<Real> &U) const
{
    BL_PROFILE("HydroState::get_cp");

    if (gamma_const && mass_const) return gamma[0]/(mass[0]*(gamma[0]-1));

    Real alpha = get_alpha_from_cons(U);
    return get_cp(alpha);
}


// in place conversion from conserved to primitive
bool HydroState::cons2prim(Vector<Real>& U, Vector<Real>& Q) const
{
    BL_PROFILE("HydroState::cons2prim");

    Real rho = U[+ConsIdx::Density];
    Real mx = U[+ConsIdx::Xmom];
    Real my = U[+ConsIdx::Ymom];
    Real mz = U[+ConsIdx::Zmom];
    Real ed = U[+ConsIdx::Eden];
    Real tr = U[+ConsIdx::Tracer];

    Real rhoinv = 1/rho;
    Real u = mx*rhoinv;
    Real v = my*rhoinv;
    Real w = mz*rhoinv;
    Real ke = 0.5*rho*(u*u + v*v + w*w);
    Real alpha = tr*rhoinv;
    Real m = get_mass(alpha);
    Real g = get_gamma(alpha);
    Real p = (ed - ke)*(g - 1);
    Real T = p*rhoinv*m;

    Q[+PrimIdx::Density] = rho;
    Q[+PrimIdx::Xvel] = u;
    Q[+PrimIdx::Yvel] = v;
    Q[+PrimIdx::Zvel] = w;
    Q[+PrimIdx::Prs] = p;
    Q[+PrimIdx::Alpha] = alpha;
    Q[+PrimIdx::Temp] = T;

    return prim_valid(Q);
}

// in place conversion from conserved to primitive
bool HydroState::cons2prim(Vector<autodiff::dual>& U, Vector<autodiff::dual>& Q) const
{
    BL_PROFILE("HydroState::cons2prim");

    autodiff::dual rho = U[+ConsIdx::Density];
    autodiff::dual mx = U[+ConsIdx::Xmom];
    autodiff::dual my = U[+ConsIdx::Ymom];
    autodiff::dual mz = U[+ConsIdx::Zmom];
    autodiff::dual ed = U[+ConsIdx::Eden];
    autodiff::dual tr = U[+ConsIdx::Tracer];

    autodiff::dual rhoinv = 1/rho;
    autodiff::dual u = mx*rhoinv;
    autodiff::dual v = my*rhoinv;
    autodiff::dual w = mz*rhoinv;
    autodiff::dual ke = 0.5*rho*(u*u + v*v + w*w);
    autodiff::dual alpha = tr*rhoinv;
    Real m = get_mass(alpha.val);
    Real g = get_gamma(alpha.val);
    autodiff::dual p = (ed - ke)*(g - 1);
    autodiff::dual T = p*rhoinv*m;

    Q[+PrimIdx::Density] = rho;
    Q[+PrimIdx::Xvel] = u;
    Q[+PrimIdx::Yvel] = v;
    Q[+PrimIdx::Zvel] = w;
    Q[+PrimIdx::Prs] = p;
    Q[+PrimIdx::Alpha] = alpha;
    Q[+PrimIdx::Temp] = T;


    if ((Q[+PrimIdx::Density].val <= 0.0) ||  (Q[+PrimIdx::Prs].val <= 0.0)) {
        return false;
    }
    return true;
}

// in-place conversion from primitive to conserved variables
void HydroState::prim2cons(Vector<Real>& Q, Vector<Real>& U) const
{
    BL_PROFILE("HydroState::prim2cons");

    Real rho = Q[+PrimIdx::Density];
    Real u = Q[+PrimIdx::Xvel];
    Real v = Q[+PrimIdx::Yvel];
    Real w = Q[+PrimIdx::Zvel];
    Real p = Q[+PrimIdx::Prs];
    Real alpha = Q[+PrimIdx::Alpha];

    Real mx = u*rho;
    Real my = v*rho;
    Real mz = w*rho;
    Real ke = 0.5*rho*(u*u + v*v + w*w);
    Real tr = alpha*rho;
    Real g = get_gamma(alpha);
    Real ed = p/(g - 1) + ke;

    U[+ConsIdx::Density] = rho;
    U[+ConsIdx::Xmom] = mx;
    U[+ConsIdx::Ymom] = my;
    U[+ConsIdx::Zmom] = mz;
    U[+ConsIdx::Eden] = ed;
    U[+ConsIdx::Tracer] = tr;

}


bool HydroState::prim_valid(Vector<Real> &Q) const
{
    if ((Q[+PrimIdx::Density] <= 0.0) ||  (Q[+PrimIdx::Prs] <= 0.0)
            ) {
        //        amrex::Abort("Primitive values outside of physical bounds!!");
        return false;
    }
    return true;
}

bool HydroState::cons_valid(Vector<Real> &U) const
{
    if ((U[+ConsIdx::Density] <= 0.0) ||  (U[+ConsIdx::Eden] <= 0.0)
            ) {
        //        amrex::Abort("Primitive values outside of physical bounds!!");
        return false;
    }
    return true;
}

Real HydroState::get_energy_from_cons(const Vector<Real> &U) const
{
    return U[+ConsIdx::Eden];
}

Real HydroState::get_temperature_from_cons(const Vector<Real> &U) const
{
    BL_PROFILE("HydroState::get_temperature_from_cons");

    Real rho = U[+ConsIdx::Density];
    Real mx = U[+ConsIdx::Xmom];
    Real my = U[+ConsIdx::Ymom];
    Real mz = U[+ConsIdx::Zmom];
    Real ed = U[+ConsIdx::Eden];
    Real tr = U[+ConsIdx::Tracer];

    Real rhoinv = 1/rho;
    Real ke = 0.5*rhoinv*(mx*mx + my*my + mz*mz);
    Real alpha = tr*rhoinv;
    Real g = get_gamma(alpha);

    Real prs = (ed - ke)*(g - 1);

    Real m = get_mass(alpha);
    return prs*rhoinv*m;

}

Real HydroState::get_temperature_from_prim(const Vector<Real> &Q) const
{
    return Q[+PrimIdx::Temp];
}

dual HydroState::get_temperature_from_prim(const Vector<dual> &Q) const
{
    dual T = Q[+PrimIdx::Temp];
    return T;
}

RealArray HydroState::get_speed_from_cons(const Vector<Real>& U) const
{
    BL_PROFILE("HydroState::get_speed_from_cons");

    Real rho = U[+ConsIdx::Density];
    Real mx = U[+ConsIdx::Xmom];
    Real my = U[+ConsIdx::Ymom];
    Real mz = U[+ConsIdx::Zmom];
    Real ed = U[+ConsIdx::Eden];
    Real tr = U[+ConsIdx::Tracer];

    Real rhoinv = 1/rho;

    Real ux = mx*rhoinv;
    Real uy = my*rhoinv;
    Real uz = mz*rhoinv;

    Real kineng = 0.5*rho*(ux*ux + uy*uy + uz*uz);
    Real alpha = tr*rhoinv;
    Real g = get_gamma(alpha);
    Real prs = (ed - kineng)*(g - 1);
    Real a = std::sqrt(g*prs*rhoinv);

    RealArray s = {AMREX_D_DECL(a + std::abs(ux), a + std::abs(uy), a + std::abs(uz))};

    return s;

}

RealArray HydroState::get_speed_from_prim(const Vector<Real>& Q) const
{
    BL_PROFILE("HydroState::get_speed_from_prim");

    Real g = get_gamma(Q[+PrimIdx::Alpha]);

    Real a = std::sqrt(g*Q[+PrimIdx::Prs]/Q[+PrimIdx::Density]);

    RealArray s = {AMREX_D_DECL(a + std::abs(Q[+PrimIdx::Xvel]),
                                a + std::abs(Q[+PrimIdx::Yvel]),
                                a + std::abs(Q[+PrimIdx::Zvel]))};


    return s;

}

RealArray HydroState::get_current_from_cons(const Vector<Real> &U) const
{
    BL_PROFILE("HydroState::get_current_from_cons");
    Real q = get_charge(U);
    Real m = get_mass(U);
    Real r = q/m;

    RealArray c = {AMREX_D_DECL(
                   r*U[+ConsIdx::Xmom],
                   r*U[+ConsIdx::Ymom],
                   r*U[+ConsIdx::Zmom]
                   )};

    return c;
}



void HydroState::calc_reconstruction(const Box& box,
                                     FArrayBox &prim,
                                     Array<FArrayBox, AMREX_SPACEDIM> &rlo,
                                     Array<FArrayBox, AMREX_SPACEDIM> &rhi
                                     EB_OPTIONAL(,const EBCellFlagFab &flag)
                                     EB_OPTIONAL(,const FArrayBox &vfrac)
                                     ) const
{
    BL_PROFILE("HydroState::calc_reconstruction");
    // if we don't want to apply extra limiting on the slopes (forced to 2nd order)
    // we can use the default reconstruction scheme
    if (!extra_slope_limits) {
        State::calc_reconstruction(box, prim, rlo, rhi EB_OPTIONAL(,flag,vfrac));
    }

    // convert pressure
    const Box &pbox = prim.box();
    const Dim3 p_lo = amrex::lbound(pbox);
    const Dim3 p_hi = amrex::ubound(pbox);

    FArrayBox gamma_minus_one(pbox);
    Array4<Real> const& src4 = prim.array();
    Array4<Real> const& gam4 = gamma_minus_one.array();

#ifdef AMREX_USE_EB
    std::vector<std::array<int,3>> grab;
    multi_dim_index({-1,AMREX_D_PICK(0,-1,-1),AMREX_D_PICK(0,0,-1)},
    {1,AMREX_D_PICK(0, 1, 1),AMREX_D_PICK(0,0, 1)},
                    grab, false);

    Array4<const EBCellFlag> const& f4 = flag.array();
    // do we need to check our stencil for covered cells?
    bool check_eb = flag.getType() != FabType::regular;
#endif

    const Dim3 lo = amrex::lbound(box);
    const Dim3 hi = amrex::ubound(box);

    int N = prim.nComp();

    Vector<Real> stencil(reconstruction->stencil_length);
    int offset = reconstruction->stencil_length/2;
    Array<int,3> stencil_index;
    Vector<Real> cell_value(N), cell_slope(N);

    Real rho_lo, rho_hi;
    Real alpha_lo, alpha_hi;
    Real abs_phi, phi_scale, coeff_eps;
    Real gam_lo, gam_hi;

    // make sure our arrays for putting lo and hi reconstructed values into
    // are the corect size
    for (int d=0; d<AMREX_SPACEDIM; ++d) {
        rlo[d].resize(box, N);
        rhi[d].resize(box, N);

#ifdef AMREX_USE_EB
        if (check_eb) {
            rlo[d].copy(prim,box);
            rhi[d].copy(prim,box);
        }
#endif
    }

    // change pressure to internal energy
    for     (int k = p_lo.z; k <= p_hi.z; ++k) {
        for   (int j = p_lo.y; j <= p_hi.y; ++j) {
            AMREX_PRAGMA_SIMD
                    for (int i = p_lo.x; i <= p_hi.x; ++i) {

#ifdef AMREX_USE_EB
                if (f4(i,j,k).isCovered()) {
                    continue;
                }
#endif

                gam4(i,j,k) = get_gamma(src4(i,j,k,+PrimIdx::Alpha)) - 1.0;

                src4(i,j,k,+PrimIdx::Prs) /= gam4(i,j,k);

            }
        }
    }

    // now do reconstruction

    // cycle over dimensions
    for (int d=0; d<AMREX_SPACEDIM; ++d) {

        Array4<Real> const& lo4 = rlo[d].array();
        Array4<Real> const& hi4 = rhi[d].array();

        for     (int k = lo.z; k <= hi.z; ++k) {
            for   (int j = lo.y; j <= hi.y; ++j) {
                AMREX_PRAGMA_SIMD
                        for (int i = lo.x; i <= hi.x; ++i) {

#ifdef AMREX_USE_EB
                    if (check_eb) {

                        // covered cell doesn't need calculating
                        if (f4(i,j,k).isCovered()) {
                            continue;
                        }

                        // cell that references a covered cell doesn't need calculating
                        bool skip = false;
                        stencil_index.fill(0);
                        for (int s=0; s<reconstruction->stencil_length; ++s) {
                            stencil_index[d] = s - offset;
                            // check if any of the stencil values are from a covered cell
                            if (f4(i+stencil_index[0], j+stencil_index[1], k+stencil_index[2]).isCovered()) {
                                skip = true;
                                break;
                            }
                        }

                        if (skip) {
                            continue;
                        }

                    }
#endif

                    // cycle over all components
                    for (int n = 0; n<N; ++n) {

                        // fill in the stencil along dimension index
                        stencil_index.fill(0);
                        for (int s=0; s<reconstruction->stencil_length; ++s) {
                            stencil_index[d] = s - offset;
                            stencil[s] = src4(i+stencil_index[0], j+stencil_index[1], k+stencil_index[2], n);
                        }

                        // perform reconstruction
                        cell_slope[n] = reconstruction->get_slope(stencil);
                        cell_value[n] = stencil[offset];

                    }


                    // apply corrections to slopes
                    // J. Sci. Comput. (2014) 60:584-611
                    // Robust Finite Volume Schemes for Two-Fluid Plasma Equations

                    Real &rho     = cell_value[+PrimIdx::Density];
                    Real &phi_rho = cell_slope[+PrimIdx::Density];

                    Real &u     = cell_value[+PrimIdx::Xvel];
                    Real &phi_u = cell_slope[+PrimIdx::Xvel];

                    Real &v     = cell_value[+PrimIdx::Yvel];
                    Real &phi_v = cell_slope[+PrimIdx::Yvel];

                    Real &w     = cell_value[+PrimIdx::Zvel];
                    Real &phi_w = cell_slope[+PrimIdx::Zvel];

                    Real &eps     = cell_value[+PrimIdx::Prs];
                    Real &phi_eps = cell_slope[+PrimIdx::Prs];

                    Real &alpha     = cell_value[+PrimIdx::Alpha];
                    Real &phi_alpha = cell_slope[+PrimIdx::Alpha];



                    // correct density slope
                    if (std::abs(phi_rho) > 2*rho) {
                        phi_rho = 2*sign(phi_rho, 0.0)*rho;
                    }

                    // get some face values
                    rho_lo = rho - 0.5*phi_rho;
                    rho_hi = rho + 0.5*phi_rho;

                    alpha_lo = alpha - 0.5*phi_alpha;
                    alpha_hi = alpha + 0.5*phi_alpha;

                    gam_lo = get_gamma(alpha_lo);
                    gam_hi = get_gamma(alpha_hi);

                    abs_phi = phi_u*phi_u + phi_v*phi_v + phi_w*phi_w;

                    // correct velocity slope
                    Real eps_face = eps - 0.5*std::abs(phi_eps);

                    if (eps_face <= 0.0) {
                        // if the reconstructed face value goes non-physical
                        // just set back to first order with zero slope
                        phi_u = 0.0;
                        phi_v = 0.0;
                        phi_w = 0.0;
                        phi_eps = 0.0;
                    } else {
                        coeff_eps = (rho/(rho_lo*rho_hi))*eps_face;
                        if ((0.125*abs_phi) > coeff_eps) {
                            phi_scale = sqrt(abs_phi);
                            coeff_eps = sqrt(8*coeff_eps);
                            phi_u = (phi_u/phi_scale)*coeff_eps;
                            phi_v = (phi_v/phi_scale)*coeff_eps;
                            phi_w = (phi_w/phi_scale)*coeff_eps;
                        }
                        // update eps
                        abs_phi = phi_u*phi_u + phi_v*phi_v + phi_w*phi_w;
                        eps -= (rho_lo*rho_hi/rho)*0.125*abs_phi;
                    }



                    // density
                    lo4(i,j,k,+PrimIdx::Density) = rho_lo;
                    hi4(i,j,k,+PrimIdx::Density) = rho_hi;

                    // x - velocity
                    lo4(i,j,k,+PrimIdx::Xvel) = u - 0.5*(rho_hi/rho)*phi_u;
                    hi4(i,j,k,+PrimIdx::Xvel) = u + 0.5*(rho_lo/rho)*phi_u;

                    // y - velocity
                    lo4(i,j,k,+PrimIdx::Yvel) = v - 0.5*(rho_hi/rho)*phi_v;
                    hi4(i,j,k,+PrimIdx::Yvel) = v + 0.5*(rho_lo/rho)*phi_v;

                    // z - velocity
                    lo4(i,j,k,+PrimIdx::Zvel) = w - 0.5*(rho_hi/rho)*phi_w;
                    hi4(i,j,k,+PrimIdx::Zvel) = w + 0.5*(rho_lo/rho)*phi_w;

                    // epsilon -> pressure
                    lo4(i,j,k,+PrimIdx::Prs) = (eps - 0.5*phi_eps)*(gam_lo - 1.0);
                    hi4(i,j,k,+PrimIdx::Prs) = (eps + 0.5*phi_eps)*(gam_hi - 1.0);

                    Real prs = lo4(i,j,k,+PrimIdx::Prs);

                    // tracer
                    lo4(i,j,k,+PrimIdx::Alpha) = alpha_lo;
                    hi4(i,j,k,+PrimIdx::Alpha) = alpha_hi;

                    // Temperature (calculate from pressure and density)
                    lo4(i,j,k,+PrimIdx::Temp) = lo4(i,j,k,+PrimIdx::Prs)/(rho_lo/get_mass(alpha_lo));
                    hi4(i,j,k,+PrimIdx::Temp) = hi4(i,j,k,+PrimIdx::Prs)/(rho_hi/get_mass(alpha_hi));

                }
            }
        }
    }


    // convert back to pressure
    for     (int k = p_lo.z; k <= p_hi.z; ++k) {
        for   (int j = p_lo.y; j <= p_hi.y; ++j) {
            AMREX_PRAGMA_SIMD
                    for (int i = p_lo.x; i <= p_hi.x; ++i) {
#ifdef AMREX_USE_EB
                if (f4(i,j,k).isCovered())
                    continue;
#endif
                src4(i,j,k,+PrimIdx::Prs) *= gam4(i,j,k);



            }
        }
    }

    return;
}

void HydroState::get_state_values(const Box& box,
                                  const FArrayBox& src,
                                  std::map<std::string,FArrayBox>& out,
                                  Vector<std::string>& updated
                                  EB_OPTIONAL(,const FArrayBox& vfrac)
                                  ) const
{
    BL_PROFILE("HydroState::get_state_values");
    const Dim3 lo = amrex::lbound(box);
    const Dim3 hi = amrex::ubound(box);

#ifdef AMREX_USE_EB
    Array4<const Real> const& vf4 = vfrac.array();
#endif

    updated.resize(0);

    // check conserved variables
    std::map<std::string,int> cons_tags;
    for (int i=0; i<n_cons(); ++i) {
        const std::string s = cons_names[i];
        const std::string var_name = s+"-"+name;
        if ( out.find(var_name) == out.end() ) continue;
        cons_tags[var_name] = i;
        updated.push_back(var_name);
    }

    // check primitive variables
    std::map<std::string,int> prim_tags;
    for (int i=0; i<n_prim(); ++i) {
        const std::string s = prim_names[i];
        if (s == cons_names[0]) continue;
        const std::string var_name = s+"-"+name;
        if ( out.find(var_name) == out.end() ) continue;
        prim_tags[var_name] = i;
        updated.push_back(var_name);
    }

    // additional variables

    Vector<std::string> other;

    const std::string charge_name = "charge-"+name;
    bool load_charge = out.find(charge_name) != out.end();
    if (load_charge) other.push_back(charge_name);

    const std::string mass_name = "mass-"+name;
    bool load_mass = out.find(mass_name) != out.end();
    if (load_mass) other.push_back(mass_name);

    const std::string gamma_name = "gamma-"+name;
    bool load_gamma = out.find(gamma_name) != out.end();
    if (load_gamma) other.push_back(gamma_name);

#ifdef AMREX_USE_EB
    const std::string vfrac_name = "vfrac-"+name;
    bool load_vfrac = out.find(vfrac_name) != out.end();
    if (load_vfrac) other.push_back(vfrac_name);
#endif

    updated.insert(updated.end(), other.begin(), other.end());

    std::map<std::string,Array4<Real>> out4;
    for (const std::string& s : updated) {
        out[s].resize(box, 1);
        out[s].setVal(0.0);
        out4[s] = out[s].array();
    }

    // temporary storage for retrieving the state data
    Vector<Real> S(n_cons()), Q(n_prim());

    Array4<const Real> const& src4 = src.array();

    for     (int k = lo.z; k <= hi.z; ++k) {
        for   (int j = lo.y; j <= hi.y; ++j) {
            AMREX_PRAGMA_SIMD
                    for (int i = lo.x; i <= hi.x; ++i) {

#ifdef AMREX_USE_EB
                if (vf4(i,j,k) == 0.0) {
                    continue;
                }
#endif

                get_state_vector(src, i, j, k, S);

                if (load_charge) out4[charge_name](i,j,k) = get_charge(S);
                if (load_mass)   out4[mass_name](i,j,k)   = get_mass(S);
                if (load_gamma)  out4[gamma_name](i,j,k)  = get_gamma(S);
            #ifdef AMREX_USE_EB
                if (load_vfrac)  out4[vfrac_name](i,j,k)  = vf4(i,j,k);
            #endif

                if (!cons_tags.empty()) {
                    for (const auto& var : cons_tags) {
                        out4[var.first](i,j,k) = S[var.second];
                    }
                }

                if (!prim_tags.empty()) {
                    cons2prim(S, Q);

                    for (const auto& var : prim_tags) {
                        out4[var.first](i,j,k) = Q[var.second];
                    }
                }
            }
        }
    }


    return;
}

void HydroState::calc_velocity(const Box& box,
                               FArrayBox& src,
                               FArrayBox& vel
                               EB_OPTIONAL(,const EBCellFlagFab& flag)
                               ) const
{
    BL_PROFILE("HydroState::calc_velocity");
    const Dim3 lo = amrex::lbound(box);
    const Dim3 hi = amrex::ubound(box);

    Array4<Real> const& src4 = src.array();
    Array4<Real> const& vel4 = vel.array();

#ifdef AMREX_USE_EB
    Array4<const EBCellFlag> const& f4 = flag.array();
#endif

    int nc = vel.nComp();

    for     (int k = lo.z; k <= hi.z; ++k) {
        for   (int j = lo.y; j <= hi.y; ++j) {
            AMREX_PRAGMA_SIMD
                    for (int i = lo.x; i <= hi.x; ++i) {

#ifdef AMREX_USE_EB
                if (f4(i,j,k).isCovered()) {
                    for (int n=0; n<nc; ++n) {
                        vel4(i,j,k,n) = 0.0;
                    }
                    continue;
                }
#endif

                Real invrho = 1.0/src4(i,j,k,+ConsIdx::Density);

                for (int n=0; n<nc; ++n) {
                    vel4(i,j,k,n) = src4(i,j,k,+ConsIdx::Xmom+n)*invrho;
                }
            }
        }
    }

    return;
}

// given all of the available face values load the ones expected by the flux calc into a vector
void HydroState::load_state_for_flux(Vector<Array4<const Real>> &face,
                                               int i, int j, int k, Vector<Real> &S) const
{
    BL_PROFILE("HydroState::load_state_for_flux");


    // first get the primitives of this state
    Array4<const Real> const &f4 = face[global_idx];

    S[+FluxIdx::Density] = f4(i,j,k,+PrimIdx::Density);
    S[+FluxIdx::Xvel] = f4(i,j,k,+PrimIdx::Xvel);
    S[+FluxIdx::Yvel] = f4(i,j,k,+PrimIdx::Yvel);
    S[+FluxIdx::Zvel] = f4(i,j,k,+PrimIdx::Zvel);
    S[+FluxIdx::Prs] = f4(i,j,k,+PrimIdx::Prs);
    S[+FluxIdx::Alpha] = f4(i,j,k,+PrimIdx::Alpha);
    S[+FluxIdx::Gamma] = get_gamma(S[+FluxIdx::Alpha]);

    return;
}

void HydroState::calc_viscous_fluxes(const Box& box,
                                     Array<FArrayBox, AMREX_SPACEDIM> &fluxes,
                                     const Box& pbox,
                                     const Vector<FArrayBox> &prim,
                                     #ifdef AMREX_USE_EB
                                     const EBCellFlagFab& flag,
                                     #endif
                                     const Real* dx) const
{
    BL_PROFILE("HydroState::calc_viscous_fluxes");
    // now calculate viscous fluxes and load them into the flux arrays
    if (viscous) {

        Viscous &V = *viscous;
        switch (V.get_type()) {
            case Viscous::Neutral :

//                plot_FAB_2d(flag, "flag", false);
//                plot_FAB_2d(prim[global_idx], 0, "prim density", false, true);

                calc_neutral_viscous_fluxes(box,
                                            fluxes,
                                            pbox,
                                            prim[global_idx],
                                            EB_OPTIONAL(flag,)
                                            dx);
                break;
            case Viscous::Ion :
                calc_ion_viscous_fluxes(box,
                                        fluxes,
                                        pbox,
                                        prim,
                                        EB_OPTIONAL(flag,)
                                        dx);
                break;
            case Viscous::Electron :
                calc_electron_viscous_fluxes(box,
                                             fluxes,
                                             pbox,
                                             prim,
                                             EB_OPTIONAL(flag,)
                                             dx);
                break;
            default :
                break;
        }
    }

}

void HydroState::calc_neutral_diffusion_terms(const Box& box,
                                              const FArrayBox& prim,
                                              FArrayBox& diff
                                              EB_OPTIONAL(,const EBCellFlagFab& flag)
                                              ) const
{
    BL_PROFILE("HydroState::calc_neutral_diffusion_terms");
    const Dim3 lo = amrex::lbound(box);
    const Dim3 hi = amrex::ubound(box);

    Array4<const Real> const& prim4 = prim.array();
    Array4<Real> const& d4 = diff.array();

#ifdef AMREX_USE_EB
    Array4<const EBCellFlag> const& f4 = flag.array();
#endif

    Viscous &V = *viscous;

    int np = n_prim();

    Vector<Real> Q(np);
    Real T, mu, kappa;

    for     (int k = lo.z; k <= hi.z; ++k) {
        for   (int j = lo.y; j <= hi.y; ++j) {
            AMREX_PRAGMA_SIMD
                    for (int i = lo.x; i <= hi.x; ++i) {

#ifdef AMREX_USE_EB
                if (f4(i,j,k).isCovered())
                    continue;
#endif

                for (int n=0; n<np; ++n) {
                    Q[n] = prim4(i,j,k,n);
                }

                V.get_neutral_coeffs(Q, T, mu, kappa);


                d4(i,j,k,Viscous::NeutralTemp) = T;
                d4(i,j,k,Viscous::NeutralKappa) = kappa;
                d4(i,j,k,Viscous::NeutralMu) = mu;
            }
        }
    }

    return;
}

#ifdef AMREX_USE_EB
void HydroState::calc_neutral_viscous_fluxes_eb(const Box& box, Array<FArrayBox,
                                                AMREX_SPACEDIM> &fluxes,
                                                const Box& pbox,
                                                const FArrayBox &prim,
                                                EB_OPTIONAL(const EBCellFlagFab& flag,)
                                                const Real* dx) const
{
    BL_PROFILE("HydroState::calc_neutral_viscous_fluxes_eb");
    FArrayBox diff(pbox, Viscous::NUM_NEUTRAL_DIFF_COEFFS);
    calc_neutral_diffusion_terms(pbox,
                                 prim,
                                 diff
                                 EB_OPTIONAL(,flag)
                                 );

    const Dim3 lo = amrex::lbound(box);
    const Dim3 hi = amrex::ubound(box);

    Array<Real, AMREX_SPACEDIM> dxinv;
    for (int d=0; d<AMREX_SPACEDIM; ++d) {
        dxinv[d] = 1/dx[d];
    }

    Array4<const Real> const& p4 = prim.array();
    Array4<const Real> const& d4 = diff.array();

    Array4<const EBCellFlag> const& f4 = flag.array();

    Real dudx=0, dudy=0, dudz=0, dvdx=0, dvdy=0, dwdx=0, dwdz=0, divu=0;
    const Real two_thirds = 2/3;

    const Array<Real,3>  weights = {0.0, 1.0, 0.5};
    Real whi, wlo;

    Real tauxx, tauxy, tauxz, dTdx, muf;

    int iTemp = Viscous::NeutralTemp;
    int iMu = Viscous::NeutralMu;
    int iKappa = Viscous::NeutralKappa;

    Vector<int> prim_vel_id = get_prim_vector_idx();

    int Xvel = prim_vel_id[0] + 0;
    int Yvel = prim_vel_id[0] + 1;
    int Zvel = prim_vel_id[0] + 2;

    Vector<int> cons_vel_id = get_cons_vector_idx();

    int Xmom = cons_vel_id[0] + 0;
    int Ymom = cons_vel_id[0] + 1;
    int Zmom = cons_vel_id[0] + 2;

    Vector<int> nrg_id = get_nrg_idx();
    int Eden = nrg_id[0];


    // X - direction
    Array4<Real> const& fluxX = fluxes[0].array();
    for     (int k = lo.z-AMREX_D_PICK(0,0,1); k <= hi.z+AMREX_D_PICK(0,0,1); ++k) {
        for   (int j = lo.y-AMREX_D_PICK(0,1,1); j <= hi.y+AMREX_D_PICK(0,1,1); ++j) {
            AMREX_PRAGMA_SIMD
                    for (int i = lo.x; i <= hi.x + 1; ++i) {

                bool covered = f4(i,j,k).isCovered();
                bool connected = f4(i,j,k).isConnected(-1,0,0);
                bool other_covered = f4(i-1,j,k).isCovered();

                // only calculate fluxes for fluid cells and between cells that are connected
                if (covered || other_covered || !connected)
                    continue;

                dTdx = (d4(i,j,k,iTemp) - d4(i-1,j,k,iTemp))*dxinv[0];

                dudx = (p4(i,j,k,Xvel) - p4(i-1,j,k,Xvel))*dxinv[0];
                dvdx = (p4(i,j,k,Yvel) - p4(i-1,j,k,Yvel))*dxinv[0];
                dwdx = (p4(i,j,k,Zvel) - p4(i-1,j,k,Zvel))*dxinv[0];

#if AMREX_SPACEDIM >= 2

                const int jhip = j + (int)f4(i,j,k).isConnected(0, 1,0);
                const int jhim = j - (int)f4(i,j,k).isConnected(0,-1,0);
                const int jlop = j + (int)f4(i-1,j,k).isConnected(0, 1,0);
                const int jlom = j - (int)f4(i-1,j,k).isConnected(0,-1,0);
                whi = weights[jhip-jhim];
                wlo = weights[jlop-jlom];
                dudy = (0.5*dxinv[1]) * ((p4(i  ,jhip,k,Xvel)-p4(i  ,jhim,k,Xvel))*whi+(p4(i-1,jlop,k,Xvel)-p4(i-1,jlom,k,Xvel))*wlo);
                dvdy = (0.50*dxinv[1]) * ((p4(i  ,jhip,k,Yvel)-p4(i  ,jhim,k,Yvel))*whi+(p4(i-1,jlop,k,Yvel)-p4(i-1,jlom,k,Yvel))*wlo);

#endif
#if AMREX_SPACEDIM == 3

                const int khip = k + (int)f4(i,j,k).isConnected(0,0, 1);
                const int khim = k - (int)f4(i,j,k).isConnected(0,0,-1);
                const int klop = k + (int)f4(i-1,j,k).isConnected(0,0, 1);
                const int klom = k - (int)f4(i-1,j,k).isConnected(0,0,-1);
                whi = weights[khip-khim];
                wlo = weights[klop-klom];
                dudz = (0.5*dxinv[2]) * ((p4(i  ,j,khip,Xvel)-p4(i  ,j,khim,Xvel))*whi + (p4(i-1,j,klop,Xvel)-p4(i-1,j,klom,Xvel))*wlo);
                dwdz = (0.5*dxinv[2]) * ((p4(i  ,j,khip,Zvel)-p4(i  ,j,khim,Zvel))*whi + (p4(i-1,j,klop,Zvel)-p4(i-1,j,klom,Zvel))*wlo);

#endif
                divu = dudx + dvdy + dwdz;

                muf = 0.5*(d4(i,j,k,iMu)+d4(i-1,j,k,iMu));
                tauxx = muf*(2*dudx-two_thirds*divu);
                tauxy = muf*(dudy+dvdx);
                tauxz = muf*(dudz+dwdx);

                fluxX(i,j,k,Xmom) -= tauxx;
                fluxX(i,j,k,Ymom) -= tauxy;
                fluxX(i,j,k,Zmom) -= tauxz;
                fluxX(i,j,k,Eden) -= 0.5*((p4(i,j,k,Xvel) +  p4(i-1,j,k,Xvel))*tauxx
                                          +(p4(i,j,k,Yvel) + p4(i-1,j,k,Yvel))*tauxy
                                          +(p4(i,j,k,Zvel) + p4(i-1,j,k,Zvel))*tauxz
                                          +(d4(i,j,k,iKappa)+d4(i-1,j,k,iKappa))*dTdx);
            }
        }
    }

    // Y - direction
#if AMREX_SPACEDIM >= 2
    Real tauyy, tauyz, dTdy;
    Real dvdz=0, dwdy=0;
    Array4<Real> const& fluxY = fluxes[1].array();
    for     (int k = lo.z-AMREX_D_PICK(0,0,1); k <= hi.z+AMREX_D_PICK(0,0,1); ++k) {
        for   (int j = lo.y; j <= hi.y + 1; ++j) {
            AMREX_PRAGMA_SIMD
                    for (int i = lo.x-1; i <= hi.x+1; ++i) {

                bool covered = f4(i,j,k).isCovered();
                bool connected = f4(i,j,k).isConnected(0,-1,0);
                bool other_covered = f4(i,j-1,k).isCovered();

                // only calculate fluxes for fluid cells and between cells that are connected
                if (covered || other_covered || !connected)
                    continue;

                dTdy = (d4(i,j,k,iTemp)-d4(i,j-1,k,iTemp))*dxinv[1];
                dudy = (p4(i,j,k,Xvel)-p4(i,j-1,k,Xvel))*dxinv[1];
                dvdy = (p4(i,j,k,Yvel)-p4(i,j-1,k,Yvel))*dxinv[1];
                dwdy = (p4(i,j,k,Zvel)-p4(i,j-1,k,Zvel))*dxinv[1];

                const int ihip = i + (int)f4(i,j,  k).isConnected( 1,0,0);
                const int ihim = i - (int)f4(i,j,  k).isConnected(-1,0,0);
                const int ilop = i + (int)f4(i,j-1,k).isConnected( 1,0,0);
                const int ilom = i - (int)f4(i,j-1,k).isConnected(-1,0,0);
                whi = weights[ihip-ihim];
                wlo = weights[ilop-ilom];

                dudx = (0.5*dxinv[0]) * ((p4(ihip,j  ,k,Xvel)-p4(ihim,j  ,k,Xvel))*whi + (p4(ilop,j-1,k,Xvel)-p4(ilom,j-1,k,Xvel))*wlo);
                dvdx = (0.5*dxinv[0]) * ((p4(ihip,j  ,k,Yvel)-p4(ihim,j  ,k,Yvel))*whi + (p4(ilop,j-1,k,Yvel)-p4(ilom,j-1,k,Yvel))*wlo);

#if AMREX_SPACEDIM == 3

                const int khip = k + (int)f4(i,j,  k).isConnected(0,0, 1);
                const int khim = k - (int)f4(i,j,  k).isConnected(0,0,-1);
                const int klop = k + (int)f4(i,j-1,k).isConnected(0,0, 1);
                const int klom = k - (int)f4(i,j-1,k).isConnected(0,0,-1);
                whi = weights[khip-khim];
                wlo = weights[klop-klom];

                dvdz = (0.5*dxinv[2]) * ((p4(i,j  ,khip,Yvel)-p4(i,j  ,khim,Yvel))*whi + (p4(i,j-1,klop,Yvel)-p4(i,j-1,klom,Yvel))*wlo);
                dwdz = (0.5*dxinv[2]) * ((p4(i,j  ,khip,Zvel)-p4(i,j  ,khim,Zvel))*whi + (p4(i,j-1,klop,Zvel)-p4(i,j-1,klom,Zvel))*wlo);

#endif
                divu = dudx + dvdy + dwdz;
                muf = 0.5*(d4(i,j,k,iMu)+d4(i,j-1,k,iMu));
                tauyy = muf*(2*dvdy-two_thirds*divu);
                tauxy = muf*(dudy+dvdx);
                tauyz = muf*(dwdy+dvdz);

                fluxY(i,j,k,Xmom) -= tauxy;
                fluxY(i,j,k,Ymom) -= tauyy;
                fluxY(i,j,k,Zmom) -= tauyz;
                fluxY(i,j,k,Eden) -= 0.5*((p4(i,j,k,Xvel)+p4(i,j-1,k,Xvel))*tauxy
                                          +(p4(i,j,k,Yvel)+p4(i,j-1,k,Yvel))*tauyy
                                          +(p4(i,j,k,Zvel)+p4(i,j-1,k,Zvel))*tauyz
                                          +(d4(i,j,k,iKappa) + d4(i,j-1,k,iKappa))*dTdy);

            }
        }
    }
#endif

    // Z - direction
#if AMREX_SPACEDIM == 3
    Real tauzz, dTdz;
    Array4<Real> const& fluxZ = fluxes[2].array();
    for     (int k = lo.z; k <= hi.z+1; ++k) {
        for   (int j = lo.y-1; j <= hi.y + 1; ++j) {
            AMREX_PRAGMA_SIMD
                    for (int i = lo.x-1; i <= hi.x+1; ++i) {

                bool covered = f4(i,j,k).isCovered();
                bool connected = f4(i,j,k).isConnected(0,0,-1);
                bool other_covered = f4(i,j,k-1).isCovered();

                // only calculate fluxes for fluid cells and between cells that are connected
                if (covered || other_covered || !connected)
                    continue;

                dTdz = (d4(i,j,k,iTemp)-d4(i,j,k-1,iTemp))*dxinv[2];
                dudz = (p4(i,j,k,Xvel)-p4(i,j,k-1,Xvel))*dxinv[2];
                dvdz = (p4(i,j,k,Yvel)-p4(i,j,k-1,Yvel))*dxinv[2];
                dwdz = (p4(i,j,k,Zvel)-p4(i,j,k-1,Zvel))*dxinv[2];

                const int ihip = i + (int)f4(i,j,k  ).isConnected( 1,0,0);
                const int ihim = i - (int)f4(i,j,k  ).isConnected(-1,0,0);
                const int ilop = i + (int)f4(i,j,k-1).isConnected( 1,0,0);
                const int ilom = i - (int)f4(i,j,k-1).isConnected(-1,0,0);
                whi = weights[ihip-ihim];
                wlo = weights[ilop-ilom];

                dudx = (0.5*dxinv[0]) * ((p4(ihip,j,k  ,Xvel)-p4(ihim,j,k  ,Xvel))*whi + (p4(ilop,j,k-1,Xvel)-p4(ilom,j,k-1,Xvel))*wlo);
                dwdx = (0.5*dxinv[0]) * ((p4(ihip,j,k  ,Zvel)-p4(ihim,j,k  ,Zvel))*whi + (p4(ilop,j,k-1,Zvel)-p4(ilom,j,k-1,Zvel))*wlo);

                const int jhip = j + (int)f4(i,j,k  ).isConnected(0 ,1,0);
                const int jhim = j - (int)f4(i,j,k  ).isConnected(0,-1,0);
                const int jlop = j + (int)f4(i,j,k-1).isConnected(0 ,1,0);
                const int jlom = j - (int)f4(i,j,k-1).isConnected(0,-1,0);
                whi = weights[jhip-jhim];
                wlo = weights[jlop-jlom];

                dvdy = (0.5*dxinv[1]) * ((p4(i,jhip,k  ,Yvel)-p4(i,jhim,k  ,Yvel))*whi + (p4(i,jlop,k-1,Yvel)-p4(i,jlom,k-1,Yvel))*wlo);
                dwdy = (0.5*dxinv[1]) * ((p4(i,jhip,k  ,Zvel)-p4(i,jhim,k  ,Zvel))*whi + (p4(i,jlop,k-1,Zvel)-p4(i,jlom,k-1,Zvel))*wlo);

                divu = dudx + dvdy + dwdz;
                muf = 0.5*(d4(i,j,k,iMu)+d4(i,j,k-1,iMu));
                tauxz = muf*(dudz+dwdx);
                tauyz = muf*(dvdz+dwdy);
                tauzz = muf*(2.*dwdz-two_thirds*divu);

                fluxZ(i,j,k,Xmom) -= tauxz;
                fluxZ(i,j,k,Ymom) -= tauyz;
                fluxZ(i,j,k,Zmom) -= tauzz;
                fluxZ(i,j,k,Eden) -= 0.5*((p4(i,j,k,Xvel)+p4(i,j,k-1,Xvel))*tauxz
                                          +(p4(i,j,k,Yvel)+p4(i,j,k-1,Yvel))*tauyz
                                          +(p4(i,j,k,Zvel)+p4(i,j,k-1,Zvel))*tauzz
                                          +(d4(i,j,k,iKappa) +d4(i,j,k-1,iKappa))*dTdz);

            }
        }
    }

#endif

}

#endif

void HydroState::calc_neutral_viscous_fluxes(const Box& box, Array<FArrayBox,
                                             AMREX_SPACEDIM> &fluxes,
                                             const Box& pbox,
                                             const FArrayBox &prim,
                                             EB_OPTIONAL(const EBCellFlagFab& flag,)
                                             const Real* dx) const
{
    BL_PROFILE("HydroState::calc_neutral_viscous_fluxes");
#ifdef AMREX_USE_EB
    if (flag.getType() != FabType::regular) {
        calc_neutral_viscous_fluxes_eb(box, fluxes, pbox, prim, flag, dx);
        return;
    }
#endif

    FArrayBox diff(pbox, Viscous::NUM_NEUTRAL_DIFF_COEFFS);
    calc_neutral_diffusion_terms(pbox,
                                 prim,
                                 diff
                             #ifdef AMREX_USE_EB
                                 ,flag
                             #endif
                                 );

    const Dim3 lo = amrex::lbound(box);
    const Dim3 hi = amrex::ubound(box);

    Array<Real, AMREX_SPACEDIM> dxinv;
    for (int d=0; d<AMREX_SPACEDIM; ++d) {
        dxinv[d] = 1/dx[d];
    }

    Array4<const Real> const& p4 = prim.array();
    Array4<const Real> const& d4 = diff.array();

#ifdef AMREX_USE_EB
    Array4<const EBCellFlag> const& f4 = flag.array();
#endif

    Real dudx=0, dudy=0, dudz=0, dvdx=0, dvdy=0, dwdx=0, dwdz=0, divu=0;
    const Real two_thirds = 2/3;

    Real tauxx, tauxy, tauxz, dTdx, muf;

    int iTemp = Viscous::NeutralTemp;
    int iMu = Viscous::NeutralMu;
    int iKappa = Viscous::NeutralKappa;

    Vector<int> prim_vel_id = get_prim_vector_idx();

    int Xvel = prim_vel_id[0] + 0;
    int Yvel = prim_vel_id[0] + 1;
    int Zvel = prim_vel_id[0] + 2;

    Vector<int> cons_vel_id = get_cons_vector_idx();

    int Xmom = cons_vel_id[0] + 0;
    int Ymom = cons_vel_id[0] + 1;
    int Zmom = cons_vel_id[0] + 2;

    Vector<int> nrg_id = get_nrg_idx();
    int Eden = nrg_id[0];

    Array4<Real> const& fluxX = fluxes[0].array();
    for     (int k = lo.z; k <= hi.z; ++k) {
        for   (int j = lo.y; j <= hi.y; ++j) {
            AMREX_PRAGMA_SIMD
                    for (int i = lo.x; i <= hi.x + 1; ++i) {

#ifdef AMREX_USE_EB
                if (f4(i,j,k).isCovered())
                    continue;
#endif

                dTdx = (d4(i,j,k,iTemp) - d4(i-1,j,k,iTemp))*dxinv[0];

                dudx = (p4(i,j,k,Xvel) - p4(i-1,j,k,Xvel))*dxinv[0];
                dvdx = (p4(i,j,k,Yvel) - p4(i-1,j,k,Yvel))*dxinv[0];
                dwdx = (p4(i,j,k,Zvel) - p4(i-1,j,k,Zvel))*dxinv[0];

#if AMREX_SPACEDIM >= 2
                dudy = (p4(i,j+1,k,Xvel)+p4(i-1,j+1,k,Xvel)-p4(i,j-1,k,Xvel)-p4(i-1,j-1,k,Xvel))*(0.25*dxinv[1]);
                dvdy = (p4(i,j+1,k,Yvel)+p4(i-1,j+1,k,Yvel)-p4(i,j-1,k,Yvel)-p4(i-1,j-1,k,Yvel))*(0.25*dxinv[1]);
#endif
#if AMREX_SPACEDIM == 3
                dudz = (p4(i,j,k+1,Xvel)+p4(i-1,j,k+1,Xvel)-p4(i,j,k-1,Xvel)-p4(i-1,j,k-1,Xvel))*(0.25*dxinv[2]);
                dwdz = (p4(i,j,k+1,Zvel)+p4(i-1,j,k+1,Zvel)-p4(i,j,k-1,Zvel)-p4(i-1,j,k-1,Zvel))*(0.25*dxinv[2]);
#endif
                divu = dudx + dvdy + dwdz;

                muf = 0.5*(d4(i,j,k,iMu)+d4(i-1,j,k,iMu));
                tauxx = muf*(2*dudx-two_thirds*divu);
                tauxy = muf*(dudy+dvdx);
                tauxz = muf*(dudz+dwdx);

                fluxX(i,j,k,Xmom) -= tauxx;
                fluxX(i,j,k,Ymom) -= tauxy;
                fluxX(i,j,k,Zmom) -= tauxz;
                fluxX(i,j,k,Eden) -= 0.5*((p4(i,j,k,Xvel) +  p4(i-1,j,k,Xvel))*tauxx
                                          +(p4(i,j,k,Yvel) + p4(i-1,j,k,Yvel))*tauxy
                                          +(p4(i,j,k,Zvel) + p4(i-1,j,k,Zvel))*tauxz
                                          +(d4(i,j,k,iKappa)+d4(i-1,j,k,iKappa))*dTdx);

            }
        }
    }

#if AMREX_SPACEDIM >= 2
    Real tauyy, tauyz, dTdy;
    Real dvdz=0, dwdy=0;
    Array4<Real> const& fluxY = fluxes[1].array();
    for     (int k = lo.z; k <= hi.z; ++k) {
        for   (int j = lo.y; j <= hi.y + 1; ++j) {
            AMREX_PRAGMA_SIMD
                    for (int i = lo.x; i <= hi.x; ++i) {

#ifdef AMREX_USE_EB
                if (f4(i,j,k).isCovered())
                    continue;
#endif

                dTdy = (d4(i,j,k,iTemp)-d4(i,j-1,k,iTemp))*dxinv[1];
                dudy = (p4(i,j,k,Xvel)-p4(i,j-1,k,Xvel))*dxinv[1];
                dvdy = (p4(i,j,k,Yvel)-p4(i,j-1,k,Yvel))*dxinv[1];
                dwdy = (p4(i,j,k,Zvel)-p4(i,j-1,k,Zvel))*dxinv[1];
                dudx = (p4(i+1,j,k,Xvel)+p4(i+1,j-1,k,Xvel)-p4(i-1,j,k,Xvel)-p4(i-1,j-1,k,Xvel))*(0.25*dxinv[0]);
                dvdx = (p4(i+1,j,k,Yvel)+p4(i+1,j-1,k,Yvel)-p4(i-1,j,k,Yvel)-p4(i-1,j-1,k,Yvel))*(0.25*dxinv[0]);
#if AMREX_SPACEDIM == 3
                dvdz = (p4(i,j,k+1,Yvel)+p4(i,j-1,k+1,Yvel)-p4(i,j,k-1,Yvel)-p4(i,j-1,k-1,Yvel))*(0.25*dxinv[2]);
                dwdz = (p4(i,j,k+1,Zvel)+p4(i,j-1,k+1,Zvel)-p4(i,j,k-1,Zvel)-p4(i,j-1,k-1,Zvel))*(0.25*dxinv[2]);
#endif
                divu = dudx + dvdy + dwdz;
                muf = 0.5*(d4(i,j,k,iMu)+d4(i,j-1,k,iMu));
                tauyy = muf*(2*dvdy-two_thirds*divu);
                tauxy = muf*(dudy+dvdx);
                tauyz = muf*(dwdy+dvdz);

                fluxY(i,j,k,Xmom) -= tauxy;
                fluxY(i,j,k,Ymom) -= tauyy;
                fluxY(i,j,k,Zmom) -= tauyz;
                fluxY(i,j,k,Eden) -= 0.5*((p4(i,j,k,Xvel)+p4(i,j-1,k,Xvel))*tauxy
                                          +(p4(i,j,k,Yvel)+p4(i,j-1,k,Yvel))*tauyy
                                          +(p4(i,j,k,Zvel)+p4(i,j-1,k,Zvel))*tauyz
                                          +(d4(i,j,k,iKappa) + d4(i,j-1,k,iKappa))*dTdy);

            }
        }
    }


#endif
#if AMREX_SPACEDIM == 3
    Real tauzz, dTdz;
    Array4<Real> const& fluxZ = fluxes[2].array();
    for     (int k = lo.z; k <= hi.z; ++k) {
        for   (int j = lo.y; j <= hi.y + 1; ++j) {
            AMREX_PRAGMA_SIMD
                    for (int i = lo.x; i <= hi.x; ++i) {

#ifdef AMREX_USE_EB
                if (f4(i,j,k).isCovered())
                    continue;
#endif

                dTdz = (d4(i,j,k,iTemp)-d4(i,j,k-1,iTemp))*dxinv[2];
                dudz = (p4(i,j,k,Xvel)-p4(i,j,k-1,Xvel))*dxinv[2];
                dvdz = (p4(i,j,k,Yvel)-p4(i,j,k-1,Yvel))*dxinv[2];
                dwdz = (p4(i,j,k,Zvel)-p4(i,j,k-1,Zvel))*dxinv[2];
                dudx = (p4(i+1,j,k,Xvel)+p4(i+1,j,k-1,Xvel)-p4(i-1,j,k,Xvel)-p4(i-1,j,k-1,Xvel))*(0.25*dxinv[0]);
                dwdx = (p4(i+1,j,k,Zvel)+p4(i+1,j,k-1,Zvel)-p4(i-1,j,k,Zvel)-p4(i-1,j,k-1,Zvel))*(0.25*dxinv[0]);
                dvdy = (p4(i,j+1,k,Yvel)+p4(i,j+1,k-1,Yvel)-p4(i,j-1,k,Yvel)-p4(i,j-1,k-1,Yvel))*(0.25*dxinv[1]);
                dwdy = (p4(i,j+1,k,Zvel)+p4(i,j+1,k-1,Zvel)-p4(i,j-1,k,Zvel)-p4(i,j-1,k-1,Zvel))*(0.25*dxinv[1]);
                divu = dudx + dvdy + dwdz;
                muf = 0.5*(d4(i,j,k,iMu)+d4(i,j,k-1,iMu));
                tauxz = muf*(dudz+dwdx);
                tauyz = muf*(dvdz+dwdy);
                tauzz = muf*(2.*dwdz-two_thirds*divu);

                fluxZ(i,j,k,Xmom) -= tauxz;
                fluxZ(i,j,k,Ymom) -= tauyz;
                fluxZ(i,j,k,Zmom) -= tauzz;
                fluxZ(i,j,k,Eden) -= 0.5*((p4(i,j,k,Xvel)+p4(i,j,k-1,Xvel))*tauxz
                                          +(p4(i,j,k,Yvel)+p4(i,j,k-1,Yvel))*tauyz
                                          +(p4(i,j,k,Zvel)+p4(i,j,k-1,Zvel))*tauzz
                                          +(d4(i,j,k,iKappa) +d4(i,j,k-1,iKappa))*dTdz);

            }
        }
    }

#endif


    return;
}

// ====================================================================================
void HydroState::calc_ion_diffusion_terms(const Box& box,const Vector<FArrayBox>& prim,
                                          State& EMstate,Array4<const Real> const& prim_EM4,
                                          State& ELEstate,Array4<const Real> const& prim_ELE4,
                                          FArrayBox& diff
                                          EB_OPTIONAL(,const EBCellFlagFab& flag)
                                          ) const {
    BL_PROFILE("HydroState::calc_ion_diffusion_terms");
    const Dim3 lo = amrex::lbound(box);
    const Dim3 hi = amrex::ubound(box);

    Array4<const Real> const& prim4 = prim[global_idx].array();
    Array4<Real> const& d4 = diff.array();
    Viscous &V = *viscous;


#ifdef AMREX_USE_EB
    Array4<const EBCellFlag> const& f4 = flag.array();
#endif

    int np = n_prim();// already includes Density, Xvel, Yvel, Zvel, Prs, Alpha,
    // where NUM is simply the number of entries
    
    Vector<Real> Q_i(np), Q_e(np), B_xyz(3);
    //prefix of p_ denotes particle characteristic
    Real alpha, T_i, eta_0, eta_1, eta_2, eta_3, eta_4, kappa_1, kappa_2, kappa_3;

    for (int k = lo.z; k <= hi.z; ++k) {
        for (int j = lo.y; j <= hi.y; ++j) {
            AMREX_PRAGMA_SIMD
                    for (int i = lo.x; i <= hi.x; ++i) {
#ifdef AMREX_USE_EB
                if (f4(i,j,k).isCovered())
                    continue;
#endif
                for (int n=0; n<np; ++n) {
                    Q_i[n] = prim4(i,j,k,n);
                    Q_e[n] = prim_ELE4(i,j,k,n);
                }

                B_xyz[0] = prim_EM4(i,j,k,+FieldState::ConsIdx::Bx);
                B_xyz[1] = prim_EM4(i,j,k,+FieldState::ConsIdx::By);
                B_xyz[2] = prim_EM4(i,j,k,+FieldState::ConsIdx::Bz);

                /*
                Print() << "\nB field in ion loop" << B_xyz[0] << "\t" 
                        << B_xyz[1] << "\t" << B_xyz[2] << "\n" ;

                */

                //V.get_ion_coeffs(Q, T, mu, kappa, p_charge, p_mass, T_e, nd_e);
                if (false && (j == -1 || j == 6)) {
                    Print() << "\nmatey i,j,k:\t" << i << " " << j << " " << k << "\n";
                } else if (false) {
                    Print() << "\ni,j,k:\t" << i << " " << j << " " << k << "\n";
                }
                V.get_ion_coeffs(EMstate,ELEstate,Q_i,Q_e,B_xyz,T_i,eta_0,eta_1,eta_2,eta_3,
                                 eta_4, kappa_1,kappa_2, kappa_3);
                //assign values to the diff (d4) matrix for usage in the superior function
                d4(i,j,k,Viscous::IonTemp) = T_i;
                d4(i,j,k,Viscous::IonKappa1) = kappa_1;
                d4(i,j,k,Viscous::IonKappa2) = kappa_2;
                d4(i,j,k,Viscous::IonKappa3) = kappa_3;
                d4(i,j,k,Viscous::IonEta0) = eta_0;
                d4(i,j,k,Viscous::IonEta1) = eta_1;
                d4(i,j,k,Viscous::IonEta2) = eta_2;
                d4(i,j,k,Viscous::IonEta3) = eta_3;
                d4(i,j,k,Viscous::IonEta4) = eta_4; // Note we could store the magnetic field but then we are doubling up on their storafe, perhpas better to just tolerate the access penalty
            }
        }
    }
    return;
}

void HydroState::calc_ion_viscous_fluxes(const Box& box, 
                                         Array<FArrayBox, AMREX_SPACEDIM> &fluxes,
                                         const Box& pbox, const Vector<FArrayBox>& prim,
                                         EB_OPTIONAL(const EBCellFlagFab& flag,)
                                         const Real* dx) const {
    BL_PROFILE("HydroState::calc_ion_viscous_fluxes");
    //data strucutre for the diffusion coefficients.
    FArrayBox diff_ion(pbox, Viscous::NUM_ION_DIFF_COEFFS);

    // from Braginskii 0 = ion, 1 = electron, 2 = em
    Vector<int> linked = viscous->get_linked_states();
    int linked_ion = linked[0];
    int linked_electron = linked[1];
    int linked_em = linked[2];

    //Electron state information
    State& ELEstate = GD::get_state(linked_electron);
    const FArrayBox& prim_ELE = prim[linked_electron];
    Array4<const Real> const& prim_ELE4 = prim_ELE.array(); // because input is const type
    //--- em state info
    State &EMstate = GD::get_state(linked_em);
    const FArrayBox& prim_EM = prim[linked_em];
    Array4<const Real> const& prim_EM4 = prim_EM.array(); // because input is const type
    /*
    int i=0, j=0, k=0;
    Print() << "\nB field in calc_ion_viscous_fluxes" 
            << prim_EM4(i,j,k,+FieldState::ConsIdx::Bx) 
            << "\t" << prim_EM4(i,j,k,+FieldState::ConsIdx::By) 
            << "\t" << prim_EM4(i,j,k,+FieldState::ConsIdx::Bz) << "\n" ;
    */

    //--- diffusion coefficients for each cell to be used
    calc_ion_diffusion_terms(pbox,prim,EMstate,prim_EM4,ELEstate,prim_ELE4,diff_ion EB_OPTIONAL(,flag));
    //handle all the generic flux calculations

#ifdef AMREX_USE_EB
    if (flag.getType() == FabType::singlevalued) {
        calc_charged_viscous_fluxes(linked_ion, linked_ion, linked_electron, linked_em,
                                    box, fluxes, pbox, prim, flag, dx, diff_ion);
        return;
    }
#endif

    calc_charged_viscous_fluxes(linked_ion, linked_ion, linked_electron, linked_em,
                                box, fluxes, pbox, prim,
                            #ifdef AMREX_USE_EB
                                flag,
                            #endif
                                dx, diff_ion);
    return;
}

// ====================================================================================

void HydroState::calc_electron_diffusion_terms(const Box& box,const Vector<FArrayBox>& prim,
                                               State& EMstate,
                                               Array4<const Real> const& prim_EM4,
                                               State& IONstate,
                                               Array4<const Real> const& prim_ION4,
                                               FArrayBox& diff
                                               EB_OPTIONAL(,const EBCellFlagFab& flag)
                                               ) const {
    BL_PROFILE("HydroState::calc_electron_diffusion_terms");

    const Dim3 lo = amrex::lbound(box);
    const Dim3 hi = amrex::ubound(box);

    Array4<const Real> const& prim4 = prim[global_idx].array();
    Array4<Real> const& d4 = diff.array();
    Viscous &V = *viscous;

#ifdef AMREX_USE_EB
    Array4<const EBCellFlag> const& f4 = flag.array();
#endif

    int np = n_prim();// already includes Density, Xvel, Yvel, Zvel, Prs, Alpha,
    // where NUM is simply the number of entries
    
    Vector<Real> Q_i(np), Q_e(np), B_xyz(3);
    //prefix of p_ denotes particle characteristic
    Real alpha, T_e, eta_0, eta_1, eta_2, eta_3, eta_4, kappa_1, kappa_2, kappa_3, beta1, beta2, beta3;

    for (int k = lo.z; k <= hi.z; ++k) {
        for (int j = lo.y; j <= hi.y; ++j) {
            AMREX_PRAGMA_SIMD
                    for (int i = lo.x; i <= hi.x; ++i) {
#ifdef AMREX_USE_EB
                if (f4(i,j,k).isCovered())
                    continue;
#endif
                for (int n=0; n<np; ++n) {
                    Q_e[n] = prim4(i,j,k,n);
                    Q_i[n] = prim_ION4(i,j,k,n);
                }
                B_xyz[0] = prim_EM4(i,j,k,+FieldState::ConsIdx::Bx);
                B_xyz[1] = prim_EM4(i,j,k,+FieldState::ConsIdx::By);
                B_xyz[2] = prim_EM4(i,j,k,+FieldState::ConsIdx::Bz);
                /*
                Print() << "\nB field in electron loop" << B_xyz[0] << "\t" 
                        << B_xyz[1] << "\t" << B_xyz[2] << "\n" ;
                */
                if (false && (j == -1 || j == 6)) {
                    Print() << "\nmatey i,j,k:\t" << i << " " << j << " " << k << "\n";
                } else if (false) {
                    Print() << "\ni,j,k:\t" << i << " " << j << " " << k << "\n";
                }

                V.get_electron_coeffs(EMstate, IONstate,Q_i,Q_e,B_xyz,T_e,eta_0,eta_1,eta_2,eta_3,
                                      eta_4, kappa_1,kappa_2, kappa_3, beta1, beta2, beta3);
                //assign values to the diff (d4) matrix for usage in the superior function
                d4(i,j,k,Viscous::EleTemp) = T_e;
                d4(i,j,k,Viscous::EleKappa1) = kappa_1;
                d4(i,j,k,Viscous::EleKappa2) = kappa_2;
                d4(i,j,k,Viscous::EleKappa3) = kappa_3;
                d4(i,j,k,Viscous::EleEta0) = eta_0;
                d4(i,j,k,Viscous::EleEta1) = eta_1;
                d4(i,j,k,Viscous::EleEta2) = eta_2;
                d4(i,j,k,Viscous::EleEta3) = eta_3;
                d4(i,j,k,Viscous::EleEta4) = eta_4;
                d4(i,j,k,Viscous::EleBeta1) = beta1;
                d4(i,j,k,Viscous::EleBeta2) = beta2;
                d4(i,j,k,Viscous::EleBeta3) = beta3;
            }
        }
    }

    return;
}

void HydroState::calc_electron_viscous_fluxes(const Box& box, 
                                              Array<FArrayBox, AMREX_SPACEDIM> &fluxes,
                                              const Box& pbox, const Vector<FArrayBox>& prim,
                                              EB_OPTIONAL(const EBCellFlagFab& flag,)
                                              const Real* dx) const {
    BL_PROFILE("HydroState::calc_electron_viscous_fluxes");
    FArrayBox diff_ele(pbox, Viscous::NUM_ELE_DIFF_COEFFS);

    // from Braginskii 0 = electron, 1 = ion, 2 = em
    Vector<int> linked = viscous->get_linked_states();
    int linked_electron = linked[0];
    int linked_ion = linked[1];
    int linked_em = linked[2];

    //Ion state information
    State& IONstate = GD::get_state(linked_ion);
    const FArrayBox& prim_ION = prim[linked_ion];
    Array4<const Real> const& prim_ION4 = prim_ION.array(); // because input is const type
    //--- em state info
    State &EMstate = GD::get_state(linked_em);
    const FArrayBox& prim_EM = prim[linked_em];
    Array4<const Real> const& prim_EM4 = prim_EM.array(); // because input is const type

    /*
    int i=0, j=0, k=0;
    Print() << "\nB field in calc_electron_viscous_fluxes" 
            << prim_EM4(i,j,k,+FieldState::ConsIdx::Bx) 
            << "\t" << prim_EM4(i,j,k,+FieldState::ConsIdx::By) 
            << "\t" << prim_EM4(i,j,k,+FieldState::ConsIdx::Bz) << "\n" ;
    */
    //--- diffusion coefficients for each cell to be used
    calc_electron_diffusion_terms(pbox,prim,EMstate,prim_EM4,IONstate,prim_ION4,diff_ele EB_OPTIONAL(,flag));
    //handle all the generic flux calculations

#ifdef AMREX_USE_EB
    if (flag.getType() == FabType::singlevalued) {
        calc_charged_viscous_fluxes(linked_electron, linked_ion, linked_electron, linked_em,
                                    box, fluxes, pbox, prim, flag, dx, diff_ele);
        return;
    }
#endif

    calc_charged_viscous_fluxes(linked_electron, linked_ion, linked_electron, linked_em,
                                box, fluxes, pbox, prim,
                                EB_OPTIONAL(flag,)
                                dx, diff_ele);
    return;
}

// ====================================================================================

void HydroState::calc_charged_viscous_fluxes(int passed_idx,
                                             int ion_idx,
                                             int electron_idx,
                                             int em_idx,
                                             const Box& box,
                                             Array<FArrayBox, AMREX_SPACEDIM> &fluxes,
                                             const Box& pbox,
                                             const Vector<FArrayBox>& prim,
                                             EB_OPTIONAL(const EBCellFlagFab& flag,)
                                             const Real* dx, FArrayBox& diff) const {
    BL_PROFILE("HydroState::calc_charged_viscous_fluxes");

    //create a box for the viscous sress tensor where we only store the 6 unique
    // elements in order of 0:tauxx, 1:tauyy, 2:tauzz, 3:tauxy, 4:tauxz, 5:tauyz
    Vector<Real> ViscTens(6);
    Vector<Real> q_flux(3);
    //--- em state info
    State &EMstate = GD::get_state(em_idx);
    const FArrayBox& prim_EM = prim[em_idx];
    Array4<const Real> const& prim_EM4 = prim_EM.array(); // because input is const type

    //---Sorting out indexing and storage access
    const Dim3 lo = amrex::lbound(box);
    const Dim3 hi = amrex::ubound(box);

    Array<Real, AMREX_SPACEDIM> dxinv;
    for (int d=0; d<AMREX_SPACEDIM; ++d) {
        dxinv[d] = 1/dx[d];
    }

    //Array4<const Real> const& p4 = prim[global_idx].array();
    Array4<const Real> const& p4 = prim[passed_idx].array();

    Array4<const Real> const& d4 = diff.array();

#ifdef AMREX_USE_EB
    Array4<const EBCellFlag> const& f4 = flag.array();
#endif

    //Array4<Real> const& d4 = diff.array();

    //Need to stage these properly once the algebraric expressions for the
    //viscous stresses are in
    Real dudx=0., dudy=0., dudz=0., dvdx=0., dvdy=0., dvdz=0., dwdx=0., dwdy=0.,
            dwdz=0., divu=0.;
    Real dTdx, dTdy, dTdz;
    const Real two_thirds = 2/3;

    State &ELEstate = GD::get_state(electron_idx); //electron state info
    const FArrayBox& prim_ELE = prim[electron_idx];
    Array4<const Real> const& prim_ELE4 = prim_ELE.array();
    State &IONstate = GD::get_state(ion_idx); //ion state info
    const FArrayBox& prim_ION = prim[ion_idx];
    Array4<const Real> const& prim_ION4 = prim_ION.array();

    //--------------Indexes for the trasnport coefficients
    int iTemp=-1, iEta0=-1, iEta1=-1, iEta2=-1, iEta3=-1, iEta4=-1, iKappa1=-1,
        iKappa2=-1, iKappa3=-1, iBeta1=-1, iBeta2=-1, iBeta3=-1;

    if (passed_idx == ion_idx) {
        iTemp = Viscous::IonTemp;
        iEta0 = Viscous::IonEta0; iEta1 = Viscous::IonEta1; iEta2 = Viscous::IonEta2;
        iEta3 = Viscous::IonEta3; iEta4 = Viscous::IonEta4;
        iKappa1=Viscous::IonKappa1;iKappa2=Viscous::IonKappa2;iKappa3 = Viscous::IonKappa3;

        State &ELEstate = GD::get_state(electron_idx); //electron state info
    } else if (passed_idx == electron_idx) {
        iTemp = Viscous::EleTemp;
        iEta0 = Viscous::EleEta0;  iEta1 = Viscous::EleEta1;   iEta2 = Viscous::EleEta2;
        iEta3 = Viscous::EleEta3;  iEta4 = Viscous::EleEta4;
        iKappa1=Viscous::EleKappa1;iKappa2=Viscous::EleKappa2;iKappa3= Viscous::EleKappa3;
        iBeta1 =Viscous::EleBeta1; iBeta2= Viscous::EleBeta2; iBeta3 = Viscous::EleBeta3;

        State &IONstate = GD::get_state(ion_idx); //ion state info
    } else {
        amrex::Abort("MFP_hydro.cpp ln 1196 - Shits fucked bruh, rogue state. ");
    }

    //--------------nondimensinalisation coefficients
    Real Debye_ref= GD::Debye, Larmor_ref = GD::Larmor;
    /*
    Real x_ref=GD::x_ref, n_ref=GD::n_ref, m_ref=GD::m_ref, rho_ref=GD::rho_ref,
    T_ref=GD::T_ref, u_ref=GD::u_ref;

    Real t_ref = x_ref/u_ref, rho_ref=m_ref/x_ref/x_ref/x_ref, mu_ref=m_ref/x_ref/t_ref;

    nd_divq  = t_ref/m_ref/n_ref/u_ref/u_ref*();
    nd_dpidx = t_ref/rho_ref/u_ref*(u_ref*mu_ref/x_ref/x_ref);
    nd_pidvdx= t_ref/m_ref/n_ref/u_ref/u_ref*()
    */
    Vector<int> prim_vel_id = get_prim_vector_idx();

    int Xvel = prim_vel_id[0] + 0;
    int Yvel = prim_vel_id[0] + 1;
    int Zvel = prim_vel_id[0] + 2;

    Vector<int> cons_vel_id = get_cons_vector_idx();

    int Xmom = cons_vel_id[0] + 0;
    int Ymom = cons_vel_id[0] + 1;
    int Zmom = cons_vel_id[0] + 2;

    Vector<int> nrg_id = get_nrg_idx();
    int Eden = nrg_id[0];
    //Magnetic field components used for Braginskii terms, `p' represents prime.
    //Real bx_pp=0.,by_pp=0.,bz_pp=0.,bx_p=0.,by_p=0.,B=0.,B_pp=0.,B_p=0.;
    Real xB, yB, zB;

    int i_disp, j_disp, k_disp;

    if (passed_idx == ion_idx) {
        iTemp = Viscous::IonTemp;
    } else if (passed_idx == electron_idx) {
        iTemp = Viscous::EleTemp;
    }
    Vector<Real> faceCoefficients(Viscous::NUM_ELE_DIFF_COEFFS); // use the beta places and leave them uninitiated if ion

    //Delete after debug 
    /*
    for (int k = lo.z; k <= hi.z; ++k) {
        for (int j = lo.y; j <= hi.y; ++j) {
            AMREX_PRAGMA_SIMD
                for (int i = lo.x; i <= hi.x + 1; ++i) {
                  Print() << "u["<<i<<"]\t"<< p4(i,j,k,Xvel) << "\n";
                }
          }
    }
    */

    //Print() << "\nfluxX call\n";
    Vector<Real> u_rel(3);
    Array4<Real> const& fluxX = fluxes[0].array();
    for (int k = lo.z; k <= hi.z; ++k) {
        for (int j = lo.y; j <= hi.y; ++j) {
            AMREX_PRAGMA_SIMD
                for (int i = lo.x; i <= hi.x + 1; ++i) {
#ifdef AMREX_USE_EB
                if (f4(i,j,k).isCovered())
                    continue;
#endif
                if (passed_idx == electron_idx) {
                    faceCoefficients[iBeta1] = 0.5*(d4(i,j,k,iBeta1)+d4(i-1,j,k,iBeta1));
                    faceCoefficients[iBeta2] = 0.5*(d4(i,j,k,iBeta2)+d4(i-1,j,k,iBeta2));
                    faceCoefficients[iBeta3] = 0.5*(d4(i,j,k,iBeta3)+d4(i-1,j,k,iBeta3));
                }
        
                //Print() << "\nKappa1 coefficient\ti, j, k\t" << i << " " << j << " " << k << " "  << d4(i,j,k,iKappa1) << "\tj\t" << d4(i-1,j,k,iKappa1) << "\n";
                faceCoefficients[iKappa1] = 0.5*(d4(i,j,k,iKappa1)+d4(i-1,j,k,iKappa1));
                faceCoefficients[iKappa2] = 0.5*(d4(i,j,k,iKappa2)+d4(i-1,j,k,iKappa2));
                faceCoefficients[iKappa3] = 0.5*(d4(i,j,k,iKappa3)+d4(i-1,j,k,iKappa3));

                faceCoefficients[iEta0] = 0.5*(d4(i,j,k,iEta0)+d4(i-1,j,k,iEta0));
                faceCoefficients[iEta1] = 0.5*(d4(i,j,k,iEta1)+d4(i-1,j,k,iEta1));
                faceCoefficients[iEta2] = 0.5*(d4(i,j,k,iEta2)+d4(i-1,j,k,iEta2));
                faceCoefficients[iEta3] = 0.5*(d4(i,j,k,iEta3)+d4(i-1,j,k,iEta3));
                faceCoefficients[iEta4] = 0.5*(d4(i,j,k,iEta4)+d4(i-1,j,k,iEta4));

                if (false && GD::verbose >= 4 ) {
                  if (passed_idx == ion_idx) {
                        Print() << "\nIon coefficients - cell:\t" << i << "\t" << j 
                                << "\t" << k << "\n";;
                    } else if (passed_idx == electron_idx) {
                        Print() << "\nElectron coefficients - cell:\t" << i << "\t" << j 
                                << "\t" << k << "\n";
                  }
                  /*
                  for (int coPrint = 0; coPrint < Viscous::NUM_ELE_DIFF_COEFFS; coPrint ++) {
                    if (coPrint > 0) {
                        Print() << "Coefficient " << coPrint << "\t" 
                                << faceCoefficients[coPrint] << "\n";
                    }
                    
                  }
                  */
                }

                xB = 0.5*(prim_EM4(i,j,k,+FieldState::ConsIdx::Bx) + prim_EM4(i-1,j,k,+FieldState::ConsIdx::Bx)); // using i j k  means you are taking the magnetic field in the cell i, not on the interface 
                yB = 0.5*(prim_EM4(i,j,k,+FieldState::ConsIdx::By) + prim_EM4(i-1,j,k,+FieldState::ConsIdx::By));
                zB = 0.5*(prim_EM4(i,j,k,+FieldState::ConsIdx::Bz) + prim_EM4(i-1,j,k,+FieldState::ConsIdx::Bz));
                //if (global_idx == ion_idx)
                u_rel[0] = 0.5*(prim_ELE4(i,j,k,Xvel) + prim_ELE4(i-1,j,k,Xvel) - prim_ION4(i,j,k,Xvel) - prim_ION4(i-1,j,k,Xvel)); //TODO fix up flux
                u_rel[1] = 0.5*(prim_ELE4(i,j,k,Yvel) + prim_ELE4(i-1,j,k,Yvel) - prim_ION4(i,j,k,Yvel) - prim_ION4(i-1,j,k,Yvel));
                u_rel[2] = 0.5*(prim_ELE4(i,j,k,Zvel) + prim_ELE4(i-1,j,k,Zvel) - prim_ION4(i,j,k,Zvel) - prim_ION4(i-1,j,k,Zvel));

                dTdx = (d4(i,j,k,iTemp) - d4(i-1,j,k,iTemp))*dxinv[0];

                dudx = (p4(i,j,k,Xvel) - p4(i-1,j,k,Xvel))*dxinv[0];
                if (GD::verbose > 4) {
                  Print() << "u_i\t" << p4(i,j,k,Xvel) << "\nu_i-1\t" << p4(i-1,j,k,Xvel) << "\n";
                }
                dvdx = (p4(i,j,k,Yvel) - p4(i-1,j,k,Yvel))*dxinv[0];
                dwdx = (p4(i,j,k,Zvel) - p4(i-1,j,k,Zvel))*dxinv[0];

#if AMREX_SPACEDIM >= 2
                dTdy = (d4(i,j+1,k,iTemp)+d4(i-1,j+1,k,iTemp)-d4(i,j-1,k,iTemp)-d4(i-1,j-1,k,iTemp))*(0.25*dxinv[1]);
                dudy = (p4(i,j+1,k,Xvel)+p4(i-1,j+1,k,Xvel)-p4(i,j-1,k,Xvel)-p4(i-1,j-1,k,Xvel))*(0.25*dxinv[1]);
                dvdy = (p4(i,j+1,k,Yvel)+p4(i-1,j+1,k,Yvel)-p4(i,j-1,k,Yvel)-p4(i-1,j-1,k,Yvel))*(0.25*dxinv[1]);

                // Put in to facilitate the matrix operations that will one day be
                // (//TODO) replaced with the explicit algebraic expression of the
                // viscous stress tensor entries hack without having correct
                // dimensional staging
                dwdy = (p4(i,j+1,k,Zvel)+p4(i-1,j+1,k,Zvel)-p4(i,j-1,k,Zvel)-p4(i-1,j-1,k,Zvel))*(0.25*dxinv[1]);
#endif

#if AMREX_SPACEDIM == 3
                dTdz = (d4(i,j,k+1,iTemp)+d4(i-1,j,k+1,iTemp)-d4(i,j,k-1,iTemp)-d4(i-1,j,k-1,iTemp))*(0.25*dxinv[1]);
                //(d4(i,j,k,iTemp)-d4(i,j,k-1,iTemp))*dxinv[2];

                dudz = (p4(i,j,k+1,Xvel)+p4(i-1,j,k+1,Xvel)-p4(i,j,k-1,Xvel)-p4(i-1,j,k-1,Xvel))*(0.25*dxinv[2]);

                dwdz = (p4(i,j,k+1,Zvel)+p4(i-1,j,k+1,Zvel)-p4(i,j,k-1,Zvel)-p4(i-1,j,k-1,Zvel))*(0.25*dxinv[2]);
                //Put in to hack without having correct dimensional staging
                dvdz = (p4(i,j,k+1,Yvel)+p4(i,j-1,k+1,Yvel)-p4(i,j,k-1,Yvel)-p4(i,j-1,k-1,Yvel))*(0.25*dxinv[2]);
#endif
                divu = dudx + dvdy + dwdz;
                if (GD::verbose>4) {
                  Print() << "divu\t" << divu << "\n";
                }
                ///TODO hacks because of transform which needs to be turned into algebra...

                //--- retrive the viscous stress tensor and heat flux vector on this face

                //TODO Print() << "Check changes to BRaginskiiViscousTensor... and calculation of xB, .. on interfaces are correct.";
                BraginskiiViscousTensorHeatFlux(passed_idx, ion_idx, electron_idx, em_idx, i, j, k, box, dxinv,
                                                xB, yB, zB, u_rel, dTdx, dTdy, dTdz,
                                                dudx, dudy, dudz, dvdx, dvdy, dvdz, dwdx, dwdy, dwdz,
                                                faceCoefficients, ViscTens, q_flux);

                if (true && GD::verbose > 1) Print() << "fluxX\ti, j, k: " << i << " " << j << " " << k << "\n";
                if (false && GD::verbose >1) {
                  //Print()<<"Check sign tensor product of viscous stress tensor and velocity";
                  Print() << "i, j, k: " << i << " " << j << " " << k << "\nfluxX Xmom\t" << ViscTens[0] 
                          << "\tfluxX Ymom\t" <<  ViscTens[3]
                          << "\tfluxX Zmom\t" <<  ViscTens[5]
                          << "\nfluxX Eden visc \t" 
                          <<  0.5*((p4(i,j,k,Xvel) + p4(i-1,j,k,Xvel))*ViscTens[0]+
                          (p4(i,j,k,Yvel) + p4(i-1,j,k,Yvel))*ViscTens[3]+
                          (p4(i,j,k,Zvel) + p4(i-1,j,k,Zvel))*ViscTens[5])
                          << "\tfluxX Eden q_flux\t" 
                          << q_flux[0] << "\n";
                }

                fluxX(i,j,k,Xmom) += ViscTens[0];
                fluxX(i,j,k,Ymom) += ViscTens[3];
                fluxX(i,j,k,Zmom) += ViscTens[5];
                //assume typo in livescue formulation
                fluxX(i,j,k,Eden) += 0.5*((p4(i,j,k,Xvel) + p4(i-1,j,k,Xvel))*ViscTens[0]+
                        (p4(i,j,k,Yvel) + p4(i-1,j,k,Yvel))*ViscTens[3]+
                        (p4(i,j,k,Zvel) + p4(i-1,j,k,Zvel))*ViscTens[5])
                        + q_flux[0];
                //+(d4(i,j,k,iKappa)+d4(i-1,j,k,iKappa))*dTdx);
            }
        }
    }

    /*
    if (GD::verbose >=3) {
      if (passed_idx == electron_idx) {
        Print() << "\nqu_e_max = " << qu_e_temp_max << "\nqt_e_max = " << qt_e_temp_max ;
      } else if (passed_idx == ion_idx) {
        Print() << "\nqt_i_max = " << qt_i_temp_max ;
      } else {
        Print() << "\nHouston...";
      }
    }
    */

#if AMREX_SPACEDIM >= 2

    //Print() << "\nfluxY call\n";
    Real tauyy, tauyz;
    Array4<Real> const& fluxY = fluxes[1].array();
    for     (int k = lo.z; k <= hi.z; ++k) {
        for   (int j = lo.y; j <= hi.y + 1; ++j) {
            AMREX_PRAGMA_SIMD
                    for (int i = lo.x; i <= hi.x; ++i) {
#ifdef AMREX_USE_EB
                if (f4(i,j,k).isCovered())
                    continue;
#endif
                if (passed_idx == electron_idx) {
                    //Print() << "\nElectron";
                    faceCoefficients[iBeta1] = 0.5*(d4(i,j,k,iBeta1)+d4(i,j-1,k,iBeta1));
                    faceCoefficients[iBeta2] = 0.5*(d4(i,j,k,iBeta2)+d4(i,j-1,k,iBeta2));
                    faceCoefficients[iBeta3] = 0.5*(d4(i,j,k,iBeta3)+d4(i,j-1,k,iBeta3));
                }
                             
                //Print() << "Kappa1 coefficient\ti, j, k\t" << i << " " << j << " " << k << " "  << d4(i,j,k,iKappa1) << "\tj\t" << d4(i,j-1,k,iKappa1) << "\n";

                faceCoefficients[iKappa1] = 0.5*(d4(i,j,k,iKappa1)+d4(i,j-1,k,iKappa1));
                faceCoefficients[iKappa2] = 0.5*(d4(i,j,k,iKappa2)+d4(i,j-1,k,iKappa2));
                faceCoefficients[iKappa3] = 0.5*(d4(i,j,k,iKappa3)+d4(i,j-1,k,iKappa3));

                faceCoefficients[iEta0] = 0.5*(d4(i,j,k,iEta0)+d4(i,j-1,k,iEta0));
                faceCoefficients[iEta1] = 0.5*(d4(i,j,k,iEta1)+d4(i,j-1,k,iEta1));
                faceCoefficients[iEta2] = 0.5*(d4(i,j,k,iEta2)+d4(i,j-1,k,iEta2));
                faceCoefficients[iEta3] = 0.5*(d4(i,j,k,iEta3)+d4(i,j-1,k,iEta3));
                faceCoefficients[iEta4] = 0.5*(d4(i,j,k,iEta4)+d4(i,j-1,k,iEta4));

                xB = 0.5*(prim_EM4(i,j,k,+FieldState::ConsIdx::Bx) + prim_EM4(i,j-1,k,+FieldState::ConsIdx::Bx)); // using i j k  means you are taking the magnetic field in the cell i, not on the interface 
                yB = 0.5*(prim_EM4(i,j,k,+FieldState::ConsIdx::By) + prim_EM4(i,j-1,k,+FieldState::ConsIdx::By));
                zB = 0.5*(prim_EM4(i,j,k,+FieldState::ConsIdx::Bz) + prim_EM4(i,j-1,k,+FieldState::ConsIdx::Bz));
                u_rel[0] = 0.5*(prim_ELE4(i,j,k,Xvel) + prim_ELE4(i,j-1,k,Xvel) - prim_ION4(i,j,k,Xvel) - prim_ION4(i,j-1,k,Xvel)); //TODO fix up flux
                u_rel[1] = 0.5*(prim_ELE4(i,j,k,Yvel) + prim_ELE4(i,j-1,k,Yvel) - prim_ION4(i,j,k,Yvel) - prim_ION4(i,j-1,k,Yvel));
                u_rel[2] = 0.5*(prim_ELE4(i,j,k,Zvel) + prim_ELE4(i,j-1,k,Zvel) - prim_ION4(i,j,k,Zvel) - prim_ION4(i,j-1,k,Zvel));

                dTdy = (d4(i,j,k,iTemp)-d4(i,j-1,k,iTemp))*dxinv[1];

                dudy = (p4(i,j,k,Xvel)-p4(i,j-1,k,Xvel))*dxinv[1];
                dvdy = (p4(i,j,k,Yvel)-p4(i,j-1,k,Yvel))*dxinv[1];
                dwdy = (p4(i,j,k,Zvel)-p4(i,j-1,k,Zvel))*dxinv[1];

                dudx = (p4(i+1,j,k,Xvel)+p4(i+1,j-1,k,Xvel)-p4(i-1,j,k,Xvel)-p4(i-1,j-1,k,Xvel))*(0.25*dxinv[0]);
                dvdx = (p4(i+1,j,k,Yvel)+p4(i+1,j-1,k,Yvel)-p4(i-1,j,k,Yvel)-p4(i-1,j-1,k,Yvel))*(0.25*dxinv[0]);
                //--- retrive the viscous stress tensor and heat flux vector on this face
                ///TODO hacks because of transform which needs to be turned into algebra...
                dwdx = (p4(i+1,j,k,Zvel)+p4(i+1,j-1,k,Zvel)-p4(i-1,j,k,Zvel)-p4(i-1,j-1,k,Zvel))*(0.25*dxinv[0]);

#if AMREX_SPACEDIM == 3
                dvdz = (p4(i,j,k+1,Yvel)+p4(i,j-1,k+1,Yvel)-p4(i,j,k-1,Yvel)-p4(i,j-1,k-1,Yvel))*(0.25*dxinv[2]);
                dwdz = (p4(i,j,k+1,Zvel)+p4(i,j-1,k+1,Zvel)-p4(i,j,k-1,Zvel)-p4(i,j-1,k-1,Zvel))*(0.25*dxinv[2]);

                //--- retrive the viscous stress tensor and heat flux vector on this face
                ///TODO hacks because of transform which needs to be turned into algebra...
                dudz = (p4(i,j,k+1,Xvel)+p4(i,j-1,k+1,Xvel)-p4(i,j,k-1,Xvel)-p4(i,j-1,k-1,Xvel))*(0.25*dxinv[2]);

#endif
                divu = dudx + dvdy + dwdz;
                //muf = 0.5*(d4(i,j,k,iMu)+d4(i,j-1,k,iMu));
                //tauyy = muf*(2*dvdy-two_thirds*divu);
                //tauxy = muf*(dudy+dvdx);
                //tauyz = muf*(dwdy+dvdz);


                BraginskiiViscousTensorHeatFlux(passed_idx, ion_idx, electron_idx, em_idx, i, j, k, box, dxinv,
                                                xB, yB, zB, u_rel, dTdx, dTdy, dTdz,
                                                dudx, dudy, dudz, dvdx, dvdy, dvdz, dwdx, dwdy, dwdz,
                                                faceCoefficients, ViscTens, q_flux);

                if (true && GD::verbose > 1) Print() << "fluxY\ti, j, k: " << i << " " << j << " " << k << "\n";
                if (false && GD::verbose >1) {
                  //Print()<<"Check sign tensor product of viscous stress tensor and velocity";
                  Print() << "i, j, k: " << i << " " << j << " " << k << "\nfluxY Xmom\t" << ViscTens[0] 
                          << "\tfluxY Ymom\t" <<  ViscTens[3]
                          << "\tfluxY Zmom\t" <<  ViscTens[5]
                          << "\nfluxY Eden visc \t" 
                          <<  0.5*((p4(i,j,k,Xvel) + p4(i,j-1,k,Xvel))*ViscTens[3]+
                          (p4(i,j,k,Yvel) + p4(i,j-1,k,Yvel))*ViscTens[1]+
                          (p4(i,j,k,Zvel) + p4(i,j-1,k,Zvel))*ViscTens[4])
                          << "\tfluxX Eden q_flux\t" 
                          << q_flux[1] << "\n";
                }
                if (GD::verbose >4) {
                  Print() << "Adding viscous terms to flux array";
                }
                fluxY(i,j,k,Xmom) += ViscTens[3];//tauxy;
                fluxY(i,j,k,Ymom) += ViscTens[1];//tauyy;
                fluxY(i,j,k,Zmom) += ViscTens[4];//tauyz;
                fluxY(i,j,k,Eden) += 0.5*((p4(i,j,k,Xvel)+p4(i,j-1,k,Xvel))*ViscTens[3]+
                        (p4(i,j,k,Yvel)+p4(i,j-1,k,Yvel))*ViscTens[1]+
                        (p4(i,j,k,Zvel)+p4(i,j-1,k,Zvel))*ViscTens[4])
                        + q_flux[1];
                //+(d4(i,j,k,iKappa) + d4(i,j-1,k,iKappa))*dTdy);
            }
        }
    }
#endif

#if AMREX_SPACEDIM == 3
    //Print() << "\nfluxZ call\n";
    Real tauzz ;
    Array4<Real> const& fluxZ = fluxes[2].array();
    for     (int k = lo.z; k <= hi.z; ++k) {
        for   (int j = lo.y; j <= hi.y + 1; ++j) {
            AMREX_PRAGMA_SIMD
                    for (int i = lo.x; i <= hi.x; ++i) {
#ifdef AMREX_USE_EB
                if (f4(i,j,k).isCovered())
                    continue;
#endif
                if (passed_idx == electron_idx) {
                    faceCoefficients[iBeta1] = 0.5*(d4(i,j,k,iBeta1)+d4(i,j,k-1,iBeta1));
                    faceCoefficients[iBeta2] = 0.5*(d4(i,j,k,iBeta2)+d4(i,j,k-1,iBeta2));
                    faceCoefficients[iBeta3] = 0.5*(d4(i,j,k,iBeta3)+d4(i,j,k-1,iBeta3));
                }
        
                faceCoefficients[iKappa1] = 0.5*(d4(i,j,k,iKappa1)+d4(i,j,k-1,iKappa1));
                faceCoefficients[iKappa2] = 0.5*(d4(i,j,k,iKappa2)+d4(i,j,k-1,iKappa2));
                faceCoefficients[iKappa3] = 0.5*(d4(i,j,k,iKappa3)+d4(i,j,k-1,iKappa3));

                faceCoefficients[iEta0] = 0.5*(d4(i,j,k,iEta0)+d4(i,j,k-1,iEta0));
                faceCoefficients[iEta1] = 0.5*(d4(i,j,k,iEta1)+d4(i,j,k-1,iEta1));
                faceCoefficients[iEta2] = 0.5*(d4(i,j,k,iEta2)+d4(i,j,k-1,iEta2));
                faceCoefficients[iEta3] = 0.5*(d4(i,j,k,iEta3)+d4(i,j,k-1,iEta3));
                faceCoefficients[iEta4] = 0.5*(d4(i,j,k,iEta4)+d4(i,j,k-1,iEta4));

                xB = 0.5*(prim_EM4(i,j,k,+FieldState::ConsIdx::Bx) + prim_EM4(i,j,k-1,+FieldState::ConsIdx::Bx)); // using i j k  means you are taking the magnetic field in the cell i, not on the interface 
                yB = 0.5*(prim_EM4(i,j,k,+FieldState::ConsIdx::By) + prim_EM4(i,j,k-1,+FieldState::ConsIdx::By));
                zB = 0.5*(prim_EM4(i,j,k,+FieldState::ConsIdx::Bz) + prim_EM4(i,j,k-1,+FieldState::ConsIdx::Bz));
                u_rel[0] = 0.5*(prim_ELE4(i,j,k,Xvel) + prim_ELE4(i,j,k-1,Xvel) - prim_ION4(i,j,k,Xvel) - prim_ION4(i,j,k-1,Xvel)); //TODO fix up flux
                u_rel[1] = 0.5*(prim_ELE4(i,j,k,Yvel) + prim_ELE4(i,j,k-1,Yvel) - prim_ION4(i,j,k,Yvel) - prim_ION4(i,j,k-1,Yvel));
                u_rel[2] = 0.5*(prim_ELE4(i,j,k,Zvel) + prim_ELE4(i,j,k-1,Zvel) - prim_ION4(i,j,k,Zvel) - prim_ION4(i,j,k-1,Zvel));

                dTdz = (d4(i,j,k,iTemp)-d4(i,j,k-1,iTemp))*dxinv[2];
                dudz = (p4(i,j,k,Xvel)-p4(i,j,k-1,Xvel))*dxinv[2];
                dvdz = (p4(i,j,k,Yvel)-p4(i,j,k-1,Yvel))*dxinv[2];
                dwdz = (p4(i,j,k,Zvel)-p4(i,j,k-1,Zvel))*dxinv[2];

                dudx = (p4(i+1,j,k,Xvel)+p4(i+1,j,k-1,Xvel)-p4(i-1,j,k,Xvel)-p4(i-1,j,k-1,Xvel))*(0.25*dxinv[0]);
                dwdx = (p4(i+1,j,k,Zvel)+p4(i+1,j,k-1,Zvel)-p4(i-1,j,k,Zvel)-p4(i-1,j,k-1,Zvel))*(0.25*dxinv[0]);
                dvdy = (p4(i,j+1,k,Yvel)+p4(i,j+1,k-1,Yvel)-p4(i,j-1,k,Yvel)-p4(i,j-1,k-1,Yvel))*(0.25*dxinv[1]);
                dwdy = (p4(i,j+1,k,Zvel)+p4(i,j+1,k-1,Zvel)-p4(i,j-1,k,Zvel)-p4(i,j-1,k-1,Zvel))*(0.25*dxinv[1]);
                divu = dudx + dvdy + dwdz;
                //muf = 0.5*(d4(i,j,k,iMu)+d4(i,j,k-1,iMu));
                //tauxz = muf*(dudz+dwdx);
                //tauyz = muf*(dvdz+dwdy);
                //tauzz = muf*(2.*dwdz-two_thirds*divu);

                ///TODO hacks because of transform which needs to be turned into algebra...
                dvdx = (p4(i+1,j,k,Yvel)+p4(i+1,j,k-1,Yvel)-p4(i-1,j,k,Yvel)-p4(i-1,j,k-1,Yvel))*(0.25*dxinv[0]);
                dudy = (p4(i,j+1,k,Xvel)+p4(i,j+1,k-1,Xvel)-p4(i,j-1,k,Xvel)-p4(i,j-1,k-1,Xvel))*(0.25*dxinv[1]);

                //--- retrive the viscous stress tensor and heat flux vector on this face
                BraginskiiViscousTensorHeatFlux(passed_idx, ion_idx, electron_idx, em_idx, i, j, k, box, dxinv,
                                                xB, yB, zB, u_rel, dTdx, dTdy, dTdz,
                                                dudx, dudy, dudz, dvdx, dvdy, dvdz,
                                                dwdx, dwdy, dwdz, faceCoefficients, ViscTens, q_flux);

                if (GD::verbose >4) {
                  Print() << "Adding viscous terms to flux array";
                }
                fluxZ(i,j,k,Xmom) += ViscTens[5];//tauxz;
                fluxZ(i,j,k,Ymom) += ViscTens[4];//tauyz;
                fluxZ(i,j,k,Zmom) += ViscTens[2];//tauzz;
                fluxZ(i,j,k,Eden) += 0.5*((p4(i,j,k,Xvel)+p4(i,j,k-1,Xvel))*ViscTens[5]+
                        (p4(i,j,k,Yvel)+p4(i,j,k-1,Yvel))*ViscTens[4]+
                        (p4(i,j,k,Zvel)+p4(i,j,k-1,Zvel))*ViscTens[2])
                        +q_flux[2];
                //+(d4(i,j,k,iKappa) +d4(i,j,k-1,iKappa))*dTdz);

            }
        }
    }

#endif

    return;
}

// ===================================================================================
// function for calculating the viscous stress tensor and heat flux vector 
// for the braginskii transport 
void HydroState::BraginskiiViscousTensorHeatFlux(int passed_idx,
                                                 int ion_idx,
                                                 int electron_idx,
                                                 int em_idx,
                                                 int i, int j, int k,
                                                 const Box& box,
                                                 Array<Real, AMREX_SPACEDIM>& dxinv,
                                                 Real xB, Real yB, Real zB, Vector<Real> u_rel,
                                                 Real dTdx, Real dTdy, Real dTdz,
                                                 Real dudx, Real dudy, Real dudz,
                                                 Real dvdx, Real dvdy, Real dvdz,
                                                 Real dwdx, Real dwdy, Real dwdz,
                                                 Vector<Real> faceCoefficients, 
                                                 //Array4<const Real> const& d4,
                                                 Vector<Real>& ViscTens,
                                                 Vector<Real>& q_flux) const {
    BL_PROFILE("HydroState::BraginskiiViscousTensorHeatFlux")
    //Note all the properties used in here need to be for the interface, not just the cell i!!!
    if (GD::verbose > 4) {
      Print() << "dudx\t" << dudx << "\ndudy\t" << dudy << "\ndudz\t" << dudz 
              << "\ndvdx\t" << dvdx << "\ndvdy\t" << dvdy << "\ndvdz\t" << dvdz
              << "\ndwdx\t" << dwdx << "\ndwdy\t" << dwdy << "\ndwdz\t" << dwdz << "\n";
    }

    //--- choose index extensions depending on the direction
    /*
    int x_hi=0, x_lo=0, y_hi=0, y_lo=0, z_hi=0, z_lo=0;
    if (flux_dim == 0) {
      x_hi = 1; x_lo = 0;
     } else if (flux_dim == 1) {
      y_hi = 1; y_lo = 0;
     } else {
      z_hi = 1; z_lo = 0;
    }
    */

    //---Sorting out indexing and storage access
    const Dim3 lo = amrex::lbound(box);
    const Dim3 hi = amrex::ubound(box);
    Real bx_pp=0.,by_pp=0.,bz_pp=0.,bx_p=0.,by_p=0.,B=0.,B_pp=0.,B_p=0.;
    Real qu_e_temp_max=0., qt_e_temp_max=0., qt_i_temp_max=0.;

    Real divu = dudx + dvdy + dwdz ;
    int i_disp, j_disp, k_disp;
    //Using the formulation of Li 2018 "High-order two-fluid plasma
    // solver for direct numerical simulations of plasma flowswith full
    // transport phenomena"  --- this is outdated, i think i was eefering
    // to the coulomb loagrithm which i just ended up taking fro braginskii

    //---get magnetic field orientated unit vectors
    B = xB*xB + yB*yB + zB*zB;
    if (B < 0.0) { //Perhaps remove this, will never happen we're not using imaginary numbers
        Print() << "MFP_hydro.cpp ln 2002 - Negative magnetic field error";
        Print() << std::to_string(B);
        amrex::Abort("Negative magnetic field - ln 2002 MFP_hydro.cpp");
    } else {
        if (B < GD::effective_zero) {
            if (GD::verbose >= 3) {
                Print() << "\nZero magnetic field \n";
            }
            //amrex::Warning("zero magnetic field check the viscous stress matrix is not transformed at all.");
            B_pp = 0.;
            B_p  = 0.;
        } else if ( (std::abs(xB) < GD::effective_zero) && (std::abs(yB) < GD::effective_zero) &&
                    (std::abs(zB) > GD::effective_zero) ) {
            if (GD::verbose >= 4) {
                Print() << "\nZero x and y magnetic field \n";
                amrex::Warning("fixed frame aligned mag field, check the viscous stress matrix is not transformed at all.");
            }
            B_pp = 1/sqrt(B); // B prime prime
            B_p  = 0.;
        } else {
            B_pp = 1/sqrt(B); // B prime prime
            B_p  = 1/sqrt(xB*xB + yB*yB); // B prime
        }

        //added for Q transformation matrix See Li 2018 "High-order
        // two-fluid plasma solver for direct numerical simulations
        // of plasma flowswith full transport phenomena"
    }

    bx_pp = xB*B_pp; bx_p = xB*B_p;
    by_pp = yB*B_pp; by_p = yB*B_p;
    bz_pp = zB*B_pp;
    
    Vector<Real> B_unit(3); //Magnetic field unit vector
    Vector<Real> u_para(3); //Velocity parallel to B_unit
    Vector<Real> u_perp(3); //Velocity perpendicular to B_unit
    Vector<Real> u_chev(3); //unit vector perp to u and B_unit
    Vector<Real> TG_para(3);//Temp grad parallel to B_unit
    Vector<Real> TG_perp(3);//Temp grad perpendicular to B_unit
    Vector<Real> TG_chev(3);//unit vector perp to gradT and B_unit

    B_unit[0] = bx_pp; B_unit[1] = by_pp; B_unit[2] = bz_pp;

    Vector<int> prim_vel_id = get_prim_vector_idx();

    int Xvel = prim_vel_id[0] + 0;
    int Yvel = prim_vel_id[0] + 1;
    int Zvel = prim_vel_id[0] + 2;

    int iTemp=-1, iEta0=-1, iEta1=-1, iEta2=-1, iEta3=-1, iEta4=-1, iKappa1=-1,
            iKappa2=-1, iKappa3=-1, iBeta1=-1, iBeta2=-1, iBeta3=-1;

    if (passed_idx == ion_idx) {
        iTemp = Viscous::IonTemp;
        iEta0 = Viscous::IonEta0; iEta1 = Viscous::IonEta1; iEta2 = Viscous::IonEta2;
        iEta3 = Viscous::IonEta3; iEta4 = Viscous::IonEta4;
        iKappa1=Viscous::IonKappa1;iKappa2=Viscous::IonKappa2;iKappa3 = Viscous::IonKappa3;

        State &ELEstate = GD::get_state(electron_idx); //electron state info
    } else if (passed_idx == electron_idx) {
        iTemp = Viscous::EleTemp;
        iEta0 = Viscous::EleEta0;  iEta1 = Viscous::EleEta1;   iEta2 = Viscous::EleEta2;
        iEta3 = Viscous::EleEta3;  iEta4 = Viscous::EleEta4;
        iKappa1=Viscous::EleKappa1;iKappa2=Viscous::EleKappa2;iKappa3= Viscous::EleKappa3;
        iBeta1 =Viscous::EleBeta1; iBeta2= Viscous::EleBeta2; iBeta3 = Viscous::EleBeta3;

        State &IONstate = GD::get_state(ion_idx); //ion state info
    } else {
        amrex::Abort("MFP_hydro.cpp ln 1196 - Shits fucked bruh, rogue state. ");
    }

    Real dot_B_unit_TG, dot_B_unit_U ;//temp variables
    
    dot_B_unit_U = bx_pp*u_rel[0] + by_pp*u_rel[1] + bz_pp*u_rel[2];
    dot_B_unit_TG= bx_pp*dTdx + by_pp*dTdy + bz_pp*dTdz;
    // Prepare the x-components components
    u_para[0] = B_unit[0]*dot_B_unit_U ;
    TG_para[0]= B_unit[0]*dot_B_unit_TG ;
    u_perp[0] = u_rel[0] - u_para[0];
    u_chev[0] = B_unit[1]*u_rel[2] - B_unit[2]*u_rel[1];
    TG_perp[0]= dTdx - TG_para[0];
    TG_chev[0]= B_unit[1]*dTdz-B_unit[2]*dTdy;
    // Prepare the y-components components
    u_para[1] = B_unit[1]*dot_B_unit_U ;
    TG_para[1]= B_unit[1]*dot_B_unit_TG ;
    u_perp[1] = u_rel[1] - u_para[1];
    u_chev[1] = -(B_unit[0]*u_rel[2]-B_unit[2]*u_rel[0]);
    TG_perp[1]= dTdy - TG_para[1];
    TG_chev[1]= -(B_unit[0]*dTdz-B_unit[2]*dTdx);
    // Prepare the z-components components
    u_para[2] = B_unit[2]*dot_B_unit_U ;
    TG_para[2]= B_unit[2]*dot_B_unit_TG ;
    u_perp[2] = u_rel[2] - u_para[2];
    u_chev[2] = B_unit[0]*u_rel[1]-B_unit[1]*u_rel[0];
    TG_perp[2]= dTdz - TG_para[2];
    TG_chev[2]= B_unit[0]*dTdy-B_unit[1]*dTdx;

    int rowFirst=3,columnFirst=3,columnSecond=3;
    /* Matrix representations of viscous stree tensor and associated
    quantities are defined as:
    Trans       - the transformation matrix Q
    Strain      - Strain rate matrix mate W
    StrainTrans - Strain rate matrix transformed into the B aligned frame
    ViscStress  - Viscous stress tensor in lab frame, PI
    ViscStressTrans - Viscous stress tensor in B aligned frame, PI'
    TransT      - Transpose of the transformation matrix, Q'
    */
    Vector<Vector<Real> > Trans(3,Vector<Real>(3));
    Vector<Vector<Real> > Strain(3,Vector<Real>(3));
    Vector<Vector<Real> > StrainTrans(3,Vector<Real>(3));
    Vector<Vector<Real> > ViscStress(3,Vector<Real>(3)), ViscStressTrans(3,Vector<Real>(3)), TransT(3,Vector<Real>(3));
    Vector<Vector<Real> > WorkingMatrix(3,Vector<Real>(3));

    //if (global_idx == electron_idx)
    if (passed_idx == electron_idx) {
        Real qu_e_temp = faceCoefficients[iBeta1]*u_para[0] + faceCoefficients[iBeta2]*u_perp[0] + faceCoefficients[iBeta3]*u_chev[0] ;
        Real qt_e_temp = -faceCoefficients[iKappa1]*TG_para[0] - faceCoefficients[iKappa2]*TG_perp[0] - faceCoefficients[iKappa3]*TG_chev[0];

        qu_e_temp_max = std::max(std::fabs(qu_e_temp_max), std::fabs(qu_e_temp));
        qt_e_temp_max = std::max(std::fabs(qt_e_temp_max), std::fabs(qt_e_temp));

        q_flux[0] = qt_e_temp + qu_e_temp ;
//#if AMREX_SPACEDIM >= 2
        q_flux[1] = faceCoefficients[iBeta1]*u_para[1] + faceCoefficients[iBeta2]*u_perp[1] + faceCoefficients[iBeta3]*u_chev[1]
                   -faceCoefficients[iKappa1]*TG_para[1] - faceCoefficients[iKappa2]*TG_perp[1] - faceCoefficients[iKappa3]*TG_chev[1];
//#endif

//#if AMREX_SPACEDIM == 3
        q_flux[2] = faceCoefficients[iBeta1]*u_para[2] + faceCoefficients[iBeta2]*u_perp[2] + faceCoefficients[iBeta3]*u_chev[2]
                   -faceCoefficients[iKappa1]*TG_para[2] - faceCoefficients[iKappa2]*TG_perp[2] - faceCoefficients[iKappa3]*TG_chev[2];

        if (true && GD::verbose > 2) {
            Print() << "ele kappa[i]\t" << faceCoefficients[iKappa1] << "\t" << faceCoefficients[iKappa2] << "\t" << faceCoefficients[iKappa3] << "\n";
            Print() << "\tTG_[0]\t" << TG_para[0] << "\t" << TG_perp[0] << "\t" << TG_chev[0] << "\n";

            Print() << "\tTG_[1]\t" << TG_para[1] << "\t" << TG_perp[1] << "\t" << TG_chev[1] << "\n";

            Print() << "\tTG_[2]\t" << TG_para[2] << "\t" << TG_perp[2] << "\t" << TG_chev[2] << "\n";
        }

        if (false) {
          Print() << "ele q_flux[i]\t" << q_flux[0] << "\t" << q_flux[1] << "\t" << q_flux[2] << "\n";  
        }

//#endif
    } else {
        q_flux[0] = -faceCoefficients[iKappa1]*TG_para[0] - faceCoefficients[iKappa2]*TG_perp[0] + faceCoefficients[iKappa3]*TG_chev[0];

        qt_i_temp_max=std::max(std::fabs(qt_i_temp_max),std::fabs(q_flux[0]));

//#if AMREX_SPACEDIM >= 2
        q_flux[1] = -faceCoefficients[iKappa1]*TG_para[1] - faceCoefficients[iKappa2]*TG_perp[1] + faceCoefficients[iKappa3]*TG_chev[1];
//#endif

//#if AMREX_SPACEDIM == 3
        q_flux[2] = -faceCoefficients[iKappa1]*TG_para[2] - faceCoefficients[iKappa2]*TG_perp[2] + faceCoefficients[iKappa3]*TG_chev[2];

        if (true && GD::verbose > 2) {
            Print() << "ion kappa[i]\t" << faceCoefficients[iKappa1] << "\t" << faceCoefficients[iKappa2] << "\t" << faceCoefficients[iKappa3] << "\n";
            Print() << "\tTG_[0]\t" << TG_para[0] << "\t" << TG_perp[0] << "\t" << TG_chev[0] << "\n";
            Print() << "\tTG_[1]\t" << TG_para[1] << "\t" << TG_perp[1] << "\t" << TG_chev[1] << "\n";
            Print() << "\tTG_[2]\t" << TG_para[2] << "\t" << TG_perp[2] << "\t" << TG_chev[2] << "\n";
        }
        if (false) {
          Print() << "ion q_flux[i]\t" << q_flux[0] << "\t" << q_flux[1] << "\t" << q_flux[2] << "\n";
        }
//#endif
    }

    //Calculate the viscous stress tensor
      //Populate strain rate tensor in B unit aligned cartesian frame
      //This is braginskii's  - the strain rate tensor multplied by negative one later to Li
      // Livescue formulation
    Strain[0][0] = 2*dudx - 2./3.*divu;
    Strain[0][1] = dudy + dvdx;
    Strain[0][2] = dwdx + dudz;
    Strain[1][0] = Strain[0][1];
    Strain[1][1] = 2*dvdy - 2./3.*divu;
    Strain[1][2] = dvdz + dwdy;
    Strain[2][0] = Strain[0][2];
    Strain[2][1] = Strain[1][2];
    Strain[2][2] = 2*dwdz - 2./3.*divu;

    for (i_disp=0; i_disp<3; ++i_disp) { // set to zero
        for (j_disp=0; j_disp<3; ++j_disp) {
            WorkingMatrix[i_disp][j_disp] = 0.;
            StrainTrans[i_disp][j_disp] = 0.;

            if (GD::verbose >= 9) Print() << "Livescue on ";
                  
            Strain[i_disp][j_disp] = - Strain[i_disp][j_disp] ; //make negtive for Li Livescue
        }
    }

    if (GD::verbose >= 9) {
        Print() << "Test each of the B field conditions to try break the code";
    }

      // Do we have a special case of B=0 or Bx=By=0 and Bz = B?
    if (1) { //(B < GD::effective_zero) { // #####################Note for now we are hard coding for non-gyro viscosity.
      if (GD::verbose >= 9) {
        Print() << "\nHard coded non-hyro viscosity\n";
      }

      ViscStress[0][0] = faceCoefficients[iEta0]*Strain[0][0];
      
      ViscStress[0][1] = faceCoefficients[iEta0]*Strain[0][1];
      ViscStress[1][0] = ViscStress[0][1];
  
      ViscStress[0][2] = faceCoefficients[iEta0]*Strain[0][2];
      ViscStress[2][0] = ViscStress[0][2];
  
      ViscStress[1][1] = faceCoefficients[iEta0]*Strain[1][1];
      ViscStress[1][2] = faceCoefficients[iEta0]*Strain[1][2];
      ViscStress[2][1] = ViscStress[1][2];
  
      ViscStress[2][2] = faceCoefficients[iEta0]*Strain[2][2];

    } else if ( (std::abs(xB) < GD::effective_zero) && (std::abs(yB) < GD::effective_zero) &&
                    (std::abs(zB) > GD::effective_zero) ) {
      ViscStress[0][0]=-1/2*faceCoefficients[iEta0]*
              (Strain[0][0] + Strain[1][1])
              -1/2*faceCoefficients[iEta1]*
              (Strain[0][0] - Strain[1][1])
              -faceCoefficients[iEta3]*(Strain[0][1]);
      
      ViscStress[0][1]=-faceCoefficients[iEta1]*Strain[0][1]
              +1/2*faceCoefficients[iEta3]*
              (Strain[0][0] - Strain[1][1]);
  
      ViscStress[1][0]= ViscStress[0][1];
  
      ViscStress[0][2]=-faceCoefficients[iEta2]*Strain[0][2]
              - faceCoefficients[iEta4]*Strain[1][2];
  
      ViscStress[2][0]= ViscStress[0][2];
  
      ViscStress[1][1]=-1/2*faceCoefficients[iEta0]*
              (Strain[0][0] + Strain[1][1])
              -1/2*faceCoefficients[iEta1]*
              (Strain[1][1] - Strain[0][0])
              +faceCoefficients[iEta3]*Strain[0][1];
  
      ViscStress[1][2]=-faceCoefficients[iEta2]*Strain[1][2] +
              faceCoefficients[iEta4]*Strain[0][2];
  
      ViscStress[2][1]=ViscStress[1][2];
  
      ViscStress[2][2]=-faceCoefficients[iEta0]*Strain[2][2];

    } else { //the generic case with non trivial magnetic field requiring a transform
      //Populate the transformation matrix from cartesian normal to B unit
      // aligned cartesian - Li 2018

      Trans[0][0]=-by_p; Trans[0][1]=-bx_p*bz_pp; Trans[0][2]=bx_pp;
  
      Trans[1][0]= bx_p; Trans[1][1]=-by_p*bz_pp; Trans[1][2] = by_pp;
  
      Trans[2][0]= 0;    Trans[2][1]= bx_p*bx_pp
              +by_p*by_pp; Trans[2][2] =bz_pp;
        //Populate the transpose of the transformation matrix
      TransT[0][0]=Trans[0][0]; TransT[0][1]=Trans[1][0]; TransT[0][2]=Trans[2][0];
  
      TransT[1][0]=Trans[0][1]; TransT[1][1]=Trans[1][1]; TransT[1][2]=Trans[2][1];
  
      TransT[2][0]=Trans[0][2]; TransT[2][1]=Trans[1][2]; TransT[2][2]=Trans[2][2];
  
      // Multiplying Q' (Transpose) by W (StressStrain)
      for (i_disp = 0; i_disp< rowFirst; ++i_disp) {
          for(j_disp = 0; j_disp< columnSecond; ++j_disp) {
              for(k_disp=0; k_disp<columnFirst; ++k_disp) {
                  WorkingMatrix[i_disp][j_disp] += TransT[i_disp][k_disp]*
                          Strain[k_disp][j_disp];
              }
          }
      }
      // Multiplying Q'W by Q
      for(i_disp = 0; i_disp< rowFirst; ++i_disp) {
          for(j_disp = 0; j_disp< columnSecond; ++j_disp) {
              for(k_disp=0; k_disp<columnFirst; ++k_disp) {
                  StrainTrans[i_disp][j_disp] += WorkingMatrix[i_disp][k_disp] *
                          Trans[k_disp][j_disp];
              }
          }
      }
      //faceCoefficients[iKappa3]
      //Populate visc stress tensor in cartesian normal frame
      ViscStressTrans[0][0]=-1/2*faceCoefficients[iEta0]*
              (StrainTrans[0][0] + StrainTrans[1][1])
              -1/2*faceCoefficients[iEta1]*
              (StrainTrans[0][0] - StrainTrans[1][1])
              -faceCoefficients[iEta3]*(StrainTrans[0][1]);
      //check the removal of the negative sign before the second bracket in ViscStressTrans00 was not a mistake
      ViscStressTrans[0][1]=-faceCoefficients[iEta1]*StrainTrans[0][1]
              +1/2*faceCoefficients[iEta3]*
              (StrainTrans[0][0] - StrainTrans[1][1]);
  
      ViscStressTrans[1][0]= ViscStressTrans[0][1];
  
      ViscStressTrans[0][2]=-faceCoefficients[iEta2]*StrainTrans[0][2]
              - faceCoefficients[iEta4]*StrainTrans[1][2];
  
      ViscStressTrans[2][0]= ViscStressTrans[0][2];
  
      ViscStressTrans[1][1]=-1/2*faceCoefficients[iEta0]*
              (StrainTrans[0][0] + StrainTrans[1][1])
              -1/2*faceCoefficients[iEta1]*
              (StrainTrans[1][1] - StrainTrans[0][0])
              +faceCoefficients[iEta3]*StrainTrans[0][1];
  
      ViscStressTrans[1][2]=-faceCoefficients[iEta2]*StrainTrans[1][2] +
              faceCoefficients[iEta4]*StrainTrans[0][2];
  
      ViscStressTrans[2][1]=ViscStressTrans[1][2];
  
      ViscStressTrans[2][2]=-faceCoefficients[iEta0]*StrainTrans[2][2];
  
      for (i_disp=0; i_disp<3; ++i_disp) {//Set to zero
          for (j_disp=0; j_disp<3; ++j_disp) {
              WorkingMatrix[i_disp][j_disp] = 0.;
              ViscStress[i_disp][j_disp] = 0.;
          }
      }
      // Multiplying Q (Trans) by PI' (ViscStressTrans)
      for(i_disp = 0; i_disp< rowFirst; ++i_disp) {
          for(j_disp = 0; j_disp< columnSecond; ++j_disp) {
              for(k_disp=0; k_disp<columnFirst; ++k_disp) {
                  WorkingMatrix[i_disp][j_disp] += Trans[i_disp][k_disp] *
                          ViscStressTrans[k_disp][j_disp];
              }
          }
      }
      // Multiplying Q*PI' by Q^T
      for(i_disp = 0; i_disp< rowFirst; ++i_disp) {
          for(j_disp = 0; j_disp< columnSecond; ++j_disp) {
              for(k_disp=0; k_disp<columnFirst; ++k_disp) {
                  ViscStress[i_disp][j_disp] += WorkingMatrix[i_disp][k_disp] *
                          TransT[k_disp][j_disp];
              }
          }
      }
    }
 
    //Storing
    //NOTE STORAGE ACCORDING TO THE TAUXX, TAUYY, TAUZZ, TAUXY OR TAUYX,
    // TAUYZ OR TAUZY, TAUXZ OR TAUZX
    ViscTens[0] = ViscStress[0][0];
    ViscTens[1] = ViscStress[1][1];
    ViscTens[2] = ViscStress[2][2];
    ViscTens[3] = ViscStress[0][1];
    ViscTens[4] = ViscStress[1][2];
    ViscTens[5] = ViscStress[0][2];

    if (true && GD::verbose >= 1) {
    /*
    Print() << "Bunit\t" << B_unit[0] << "\nBunit\t" << B_unit[1] << "\nBunit\t" << B_unit[2] << "\n";

    Print() << "u_rel:\t" << u_rel[0] << "\nu_rel:\t" << u_rel[1] << "\nu_rel:\t" 
            << u_rel[2] << "\n";
    Print() << "dot_B_unit_U\t" << dot_B_unit_U<< "\n";
    Print() << "dot_B_unit_TG\t" << dot_B_unit_TG<< "\n";
    Print() << "u_para\t" << u_para[0] << "\nu_para\t" << u_para[1] << "\nu_para\t" 
            << u_para[2] << "\n";
    Print() << "u_perp\t" << u_perp[0] << "\nu_perp\t" << u_perp[1] << "\nu_perp\t" 
            << u_perp[2] << "\n";
    Print() << "u_chev\t" << u_chev[0] << "\nu_chev\t" << u_chev[1] << "\nu_chev\t" 
            << u_chev[2]  << "\n";
    Print() << "TG_para\t" << TG_para[0] << "\nTG_para\t" << TG_para[1] << "\nTG_para\t" 
            << TG_para[2] << "\n";
    Print() << "TG_perp\t" << TG_perp[0] << "\nTG_perp\t" << TG_perp[1] << "\nTG_perp\t" 
            << TG_perp[2] << "\n";
    Print() << "TG_chev\t" << TG_chev[0] << "\nTG_chev\t" << TG_chev[1] << "\nTG_chev\t" 
            << TG_chev[2] << "\n";
    */
    if (true && GD::verbose>1) {
      Print() << "qVector\t" << q_flux[0] << "\nqVector\t" << q_flux[1] 
              << "\nqVector\t" << q_flux[2] << "\n";

      Print() <<  "ViscTens\t" << ViscTens[0] << "\nviscTens\t" << ViscTens[1] 
              << "\nviscTens\t" << ViscTens[2] << "\nviscTens\t" <<ViscTens[3] 
              << "\nviscTens\t" << ViscTens[4] << "\nviscTens\t" << ViscTens[5] << "\n";
      }
    }
    
    return ;
}


// ====================================================================================
void HydroState::write_info(nlohmann::json& js) const
{

    State::write_info(js);

    js["mass"] = mass;
    js["charge"] = charge;
    js["gamma"] = gamma;

    if (viscous) {

        auto& grp = js["viscosity"];

        int tp = viscous->get_type();
        grp["type"] = tp;

        const auto coeffs = viscous->get_refs();

        for (const auto& cf : coeffs) {
            grp[cf.first] = cf.second;
        }
    }
}

std::string HydroState::str() const
{
    std::stringstream msg;

    msg << State::str();

    if (viscous) {

        msg << "    viscosity : " << viscous->str() << "\n";

    }

    return msg.str();
}

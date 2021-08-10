#include "MFP_viscous.H"

#include <cmath>

#include "MFP_global.H"
#include "MFP_source.H"
#include "MFP_braginskii.H"

using GD = GlobalData;

Viscous::Viscous(){}
Viscous::~Viscous(){}

int Viscous::get_type(){return -1;}
int Viscous::get_BraginskiiIdentity(){return -1;} 
int Viscous::get_num(){return 0;}

void Viscous::get_neutral_coeffs(const Vector<Real> &Q, Real &T, Real &mu, 
                                 Real &kappa){return;}
//Braginskii Transport coefficients 
void Viscous::get_ion_coeffs(State& EMstate,State& ELEstate,const Vector<Real>& Q_i,
                             const Vector<Real>& Q_e,const Vector<Real>& B_xyz,Real& T_i,
                             Real& eta0,Real& eta1,Real& eta2,Real& eta3,Real& eta4,
                             Real& kappa1,Real& kappa2,Real& kappa3, int& truncatedTau){return;}
void Viscous::get_electron_coeffs(State& EMstate,State& IONstate,
                                  const Vector<Real>& Q_i,const Vector<Real>& Q_e,
                                  const Vector<Real>& B_xyz,Real& T_e,Real& eta0,
                                  Real& eta1,Real& eta2,Real& eta3,Real& eta4,
                                  Real& kappa1,Real& kappa2,Real& kappa3,
                                  Real& beta1,Real& beta2,Real& beta3, int& truncatedTau){return;}

Real Viscous::get_max_speed(const Vector<Vector<Real>> &U){return 0.0;}

Real Viscous::get_coulomb_logarithm(const Real& T_i,const Real& T_e,const Real& nd_e){
    // Coulomb logairthm as reported in BRaginskii OG paper.
    // Alternative is to use the formulation from
    // "Ionic transport in high-energy-density matter" Stanton 2016
    // Better function in from Li and Livescue 2018
    return 10.;
    Real T_ref = GD::T_ref;
    Real n_ref = GD::n_ref;
}

void Viscous::update_linked_states()
{
    State& istate = GD::get_state(idx);
    istate.set_num_grow(2);
}

PhysicsFactory<Viscous>& GetViscousFactory() {
    static PhysicsFactory<Viscous> F;
    return F;
}

std::string Viscous::str() const
{
    std::stringstream msg;

    const auto coeffs = get_refs();

    msg << get_tag() << "(";

    for (const auto& cf : coeffs) {
        msg << cf.first << "=" << cf.second << ", ";
    }

    msg.seekp(-2,msg.cur);
    msg << ")";

    return msg.str();
}

// ====================================================================================

Sutherland::Sutherland(){}
Sutherland::~Sutherland(){}

std::string Sutherland::tag = "Sutherland";
bool Sutherland::registered = GetViscousFactory().Register(Sutherland::tag, ViscousBuilder<Sutherland>);

Sutherland::Sutherland(const int global_idx, const sol::table& def)
{

    // check valid state
    if (!valid_state(global_idx))
        Abort("Sutherland viscosity is not valid for "+GD::get_state(global_idx).name);

    Real mu_ref = def["mu0"];
    Real T_ref = def["T0"];
    Real S_ref = def["S"];
    Prandtl = def["Pr"];
    cfl = def.get_or("cfl",1.0);

    idx = global_idx;

    Real r_r = GD::rho_ref;
    Real u_r = GD::u_ref;
    Real x_r = GD::x_ref;
    Real T_r = GD::T_ref;

    mu_0 = mu_ref/(r_r*x_r*u_r);
    T0 = T_ref/T_r;
    S = S_ref/T_r;

    if ((mu_0 <= 0) || (T0 <= 0) || (S <= 0)) {
        amrex::Abort("Sutherland input coefficients non-physical");
    }
}

int Sutherland::get_type(){return Neutral;}
int Sutherland::get_BraginskiiIdentity(){return 0;} 
int Sutherland::get_num(){return NUM_NEUTRAL_DIFF_COEFFS;}

void Sutherland::get_neutral_coeffs(const Vector<Real> &Q, Real &T, Real &mu, Real &kappa)
{
    BL_PROFILE("Sutherland::get_neutral_coeffs");
    State &istate = GD::get_state(idx);

    T = istate.get_temperature_from_prim(Q);
    Real alpha = istate.get_alpha_from_prim(Q);
    Real cp = istate.get_cp(alpha);

    Real T_ = T/T0;
    mu = mu_0*T_*sqrt(T_)*(T0+S)/(T+S);
    kappa = mu*cp/Prandtl;

    return;
}


Real Sutherland::get_max_speed(const Vector<Vector<amrex::Real> > &U)
{
    BL_PROFILE("Sutherland::get_max_speed");
    State &istate = GD::get_state(idx);

    Real rho = istate.get_density_from_cons(U[0]);
    Real T = istate.get_temperature_from_cons(U[0]);
    Real gamma = istate.get_gamma(U[0]);


    Real T_ = T/T0;
    Real mu = mu_0*T_*sqrt(T_)*(T0+S)/(T+S);

    return (4*mu*gamma/(Prandtl*rho))/cfl;
}

bool Sutherland::valid_state(const int idx)
{
    int s = GD::get_state(idx).get_type();

    if (s != +StateType::isHydro) {
        return false;
    }
    return true;
}

// ====================================================================================

PowerLaw::PowerLaw(){}
PowerLaw::~PowerLaw(){}

std::string PowerLaw::tag = "PowerLaw";
bool PowerLaw::registered = GetViscousFactory().Register(PowerLaw::tag, ViscousBuilder<PowerLaw>);

int PowerLaw::get_type(){return Neutral;}
int PowerLaw::get_BraginskiiIdentity(){return 0;}
int PowerLaw::get_num(){return NUM_NEUTRAL_DIFF_COEFFS;}

PowerLaw::PowerLaw(const int global_idx, const sol::table& def)
{

    // check valid state
    if (!valid_state(global_idx))
        Abort("Power Law viscosity is not valid for "+GD::get_state(global_idx).name);

    Real mu_ref = def["mu0"];
    Real T_ref = def["T0"];
    n = def["n"];
    Prandtl = def["Pr"];
    cfl = def.get_or("cfl",1.0);

    idx = global_idx;

    Real r_r = GD::rho_ref;
    Real u_r = GD::u_ref;
    Real x_r = GD::x_ref;
    Real T_r = GD::T_ref;

    mu_0 = mu_ref/(r_r*x_r*u_r);
    T0 = T_ref/T_r;


    if ((mu_0 <= 0) || (T0 <= 0) || (n <= 0)) {
        amrex::Abort("Power Law input coefficients non-physical");
    }
}

void PowerLaw::get_neutral_coeffs(const Vector<Real> &Q, Real &T, Real &mu, Real &kappa)
{
    BL_PROFILE("PowerLaw::get_neutral_coeffs");
    State &istate = GD::get_state(idx);

    T = istate.get_temperature_from_prim(Q);
    Real alpha = istate.get_alpha_from_prim(Q);
    Real cp = istate.get_cp(alpha);


    mu = mu_0*pow(T/T0,n);
    kappa = mu*cp/Prandtl;

    return;
}

Real PowerLaw::get_max_speed(const Vector<Vector<amrex::Real> > &U)
{
    BL_PROFILE("PowerLaw::get_max_speed");
    State &istate = GD::get_state(idx);

    Real rho = istate.get_density_from_cons(U[0]);
    Real T = istate.get_temperature_from_cons(U[0]);
    Real mu = mu_0*pow(T/T0,n);
    Real gamma = istate.get_gamma(U[0]);

    return (4*mu*gamma/(Prandtl*rho))/cfl;
}

bool PowerLaw::valid_state(const int idx)
{
    int s = GD::get_state(idx).get_type();

    if (s != +StateType::isHydro) {
        return false;
    }
    return true;
}

// ====================================================================================
BraginskiiIon::BraginskiiIon(){}
BraginskiiIon::~BraginskiiIon(){}

std::string BraginskiiIon::tag = "BraginskiiIon";
bool BraginskiiIon::registered = GetViscousFactory().Register(BraginskiiIon::tag, ViscousBuilder<BraginskiiIon>);

BraginskiiIon::BraginskiiIon(const int global_idx, const sol::table& def)
{
    // check valid state
    if (!valid_state(global_idx))
        Abort("BraginskiiIon viscosity is not valid for "+GD::get_state(global_idx).name);

    idx = global_idx;

    cfl = def.get_or("cfl",1.0); 
    //Print() << "\nln 253 - cfl viscous ion: " << cfl << "\n"; 
}

void BraginskiiIon::update_linked_states()
{
    // go through the attached sources and find if there is a Braginskii source term to support this object
    linked_em = -1;
    linked_electron = -1;
    State& istate = GD::get_state(idx);

    if (istate.associated_state.at(AssociatedType::Field).size() == 1) {
        linked_em = istate.associated_state[AssociatedType::Field][0];
    }

    if (istate.associated_state.at(AssociatedType::Electron).size() == 1) {
        linked_electron = istate.associated_state[AssociatedType::Electron][0];
    }

    if (linked_electron < 0)
        Abort("BraginskiiIon linking to electron state failed");

    if (linked_em < 0)
        Abort("BraginskiiIon linking to field state failed");

    // make sure that the this and attached states have enough ghost cells for the calulation of slopes
    for (int linked_idx : {idx, linked_electron, linked_em}) {
        State& linked_state = GD::get_state(linked_idx);
        linked_state.set_num_grow(2);
    }

}

int BraginskiiIon::get_type(){return Ion;}
int BraginskiiIon::get_BraginskiiIdentity(){return 1;} 
int BraginskiiIon::get_num(){return NUM_ION_DIFF_COEFFS;}

void BraginskiiIon::get_ion_coeffs(State& EMstate,State& ELEstate,
                                   const Vector<Real>& Q_i,const Vector<Real>& Q_e,
                                   const Vector<Real>& B_xyz,Real& T_i,Real& eta0,
                                   Real& eta1,Real& eta2,Real& eta3,Real& eta4,
                                   Real& kappa1,Real& kappa2,Real& kappa3, int& truncatedTau){
    BL_PROFILE("BraginskiiIon::get_ion_coeffs");

    truncatedTau = 0;
    Real mass_i,mass_e,charge_i,charge_e,T_e,nd_i,nd_e,alpha_e,alpha_i;
    State &istate = GD::get_state(idx);

    //Extract and assign parameters from Q_i and Q_e
    //--- electron state and properties required for calcs -------Note move this
    alpha_e = ELEstate.get_alpha_from_prim(Q_e);
    charge_e= ELEstate.get_charge(Q_e); // electron propertis
    mass_e  = ELEstate.get_mass(Q_e);
    T_e     = ELEstate.get_temperature_from_prim(Q_e);
    nd_e    = Q_e[+HydroState::ConsIdx::Density]/mass_e;
    //--- ion state and properties required for calcs
    alpha_i = istate.get_alpha_from_prim(Q_i);
    charge_i= istate.get_charge(Q_i);
    mass_i  = istate.get_mass(Q_i);
    T_i     = istate.get_temperature_from_prim(Q_i); //
    nd_i    = Q_i[+HydroState::PrimIdx::Density]/mass_i;
    //Magnetic field
    Real Bx=B_xyz[0], By=B_xyz[1], Bz=B_xyz[2];

    // See page 215 (document numbering) of Braginskii's original transport paper
    Real t_collision_ion, lambda_i, p_lambda, omega_ci, omega_p;
    p_lambda = get_coulomb_logarithm(T_i,T_e,nd_e);
    // absence of the boltzmann constant due to usage of nondimensional variables.
    // Note that here charge_i = e^4*Z^4

    Real Debye = GD::Debye, Larmor = GD::Larmor;

    Real x_ref=GD::x_ref, n0_ref=GD::n0, m_ref=GD::m_ref, rho_ref=GD::rho_ref,
            T_ref=GD::T_ref, u_ref=GD::u_ref;

    Real t_ref = x_ref/u_ref;
    t_collision_ion = std::pow(Debye,4)*n0_ref
                      *(12*std::sqrt(mass_i)*std::pow(3.14159*T_i, 3./2.)) /
                      (p_lambda * std::pow(charge_i,4) * nd_i);

    omega_ci = charge_i * std::sqrt( Bx*Bx + By*By + Bz*Bz ) / mass_i / Larmor;
    omega_p  = std::sqrt(nd_i*charge_i*charge_i/mass_i/Debye/Debye) ;

    if (1/t_collision_ion < GD::effective_zero) {
        if (false && GD::verbose > 1) {
            Print() << "Less than effective zero\t1/tau_i = " << 1/t_collision_ion << "\tomega_ci = " 
                    << omega_ci << "\tomega_p = " << omega_p << "\n"; 
        }

        t_collision_ion = 1/GD::effective_zero;

        if (true && GD::verbose > 1) Print() << "\t1/tau_i\t" << 1/t_collision_ion << "\n";
    }

    if (GD::srin_switch && (1/t_collision_ion < omega_ci/10/2/3.14159) && (1/t_collision_ion < omega_p/10/2/3.14159)) {
        if (GD::verbose >= 2) {
        //   Print() << "\nMFP_viscous.cpp ln 334 --- Ion collision frequency limited to minimum of plasma and cyclotron frequency\n";
            Print() << "1/tau_i = " << 1/t_collision_ion << "\tomega_ci = "  
                  << omega_ci << "\tomega_p = " << omega_p << "\n"; 
        }
        t_collision_ion = 1/std::min(omega_ci/2/3.14159, omega_p/2/3.14159); truncatedTau = 1; 
        if (GD::verbose > 1) Print() << "1/tau_i = " << 1/t_collision_ion << "\n";
    } 

    // coefficients used exclusively in the braginskii transport
    Real delta_kappa, delta_eta, delta_eta2, x_coef ;

    x_coef = omega_ci*t_collision_ion;

    //TODO fix up coefficients here also with tabled depending atomic number

    delta_kappa = x_coef*x_coef*x_coef*x_coef + 2.700*x_coef*x_coef + 0.677;
    delta_eta   = x_coef*x_coef*x_coef*x_coef + 4.030*x_coef*x_coef + 2.330;
    delta_eta2  = 16*x_coef*x_coef*x_coef*x_coef + 4*4.030*x_coef*x_coef + 2.330;

    eta0 = 0.96*nd_i*T_i*t_collision_ion ;//* n0_ref;
    if (GD::braginskii_anisotropic) {
      eta2 = nd_i*T_i*t_collision_ion*(6./5.*x_coef*x_coef+2.23)/delta_eta;
      eta1 = nd_i*T_i*t_collision_ion*(6./5.*(2*x_coef)*(2*x_coef)+2.23)/delta_eta2;
      eta4 = nd_i*T_i*t_collision_ion*x_coef*(x_coef*x_coef + 2.38)/delta_eta;
      eta3 = nd_i*T_i*t_collision_ion*(2*x_coef)*((2*x_coef)*(2*x_coef) + 2.38)/delta_eta2;
      if (x_coef < 1e-8 ) {
        eta2 = eta2/x_coef/x_coef;
        eta1 = eta1/x_coef/x_coef;
        eta4 = eta4/x_coef;
        eta3 = eta3/x_coef; }
    } else {
      eta2 = 0;
      eta1 = 0;
      eta4 = 0;
      eta3 = 0; }

    if (GD::verbose >= 9 ) {
        Print() << "\nIon 5 dynamic viscosity coefficients\t" << eta0 <<"\t" << eta1 
                <<"\t" << eta2 <<"\t" << eta3 <<"\t" << eta4 << "\n"; 
    }
    //
    if ((eta0<0) || (eta1<0) || (eta2<0) || (eta3<0) || (eta4<0)) {
        if (eta0<=0) {
          Print() << "\neta0 = " << eta0 << "\n";
        } else if (eta1<=0) {
          Print() << "\neta1 = " << eta1 << "\n";
        } else if (eta2<=0) {
          Print() << "\neta2 = " << eta2 << "\n";
        } else if (eta3<=0) {
          Print() << "\neta3 = " << eta3 << "\n";
        } else if (eta4<=0) {
          Print() << "\neta4 = " << eta4 << "\n";
        }
        amrex::Abort("mfp_viscous.cpp ln: 334 - Braginski Ion coefficients are non-physical");
        //Print() << "\nmfp_viscous.cpp ln: 273 - Braginski Ion coefficients are non-physical\n";
    }//
    
    //From Braginskii OG paper page 250 of paper in journal heading 4
    // Kinetics of a simple plasma (Quantitative Analyis)
    // TODO Add in flexibility for different atomic numbers of the ion species used,
    // see Braginskii

    //TODO Print() << "\nget_ion_coefficients - Test if 1/no_ref on Diverence(q) and n0_ref of Kappa makes a difference (they cancel out overall but perhaps the numerics?\n";
    kappa1 = 3.906*nd_i*T_i*t_collision_ion/mass_i;
    if (GD::braginskii_anisotropic) {
      kappa2 = (2.*x_coef*x_coef + 2.645)/delta_kappa*nd_i*T_i*t_collision_ion/mass_i;
      kappa3 = (5./2.*x_coef*x_coef + 4.65)*x_coef*nd_i*T_i*t_collision_ion/mass_i/delta_kappa;
    } else {kappa2 = 0; kappa3 = 0; }

    //TODO remove after comparison to plasmapy
    if (false && GD::verbose >= 1) {
        Print() << "\n\nIon viscous coefficients and thermal conductivity coefficients";
        Print() << "\nnd_i = " << nd_i << "\tT_i = " << T_i << "\tt_i = " << t_collision_ion ;
        Print() << "\neta0 = " << eta0 ;
        Print() << "\neta1 = " << eta1 ;
        Print() << "\neta2 = " << eta2 ;
        Print() << "\neta3 = " << eta3 ;
        Print() << "\neta4 = " << eta4 ;
        Print() << "\nkappa_para\t" << kappa1*n0_ref << "\nkappa_perp\t" << kappa2*n0_ref << "\nkappa_chev\t" << kappa3*n0_ref ;
    }

    if ((kappa1<0) || (kappa2<0) || (kappa3 < 0)) {
        amrex::Abort("mfp_viscous.cpp ln: 350 - Braginski Ion coefficients are non-physical");
    }

    if (kappa1 < kappa2) {
        if (GD::verbose >= 4 ) {
            Print() << "\nmfp_viscous.cpp ln: 401 - ion kappa2 exceed kappp1\n";
        }
        //Print() << "\nmfp_viscous.cpp ln: 285 - Braginski Ion coefficients are non-physical\n";
        kappa2 = kappa1;
    }
    if (kappa1 < kappa3) {
        if (GD::verbose >= 4 ) {
            Print() << "\nmfp_viscous.cpp ln: 404 - ion kappa3 exceed kappp1\n";
        }
        kappa3 = kappa1;
    }

    //Print() << "\nln 440 - cfl viscous ion: " << cfl << "\n";
    return;
}

Real BraginskiiIon::get_max_speed(const Vector<Vector<amrex::Real>>&U) {
    BL_PROFILE("BraginskiiIon::get_max_speed");

    State &ELEstate = GD::get_state(linked_electron);
    State &EMstate = GD::get_state(linked_em);
    State &IONstate = GD::get_state(idx);

    const Vector<Real>& U_i = U[0]; //TODO check the indexing of incoming U as its not according to anythng sensible 
    const Vector<Real>& U_e = U[1];
    const Vector<Real>& U_em = U[2];


    //Print() << "\nIon getMaxSpeed\n" << idx << "\n" << linked_em << "\n" << linked_electron << "\n";
    //Print() << "\n" << U_e[0] << "\n" << U_i[0] << "\n" << U_em[+FieldState::ConsIdx::Bx] << "\n"; 
    //---Calculate the coefficients from scratch
    Real mass_i,mass_e,charge_i,charge_e,T_e,T_i,nd_i,nd_e,alpha_i,alpha_e;
    Real eta0, eta1, eta2, eta3, eta4, kappa1, kappa2, kappa3;


    //Extract and assign parameters from Q_i and Q_e
    //--- electron state and properties required for calcs -------Note move this
    alpha_e = ELEstate.get_alpha_from_cons(U_e);
    charge_e= ELEstate.get_charge(U_e); // electron propertis
    mass_e  = ELEstate.get_mass(U_e);
    //Print() << "\nmass_e" << mass_e << "\n";
    T_e     = ELEstate.get_temperature_from_cons(U_e);
    nd_e    = ELEstate.get_density_from_cons(U_e)/mass_e;

    //--- ion state and properties required for calcs
    alpha_i = IONstate.get_alpha_from_cons(U_i);
    charge_i= IONstate.get_charge(U_i);
    mass_i  = IONstate.get_mass(U_i);
    //Print() << "\nmass_i" << mass_i << "\n";
    T_i     = IONstate.get_temperature_from_cons(U_i); //
    nd_i    = IONstate.get_density_from_cons(U_i)/mass_i;

    //Magnetic field
    Real Bx = U_em[+FieldState::ConsIdx::Bx];
    Real By = U_em[+FieldState::ConsIdx::By];
    Real Bz = U_em[+FieldState::ConsIdx::Bz];


    // See page 215 (document numbering) of Braginskii's original transport paper
    Real t_collision_ion, lambda_i,p_lambda, omega_ci, omega_p;
    p_lambda = get_coulomb_logarithm(T_i,T_e,nd_e);
    // absence of the boltzmann constant due to usage of nondimensional variables.
    // Note that here charge_i = e^4*Z^4

    Real Debye = GD::Debye, Larmor = GD::Larmor, x_ref=GD::x_ref, n0_ref=GD::n0,
            m_ref=GD::m_ref, rho_ref=GD::rho_ref, T_ref=GD::T_ref, u_ref=GD::u_ref;

    Real t_ref = x_ref/u_ref;
    t_collision_ion = std::pow(Debye,4)*n0_ref
                      *(12*std::sqrt(mass_i)*std::pow(3.14159*T_i, 3./2.)) /
                      (p_lambda * std::pow(charge_i,4) * nd_i);

    omega_ci = charge_i * std::sqrt( Bx*Bx + By*By + Bz*Bz ) / mass_i / Larmor;
    omega_p  = std::sqrt(nd_i*charge_i*charge_i/mass_i/Debye/Debye) ;

    if (1/t_collision_ion < GD::effective_zero) t_collision_ion = 1/GD::effective_zero;

    if (GD::srin_switch && (1/t_collision_ion < omega_ci/10/2/3.14159) && (1/t_collision_ion < omega_p/10/2/3.14159)) {
        if (GD::verbose >= 2) {
        //    Print() << "\nMFP_viscous.cpp ln 334 --- Ion collision frequency limited to minimum of plasma and cyclotron frequency\n";
            Print() << "1/tau_i = " << 1/t_collision_ion << "\tomega_ci = "  
                  << omega_ci << "\tomega_p = " << omega_p << "\n"; 
        }
        t_collision_ion = 1/std::min(omega_ci/2/3.14159, omega_p/2/3.14159) ;
        // else {
        //    t_collision_ion = 1/(1/t_collision_ion + GD::effective_zero);
        //}
        //Print()<< "\t" << t_collision_ion;
    } 

    //-- simple plasma with magnetic field
    Real delta_kappa, delta_eta, delta_eta2, x_coef ;

    x_coef = omega_ci*t_collision_ion;

    //TODO fix up coefficients here also with tabled depending atomic number
    delta_kappa = x_coef*x_coef*x_coef*x_coef + 2.700*x_coef*x_coef + 0.677;
    delta_eta   = x_coef*x_coef*x_coef*x_coef + 4.030*x_coef*x_coef + 2.330;
    delta_eta2  = 16*x_coef*x_coef*x_coef*x_coef + 4*4.030*x_coef*x_coef + 2.330;

    eta0 = 0.96*nd_i*T_i*t_collision_ion ;//* n0_ref;
    if (GD::braginskii_anisotropic) {
      eta2 = nd_i*T_i*t_collision_ion*(6./5.*x_coef*x_coef+2.23)/delta_eta;
      eta1 = nd_i*T_i*t_collision_ion*(6./5.*(2*x_coef)*(2*x_coef)+2.23)/delta_eta2;
      eta4 = nd_i*T_i*t_collision_ion*x_coef*(x_coef*x_coef + 2.38)/delta_eta;
      eta3 = nd_i*T_i*t_collision_ion*(2*x_coef)*((2*x_coef)*(2*x_coef) + 2.38)/delta_eta2;
      if (x_coef < 1e-8 ) {
        eta2 = eta2/x_coef/x_coef;
        eta1 = eta1/x_coef/x_coef;
        eta4 = eta4/x_coef;
        eta3 = eta3/x_coef; }
    } else {
      eta2 = 0;
      eta1 = 0;
      eta4 = 0;
      eta3 = 0; }

    if ((eta0<0) || (eta1<0) || (eta2<0) || (eta3<0) || (eta4<0)) {
        Print () << "\nNon-physical eta\n" << eta0;
        Print () << "\n" << eta1;
        Print () << "\n" << eta2;
        Print () << "\n" << eta3;
        Print () << "\n" << eta4;
        amrex::Abort("mfp_viscous.cpp ln: 347 - Braginski Ion coefficients are non-physical");
        Print() << "\nmfp_viscous.cpp ln: 347 - Braginski Ion coefficients are non-physical\n";
    }

    //Srinivasan recomendatiosn
    
    if (std::abs(eta0) < std::abs(eta1)) {
        Print() << "\nion viscous coefficient eta1 greater than eta0\n";
        Print() << "\n" + std::to_string(eta0) + "\n";
        Print() << "\n" + std::to_string(eta1) + "\n";
    } else if (std::abs(eta0) < std::abs(eta2)) {
        Print() << "\nion viscous coefficient eta2 greater than eta0\n";
        Print() << "\n" + std::to_string(eta0) + "\n";
        Print() << "\n" + std::to_string(eta2) + "\n";
    } else if (std::abs(eta0) < std::abs(eta3)) {
        Print() << "\nion viscous coefficient eta3 greater than eta0\n";
        Print() << "\n" + std::to_string(eta0) + "\n";
        Print() << "\n" + std::to_string(eta3) + "\n";
    } else if (std::abs(eta0) < std::abs(eta4)) {
        Print() << "\nion viscous coefficient eta4 greater than eta0\n";
        Print() << "\n" + std::to_string(eta0) + "\n";
        Print() << "\n" + std::to_string(eta4) + "\n";
    }
    
    //From Braginskii OG paper page 250 of paper in journal heading 4
    // Kinetics of a simple plasma (Quantitative Analyis)
    // TODO Add in flexibility for different atomic numbers of the ion species used,
    // see Braginskii
    kappa1 = 3.906*nd_i*T_i*t_collision_ion/mass_i;
    if (GD::braginskii_anisotropic) {
      kappa2 = (2.*x_coef*x_coef + 2.645)/delta_kappa*nd_i*T_i*t_collision_ion/mass_i;
      kappa3 = (5./2.*x_coef*x_coef + 4.65)*x_coef*nd_i*T_i*t_collision_ion/mass_i/delta_kappa;
    } else {kappa2 = 0; kappa3 = 0; }

    if (GD::verbose >= 9 ) {
        Print() << "\nIon heat conductivity coefficients\n"
                << kappa1 << "\n" << kappa2 << "\n" << kappa3 ;
    }

    if ((kappa1<0) || (kappa2<0) || (kappa3 < 0)) {
        amrex::Abort("mfp_viscous.cpp ln: 350 - Braginski Ion coefficients are non-physical");
    }

    if (kappa1 < kappa2) {
        if (GD::verbose >= 4 ) {
            Print() << "\nmfp_viscous.cpp ln: 401 - ion kappa2 exceed kappp1\n";
        }
        //Print() << "\nmfp_viscous.cpp ln: 285 - Braginski Ion coefficients are non-physical\n";
        kappa2 = kappa1;
    }
    if (kappa1 < kappa3) {
        if (GD::verbose >= 4 ) {
            Print() << "\nmfp_viscous.cpp ln: 404 - ion kappa3 exceed kappp1\n";
        }
        kappa3 = kappa1;
    }

    //TODO remove after debugging    
    if (GD::verbose >= 3 ) {
        if (omega_ci==0) {
            Print() << "\nomega_ci = " << std::to_string(omega_ci) + "\n";
            Print() << "\nZero magnetic field?? MFP_viscous.cpp\n";
        }
    }

    Real rho = IONstate.get_density_from_cons(U_i);
    Real cp_ion = IONstate.get_cp(alpha_i);
    Real gamma_ion = IONstate.get_gamma(alpha_i);
    //Real Prandtl = cp_ion*eta0/kappa1;
    //Real nu_thermal = (4*eta0*gamma_ion/(Prandtl*rho))/cfl/n0_ref;
    Real nu_thermal = kappa1/rho/cp_ion/cfl; //thermal diffusivity 
    //Real nu_visc = (eta0/rho)/cfl/n0_ref;
    Real nu_visc = (eta0/rho)/cfl;
    if (GD::verbose>4) {
      Print() << "Debug and test Prandtl viscous time step ln 553 feature\nnu_visc\t" 
              << nu_visc << "\tnu_thermal\t" << nu_thermal << "\teta0\t" << eta0 << "\n";
    }
    //TODO delete 
    if (false) {
      Print() << "rho etc.\n" <<  rho << "\n";
      Print() << cp_ion<< "\n";
      Print() << gamma_ion<< "\n";
      Print() << nu_thermal << "\n";
      Print() << nu_visc << "\n";
    }
    Real nu; 
    if (nu_thermal> nu_visc) {  
      nu = nu_thermal ;
    } else if (nu_thermal <= nu_visc) {
      nu = nu_visc ;
    }

    //Print() << "\nln 622 - cfl viscous ion: " << cfl;
    return 2*nu;
    //return t_collision_ion ;
    //return (eta0/rho)/cfl;
    //return std::max(eta0, std::max(eta1, std::max(eta2, std::max(eta3,eta4))))/rho;
}


bool BraginskiiIon::valid_state(const int idx)
{
    int s = GD::get_state(idx).get_type();

    if (s == +StateType::isField) {
        return false;
    }
    return true;
}

// ====================================================================================
BraginskiiEle::BraginskiiEle(){}
BraginskiiEle::~BraginskiiEle(){}

std::string BraginskiiEle::tag = "BraginskiiEle";
bool BraginskiiEle::registered = GetViscousFactory().Register(BraginskiiEle::tag, ViscousBuilder<BraginskiiEle>);

//BraginskiiEle::BraginskiiEle(const Real mu_ref, const Real T_ref, const int BT, const int i)
BraginskiiEle::BraginskiiEle(const int global_idx, const sol::table& def)
{
    // check valid state
    if (!valid_state(global_idx))
        Abort("BraginskiiEle viscosity is not valid for "+GD::get_state(global_idx).name);

    idx = global_idx;

    cfl = def.get_or("cfl",1.0);
    //Print() << "\nln 656 - cfl viscous ele: " << cfl << "\n";
}

void BraginskiiEle::update_linked_states()
{
    linked_ion = -1;
    linked_em = -1;

    // go through the attached sources and find if there are attached fields and ions
    State& istate = GD::get_state(idx);

    if (istate.associated_state.at(AssociatedType::Field).size() == 1) {
        linked_em = istate.associated_state[AssociatedType::Field][0];
    }

    if (istate.associated_state.at(AssociatedType::Ion).size() == 1) {
        linked_ion = istate.associated_state[AssociatedType::Ion][0];
    }

    if (linked_ion < 0)
        Abort("BraginskiiIon linking to electron state failed");

    if (linked_em < 0)
        Abort("BraginskiiIon linking to field state failed");

    // make sure that the this and attached states have enough ghost cells for the calulation of slopes
    for (int linked_idx : {idx, linked_ion, linked_em}) {
        State& linked_state = GD::get_state(linked_idx);
        linked_state.set_num_grow(2);
    }

}

int BraginskiiEle::get_type(){return Electron;}
int BraginskiiEle::get_BraginskiiIdentity(){return 2;} 
int BraginskiiEle::get_num(){return NUM_ELE_DIFF_COEFFS;}

void BraginskiiEle::get_electron_coeffs(State& EMstate,State& IONstate,
                                        const Vector<Real>& Q_i,const Vector<Real>& Q_e,
                                        const Vector<Real>& B_xyz,Real& T_e,Real& eta0,
                                        Real& eta1,Real& eta2,Real& eta3,Real& eta4,
                                        Real& kappa1,Real& kappa2,Real& kappa3,
                                        Real& beta1,Real& beta2,Real& beta3, int& truncatedTau){
    BL_PROFILE("BraginskiiEle::get_electron_coeffs");

    truncatedTau = 0;
    Real mass_i,mass_e,charge_i,charge_e,T_i,nd_i,nd_e,alpha_e,alpha_i;
    State &istate = GD::get_state(idx);

    //Extract and assign parameters from Q_i and Q_e
    //--- electron state and properties required for calcs -------Note move this
    alpha_i = IONstate.get_alpha_from_prim(Q_i);
    charge_i= IONstate.get_charge(Q_i); // electron propertis
    mass_i  = IONstate.get_mass(Q_i);
    T_i     = IONstate.get_temperature_from_prim(Q_i);
    nd_i    = Q_i[+HydroState::PrimIdx::Density]/mass_i;
    //--- ion state and properties required for calcs
    alpha_e = istate.get_alpha_from_prim(Q_e);
    charge_e= istate.get_charge(Q_e);
    mass_e  = istate.get_mass(Q_e);
    T_e     = istate.get_temperature_from_prim(Q_e); //
    nd_e    = Q_e[+HydroState::PrimIdx::Density]/mass_e;
    //Magnetic field
    Real Bx=B_xyz[0], By=B_xyz[1], Bz=B_xyz[2];

    // See page 215 (document numbering) of Braginskii's original transport paper
    Real t_collision_ele, lambda_e, p_lambda, omega_ce, omega_p;
    p_lambda = get_coulomb_logarithm(T_i,T_e,nd_e);

    Real Debye = GD::Debye, Larmor = GD::Larmor, x_ref=GD::x_ref, n0_ref=GD::n0,
            m_ref=GD::m_ref, rho_ref=GD::rho_ref, T_ref=GD::T_ref, u_ref=GD::u_ref;
    Real t_ref = x_ref/u_ref;

    //t_collision_ele = std::pow(Debye,4)*n0_ref
    //                  *(6*std::sqrt(2*mass_e)*std::pow(3.14159*T_e, 3./2.)) /
    //                  (p_lambda*std::pow(charge_e,4)*(charge_i/-charge_e)*nd_e);
    //Print() << "Note the changed t_collision_e_i - kn 678";
    t_collision_ele = std::pow(Debye,4)*n0_ref
                      *(6*std::sqrt(2*mass_e)*std::pow(3.14159*T_e, 3./2.)) /
                      (p_lambda*std::pow((charge_i/-charge_e), 2)*nd_i);

    omega_ce = -charge_e * std::sqrt( Bx*Bx + By*By + Bz*Bz ) / mass_e/ Larmor;
//    Print() << charge_e << "\t" << Bx << "\t"<< By << "\t"<< Bz << "\t"<< mass_e << "\t"<< Larmor << "\n";
    omega_p  = std::sqrt(nd_e*charge_e*charge_e/mass_e/Debye/Debye) ;

    if (1/t_collision_ele < GD::effective_zero) {
        if (false and GD::verbose > 1) {
            Print() << "1/tau_e = " << 1/t_collision_ele << "\tomega_ce = "
                    << omega_ce << "\tomega_p = " << omega_p << "\n";
        }
        t_collision_ele = 1/GD::effective_zero;

        if (true and GD::verbose > 1) Print() << "\t1/tau_e\t" << 1/t_collision_ele << "\n";
    }

    if (GD::srin_switch && (1/t_collision_ele < omega_ce/10/2/3.14159) && (1/t_collision_ele < omega_p/10/2/3.14159)) {
        if (GD::verbose >= 2 ) { //TODO bebug the limiting function - just use a delta value for any potential divide by zeros
            //Print() << "\nMFP_viscous.cpp ln 658 --- Electron collision frequency limited to minimum of plasma and cyclotron frequency\n";
            Print() << "1/tau_e = " << 1/t_collision_ele << "\tomega_ce = "
                  << omega_ce << "\tomega_p = " << omega_p << "\n";
        }
        t_collision_ele = 1/std::min(omega_ce/2/3.14159, omega_p/2/3.14159); truncatedTau = 1;  
        if (GD::verbose > 1 ) Print() << "1/tau_e = " << 1/t_collision_ele << "\n" ;
    }    

    Real delta_kappa, delta_eta, delta_eta2, x_coef;// coefficients used exclusively in the braginskii
    x_coef = omega_ce*t_collision_ele; 
    // TODO fix up these table 2 page 251 BT
    Real delta_0=3.7703, delta_1=14.79;
    delta_kappa= x_coef*x_coef*x_coef*x_coef+delta_1*x_coef*x_coef + delta_0;
    delta_eta  = x_coef*x_coef*x_coef*x_coef+13.8*x_coef*x_coef + 11.6;
    delta_eta2 = 16*x_coef*x_coef*x_coef*x_coef+4*13.8*x_coef*x_coef + 11.6;

    //Print() << "nd_e  = " << nd_e << "\tT_e = " << T_e << "\tt_collision_ele = " 
    //        << t_collision_ele << "\n";

    eta0 = 0.733*nd_e *T_e * t_collision_ele;
    if (GD::braginskii_anisotropic) {
      eta2 = nd_e *T_e*t_collision_ele*(2.05*x_coef*x_coef+8.5)/delta_eta;
      eta1 = nd_e *T_e*t_collision_ele*(2.05*(2*x_coef)*(2*x_coef)+8.5)/delta_eta2;
      eta4 = -nd_e*T_e*t_collision_ele*x_coef*(x_coef*x_coef+7.91)/delta_eta;
      eta3 = -nd_e*T_e*t_collision_ele*(2*x_coef)*((2*x_coef)*(2*x_coef)+7.91)/delta_eta2;
      if (x_coef < 1e-8 ) {
        eta2 = eta2/x_coef/x_coef;
        eta1 = eta1/x_coef/x_coef;
        eta4 = eta4/x_coef;
        eta3 = eta3/x_coef; }
    } else {
      eta2 = 0;
      eta1 = 0;
      eta4 = 0;
      eta3 = 0; }

    if (GD::verbose >= 9 ) {
        Print() << "\nElectron 5 dynamic viscosity coefficients\t" <<  eta0 << "\t" 
                <<  eta1 << "\t" <<  eta2 << "\t" <<  eta3 << "\t" <<  eta4 << "\n"; 
    }
    //maybe benchmark to see if pow should be used idunnobruda

    //From Braginskii OG paper page 250 of paper in journal heading 4
    // Kinetics of a simple plasma (Quantitative Analyis)
    //TODO change coefficient values for different Z values
    // Currently set for a hydrogen  plasma
    Real BT_gamma_0=11.92/3.7703,BT_gamma_0_p=11.92,BT_gamma_1_p=4.664,BT_gamma_1_pp=5./2.;
    Real BT_gamma_0_pp=21.67;

    kappa1=nd_e*T_e*t_collision_ele/mass_e*BT_gamma_0;
    if (GD::braginskii_anisotropic) {
      kappa2=(BT_gamma_1_p*x_coef*x_coef+BT_gamma_0_p)/delta_kappa*nd_e*T_e*t_collision_ele/mass_e;
      kappa3=(BT_gamma_1_pp*x_coef*x_coef+BT_gamma_0_pp)*x_coef*nd_e*T_e*t_collision_ele
           /mass_e/delta_kappa;
    } else {kappa2 = 0; kappa3 = 0;}

    //TODO remove after comparison to plasmapy
    if (false && GD::verbose >= 1) {
        Print() << "\n\nElectron viscous coefficients and thermal conductivity coefficients";
        Print() << "\nnd_e = " << nd_e << "\tT_e = " << T_e << "\tt_e = " << t_collision_ele ;
        Print() << "\neta0 = " << eta0 ;
        Print() << "\neta1 = " << eta1 ;
        Print() << "\neta2 = " << eta2 ;
        Print() << "\neta3 = " << eta3 ;
        Print() << "\neta4 = " << eta4 ;
        Print() << "\nkappa_para\t" << kappa1*n0_ref << "\nkappa_perp\t" << kappa2*n0_ref << "\nkappa_chev\t" << kappa3*n0_ref ;
    }

    if ((kappa1<0.) || (kappa2<0.) || (kappa3 < 0.)) {
        //amrex::Abort("mfp_viscous.cpp ln: 673 - Braginski Ion coefficients are non-physical");
        Print() << "\nmfp_viscous.cpp ln: 673 - Braginski Ion coefficients are non-physical\n";
        Print() << "\n" << kappa1 << "\n" << kappa2 << "\n" << kappa3 << "\nomega_ce = " << omega_ce;
    }

    if (kappa1 < kappa2) {
        if (GD::verbose >= 4) {
            Print() << "mfp_viscous.cpp ln: 688 - electron kappa2 exceed kappp1";
        }
        kappa2 = kappa1;
    }
    if (kappa1 < kappa3) {
        if (GD::verbose >= 4) {
            Print() << "mfp_viscous.cpp ln: 694 - electron kappa3 exceed kappp1";
        }
        kappa3 = kappa1;
    }



    //--- beta terms for the thermal component of thermal heat flux of the electrons.

    Real b_0 = 0.711, b_0_pp = 3.053, b_0_p=2.681, b_1_p=5.101, b_1_pp=3./2.;
    beta1 = nd_e*b_0*T_e;
    if (GD::braginskii_anisotropic) {
    beta2 = nd_e*(b_1_p*x_coef*x_coef+b_0_p)/delta_kappa*T_e;
    beta3 = nd_e*x_coef*(b_1_pp*x_coef*x_coef+b_0_pp)/delta_kappa*T_e; 
    } else {beta2 = 0; beta3 = 0;}

    //Print() << "\nln 844 - cfl viscous ele: " << cfl << "\n";
    return;
}

Real BraginskiiEle::get_max_speed(const Vector<Vector<amrex::Real> > &U) {
    BL_PROFILE("BraginskiiEle::get_max_speed");

    State &ELEstate = GD::get_state(idx);
    State &EMstate = GD::get_state(linked_em);
    State &IONstate = GD::get_state(linked_ion);

    const Vector<Real>& U_e = U[0];
    const Vector<Real>& U_i = U[1];
    const Vector<Real>& U_em = U[2];

    //Print() << "\nEle getMaxSpeed\n" << idx << "\n" << linked_em << "\n" << linked_ion << "\n";
    //Print() << "\n" << U_e[0] << "\n" << U_i[0] << "\n" << U_em[+FieldState::ConsIdx::Bx] << "\n"; 
    //TODO Fix get_max_speed to be compatible with electrona and ion
    //---Calculate the coefficients from scratch
    Real mass_i,mass_e,charge_i,charge_e,T_e,T_i,nd_i,nd_e,alpha_i,alpha_e;


    //Real eta0, eta1, eta2, eta3, eta4, kappa1, kappa2, kappa3, beta1, beta2, beta3;
    //Extract and assign parameters from Q_i and Q_e

    //--- electron state and properties required for calcs -------Note move this
    alpha_e = ELEstate.get_alpha_from_cons(U_e);
    charge_e= ELEstate.get_charge(U_e); // electron propertis
    mass_e  = ELEstate.get_mass(U_e);

    //Print() << "\nmass_i" << mass_i << "\n";
    T_e     = ELEstate.get_temperature_from_cons(U_e);
    nd_e    = ELEstate.get_density_from_cons(U_e)/mass_e;

    //--- ion state and properties required for calcs
    alpha_i = IONstate.get_alpha_from_cons(U_i);
    charge_i= IONstate.get_charge(U_i);
    mass_i  = IONstate.get_mass(U_i);
    //Print() << "\nmass_i" << mass_i << "\n";
    T_i     = IONstate.get_temperature_from_cons(U_i); //
    nd_i    = IONstate.get_density_from_cons(U_i)/mass_i;

    //Magnetic field
    Real Bx = U_em[+FieldState::ConsIdx::Bx];
    Real By = U_em[+FieldState::ConsIdx::By];
    Real Bz = U_em[+FieldState::ConsIdx::Bz];

    Real t_collision_ele, lambda_e, p_lambda, omega_ce, omega_p;
    p_lambda = get_coulomb_logarithm(T_i,T_e,nd_e);

    //collision time nondimensional
    Real Debye = GD::Debye, Larmor = GD::Larmor, x_ref=GD::x_ref, n0_ref=GD::n0,
            m_ref=GD::m_ref, rho_ref=GD::rho_ref, T_ref=GD::T_ref, u_ref=GD::u_ref;
    Real t_ref = x_ref/u_ref;

    //t_collision_ele = std::pow(Debye,4)*n0_ref
    //                  *(6*std::sqrt(2*mass_e)*std::pow(3.14159*T_e, 3./2.)) /
    //                  (p_lambda*std::pow(charge_e,4)*(charge_i/-charge_e)*nd_e);
    //Print() << "Note the changed t_collision_e_i - ln 815";
    t_collision_ele = std::pow(Debye,4)*n0_ref
                      *(6*std::sqrt(2*mass_e)*std::pow(3.14159*T_e, 3./2.)) /
                      (p_lambda*std::pow((charge_i/-charge_e), 2)*nd_i);
    // the issue with the 'c' value in the

    omega_ce = -charge_e * std::sqrt( Bx*Bx + By*By + Bz*Bz ) / mass_e/Larmor;
    omega_p  = std::sqrt(nd_e*charge_e*charge_e/mass_e/Debye/Debye) ;

    if (1/t_collision_ele < GD::effective_zero) t_collision_ele = 1/GD::effective_zero;

    if (GD::srin_switch && (1/t_collision_ele < omega_ce/10/2/3.14159) && (1/t_collision_ele < omega_p/10/2/3.14159)) {

        if (false && GD::verbose >= 2 ) { //TODO bebug the limiting function - correct strat needed, or a flag fpr the viscous hammer as it were just use a delta value for any potential divide by zeros
            //Print() << "\nMFP_viscous.cpp ln 658 --- Electron collision frequency limited to minimum of plasma and cyclotron frequency\n";
            Print() << "1/tau_e = " << 1/t_collision_ele << "\tomega_ce = "
                  << omega_ce << "\tomega_p = " << omega_p << "\n";
        }
        t_collision_ele = 1/std::min(omega_ce/2/3.14159, omega_p/2/3.14159) ;
        //t_collision_ele = 1/(1/t_collision_ele + GD::effective_zero);
        //Print()<< "\t" << t_collision_ion;
    }    

    Real delta_kappa, delta_eta, delta_eta2,x_coef;// coefficients used exclusively in the braginskii
    x_coef = omega_ce*t_collision_ele;
    // TODO fix up these table 2 page 251 BT
    Real delta_0=3.7703, delta_1=14.79;
    delta_kappa= x_coef*x_coef*x_coef*x_coef+delta_1*x_coef*x_coef + delta_0;
    delta_eta  = x_coef*x_coef*x_coef*x_coef+13.8*x_coef*x_coef + 11.6;
    delta_eta2 = 16*x_coef*x_coef*x_coef*x_coef+4*13.8*x_coef*x_coef + 11.6;

    eta0 = 0.733*nd_e *T_e * t_collision_ele;
    if (GD::braginskii_anisotropic) {
      eta2 = nd_e *T_e*t_collision_ele*(2.05*x_coef*x_coef+8.5)/delta_eta;
      eta1 = nd_e *T_e*t_collision_ele*(2.05*(2*x_coef)*(2*x_coef)+8.5)/delta_eta2;
      eta4 = -nd_e*T_e*t_collision_ele*x_coef*(x_coef*x_coef+7.91)/delta_eta;
      eta3 = -nd_e*T_e*t_collision_ele*(2*x_coef)*((2*x_coef)*(2*x_coef)+7.91)/delta_eta2;
      if (x_coef < 1e-8 ) {
        eta2 = eta2/x_coef/x_coef;
        eta1 = eta1/x_coef/x_coef;
        eta4 = eta4/x_coef;
        eta3 = eta3/x_coef; }
    } else {
      eta2 = 0;
      eta1 = 0;
      eta4 = 0;
      eta3 = 0; }

    //Srinivasan recomendation
    if (std::abs(eta0) < std::abs(eta1)) {
        Print() << "\nelectron viscous coefficient eta1 greater than eta0\n";
        Print() << "\n" + std::to_string(eta0) + "\n";
        Print() << "\n" + std::to_string(eta1) + "\n";
    } else if (std::abs(eta0) < std::abs(eta2)) {
        Print() << "\nelectron viscous coefficient eta2 greater than eta0\n";
        Print() << "\n" + std::to_string(eta0) + "\n";
        Print() << "\n" + std::to_string(eta2) + "\n";
    } else if (std::abs(eta0) < std::abs(eta3)) {
        Print() << "\nelectron viscous coefficient eta3 greater than eta0\n";
        Print() << "\n" + std::to_string(eta0) + "\n";
        Print() << "\n" + std::to_string(eta3) + "\n";
    } else if (std::abs(eta0) < std::abs(eta4)) {
        Print() << "\nelectron viscous coefficient eta4 greater than eta0\n";
        Print() << "\n" + std::to_string(eta0) + "\n";
        Print() << "\n" + std::to_string(eta4) + "\n";
    }
    //From Braginskii OG paper page 250 of paper in journal heading 4
    // Kinetics of a simple plasma (Quantitative Analyis)
    //TODO change coefficient values for different Z values
    // Currently set for a hydrogen  plasma
    Real BT_gamma_0=11.92/3.7703,BT_gamma_0_p=11.92,BT_gamma_1_p=4.664,BT_gamma_1_pp=5./2.;
    Real BT_gamma_0_pp=21.67;

    kappa1=nd_e*T_e*t_collision_ele/mass_e*BT_gamma_0;
    if (GD::braginskii_anisotropic) {
      kappa2=(BT_gamma_1_p*x_coef*x_coef+BT_gamma_0_p)/delta_kappa*nd_e*T_e*t_collision_ele/mass_e;
      kappa3=(BT_gamma_1_pp*x_coef*x_coef+BT_gamma_0_pp)*x_coef*nd_e*T_e*t_collision_ele
           /mass_e/delta_kappa;}

    if ((kappa1<0.) || (kappa2<0.) || (kappa3 < 0.)) {
        //amrex::Abort("mfp_viscous.cpp ln: 673 - Braginski Ion coefficients are non-physical");
        Print() << "\nmfp_viscous.cpp ln: 673 - Braginski Ion coefficients are non-physical\n";
        Print() << "\n" << kappa1 << "\n" << kappa2 << "\n" << kappa3 << "\nomega_ce = " << omega_ce;
    }

    if (kappa1 < kappa2) {
        if (GD::verbose >= 4) {
            Print() << "mfp_viscous.cpp ln: 688 - electron kappa2 exceed kappp1";
        }
        kappa2 = kappa1;
    }
    if (kappa1 < kappa3) {
        if (GD::verbose >= 4) {
            Print() << "mfp_viscous.cpp ln: 694 - electron kappa3 exceed kappp1";
        }
        kappa3 = kappa1;
    }

    // add in the kappa and beta, i have tio figure out how to take these into account,
    // the viscous characteristic speed is not properly taken account of.
    Real rho = ELEstate.get_density_from_cons(U_e);
    Real cp_ele = ELEstate.get_cp(alpha_e);
    Real gamma_ele = ELEstate.get_gamma(alpha_e);
    //Real Prandtl = cp_ele*eta0/kappa1;
    //Real nu_thermal = (4*eta0*gamma_ele/(Prandtl*rho))/cfl/n0_ref;
    //Real nu_visc = (eta0/rho)/cfl/n0_ref;
    Real nu_thermal = kappa1/rho/cp_ele/cfl; //thermal diffusivity 
    Real nu_visc = (eta0/rho)/cfl;
    //TODO delete 
    if (false) { //TODO delete
      Print() << "rho etc\n" <<  rho << "\n";
      Print() << cp_ele<< "\n";
      Print() << gamma_ele<< "\n";
      Print() << nu_thermal << "\n";
      Print() << nu_visc << "\n";
    }
    if (GD::verbose>4) {
      Print() << "Debug and test Prandtl viscous time step ln 867 feature\nnu_visc\t" 
              << nu_visc << "\tnu_thermal\t" << nu_thermal << "\teta0\t" << eta0 << "\n";
    }

    Real nu;
    if (nu_thermal> nu_visc) {
      nu = nu_thermal ;
    } else if (nu_thermal <= nu_visc) {
      nu = nu_visc ;
    }

    //Print() << "\nln 1011 - cfl viscous ele: " << cfl;
    return 2*nu;
    //TODO verify the expression used for this
    //return t_collision_ele ;
    //return (eta0/rho)/cfl;
    //return std::max(eta0, std::max(eta1, std::max(eta2, std::max(eta3,eta4))))/rho;
}

bool BraginskiiEle::valid_state(const int idx)
{
    int s = GD::get_state(idx).get_type();

    if (s == +StateType::isField) {
        return false;
    }
    return true;
}

// ====================================================================================

UserDefinedViscosity::UserDefinedViscosity(){}
UserDefinedViscosity::~UserDefinedViscosity(){}

std::string UserDefinedViscosity::tag = "UserDefined";
bool UserDefinedViscosity::registered = GetViscousFactory().Register(UserDefinedViscosity::tag, ViscousBuilder<UserDefinedViscosity>);

int UserDefinedViscosity::get_type(){return Neutral;}
int UserDefinedViscosity::get_num(){return NUM_NEUTRAL_DIFF_COEFFS;}

UserDefinedViscosity::UserDefinedViscosity(const int global_idx, const sol::table& def)
{

    // check valid state
    if (!valid_state(global_idx))
        Abort("Constant viscosity is not valid for "+GD::get_state(global_idx).name);

    mu_0 = def["mu0"];
    Prandtl = def["Pr"];
    cfl = def.get_or("cfl",1.0);

    idx = global_idx;

    if (mu_0 <= 0) {
        amrex::Abort("Constant viscosity input coefficients non-physical");
    }
}

void UserDefinedViscosity::get_neutral_coeffs(const Vector<Real> &Q, Real &T, Real &mu, Real &kappa)
{
    BL_PROFILE("UserDefinedViscosity::get_neutral_coeffs");
    State &istate = GD::get_state(idx);

    T = istate.get_temperature_from_prim(Q);
    Real alpha = istate.get_alpha_from_prim(Q);
    Real cp = istate.get_cp(alpha);


    mu = mu_0;
    kappa = mu*cp/Prandtl;

    return;
}

Real UserDefinedViscosity::get_max_speed(const Vector<Vector<amrex::Real> >& U)
{
    BL_PROFILE("UserDefinedViscosity::get_max_speed");
    State &istate = GD::get_state(idx);

    Real rho = istate.get_density_from_cons(U[0]);
    Real mu = mu_0;
    Real gamma = istate.get_gamma(U[0]);

    return (4*mu*gamma/(Prandtl*rho))/cfl;
}

bool UserDefinedViscosity::valid_state(const int idx)
{
    int s = GD::get_state(idx).get_type();

    if (s != +StateType::isHydro) {
        return false;
    }
    return true;
}


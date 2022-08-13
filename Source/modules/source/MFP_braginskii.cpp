#include "MFP_braginskii.H"
#include "MFP_global.H"

using GD = GlobalData;

std::string BraginskiiSource::tag = "braginskii";
bool BraginskiiSource::registered = GetSourceTermFactory().Register(BraginskiiSource::tag, SourceTermBuilder<BraginskiiSource>);

BraginskiiSource::BraginskiiSource(){}

BraginskiiSource::BraginskiiSource(const sol::table& def) 
{
    name = def.get<std::string>("name"); 
    Debye = def["DebyeReference"]; //reference valuesfor the braginskii source specifically, not the simulation
    Larmor = def["LarmorReference"];

    if (!BraginskiiSource::valid_solver(def["solver"])) {
        Abort("Error: Source '"+name+"' needs a different solver");
    }

    Vector<int> index;
    Vector<std::string> includes;

    get_includes(def, &BraginskiiSource::valid_state, includes, index);

    offsets.resize(index.size());
    for (int idx=0; idx<index.size(); ++idx) {
        offsets[idx].local = idx;
        offsets[idx].global = index[idx];
    }

    linked_em = -1;
    linked_ele = -1;
    linked_ion = -1;
    Real q0, q1;
    for (const auto &idx : offsets) {
        State &istate = GD::get_state(idx.global);

        q0 = istate.charge[0];
        q1 = istate.charge[1];

        // also check if we have an active field
        if (istate.get_type() == +StateType::isField) { //is it a field?
            if (linked_em> -1) Abort("Source '"+tag+"' has more than one EM state linked to it");
            linked_em = istate.global_idx;
        } else if ((std::abs(q0) == 0) || (std::abs(q1) == 0)) { //is it a neutral?
            Abort("Source '"+tag+"' applied to state '"+istate.name+"' with zero charge");
        } else if (istate.get_type() == +StateType::isHydro) {
            if ((q0 < 0) || (q1 < 0)) {
                linked_ele = istate.global_idx;
            } else if ((q0 > 0) || (q1 > 0)) {
                linked_ion = istate.global_idx;
            }
        } else {
            amrex::Abort("Unknown state type (no state type or charge");
        }
    }

    // get the reconstruction option
    set_reconstruction(def);
    if (!reconstruction)
        Abort("Source '"+tag+"' requires a reconstruction method to be defined (reconstruction=)");

    Print() << "\n\n========== BraginskiiSource ==============\nlinked_em = "  
            << linked_em << "\nlinked_ele = " << linked_ele << "\nlinked_ion = " 
            << linked_ion << "\n" ;
}

BraginskiiSource::~BraginskiiSource()
{
    // do nothing
}

int BraginskiiSource::num_slopes() const
{
    return 3*AMREX_SPACEDIM; // electron temperature gradients 
}

// calculate slopes and pack them serially into a vector
void BraginskiiSource::calc_slopes(const Box& box,
                                   Vector<FArrayBox*>& src_dat,
                                   Vector<FArrayBox>& slopes,
                                   EB_OPTIONAL(Vector<const EBCellFlagFab*> &flags,)
                                   const Real *dx) const
{
    BL_PROFILE("BraginskiiSource::num_slopes"); 
    slopes.resize(num_slopes());
    int cnt = 0;

    const Box slope_box = grow(box, 1);

    const Dim3 lo = amrex::lbound(slope_box);
    const Dim3 hi = amrex::ubound(slope_box);

    // First calculate the primitive variables magnetic field
    const int nc = +HydroState::ConsIdx::NUM;
    const int np = +HydroState::PrimIdx::NUM;

    FArrayBox buffer(slope_box, 1); //num_slopes() box with spatial info i,j,k and the slopes
    Array4<Real> const& b4 = buffer.array();

    State& istate = GD::get_state(linked_ele);
    Array4<const Real> const& h4 = src_dat[linked_ele]->array();
    Vector<Real> U(nc), Q(np);

    for     (int k = lo.z; k <= hi.z; ++k) {
        for   (int j = lo.y; j <= hi.y; ++j) {
            AMREX_PRAGMA_SIMD
                    for (int i = lo.x; i <= hi.x; ++i) {

                // grab the conserved quantities
                for (int n=0; n<nc; ++n) {
                    U[n] = h4(i,j,k,n);
                }

                // convert to primitive
                istate.cons2prim(U, Q);

                // load into buffer
                //for (int n=0; n<+HydroState::PrimIdx::Temp; ++n) {
                b4(i,j,k,0) = U[+HydroState::PrimIdx::Temp]; // just temperature
                //}


            }
        }
    }

    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        for (int n=0; n< 1 ; ++n) { // limit is property index you want up to in buffer
            State::calc_slope(box, buffer, slopes[cnt], EB_OPTIONAL(*flags[linked_ele],) dx, n, d, *reconstruction); cnt++;
        }
    }
}

void BraginskiiSource::retrieve_slopes(
        Vector<FArrayBox>& slopes,
        const int i,
        const int j,
        const int k)
{
    BL_PROFILE("BraginskiiSource::retrieve_slopes");
    slope.resize(num_slopes());
    int cnt = 0;

    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        for (int n=0; n<1; ++n) {
            slope[cnt] = slopes[cnt].array()(i,j,k); cnt++;
        }
    }
}


Real BraginskiiSource::get_coulomb_logarithm(Real T_i,Real T_e,Real nd_e){
  BL_PROFILE("BraginskiiSource::get_coulomb_logarithm");
  // Coulomb logairthm as reported in BRaginskii OG paper. 
  // Alternative is to use the formulation from 
  // "Ionic transport in high-energy-density matter" Stanton 2016
  return 10.;

  Real T_ref = GD::T_ref;
  Real n_ref = GD::n_ref; 

  if (T_e < 50*11600/T_ref) {//where refernce value is in K and conversion 1 eV = 11600K 
    return 23.4 - 1.15 * log10( nd_e*n_ref ) + 3.45 * log10( T_e*T_ref/11600 );

    }
  else {
    Real val = 25.3 - 1.15 * log10( nd_e*n_ref ) + 2.3*log10( T_e*T_ref/11600 );
    return val; 
    //return 25.3 - 1.15 * log10( nd_e*n_ref ) + 2.3*log10( T_e*T_ref/11600 );
    }
  }

Vector<Real> BraginskiiSource::source(const Vector<Real>& y, const Vector<OffsetIndex> &index) const {
    BL_PROFILE("BraginskiiSource::source");
    /*Would like to have kept this general like Daryls above but BRaginskii stuff is very 
    prescriptive and only really accounts for a fully ionised simple plasma or a three 
    component plasam of ion, neutral, and electrons. The current set up is to allow reversion
    to the for loop arrangment. `a' represents the electrons, `b' represents the ions. */

    Vector<Real> ydot(y.size());

    Real q_a, q_a2, m_a;
    Real rho_a, n_a, u_a, v_a, w_a, p_a, T_a, alpha_a, gam_a;
    Real q_b, q_b2, m_b;
    Real rho_b, n_b, u_b, v_b, w_b, p_b, T_b, alpha_b, gam_b;

    Real du, dv, dw;
    Real Rx, Ry, Rz;
    Real Qe, Qf, Q;
    /*----- number of hydrdynamic states
    note idx_EM_state_index indicates which entry of the index data structure holds the 
    EM state, if any. */
    int num_hydroBT=0, num_field=0, idx_EM_state_index=-1; 
    Vector<Real> hydroBT_index_global_map; //map hydro number to the index structure global index
    Vector<Real> hydroBT_index_local_map; //map hydro number to the index structure local index
    int ele_a_idx, ion_b_idx;
    for (const auto &idx : index) {// for each state we are looking at. 
      State &istate = *GD::states[idx.global];
      if (istate.get_type() == +StateType::isHydro) {
        if (istate.charge[0] < 0.) {
          ele_a_idx = num_hydroBT;
        } else if (istate.charge[0]>0.) {
          ion_b_idx = num_hydroBT;
        } else {
          amrex::Abort("hydro state has neutral charge"); 
        }
        hydroBT_index_global_map.push_back (idx.global);
        hydroBT_index_local_map.push_back (idx.local);
        num_hydroBT += 1;
      } else if (istate.get_type() == +StateType::isField) {
        num_field   += 1;
        idx_EM_state_index = idx.local;
      }
    }

    if (num_field >1) {
      amrex::Abort("MFP_source.cpp - BraginskiiSource::source() There are more than one field states, the Braginskii implemetation coded only accepts one."); 
    } else if (num_field ==0) {
      Print() << "\nno field supplied to Braginskii source terms\n";
    }
 
    if (num_hydroBT != 2) {
      amrex::Abort("MFP_source.cpp - BraginskiiSource::source() There are not exactly two hydroBT states, the Braginskii implemetation coded only accepts one."); 
    }

    // vector for hydro primitive values
    Vector<Vector<Real> > hydro_prim(num_hydroBT);//(index.size());
    Vector<Real> EM_prim(num_field);
    int nc, np;
    //TODO find away around the variable declaration in the case of isotropic - may just need to bite the bullet and hav a spearate function 
    Vector<Real> B_unit(3); //Magnetic field unit vector 
    Vector<Real> u_para(3); //Velocity parallel to B_unit
    Vector<Real> u_perp(3); //Velocity perpendicular to B_unit
    Vector<Real> u_chev(3); //unit vector perp to u and B_unit
    Vector<Real> TG_para(3);//Temp grad parallel to B_unit 
    Vector<Real> TG_perp(3);//Temp grad perpendicular to B_unit
    Vector<Real> TG_chev(3);//unit vector perp to gradT and B_unit

    Real B_p=0.,B_pp=0.,bx_pp=0.,by_pp=0.,bz_pp=0.,bx_p=0.,by_p=0., xB=0, yB=0, zB=0; 

    int hydro_counter = 0;
    for (const auto &idx : index) {// for each state we are looking at. 
        State &istate = *GD::states[idx.global];
        nc = istate.n_cons(); np = istate.n_prim();
        if (istate.get_type() != +StateType::isField) {
          // get a copy of the hydro state conserved variables
          Vector<Real> U(nc); Vector<Real> Q(np);
          for (int i=0; i<U.size(); ++i) {
              U[i] = y[idx.solver+i]; // note that the idx.solver here is not a simple interger representing the location of the state etc. iot is anm offset so that the variabeles can all be stoered in the y vector and apsssed, the idx.solver is an offst for all the precerding varianbles.
          }
  
          istate.cons2prim(U, Q); // convert to primitive
            
          hydro_prim[hydro_counter].resize(Q.size()); // save for later - changed from previous implementation 
          std::copy(Q.begin(), Q.end(), hydro_prim[hydro_counter].begin());
          hydro_counter += 1;
        } else if ((GD::braginskii_anisotropic) && (istate.get_type() == +StateType::isField)) {
          // magnetic field
          xB = y[idx.solver + +FieldState::ConsIdx::Bx];
          yB = y[idx.solver + +FieldState::ConsIdx::By];
          zB = y[idx.solver + +FieldState::ConsIdx::Bz];

          Real B = xB*xB + yB*yB + zB*zB;

          if (B < 0.0) {
              Print() << 
              "MFP_braginskii.cpp ln 203 - Negative magnetic field error";
              amrex::Abort("ln 204 MFP_braginskii.cpp");
              Print() << std::to_string(B);
              B = 0.0;
          } else {
              if (B < GD::effective_zero) {
                if (GD::verbose >= 4 ) {
                  Print() << "\nzero magnetic field\n";
                }
                B_pp = 0.;
                B_p  = 0.;
              } else if ( (std::abs(xB) < GD::effective_zero) && (std::abs(yB) < GD::effective_zero) 
                       && (std::abs(zB) > GD::effective_zero) ) {
                if (GD::verbose >= 4 ) {
                  Print() << "\nzero x and y magnetic field\n";
                }
                B_pp = 1/sqrt(B); // B prime prime 
                B_p  = 0.;
              } else {  
                B_pp = 1/sqrt(B); // B prime prime 
                B_p  = 1/sqrt(xB*xB + yB*yB); // B prime 
              }
          }
          
          bx_pp = xB*B_pp; bx_p = xB*B_p;
          by_pp = yB*B_pp; by_p = yB*B_p;
          bz_pp = zB*B_pp; 
          
          B_unit[0] = bx_pp; B_unit[1] = by_pp; B_unit[2] = bz_pp; 
          }
    } 

    // begin loop
    //for (int a = 0; a < num_hydro; ++a) {// will have to change this to just be the electrons 
    int a = ele_a_idx;// represents the electron state.// this is superfluous  
    State &state_a = GD::get_state(hydroBT_index_global_map[a]);// check mapping is done correctly 
    OffsetIndex  offset_a = index[hydroBT_index_local_map[a]];

    rho_a =   hydro_prim[a][+HydroState::PrimIdx::Density];

    if (rho_a <= GD::effective_zero) {
      amrex::Abort(
        "MFP_source.cpp - BraginskiiSource::source() rho less than effective zero, oopsy"); 
    }

    u_a =     hydro_prim[a][+HydroState::PrimIdx::Xvel];
    v_a =     hydro_prim[a][+HydroState::PrimIdx::Yvel];
    w_a =     hydro_prim[a][+HydroState::PrimIdx::Zvel];
    p_a =     hydro_prim[a][+HydroState::PrimIdx::Prs];
    T_a =     hydro_prim[a][+HydroState::PrimIdx::Temp];

    alpha_a = hydro_prim[a][+HydroState::PrimIdx::Alpha]; 
    m_a = state_a.get_mass(alpha_a);
    q_a = state_a.get_charge(alpha_a);

    n_a = rho_a/m_a;
    
    if (p_a < 0. ) {
        Print()<<"\n==========\nMFP_braginskii.cpp ln 251 - fluid a properties\n" << "rho = " << rho_a
               << "\nu_a=" << u_a << "\nv_a=" << v_a << "\nw_a=" << w_a << "\np_a=" 
               << p_a << "\nalpha = " << alpha_a;
        Print() << "\ncharge = " << q_a << "\nmass = " << m_a ;
        amrex::Abort("Negative pressure");
    }
    if (T_a < 0.) {
        Print() << "\nT_a = " + std::to_string(T_a);
    }
    q_a2 = q_a*q_a;
    gam_a= state_a.get_gamma(alpha_a);

    //pull out the gradient values from the slopes stuff

    /*retrieve the velocity slopes - note here that idx.local for this entry 
    (the electron) is the hydroBT_index_local_map[a] entry 
    */

    //TODO Only the temperature gradient is needed
    int cnt = 0;

    //const Real drho_dx = slope[cnt]; cnt++;
    //const Real du_dx   = slope[cnt]; cnt++;
    //const Real dv_dx   = slope[cnt]; cnt++;
    //const Real dw_dx   = slope[cnt]; cnt++;
    //const Real dp_dx   = slope[cnt]; cnt++;
    const Real dT_dx = slope[cnt]; cnt++;

#if AMREX_SPACEDIM >= 2
    //const Real drho_dy = slope[cnt]; cnt++;
    //const Real du_dy   = slope[cnt]; cnt++;
    //const Real dv_dy   = slope[cnt]; cnt++;
    //const Real dw_dy   = slope[cnt]; cnt++;
    //const Real dp_dy   = slope[cnt]; cnt++;
    const Real dT_dy = slope[cnt]; cnt++;
#else
    const Real drho_dy=0, du_dy=0, dv_dy=0, dw_dy=0, dp_dy=0, dT_dy=0;
#endif

#if AMREX_SPACEDIM == 3
    //const Real drho_dz = slope[cnt]; cnt++;
    //const Real du_dz   = slope[cnt]; cnt++;
    //const Real dv_dz   = slope[cnt]; cnt++;
    //const Real dw_dz   = slope[cnt]; cnt++;
    //const Real dp_dz   = slope[cnt]; cnt++;
    const Real dT_dz = slope[cnt]; cnt++;
#else
    const Real drho_dz=0, du_dz=0, dv_dz=0, dw_dz=0, dp_dz=0, dT_dz=0;
#endif

    //for (int b = a+1; b < num_hydro; ++b) {// will have to changethis to just be the ions 
    int b = ion_b_idx;// represents the ion state. 

    State &state_b = GD::get_state(hydroBT_index_global_map[b]);

    OffsetIndex offset_b = index[hydroBT_index_local_map[b]];

    rho_b =   hydro_prim[b][+HydroState::PrimIdx::Density];

    u_b =     hydro_prim[b][+HydroState::PrimIdx::Xvel];
    v_b =     hydro_prim[b][+HydroState::PrimIdx::Yvel];
    w_b =     hydro_prim[b][+HydroState::PrimIdx::Zvel];
    p_b =     hydro_prim[b][+HydroState::PrimIdx::Prs];
    T_b =     hydro_prim[b][+HydroState::PrimIdx::Temp];
    alpha_b = hydro_prim[b][+HydroState::PrimIdx::Alpha];

    m_b = state_b.get_mass(alpha_b);
    q_b = state_b.get_charge(alpha_b);

    n_b =   rho_b/m_b;
    T_b =   p_b/n_b;
    q_b2 =  q_b*q_b;
    gam_b = state_b.get_gamma(alpha_b); 
    
    // Get charge for braginskii table of constants 
    Real Z_i = -q_b/q_a ; // electrong charge is negative

    du = u_a - u_b; dv = v_a - v_b; dw = w_a - w_b; //dU2 = du*du + dv*dv + dw*dw;

    if (p_b < 0. ) {
        Print()<<"\nMFP_braginskii.cpp ln 326 - fluid b properties\n" << "rho = " << rho_b 
               << "\nu_b=" << u_b << "\nv_b=" << v_b << "\nw_b=" << w_b << "\np_b=" 
               << p_b << "\n" << alpha_b;
        Print() << "\ncharge = " << q_b << "\nmass = " << m_b ;
        amrex::Abort("Negative pressure");
    }
    if (T_b < 0.) {
        Print() << "\nT_b = " + std::to_string(T_b);
    }


    //Braginskii directionality stuff.  
    if (GD::braginskii_anisotropic) {    
      Real dot_B_unit_TG, dot_B_unit_U ;//temp variables
  
      dot_B_unit_U = bx_pp*du + by_pp*dv + bz_pp*dw;
      dot_B_unit_TG = bx_pp*dT_dx + by_pp*dT_dy + bz_pp*dT_dz;
  
      for (int i_disp = 0; i_disp < 3; ++i_disp) {//i_disp - i disposable 
        u_para[i_disp] = B_unit[i_disp]*dot_B_unit_U ;
        TG_para[i_disp]= B_unit[i_disp]*dot_B_unit_TG ;
        if (i_disp==0){
          u_perp[i_disp] = du - u_para[0]; // couuld be automated with 
                                                  // index = prim_vel_id[0] + i_disp 
          u_chev[i_disp] = B_unit[1]*dw-B_unit[2]*dv;
          TG_perp[i_disp]= dT_dx - TG_para[0];  //...automated with dT_i vector...
          TG_chev[i_disp]= B_unit[1]*dT_dz-B_unit[2]*dT_dy;
        }
        else if (i_disp==1) {
          u_perp[1] = dv - u_para[1];
          u_chev[1] = -(B_unit[0]*dw-B_unit[2]*du);
          TG_perp[1] = dT_dy - TG_para[1];
          TG_chev[1]= -(B_unit[0]*dT_dz-B_unit[2]*dT_dx);
        }
        else {
          u_perp[i_disp] = dw - u_para[2];
          u_chev[i_disp] = B_unit[0]*dv-B_unit[1]*du;
          TG_perp[i_disp] = dT_dz - TG_para[2];
          TG_chev[i_disp]= B_unit[0]*dT_dy-B_unit[1]*dT_dx;
        }
      }
      /*
      if (GD::verbose > 4) {
        Print() << "\nBunit\t" << B_unit[0] << "\nBunit\t" << B_unit[1] 
                << "\nBunit\t" << B_unit[2] << "\n";
        Print() << "u_rel\t" << du << "\nu_rel\t" << dv << "\nu_rel\t" << dw << "\n";
        Print() << "dot_B_unit_U\t" << dot_B_unit_U << "\n";
        Print() << "dot_B_unit_TG\t" << dot_B_unit_TG << "\n";
        Print() << "u_para[0]\t" << u_para[0] << "\nu_para\t" << u_para[1] 
                << "\nu_para\t" << u_para[2] << "\n";
        Print() << "u_perp[0]\t" << u_perp[0] << "\nu_perp\t" << u_perp[1] 
                << "\nu_perp\t" << u_perp[2] << "\n";  
        Print() << "u_chev[0]\t" << u_chev[0] << "\nu_chev[1]\t" << u_chev[1] 
                << "\nu_chev[2]\t" << u_chev[2] << "\n";
        Print() << "TG_para[0]\t" << TG_para[0] << "\nTG_para[1]\t" << TG_para[1] 
                << "\nTG_para[2]\t" << TG_para[2] << "\n";
        Print() << "TG_perp[0]\t" << TG_perp[0] << "\nTG_perp[1]\t" << TG_perp[1] 
                << "\nTG_perp[2]\t" << TG_perp[2] << "\n";
        Print() << "TG_chev[0]\t" << TG_chev[0] << "\nTG_chev[1]\t" << TG_chev[1]   
                << "\nTG_chev[2]\t" << TG_chev[2] << "\n";
      }
      */
    }
    //---------------Braginskii Momentum source
    Real alpha_0, alpha_1, alpha_2, beta_0, beta_1, beta_2, t_c_a;
    Real p_lambda = get_coulomb_logarithm(T_b,T_a,n_a);

    get_alpha_beta_coefficients(Z_i, Debye, Larmor, m_a, T_a, q_a, q_b, n_a, n_b, alpha_0, 
        alpha_1, alpha_2, beta_0, beta_1, beta_2, t_c_a, p_lambda, xB, yB, zB); 

    /*
    if (GD::verbose>4) {
      Print() << "alpha_0\t" << alpha_0 << "\nalpha_1\t" << alpha_1 << "\nalpha_2\t" << alpha_2   
              << "\nbeta_0\t" << beta_0 << "\nbeta_1\t" << beta_1 << "\nbeta_2\t" << beta_2 
              << "\nt_c_a\t" << t_c_a << "\n";
    }
    */

    Vector<Real> R_u(3), R_T(3);
    if (GD::braginskii_anisotropic) {
    //frictional force
      R_u[0] = -alpha_0*u_para[0] - alpha_1*u_perp[0] + alpha_2*u_chev[0];
      R_u[1] = -alpha_0*u_para[1] - alpha_1*u_perp[1] + alpha_2*u_chev[1];
      R_u[2] = -alpha_0*u_para[2] - alpha_1*u_perp[2] + alpha_2*u_chev[2];
      //Thermal force
      R_T[0] = -beta_0*TG_para[0] - beta_1*TG_perp[0] - beta_2*TG_chev[0];
      R_T[1] = -beta_0*TG_para[1] - beta_1*TG_perp[1] - beta_2*TG_chev[1];
      R_T[2] = -beta_0*TG_para[2] - beta_1*TG_perp[2] - beta_2*TG_chev[2];
    } else {
      R_u[0] = -alpha_0*du;
      R_u[1] = -alpha_0*dv;
      R_u[2] = -alpha_0*dw;
      //Thermal force
      //Print() << "\nR_T set to zero\n";
      R_T[0] = -beta_0*dT_dx;
      R_T[1] = -beta_0*dT_dy;
      R_T[2] = -beta_0*dT_dz;
    }
    //Thermal equilibration
    Real Q_delta = 3*m_a/m_b*n_a/t_c_a*(T_a-T_b);
    //Real Q_fric  = (R_u[0]+R_T[0])*du + (R_u[1]+R_T[1])*dv + (R_u[2]+R_T[2])*dw; //TODO incorrect bc assumes internal energy cons, not total energy cons 
    Real Q_fric  = (R_u[0]+R_T[0])*u_b + (R_u[1]+R_T[1])*v_b + (R_u[2]+R_T[2])*w_b;

    /*
    Real x_ref=GD::x_ref, n_ref=GD::n_ref, m_ref=GD::m_ref, rho_ref=GD::rho_ref, 
    T_ref=GD::T_ref, u_ref=GD::u_ref;

    Real t_ref = x_ref/u_ref, rho_ref=m_ref/x_ref/x_ref/x_ref;

    nd_Ru = t_ref/rho_ref/u_ref*(m_ref*u_ref*n_ref/t_ref);
    nd_RT = t_ref/rho_ref/u_ref*(n_ref*T_ref/x_ref);
    nd_Qe = t_ref/m_ref/n_ref/u_ref/u_ref*()
    nd_Qdelta = t_ref/m_ref/n_ref/u_ref/u_ref*()
    if ( (nd_Ru != 1.) ||  (nd_RT != 1.) ||  (nd_Qe != 1.) ||  (nd_Qdelta != 1.)) {
        Print() << "\n" + std::to_string
    } 
    */
    ydot[offset_a.solver + +HydroState::ConsIdx::Xmom] += R_u[0]+R_T[0];
    ydot[offset_a.solver + +HydroState::ConsIdx::Ymom] += R_u[1]+R_T[1];
    ydot[offset_a.solver + +HydroState::ConsIdx::Zmom] += R_u[2]+R_T[2];
    //ydot[offset_a.solver + +HydroState::ConsIdx::Eden] -= Q_delta + Q_fric;//TODO wrong frame 
    ydot[offset_a.solver + +HydroState::ConsIdx::Eden] += -Q_delta + Q_fric; //changed for total energy frame 
    //note here the b is the ion
    ydot[offset_b.solver + +HydroState::ConsIdx::Xmom] -= R_u[0]+R_T[0];
    ydot[offset_b.solver + +HydroState::ConsIdx::Ymom] -= R_u[1]+R_T[1];
    ydot[offset_b.solver + +HydroState::ConsIdx::Zmom] -= R_u[2]+R_T[2];
    //ydot[offset_b.solver + +HydroState::ConsIdx::Eden] += Q_delta + Q_fric; //TODO incorrect bc assumes internal energy cons, not total energy cons 
    ydot[offset_b.solver + +HydroState::ConsIdx::Eden] += Q_delta - Q_fric; //correct total energy frame
    /*
    if (GD::viewFluxSrcContributions == 1) Print() << "\tElectron\t" << R_u[2]+R_T[2] << "\tIon:\t" << -R_u[2]-R_T[2]; 

    if (true && GD::verbose > 2) {
        Print() << "\nQ_fric\t" << Q_fric << "\nQ_delta\t" << Q_delta << "\n";
        for (int i_disp = 0; i_disp < 3; i_disp ++) {
            Print() << "R_u[" << i_disp << "]\t" << R_u[i_disp] << "\n";
        }
        for (int i_disp = 0; i_disp < 3; i_disp ++) {
            Print() << "R_T[" << i_disp << "]\t" << R_T[i_disp] << "\n";
        } 
        Print() << "T grad ele\t" << dT_dx << "\nT grad ele\t" << dT_dy << "\nT grad ele\t" << dT_dz << "\n"; 
    }
    */

    return ydot;
}

int BraginskiiSource::fun_rhs(Real x, Real y, Real z, Real t, Vector<Real> &y0, Vector<Real> &ydot, Real dt) const
{
    BL_PROFILE("BraginskiiSource::fun_rhs");
    ydot = source(y0, offsets);

    return 0;
}

int BraginskiiSource::fun_jac(Real x, Real y, Real z, Real t, Vector<Real> &y0, Vector<Real> &J) const
{
    BL_PROFILE("BraginskiiSource::fun_jac");
    /* Doesn't work, I will have to fix this if I want to use an implicit method (cvode) with the
       Braginskii stuff at some point.
    */
    return 0;
}

void BraginskiiSource::get_alpha_beta_coefficients(const Real& Z_i, Real Debye, Real Larmor, 
      Real mass_e, Real T_e, Real charge_e, Real charge_i, Real nd_e, Real nd_i, Real& alpha_0, 
      Real& alpha_1, Real& alpha_2, Real& beta_0, Real& beta_1, Real& beta_2, Real& t_c_e, 
      Real p_lambda, Real Bx, Real By, Real Bz) {

  //collision time nondimensional
  Real x_ref=GD::x_ref, n0_ref=GD::n0, m_ref=GD::m_ref, rho_ref=GD::rho_ref, 
    T_ref=GD::T_ref, u_ref=GD::u_ref;

  Real t_ref = x_ref/u_ref;
  Real pi_num = 3.14159265358979323846;
  //t_c_e = std::pow(Debye,4)*n0_ref
  //                  *(6*std::sqrt(2*mass_e)*std::pow(pi_num*T_e, 3./2.)) / 
  //                  (p_lambda*std::pow(charge_e,4)*(charge_i/-charge_e)*nd_e); 
  //Print() << "Note the change in t_c_e - ln 564";

  t_c_e = std::pow(Debye,4)*n0_ref
                    *(6*std::sqrt(2*mass_e)*std::pow(pi_num*T_e, 3./2.)) / 
                    (p_lambda*std::pow((charge_i/-charge_e),2)*nd_i); 

  /*
  t_c_e = (3*std::sqrt(mass_e) * std::pow(T_e, 3./2.)) / 
                      (4*std::sqrt(pi_num) * p_lambda * std::pow(charge_e,4) * nd_e); 
  */
  //Real omega_ce =  -charge_e*std::sqrt( Bx*Bx + By*By + Bz*Bz )/mass_e/2/pi_num/Larmor; 
  Real omega_ce = -charge_e * std::sqrt( Bx*Bx + By*By + Bz*Bz ) / mass_e / Larmor; 
  Real omega_p  = std::sqrt(nd_e*charge_e*charge_e/mass_e/Debye/Debye) ;

  if (1/t_c_e < GD::effective_zero) t_c_e = 1/GD::effective_zero;

  if (GD::srin_switch && (1/t_c_e < omega_ce/10/2/pi_num) && (1/t_c_e < omega_p/10/2/pi_num)) {
      if  (GD::verbose > 2) {
      Print() << "1/tau_e = " << 1/t_c_e << "\tomega_ce = " << omega_ce 
            << "\tomega_p = " << omega_p << "\n";
      }
      t_c_e = 1/std::min(omega_ce/2/pi_num, omega_p/2/pi_num) ;
      //t_c_e = 1/( 1/t_c_e + GD::effective_zero);

      if  (GD::verbose > 2) Print() << "1/tau_e correction: " << 1/t_c_e;
  }
  if (false && GD::verbose > 1) Print() << "\t" << t_c_e << "\n";
 
  //TODO create a table lookup for the coefficients based off the plasma constituents
  Real delta_coef, x_coef ;// coefficients used exclusively in the braginskii 
                           // transport terms 
  x_coef = omega_ce*t_c_e;

  Real delta_0, delta_1, a_0, a_0_p, a_1_p, a_0_pp, a_1_pp, b_0, b_0_pp, b_0_p, b_1_p, b_1_pp;
  // check Z_i and round 
  if (Z_i < 0) Abort("\nNegative Z number ln 746 MFP_viscous.cpp\n");
  Real Z_i_rounded = Z_i;
  Z_i_rounded = std::roundf(Z_i_rounded);
  // assign based on charge 
  if (Z_i_rounded == 1) {
    a_0 = 0.5129;
    b_0 = 0.7110;
    delta_0 = 3.7703;
    delta_1 = 14.79;
    b_1_p = 5.101;
    b_0_p = 2.681;
    b_1_pp = 3./2.;
    b_0_pp = 3.053;
    a_1_p = 6.416;
    a_0_p = 1.837;
    a_1_pp = 1.704;
    a_0_pp = 0.7796;
  } else if (Z_i_rounded == 2) {
    a_0 = 0.4408;
    b_0 = 0.9052;
    delta_0 = 1.0465;
    delta_1 = 10.80;
    b_1_p = 4.450;
    b_0_p = 0.9473;
    b_1_pp = 3./2.;
    b_0_pp = 1.784;
    a_1_p = 5.523;
    a_0_p = 0.5956;
    a_1_pp = 1.704;
    a_0_pp = 0.3439;
  } else if (Z_i_rounded == 3) {
    a_0 = 0.3965;
    b_0 = 1.016;
    delta_0 = 0.5814;
    delta_1 = 9.618;
    b_1_p = 4.233;
    b_0_p = 0.5905;
    b_1_pp = 3./2.;
    b_0_pp = 1.442;
    a_1_p = 5.226;
    a_0_p = 0.3515;
    a_1_pp = 1.704;
    a_0_pp = 0.2400;
  } else if (Z_i_rounded == 4) {
    a_0 = 0.3752;
    b_0 = 1.090;
    delta_0 = 0.4106;
    delta_1 = 9.055;
    b_1_p = 4.124;
    b_0_p = 0.4478;
    b_1_pp = 3./2.;
    b_0_pp = 1.285;
    a_1_p = 5.077;
    a_0_p = 0.2566;
    a_1_pp = 1.704;
    a_0_pp = 0.1957;
  } else { 
    a_0 = 0.2949;
    b_0 = 1.521;
    delta_0 = 0.0961;
    delta_1 = 7.482;
    b_1_p = 3.798;
    b_0_p = 0.1461;
    b_1_pp = 3./2.;
    b_0_pp = 0.877;
    a_1_p = 4.63;
    a_0_p = 0.0678;
    a_1_pp = 1.704;
    a_0_pp = 0.0940;
  }
  /*
  if ( (Z_i > 2 ) and (Z_i < 3)) {
    Print() << "\nZ_i\t" << Z_i << "\t" << Z_i_rounded;

    Print() << "\n" << a_0 << "\t" << b_0 << "\t" << delta_0 << "\t" 
      << delta_1 << "\t" 
      << a_1_p << "\t" << a_0_p << "\t" << a_1_pp << "\t" << a_0_pp
      << b_1_p << "\t" << b_0_p << "\t" << b_1_pp << "\t" << b_0_pp ;
  }
  */

  //Real delta_0 =3.7703, delta_1 = 14.79;
  delta_coef = x_coef*x_coef*x_coef*x_coef+delta_1*x_coef*x_coef + delta_0;// TODO delta0 tables 

  //Real a_0 = 0.5129, a_0_p =1.837, a_1_p =6.416, a_0_pp =0.7796, a_1_pp =1.704;

  alpha_0 = mass_e*nd_e/t_c_e*a_0;

  if (GD::braginskii_anisotropic) {    
    alpha_1 = mass_e*nd_e/t_c_e*( 1 - (a_1_p*x_coef*x_coef + a_0_p)/delta_coef);
    alpha_2 = mass_e*nd_e/t_c_e*x_coef*(a_1_pp*x_coef*x_coef+a_0_pp)/delta_coef;
  } else { alpha_1 = 0; alpha_2 = 0;}

  //Real b_0 = 0.711, b_0_pp = 3.053, b_0_p=2.681, b_1_p=5.101, b_1_pp=3./2.;
  
  beta_0 = nd_e*b_0;
  if (GD::braginskii_anisotropic) {    
  beta_1 = nd_e*(b_1_p*x_coef*x_coef+b_0_p)/delta_coef;
  beta_2 = nd_e*x_coef*(b_1_pp*x_coef*x_coef+b_0_pp)/delta_coef;
  } else { beta_1 = 0; beta_2 = 0; }

  //TODO remove after comparison with plasmapy braginskii
  /*
  if (false && GD::verbose >= 2) {
      Print() << "\n\nResistivity\nrhor_para\t" <<1/(nd_e*nd_e*charge_e*charge_e/alpha_0);
      Print() << "\nrhor_perp\t" << 1/(nd_e*nd_e*charge_e*charge_e/alpha_1);
      Print() << "\nrhor_chev\t" << 1/(nd_e*nd_e*charge_e*charge_e/alpha_2);
      
      Print() << "\n\nThermoelectric conductivity\nbeta_para\t" << beta_0 << "\nbeta_perp\t" << beta_1 << "\nbeta_chev\t" << beta_2 ;
  }
  */
  return ;
  } 

Real BraginskiiSource::get_max_freq(Vector<Real> &y) const
{
    BL_PROFILE("BraginskiiSource::get_max_freq");
    // get any magnetic field

    Real Bx=0., By=0., Bz=0.;

    int field_offset;

    for (const auto &idx : offsets) {
        State &istate = GD::get_state(idx.global);
        int t = istate.get_type();

        if (t != +StateType::isField)
            continue;

        field_offset = idx.solver;

        // magnetic field
        Bx = y[field_offset + +FieldState::ConsIdx::Bx];
        By = y[field_offset + +FieldState::ConsIdx::By];
        Bz = y[field_offset + +FieldState::ConsIdx::Bz];

        break;

    }



    Real B = std::sqrt(Bx*Bx + By*By + Bz*Bz);
    if (B<0.) {
    amrex::Abort("Negative B field in Braginskii source");
    }

    Real q, m, r;
    Real rho, alpha;
    Real omega_p, omega_c;
    Real Debye = GD::Debye;
    Real n0_ref = GD::n0;
    Real D2 = GD::Debye*GD::Debye; // should this be the simulation D2 and L for the reference parameters + cdim, or should t be from the source terms own D2 and L
    Real L = GD::Larmor;

    // Variables for the collision time scale in the cell
    Real mass_e, T_e, charge_e, nd_e, mass_i, T_i, charge_i, nd_i; 

    Real f = 0;
    for (const auto &idx : offsets) {

        State &istate = GD::get_state(idx.global);
        int t = istate.get_type();

        if (t == +StateType::isField)
            continue;

        if (!idx.valid) Abort("State '"+istate.name+"' is unavailable for source of type '"+tag+"'");

        rho =   y[idx.solver + +HydroState::ConsIdx::Density];
        alpha = y[idx.solver + +HydroState::ConsIdx::Tracer]/rho;

        m = istate.get_mass(alpha);
        q = istate.get_charge(alpha);
        Real g = istate.get_gamma(alpha);
        Real mx = y[idx.solver + +HydroState::ConsIdx::Xmom];
        Real my = y[idx.solver + +HydroState::ConsIdx::Ymom];
        Real mz = y[idx.solver + +HydroState::ConsIdx::Zmom];
        Real ed = y[idx.solver + +HydroState::ConsIdx::Eden];
    
        Real rhoinv = 1/rho;
        Real ke = 0.5*rhoinv*(mx*mx + my*my + mz*mz);
        Real prs = (ed - ke)*(g - 1);

        r = q/m;
        if (GD::source_collision_frequency_constraint == 1) {
          if (q < 0) {
            mass_e = m;
            T_e = prs*rhoinv*m;
            charge_e = q;
            nd_e = rho/m;
          } else if (q > 0) {
            mass_i = m;
            T_i = prs*rhoinv*m;
            charge_i = q;
            nd_i = rho/m;
          } else {
            Abort("Error: braginskii neutral species selected");
          }
        }

        omega_p = 10*std::sqrt(rho*r*r/D2)/(2*PI);
        omega_c = 10*(std::abs(r)*B)/(L*2*PI);
        f = std::max(f, omega_p);
        f = std::max(f, omega_c);
    }

    if (GD::source_collision_frequency_constraint == 1) {
      Real p_lambda = get_coulomb_logarithm(T_i,T_e,nd_e);
      // checking the collision time scales are adhered to 
      //Print() << "\n\t" << 1/f ;
      Real t_c_e = std::pow(Debye,4)*n0_ref
                  *(6*std::sqrt(2*mass_e)*std::pow(3.14159265358979323846*T_e, 3./2.)) / 
                  (p_lambda*std::pow((charge_i/-charge_e),2)*nd_i); 
  
      Real t_c_i = std::pow(Debye,4)*n0_ref
                    *(12*std::sqrt(mass_i)*std::pow(3.14159265358979323846*T_i, 3./2.)) /
                    (p_lambda * std::pow(charge_i,4) * nd_i);
      //Print() << "\n" << 1/(10/t_c_e) << "\t" << 1/(10/t_c_i);
  
      f = std::max(f, 10/t_c_e);
      f = std::max(f, 10/t_c_i);
    }

    return f;
    }

bool BraginskiiSource::valid_state(const int global_idx)
{

    State &istate = GD::get_state(global_idx);

    switch (istate.get_type()) {
    case +StateType::isField:
        return true;
    case +StateType::isHydro:
        return true;
    default:
        return false;
    }
}

bool BraginskiiSource::valid_solver(const int solver_idx)
{
    return true;
}

std::string BraginskiiSource::print() const
{
    std::stringstream msg;

    msg << tag << " : " << name;
    return msg.str();

}





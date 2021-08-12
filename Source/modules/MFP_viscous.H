#ifndef MFP_VISCOUS_H
#define MFP_VISCOUS_H

#include <AMReX_REAL.H>
#include <AMReX_Vector.H>

#include "MFP_factory.H"
#include "sol.hpp"
// Forward declaration of class State for use in Braginskii viscosity
class State;

using amrex::Real;
using amrex::Vector;
using amrex::Array;
// ====================================================================================

class Viscous
{
public:
  Viscous();
  ~Viscous();

  enum DiffusionType {
    Neutral = 0,
    Ion,
    Electron
  };

  enum NeutralDiffusionCoeffs {
    NeutralTemp = 0,
    NeutralMu,
    NeutralKappa,
    NUM_NEUTRAL_DIFF_COEFFS
  };

  enum IonDiffusionCoeffs {
    IonTemp = 0,
    IonEta0,
    IonEta1,
    IonEta2,
    IonEta3,
    IonEta4,
    IonKappa1,
    IonKappa2,
    IonKappa3,
    NUM_ION_DIFF_COEFFS
  };

  enum ElectronDiffusionCoeffs {
    EleTemp = 0,
    EleEta0,
    EleEta1,
    EleEta2,
    EleEta3,
    EleEta4,
    EleKappa1,
    EleKappa2,
    EleKappa3,
    EleBeta1, 
    EleBeta2,
    EleBeta3,
    NUM_ELE_DIFF_COEFFS
  };

  virtual int get_type();
  virtual int get_BraginskiiIdentity();
  virtual int get_num();
  virtual void get_neutral_coeffs(const Vector<Real> &Q, Real &T, Real &mu, Real &kappa);
  virtual void get_ion_coeffs(State& EMstate, State& ELEstate,
                              const Vector<Real>& Q_i,const Vector<Real>& Q_e,
                              const Vector<Real>& B_xyz,Real& T_i,Real& eta0,
                              Real& eta1,Real& eta2,Real& eta3,Real& eta4,
                              Real& kappa1,Real& kappa2,Real& kappa3, int& truncatedTau);
  virtual void get_electron_coeffs(State& EMstate, State& IONstate,
                                   const Vector<Real>& Q_i,const Vector<Real>& Q_e,
                                   const Vector<Real>& B_xyz,Real& T_e,Real& eta0,
                                   Real& eta1,Real& eta2,Real& eta3,Real& eta4,
                                   Real& kappa1,Real& kappa2,Real& kappa3, 
                                   Real& beta1,Real& beta2,Real& beta3, int& truncatedTau);
  virtual void update_linked_states();
  virtual Vector<int> get_linked_states() {return {idx};}
  virtual Real get_max_speed(const Vector<Vector<Real>> &U);
  
  virtual const std::string& get_tag() const = 0;
  virtual const std::map<std::string, Real> get_refs() const {return {};}
  virtual std::string str() const;

  virtual Real get_coulomb_logarithm(const Real& T_i,const Real& T_e,const Real& nd_e);
  int idx;
  Real cfl;
};

// ====================================================================================

class Sutherland : public Viscous
{
public:
  Sutherland();
  Sutherland(const int global_idx, const sol::table &def);
  ~Sutherland();

  Real mu_0, T0, S, Prandtl;

  virtual int get_type() override;
  virtual int get_BraginskiiIdentity() override;
  virtual int get_num() override;
  virtual void get_neutral_coeffs(const Vector<Real> &Q, Real &T, Real &mu, Real &kappa) override;
  virtual Vector<int> get_linked_states() override {return {idx};}
  virtual Real get_max_speed(const Vector<Vector<Real>> &U) override;
  static bool valid_state(const int idx);
  virtual const std::string& get_tag() const override {return tag;}
  virtual const std::map<std::string, Real> get_refs() const override {
      return {{"mu_0", mu_0},{"T0", T0}, {"S",S}, {"Pr",Prandtl}};
  }
  static bool registered;
  static std::string tag;
};

// ====================================================================================

class PowerLaw : public Viscous
{
public:
  PowerLaw();
  PowerLaw(const int global_idx, const sol::table& def);
  ~PowerLaw();

  enum NeutralDiffusiveVariable {
    Temp = 0,
    Mu,
    Kappa,
    NUM_NEUTRAL_DIFF
  };

  Real mu_0, T0, n, Prandtl;

  virtual int get_type() override;
  virtual int get_BraginskiiIdentity() override;
  virtual int get_num() override;
  virtual void get_neutral_coeffs(const Vector<Real> &Q, Real &T, Real &mu, Real &kappa) override;
  virtual Vector<int> get_linked_states() override {return {idx};}
  virtual Real get_max_speed(const Vector<Vector<Real>> &U) override;
  static bool valid_state(const int idx);
  virtual const std::string& get_tag() const override {return tag;}
  virtual const std::map<std::string, Real> get_refs() const override {
      return {{"mu_0", mu_0},{"T0", T0}, {"n",n}, {"Pr",Prandtl}};
  }
  static bool registered;
  static std::string tag;
};


// ====================================================================================

class BraginskiiIon : public Viscous
{
public:
  BraginskiiIon();
  BraginskiiIon(const int global_idx, const sol::table& def);
  //BraginskiiIon(const Real mu_ref, const Real T_ref, const int BT, const int i);
  ~BraginskiiIon(); 

  enum IonDiffusiveVariable {
    Temp = 0,
    Eta0,
    Eta1,
    Eta2,
    Eta3,
    Eta4,
    Kappa1,
    Kappa2,
    Kappa3,
    NUM_ION_DIFF_COEFFS
  };

  Real mu_0, T0, BT, forceViscosityValue;
  bool forceViscosity;
  Real T_i,eta0,eta1,eta2,eta3,eta4,kappa1,kappa2,kappa3;

  int linked_electron, linked_em;

  virtual int get_type() override;
  virtual int get_BraginskiiIdentity() override;
  virtual int get_num() override;
  virtual void get_ion_coeffs(State & EMstate,State & ELEstate,
                              const Vector<Real>& Q_i,const Vector<Real>& Q_e,
                              const Vector<Real>& B_xyz,Real& T_i,Real& eta0,
                              Real& eta1,Real& eta2,Real& eta3,Real& eta4,
                              Real& kappa1,Real& kappa2,Real& kappa3, int& truncatedTau) override;
  virtual void update_linked_states() override;

  virtual Vector<int> get_linked_states() override {return {idx, linked_electron, linked_em};}//Note last must always eb linked_em for dependencies in viscous time step estimate in MFP_stepsize.cpp

  virtual Real get_max_speed(const Vector<Vector<Real>>& U) override;
  static bool valid_state(const int idx);
  virtual const std::string& get_tag() const override {return tag;}
  virtual const std::map<std::string, Real> get_refs() const override {
      return {{"cfl", cfl}, {"forceViscosity",forceViscosity},{"forceViscosityValue",forceViscosityValue} };
  }
  static bool registered;
  static std::string tag;
};

// ====================================================================================

class BraginskiiEle : public Viscous
{
public:
  BraginskiiEle();
  BraginskiiEle(const int global_idx, const sol::table& def);
  //BraginskiiEle(const Real mu_ref, const Real T_ref, const int BT, const int i);
  ~BraginskiiEle(); 

  enum EleDiffusiveVariable {
    Temp = 0,
    Eta0,
    Eta1,
    Eta2,
    Eta3,
    Eta4,
    Kappa1,
    Kappa2,
    Kappa3,
    Beta1, 
    Beta2,
    Beta3, 
    NUM_ELE_DIFF_COEFFS
  };

  Real mu_0, T0, BT, forceViscosityValue;
  bool forceViscosity;

  Real T_e,eta0,eta1,eta2,eta3,eta4,kappa1,kappa2,kappa3,beta1,beta2,beta3;

  int linked_ion, linked_em;

  virtual int get_type() override;
  virtual int get_BraginskiiIdentity() override;
  virtual int get_num() override;
  virtual void get_electron_coeffs(State& EMstate,State& IONstate,
                                   const Vector<Real>& Q_i,const Vector<Real>& Q_e,
                                   const Vector<Real>& B_xyz,Real& T_e,Real& eta0,
                                   Real& eta1,Real& eta2,Real& eta3,Real& eta4,
                                   Real& kappa1,Real& kappa2,Real& kappa3, 
                                   Real& beta1,Real& beta2,Real& beta3, int& truncatedTau) override;
  virtual void update_linked_states() override;

  virtual Vector<int> get_linked_states() override {return {idx, linked_ion, linked_em};}//Note last must always eb linked_em for dependencies in viscous time step estimate in MFP_stepsize.cpp

  virtual Real get_max_speed(const Vector<Vector<Real>>& U) override;
  static bool valid_state(const int idx);
  virtual const std::string& get_tag() const override {return tag;}
  virtual const std::map<std::string, Real> get_refs() const override {
      return {{"cfl", cfl}, {"forceViscosity",forceViscosity},{"forceViscosityValue",forceViscosityValue} };
  }
  static bool registered;
  static std::string tag;

};



// ====================================================================================

class UserDefinedViscosity : public Viscous
{
public:
  UserDefinedViscosity();
  UserDefinedViscosity(const int global_idx, const sol::table& def);
  ~UserDefinedViscosity();

  enum NeutralDiffusiveVariable {
    Temp = 0,
    Mu,
    Kappa,
    NUM_NEUTRAL_DIFF
  };

  Real mu_0, T0, n, Prandtl;

  virtual int get_type() override;
  virtual int get_num() override;
  virtual void get_neutral_coeffs(const Vector<Real> &Q, Real &T, Real &mu, Real &kappa) override;
  virtual Real get_max_speed(const Vector<Vector<Real>>& U) override;
  static bool valid_state(const int idx);
  virtual const std::string& get_tag() const override {return tag;}
  virtual const std::map<std::string, Real> get_refs() const override {
      return {{"mu_0", mu_0}, {"Pr",Prandtl}};
  }
  static bool registered;
  static std::string tag;
};

// ====================================================================================


template <typename D>
std::unique_ptr<Viscous> ViscousBuilder(const sol::table& def)
{
    std::unique_ptr<Viscous> viscous;

    const int global_idx = def["global_idx"];
    const sol::table& visc = def["viscosity"].get_or(sol::table());

    if (visc.valid()) {

        // load the diffusion coefficients

        if (visc["type"] == D::tag) {
            return std::unique_ptr<D>(new D(global_idx, visc));
        } else {
            return nullptr;
        }
    } else {
        return nullptr;
    }
}



PhysicsFactory<Viscous>& GetViscousFactory();

#endif // MFP_DIFFUSION_H

#ifndef MFP_BRAGINSKII_H
#define MFP_BRAGINSKII_H

 #include "MFP_source.H" 

class BraginskiiSource : public SourceTerm
{
public:
    BraginskiiSource();
    BraginskiiSource(const sol::table& def);
    ~BraginskiiSource();

    static void get_alpha_beta_coefficients(Real Debye, Real Larmor, Real mass_e, Real T_e, Real charge_e, Real charge_i 
                        ,Real nd_e, Real nd_i, Real& alpha_0, Real& alpha_1, Real& alpha_2 
                        ,Real& beta_1, Real& beta_2, Real& beta_3, Real& t_c_e, Real p_lambda 
                        ,Real Bx, Real By, Real Bz);

    static Real get_coulomb_logarithm(Real T_i, Real T_e, Real nd_e); 
    Vector<Real> get_slopes() const;

    Vector<Real> source(const Vector<Real> &y, const Vector<OffsetIndex> &apply) const;
    virtual int fun_rhs(Real x, Real y, Real z, Real t, Vector<Real> &y0, Vector<Real> &ydot, Real dt=0) const override;
    virtual int fun_jac(Real x, Real y, Real z, Real t, Vector<Real> &y0, Vector<Real> &J) const override;

    virtual int num_slopes() const override;

    virtual void calc_slopes(const Box& box,
                             Vector<FArrayBox*>& src_dat,
                             Vector<FArrayBox>& slopes,
                             EB_OPTIONAL(Vector<const EBCellFlagFab*> &flags,)
                             const Real *dx) const override;

    virtual void retrieve_slopes(
            Vector<FArrayBox>& slopes,
            const int i,
            const int j,
            const int k) override;

    virtual Real get_max_freq(Vector<Real> &y) const override;
    virtual bool has_freq() const override {return true;}

    static bool valid_state(const int global_idx);
    static bool valid_solver(const int solve_idx);

    virtual SourceType get_type() const override {return SourceType::Braginskii;}
    virtual const std::string& get_tag() const override {return tag;}
    virtual std::string print() const override;

    static std::string tag;
    static bool registered;


    int linked_em, linked_ele, linked_ion ; //linked_hydro; replaced hydro with ele and ion
    Real Debye, Larmor;

};

#endif // MFP_BRAGINSKII_H

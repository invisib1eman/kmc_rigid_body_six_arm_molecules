//MIXING: enegies.h Energy Class (Revision Date: Feb 2, 2012)
#ifndef _ENERGY_H
#define _ENERGY_H
#include "system.h"
#include "utils.h"
#include "molecule.h"
#include "hbond.h"
#include "xyz.h"
#include "quarternion.h"
class Energy
{
public:
    double Kr=20.0;
    double Kalpha=300.0;
    double Kxhi=50.0;
    double r0=0.28;
    double fene_R0=0.58;//max_bond_extension
    double alpha0=0.0;
    double beta0=0;
    double xhi0=0.0;
    double L;//box size
    double sigma=1.8;//2*coresize
    double sigmaarm=0.28;//FENE hardcore diameter
    double epsilon=20;//LJ strength
    double feneepsilon=1;
    double bond_epsilon=10.0;
    double bond_r0=0.305638339862;
    double bond_cutoff=0.5;
    double sigma_angle=0.3;
    double sigma_torsion=0.6;
    double interaction_cutoff=0.0;

    void Configure(const System& S)
    {
        L=S.L;
        sigma=S.core_sigma;
        epsilon=S.core_epsilon;
        bond_epsilon=S.bond_epsilon;
        bond_r0=S.bond_r0;
        bond_cutoff=S.bond_cutoff;
        sigmaarm=bond_r0/pow(2.0,1.0/6.0);
        feneepsilon=S.arm_epsilon;
        sigma_angle=S.sigma_angle;
        sigma_torsion=S.sigma_torsion;
        alpha0=beta0=S.angle0;
        xhi0=S.torsion0;
        interaction_cutoff=S.interaction_cutoff;
    }

    // WCA repulsion for every tip pair, independent of alignment or occupancy.
    double tip_repulsion(double r2) const
    {
        if (r2 >= bond_r0*bond_r0) return 0.0;
        if (r2 <= 0) return numeric_limits<double>::infinity();
        double q2=bond_r0*bond_r0/r2;
        double q6=q2*q2*q2;
        return feneepsilon*(q6-1)*(q6-1);
    }

    // Attractive LJ branch with a C2 taper; the minimum remains -bond_epsilon.
    double radial_attraction(double r) const
    {
        if (r >= bond_cutoff || bond_epsilon == 0) return 0.0;
        if (r <= bond_r0) return -bond_epsilon;
        double q=bond_r0/r;
        double q6=pow(q,6);
        double t=(r-bond_r0)/(bond_cutoff-bond_r0);
        double taper=pow(1-t,3)*(1+3*t+6*t*t);
        return -bond_epsilon*(2*q6-q6*q6)*taper;
    }

    double angular_factor(double alpha, double beta) const
    {
        if (!isfinite(alpha) || !isfinite(beta)) return 0.0;
        double scaled_alpha=(alpha-alpha0)/sigma_angle;
        double scaled_beta=(beta-beta0)/sigma_angle;
        return exp(-0.5*(scaled_alpha*scaled_alpha+scaled_beta*scaled_beta));
    }

    double torsional_factor(double xhi) const
    {
        if (!isfinite(xhi)) return 0.0;
        double deviation=remainder(xhi-xhi0, 2*M_PI);
        double scaled_deviation=deviation/sigma_torsion;
        return exp(-0.5*scaled_deviation*scaled_deviation);
    }

    double patch_attraction(const Molecule& M1, const Molecule& M2, int a, int b) const
    {
        XYZ separation=real_vector(M2.ver[b]-M1.ver[a],L);
        double r=separation.norm();
        if (r >= bond_cutoff || r == 0 || bond_epsilon == 0) return 0.0;
        XYZ reverse=XYZ(-separation.x,-separation.y,-separation.z);
        double alpha=angle_vectors(M1.ver[a]-M1.centre,separation);
        double beta=angle_vectors(M2.ver[b]-M2.centre,reverse);
        XYZ axis=real_vector(M2.centre-M1.centre,L);
        // Retain the existing neighboring-arm torsion convention and preferred zero.
        double xhi=dihedral_vectors(M1.ver[neighborarm(a)]-M1.centre,axis,
                                   M2.ver[neighborarm(b)]-M2.centre);
        return radial_attraction(r)*angular_factor(alpha,beta)*torsional_factor(xhi);
    }

    double PairRepulsion(const Molecule& M1, const Molecule& M2)
    {
        if (M1.MOL_ID > M2.MOL_ID) return PairRepulsion(M2,M1);
        double r2=min_d2(M1.centre,M2.centre,L);
        if (r2 >= interaction_cutoff*interaction_cutoff) return 0.0;
        double repulsion=WCA(r2);
        for (int a=0; a<M1.N_VER; ++a)
            for (int b=0; b<M2.N_VER; ++b)
                repulsion+=tip_repulsion(min_d2(M1.ver[a],M2.ver[b],L));
        return repulsion;
    }

    // Like the patch-particle model, only the strongest patch pair contributes.
    double PairAttraction(const Molecule& M1, const Molecule& M2, hbond* contact=nullptr) const
    {
        // Canonical ordering makes tied patch choices identical in both move directions.
        if (M1.MOL_ID > M2.MOL_ID)
        {
            hbond reverse;
            double attraction=PairAttraction(M2,M1,contact ? &reverse : nullptr);
            if (contact && attraction < 0)
                contact->set(M1.MOL_ID,M2.MOL_ID,reverse.arm2,reverse.arm1);
            return attraction;
        }
        if (min_d2(M1.centre,M2.centre,L) >= interaction_cutoff*interaction_cutoff)
            return 0.0;
        double attraction=0.0;
        for (int a=0; a<M1.N_VER; ++a)
            for (int b=0; b<M2.N_VER; ++b)
            {
                double candidate=patch_attraction(M1,M2,a,b);
                if (candidate < attraction)
                {
                    attraction=candidate;
                    if (contact) contact->set(M1.MOL_ID,M2.MOL_ID,a,b);
                }
            }
        return attraction;
    }

    double PairEnergy(const Molecule& M1, const Molecule& M2)
    {
        return PairRepulsion(M1,M2)+PairAttraction(M1,M2);
    }

    // Legacy bonded helpers remain available; new MC moves use PairEnergy.
    double hbonde(Molecule M1,Molecule M2,int n)
    {
        int armindex1=M1.hbond_list[n].arm1;
        XYZ arm1=image(M1.ver[armindex1],L);
        XYZ centre1=image(M1.centre,L);
        XYZ neighborarm1=image(M1.ver[neighborarm(armindex1)],L);
        int armindex2=M1.hbond_list[n].arm2;
        XYZ arm2=image(M2.ver[armindex2],L);
        XYZ centre2=image(M2.centre,L);
        XYZ neighborarm2=image(M2.ver[neighborarm(armindex2)],L);
        double r=sqrt(min_d2(arm1,arm2,L));
        double alpha=angle_vectors(real_vector(arm1-centre1,L),real_vector(arm2-arm1,L));
        double beta=angle_vectors(real_vector(arm2-centre2,L),real_vector(arm1-arm2,L));
        double xhi=dihedral_vectors(real_vector(neighborarm1-centre1,L),real_vector(centre2-centre1,L),real_vector(neighborarm2-centre2,L));
        return 0.5*Kr*(r-r0)*(r-r0)+0.5*Kxhi*(xhi-xhi0)*(xhi-xhi0)+0.5*Kalpha*(alpha-alpha0)*(alpha-alpha0)+0.5*Kalpha*(beta-beta0)*(beta-beta0);
        
    };
    double hbonde_fene(Molecule M1,Molecule M2,int armindex1,int armindex2)
    {
        
        XYZ arm1=image(M1.ver[armindex1],L);
        
        
        XYZ arm2=image(M2.ver[armindex2],L);
        double wca=0;
        double fene=0;
        double r=sqrt(min_d2(arm1,arm2,L));
        double r2=r*r;
        double nr2=r2/(sigmaarm*sigmaarm);
        double nr2cut=1.12*1.12;
        double nr6=nr2*nr2*nr2;
        double nr6cut=nr2cut*nr2cut*nr2cut;
        if(nr2<nr2cut)//2^(1/6)=1.12
        {
            
            wca=4*feneepsilon*(1/(nr6*nr6)-(1/nr6))-4*feneepsilon*(1/(nr6cut*nr6cut)-(1/nr6cut));
        }
        else
        {
            
            
            wca=0;
        }
        if (r>=fene_R0)
        {
            fene=numeric_limits<double>::infinity();//outside the legacy FENE domain
        }
        else
        {
            
            fene=-0.5*Kr*fene_R0*fene_R0*log(1-(r/fene_R0)*(r/fene_R0));
        }
        return wca+fene;
        
    };
    double hbonde_angle(Molecule M1,Molecule M2,int armindex1,int armindex2)
    {
        
        XYZ arm1=image(M1.ver[armindex1],L);
        XYZ centre1=image(M1.centre,L);
        
        
        XYZ arm2=image(M2.ver[armindex2],L);
        XYZ centre2=image(M2.centre,L);
        
       
        
    
        
        double alpha=angle_vectors(real_vector(arm1-centre1,L),real_vector(arm2-arm1,L));
        double beta=angle_vectors(real_vector(arm2-centre2,L),real_vector(arm1-arm2,L));
            
        return 0.5*Kalpha*(alpha-alpha0)*(alpha-alpha0)+0.5*Kalpha*(beta-beta0)*(beta-beta0);
        
    }
    double hbonde_dihedral(Molecule M1,Molecule M2,int armindex1,int armindex2)
    {
        
        XYZ centre1=image(M1.centre,L);
        XYZ neighborarm1=image(M1.ver[neighborarm(armindex1)],L);
        
        XYZ centre2=image(M2.centre,L);
        XYZ neighborarm2=image(M2.ver[neighborarm(armindex2)],L);
        double xhi=dihedral_vectors(real_vector(neighborarm1-centre1,L),real_vector(centre2-centre1,L),real_vector(neighborarm2-centre2,L));
        return 0.5*Kxhi*(xhi-xhi0)*(xhi-xhi0);
        
    }
    double LJ(double r2)
    {
        double nr2=r2/(sigma*sigma);
        double nr6=nr2*nr2*nr2;
        return 4*epsilon*(1/(nr6*nr6)-(1/nr6));

    };
    double WCA(double r2)
    {
        if (epsilon == 0) return 0.0;
        if (r2 >= pow(2.0,1.0/3.0)*sigma*sigma) return 0.0;
        if (r2 <= 0) return numeric_limits<double>::infinity();
        double q2=sigma*sigma/r2;
        double q6=q2*q2*q2;
        return 4*epsilon*(q6-0.5)*(q6-0.5);
    };

};
#endif


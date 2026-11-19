//NANOROD: system.h System Class (Revision Date: Oct 27,2023)
#ifndef _SYSTEM_H
#define _SYSTEM_H
#include "header.h"
#include "xyz.h"
#include "quarternion.h"
#include "molecule.h"
#include "utils.h"
#include "grid.h"
#include "aggregate.h"
class System
{
  public:
    vector<Molecule> M; //List of molecules
    vector<Grid> G;//grid list
    list<hbond> H;//hbond list
    //list<Aggregate> Ag;//list of aggregates
    
    int NGRID,NGRID3;//NGRID=#cell in one direction, NGRID3=total # of cells of grid, NCELL=#particles per cell
    double NCELL;
    double GRIDL;//Grid length in one direction
    const gsl_rng_type * gsl_T=nullptr;
    gsl_rng * gsl_r=nullptr;
    string Description;
    string initial_state_file; //Optional rigid-body snapshot input
    int NMOL; //Number of molecules
    double coresize=0.2;
    double arm_L=1.1;
    double bond_length=0.3;//minimum of fene potential, see energy.h
    double bond_extension=0.2;//max init energy is 6.4kT,see energy.h
    double searchl2_bond=pow(bond_length+bond_extension,2);//the length range of forming a bond
    double cm_L=arm_L*2+bond_length;
    double search2_cm=pow(cm_L+bond_extension,2);//the cm distance when it is possible to form a bond
    double search_angle=0.3;//max init energy is 13.5kT at Kalpha=300
    double search_xhi=0.3;
    double L; //Length of box
    int GSL_SEED; //Seed of random number generator
    int nsweep; //Number of MC sweeps
    double deltat; //Timestep
    double MCstep; //Step size of translation
    double MCstep_rotation;//Maximum trial rotation, in radians
    double core_sigma=1.8;
    double core_epsilon=20.0;
    double bond_epsilon=10.0;//Depth of the complete double-hydrogen-bond contact
    double bond_r0=0.305638339862;
    double bond_cutoff=0.5;
    double arm_epsilon=1.0;//Orientation-independent tip repulsion
    double sigma_angle=0.3;
    double sigma_torsion=0.6;
    double angle0=0.0;
    double torsion0=0.0;
    double interaction_cutoff;//Maximum center separation requiring evaluation
    bool single_partner=true;//Hard valence constraint on sufficiently strong contacts
    double contact_fraction=0.5;//Attraction magnitude / bond_epsilon defining a contact
    double E_1=10;//hbond dis enthalpy
    double free_bond_freeenergy=-1;//free bond entropy
    double omega_B=0.001;//arrhenius prefactor
    double omega_T=0.1;//frequency of change types
    void ReadInput(int argc, char *argv[])
    {
        double total_time;
        
        options_description desc("Usage:\nNANOROD <options>");
    
        desc.add_options()
        ("help,h", "print usage message")
        ("initial_state", value<string>(&initial_state_file)->default_value(""), "rigid-body snapshot file (sets NMOL and box_length)")
        ("NGRID,G",value<int>(&NGRID)->default_value(10),"grids(default 10)")
        ("NMOL,N", value<int>(&NMOL)->default_value(400), "#molecules (default 400)")
        ("box_length,L", value<double>(&L)->default_value(20.0), "length of box (default 20.0)")
        ("time,s", value<double>(&total_time)->default_value(100.0), "total time in tau_0 units (default 100.0)")
        ("MCstep,m", value<double>(&MCstep)->default_value(0.1), "translation step in nm")
        ("rotation_step", value<double>(&MCstep_rotation)->default_value(0.1), "maximum rotation step in radians")
        ("core_sigma", value<double>(&core_sigma)->default_value(1.8), "core WCA diameter in nm")
        ("core_epsilon", value<double>(&core_epsilon)->default_value(20.0), "core WCA strength in kBT")
        ("bond_epsilon", value<double>(&bond_epsilon)->default_value(10.0), "full COOH dimer well depth in kBT")
        ("bond_r0", value<double>(&bond_r0)->default_value(0.305638339862), "preferred arm-tip separation in nm")
        ("bond_cutoff", value<double>(&bond_cutoff)->default_value(0.5), "arm-tip attraction cutoff in nm")
        ("arm_epsilon", value<double>(&arm_epsilon)->default_value(1.0), "tip WCA repulsion strength in kBT")
        ("sigma_angle", value<double>(&sigma_angle)->default_value(0.3), "Gaussian alignment width in radians")
        ("sigma_torsion", value<double>(&sigma_torsion)->default_value(0.6), "Gaussian torsional width in radians")
        ("single_partner", value<bool>(&single_partner)->default_value(true), "reject multiple bound partners per arm (0 or 1)")
        ("contact_fraction", value<double>(&contact_fraction)->default_value(0.5), "minimum fraction of well depth defining a bound contact, (0,1)")
        ("angle0", value<double>(&angle0)->default_value(0.0), "preferred alignment deviation in radians")
        ("torsion0", value<double>(&torsion0)->default_value(0.0), "preferred torsion in the existing convention, radians")
        ("GSL_SEED,g", value<int>(&GSL_SEED)->default_value(10), "seed for the RNG (default 10)")
        ("Description,D", value<string>(&Description)->default_value("nanorod"), "Description (default nanorod)");

        
        variables_map vm;
        store(parse_command_line(argc, argv, desc), vm);
        notify(vm);
        
        if (vm.count("help"))
        {
            cout << desc << "\n";
            exit(0);
        }
        if (!initial_state_file.empty())
        {
            int snapshot_molecules;
            double snapshot_length;
            ReadInitialStateHeader(snapshot_molecules,snapshot_length);
            if (!vm["NMOL"].defaulted() && NMOL != snapshot_molecules)
                throw invalid_argument("NMOL conflicts with initial_state molecule count.");
            if (!vm["box_length"].defaulted() &&
                fabs(L-snapshot_length) > 1e-10*std::max(1.0,fabs(snapshot_length)))
                throw invalid_argument("box_length conflicts with initial_state box length.");
            NMOL=snapshot_molecules;
            L=snapshot_length;
        }
        if (NMOL < 1 || NGRID < 1 || NGRID > 1000 ||
            !isfinite(L) || L <= 0 || !isfinite(total_time) || total_time < 0 ||
            !isfinite(MCstep) || MCstep <= 0 ||
            !isfinite(MCstep_rotation) || MCstep_rotation < 0 || MCstep_rotation > M_PI ||
            !isfinite(core_sigma) || core_sigma <= 0 ||
            !isfinite(core_epsilon) || core_epsilon < 0 ||
            !isfinite(bond_epsilon) || bond_epsilon < 0 ||
            !isfinite(arm_epsilon) || arm_epsilon <= 0 ||
            !isfinite(bond_r0) || bond_r0 <= 0 ||
            !isfinite(bond_cutoff) || bond_cutoff <= bond_r0 ||
            !isfinite(sigma_angle) || sigma_angle <= 0 ||
            !isfinite(sigma_torsion) || sigma_torsion <= 0 ||
            !isfinite(contact_fraction) || contact_fraction <= 0 || contact_fraction >= 1 ||
            !isfinite(angle0) || angle0 < 0 || angle0 > M_PI || !isfinite(torsion0))
            throw invalid_argument("Invalid simulation or energy parameter; see --help for units and ranges.");

        Molecule reference;
        double longest_arm=0;
        for (const XYZ& arm : reference.aorigin)
            longest_arm=std::max(longest_arm, arm.norm());
        interaction_cutoff=std::max(pow(2.0,1.0/6.0)*core_sigma, 2*longest_arm+bond_cutoff);
        if (L <= 2*interaction_cutoff)
            throw invalid_argument("box_length must exceed twice the maximum center interaction cutoff.");
        deltat=MCstep*MCstep/12.0;
        if (!isfinite(deltat) || deltat <= 0 || total_time/deltat > numeric_limits<int>::max())
            throw invalid_argument("Requested time and MCstep give an invalid timestep or too many sweeps.");
        gsl_rng_env_setup();
          
        gsl_T = gsl_rng_default;
        gsl_r = gsl_rng_alloc(gsl_T);
        gsl_rng_set(gsl_r, GSL_SEED);
        
        nsweep=int(ceil(total_time/deltat));
        NGRID3=NGRID*NGRID*NGRID;
        
    try
    {
      G.reserve(NGRID3);
    }
    catch (int e)
    {
	cout<<"Memory issues in cell list allocation.. exiting"<<endl;
	exit(1);
    }
        GRIDL=L/NGRID;
        NCELL=double(NMOL)/NGRID3;
    }
    
    void Create();
    void ReadInitialStateHeader(int& molecule_count, double& box_length) const;
    void LoadInitialState();
    void WriteMol2(int timestep);
    void WriteDump(int timestep);
    void WriteBond(int timestep);
    void WriteOrientation(int timestep);
    void WriteGrid(int timestep);
    void UpdateGrid();
};
#endif

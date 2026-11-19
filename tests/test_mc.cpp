// Grid, contact bookkeeping, and incremental-energy integration checks.
#include "mc.h"
#include <cassert>
#include <iostream>
#include <set>
#include <tuple>

namespace
{
typedef std::tuple<int,int,int,int> Contact;

void expect_close(double actual, double expected)
{
    assert(std::isfinite(actual));
    assert(std::abs(actual-expected)<1e-8*(1.0+std::abs(expected)));
}

void initialize(MC& mc, int grids, int molecules)
{
    System& system=mc.S;
    system.L=12.0;
    system.NGRID=grids;
    system.NGRID3=grids*grids*grids;
    system.GRIDL=system.L/grids;
    system.NMOL=molecules;
    system.MCstep=0.3;
    system.MCstep_rotation=0.2;
    system.interaction_cutoff=2.0*Molecule().aorigin[0].norm()+system.bond_cutoff;
    system.gsl_r=gsl_rng_alloc(gsl_rng_mt19937);
    gsl_rng_set(system.gsl_r,7141);
    system.Create();
    mc.E.Configure(system);
}

void assign_cells(System& system)
{
    for (Grid& cell : system.G)
    {
        cell.plist.clear();
        cell.n=0;
    }
    for (int i=0;i<system.NMOL;++i)
    {
        Molecule& molecule=system.M[i];
        molecule.MOL_ID=i;
        molecule.centre=image(molecule.centre,system.L);
        molecule.UpdateVertices();
        molecule.gID=GridIndex_xyz(molecule.centre,system.NGRID,system.GRIDL,system.L);
        system.G[molecule.gID].plist.push_back(i);
        ++system.G[molecule.gID].n;
    }
}

void place_dimer(MC& mc)
{
    Molecule first,second;
    first.centre=XYZ(5.8,0.0,0.0);
    XYZ axis=first.aorigin[0]/first.aorigin[0].norm();
    double separation=2.0*first.aorigin[0].norm()+mc.S.bond_r0;
    second.centre=first.centre+XYZ(axis.x*separation,axis.y*separation,axis.z*separation);
    second.orientation=angle_to_quarternion(M_PI,M_PI/2.0,M_PI/2.0);
    mc.S.M[0]=first;
    mc.S.M[1]=second;
}

double all_pairs_energy(MC& mc)
{
    double result=0.0;
    for (int i=0;i<mc.S.NMOL;++i)
        for (int j=i+1;j<mc.S.NMOL;++j)
            result+=mc.E.PairEnergy(mc.S.M[i],mc.S.M[j]);
    return result;
}

void check_grid_and_contacts(MC& mc)
{
    System& system=mc.S;
    std::vector<int> seen(system.NMOL,0);
    for (const Grid& cell : system.G)
    {
        std::set<int> unique_cells(cell.nbr.begin(),cell.nbr.end());
        assert(unique_cells.size()==cell.nbr.size());
        assert(cell.n==static_cast<int>(cell.plist.size()));
        for (int i : cell.plist)
        {
            assert(system.M[i].gID==cell.g_index);
            ++seen[i];
        }
    }
    for (int count : seen) assert(count==1);
    expect_close(mc.TotalEnergy(),all_pairs_energy(mc));

    std::set<Contact> expected, actual;
    for (int i=0;i<system.NMOL;++i)
        for (int j=i+1;j<system.NMOL;++j)
        {
            hbond contact;
            if (mc.IsContact(mc.E.PairAttraction(system.M[i],system.M[j],&contact)))
                expected.insert(Contact(i,j,contact.arm1,contact.arm2));
        }
    for (const hbond& contact : system.H)
        actual.insert(Contact(contact.M1,contact.M2,contact.arm1,contact.arm2));
    assert(actual==expected);
    assert(actual.size()==system.H.size());
    for (int i=0;i<system.NMOL;++i)
    {
        const Molecule& molecule=system.M[i];
        assert(molecule.nbonds==static_cast<int>(molecule.hbond_list.size()));
        std::vector<int> counts(molecule.N_VER,0);
        std::set<Contact> local;
        for (const hbond& contact : molecule.hbond_list)
        {
            assert(contact.M1==i);
            ++counts[contact.arm1];
            Contact canonical=i<contact.M2 ? Contact(i,contact.M2,contact.arm1,contact.arm2)
                                          : Contact(contact.M2,i,contact.arm2,contact.arm1);
            assert(expected.count(canonical)==1);
            local.insert(canonical);
        }
        size_t expected_count=0;
        for (const Contact& contact : expected)
            if (std::get<0>(contact)==i || std::get<1>(contact)==i) ++expected_count;
        assert(local.size()==expected_count);
        for (int arm=0;arm<molecule.N_VER;++arm)
        {
            assert(molecule.vertype[arm]==(counts[arm]>0 ? 'I' : 'A'));
            if (system.single_partner) assert(counts[arm]<=1);
        }
    }
}

void test_grids_and_moves()
{
    int crossings=0;
    const int grid_sizes[]={1,2,3,6,8,12};
    for (int grids : grid_sizes)
    {
        MC mc;
        initialize(mc,grids,5);
        place_dimer(mc);
        mc.S.M[2].centre=XYZ(-2.0,-3.0,-3.0);
        mc.S.M[3].centre=XYZ(-0.05,-3.0,-3.0);
        mc.S.M[4].centre=XYZ(3.0,4.0,4.0);
        assign_cells(mc.S);
        mc.RebuildContacts();
        std::vector<std::vector<int> > neighbors;
        for (const Grid& cell : mc.S.G) neighbors.push_back(cell.nbr);
        mc.S.UpdateGrid();
        for (size_t i=0;i<mc.S.G.size();++i) assert(mc.S.G[i].nbr==neighbors[i]);
        assert(mc.S.H.size()==1);
        expect_close(mc.bond_energy(),-mc.S.E_1);
        expect_close(mc.bond_freeze_freenerngy(),-4.0*mc.S.free_bond_freeenergy);
        mc.energy=mc.TotalEnergy();
        check_grid_and_contacts(mc);
        for (int sweep=0;sweep<80;++sweep)
        {
            std::vector<int> old_cells;
            for (const Molecule& molecule : mc.S.M) old_cells.push_back(molecule.gID);
            double accepted=mc.MoveMolecule();
            assert(accepted>=0.0 && accepted<=1.0);
            for (int i=0;i<mc.S.NMOL;++i)
                crossings+=(old_cells[i]!=mc.S.M[i].gID);
            expect_close(mc.energy,mc.TotalEnergy());
            check_grid_and_contacts(mc);
        }
        gsl_rng_free(mc.S.gsl_r);
    }
    assert(crossings>0);
}

void test_saturation_during_moves()
{
    int maximum_unrestricted=0;
    for (bool constrained : {true,false})
    {
        MC mc;
        initialize(mc,3,3);
        mc.S.single_partner=constrained;
        mc.S.MCstep=0.008;
        mc.S.MCstep_rotation=0.0;
        // Weak repulsion makes competing contacts accessible in this constraint test.
        mc.S.core_epsilon=0.0;
        mc.S.arm_epsilon=1e-16;
        mc.S.bond_epsilon=1e-3;
        mc.S.sigma_angle=0.8;
        mc.E.Configure(mc.S);
        place_dimer(mc);
        XYZ axis=mc.S.M[0].aorigin[0]/mc.S.M[0].aorigin[0].norm();
        double offset=0.38-mc.S.bond_r0;
        mc.S.M[2]=mc.S.M[1];
        mc.S.M[1].centre=mc.S.M[1].centre+XYZ(0.0,0.02,0.0);
        mc.S.M[2].centre=mc.S.M[2].centre+XYZ(axis.x*offset,-0.02,axis.z*offset);
        assign_cells(mc.S);
        mc.RebuildContacts();
        assert(mc.S.H.size()==1);
        mc.energy=mc.TotalEnergy();
        for (int sweep=0;sweep<400;++sweep)
        {
            mc.MoveMolecule();
            expect_close(mc.energy,mc.TotalEnergy());
            check_grid_and_contacts(mc);
            if (!constrained)
                for (const Molecule& molecule : mc.S.M)
                {
                    std::vector<int> counts(molecule.N_VER,0);
                    for (const hbond& contact : molecule.hbond_list)
                        maximum_unrestricted=std::max(maximum_unrestricted,++counts[contact.arm1]);
                }
        }
        gsl_rng_free(mc.S.gsl_r);
    }
    assert(maximum_unrestricted>=2);
}

void test_shared_arm_contacts()
{
    MC mc;
    initialize(mc,3,3);
    place_dimer(mc);
    mc.S.M[2]=mc.S.M[1];
    mc.S.M[1].centre=mc.S.M[1].centre+XYZ(0.0,0.02,0.0);
    mc.S.M[2].centre=mc.S.M[2].centre+XYZ(0.0,-0.02,0.0);
    assign_cells(mc.S);
    bool rejected=false;
    try { mc.RebuildContacts(); }
    catch (const std::runtime_error&) { rejected=true; }
    assert(rejected);
    mc.S.single_partner=false;
    mc.RebuildContacts();
    assert(mc.S.H.size()==2);
    assert(mc.S.M[0].nbonds==2);
    check_grid_and_contacts(mc);

    // Removing one partner must leave the shared arm occupied by the other.
    mc.S.M[1].centre=XYZ(-4.0,-4.0,-4.0);
    assign_cells(mc.S);
    mc.UpdateContacts(1,std::vector<hbond>());
    assert(mc.S.M[0].vertype[0]=='I');
    assert(mc.S.M[0].nbonds==1);
    check_grid_and_contacts(mc);
    mc.S.M[2].centre=XYZ(0.0,-4.0,-4.0);
    assign_cells(mc.S);
    mc.UpdateContacts(2,std::vector<hbond>());
    assert(mc.S.M[0].vertype[0]=='A');
    assert(mc.S.H.empty());
    check_grid_and_contacts(mc);
    expect_close(mc.bond_energy(),0.0);
    expect_close(mc.bond_freeze_freenerngy(),0.0);
    gsl_rng_free(mc.S.gsl_r);
}
}

int main()
{
    test_grids_and_moves();
    test_shared_arm_contacts();
    test_saturation_during_moves();
    MC mc;
    assert(!mc.Glauber(std::numeric_limits<double>::quiet_NaN(),0.0));
    assert(!mc.Glauber(1000.0,0.1));
    assert(mc.Glauber(-1000.0,0.9));
    std::ifstream events("events.txt");
    std::string first_event;
    if (!std::getline(events,first_event))
    {
        std::cerr << "MC tests could not read the expected contact-event output.\n";
        return 1;
    }
    std::cout << "MC integration tests passed.\n";
}

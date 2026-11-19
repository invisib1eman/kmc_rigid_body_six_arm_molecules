// Deterministic checks for the nonbonded six-arm pair potential.
#include "energies.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>

namespace
{
void expect_close(double actual, double expected, double tolerance=1e-9)
{
    assert(std::isfinite(actual));
    assert(std::abs(actual-expected) <= tolerance*(1.0+std::abs(expected)));
}

System parameters()
{
    System system;
    system.L=20.0;
    system.core_sigma=1.75;
    system.core_epsilon=12.0;
    system.bond_epsilon=7.5;
    system.bond_r0=0.31;
    system.bond_cutoff=0.55;
    system.arm_epsilon=1.3;
    system.sigma_angle=0.21;
    system.sigma_torsion=0.42;
    double arm_length=Molecule().aorigin[0].norm();
    system.interaction_cutoff=2.0*arm_length+system.bond_cutoff;
    return system;
}

void ideal_dimer(const Energy& energy, Molecule& first, Molecule& second)
{
    first.MOL_ID=0;
    second.MOL_ID=1;
    first.UpdateVertices();
    XYZ axis=first.aorigin[0]/first.aorigin[0].norm();
    double separation=2.0*first.aorigin[0].norm()+energy.bond_r0;
    second.centre=XYZ(axis.x*separation,axis.y*separation,axis.z*separation);
    second.orientation=angle_to_quarternion(M_PI,M_PI/2.0,M_PI/2.0);
    second.UpdateVertices();
}

void rotate_molecule(Molecule& molecule, quarternion rotation)
{
    molecule.centre=quarterrotation(molecule.centre,rotation);
    molecule.orientation=quartermulti(rotation,molecule.orientation);
    molecule.UpdateVertices();
}

void test_radial_and_gates(Energy& energy)
{
    expect_close(energy.radial_attraction(energy.bond_r0),-energy.bond_epsilon);
    expect_close(energy.tip_repulsion(energy.bond_r0*energy.bond_r0),0.0);
    assert(energy.tip_repulsion(pow(0.9*energy.bond_r0,2))>0.0);
    assert(std::isinf(energy.tip_repulsion(0.0)));
    assert(energy.radial_attraction(1.05*energy.bond_r0)>-energy.bond_epsilon);
    expect_close(energy.radial_attraction(energy.bond_cutoff),0.0);
    expect_close(energy.radial_attraction(energy.bond_cutoff+0.1),0.0);

    // Both the energy and its left derivative approach zero at the cutoff.
    double h=(energy.bond_cutoff-energy.bond_r0)*1e-4;
    double near_cutoff=energy.radial_attraction(energy.bond_cutoff-h);
    assert(std::abs(near_cutoff)<1e-8*energy.bond_epsilon);
    assert(std::abs(near_cutoff/h)<1e-5*energy.bond_epsilon);
    expect_close(energy.angular_factor(0.0,0.0),1.0);
    expect_close(energy.angular_factor(energy.sigma_angle,0.0),std::exp(-0.5));
    expect_close(energy.angular_factor(energy.sigma_angle,energy.sigma_angle),std::exp(-1.0));
    expect_close(energy.torsional_factor(energy.sigma_torsion),std::exp(-0.5));
    energy.xhi0=2.9;
    expect_close(energy.torsional_factor(2.9),1.0);
    expect_close(energy.torsional_factor(3.1),energy.torsional_factor(3.1-2*M_PI));
    expect_close(energy.torsional_factor(-2.9),energy.torsional_factor(-2.9+4*M_PI));
    energy.xhi0=0.0;
    double undefined=std::numeric_limits<double>::quiet_NaN();
    expect_close(energy.angular_factor(undefined,0.0),0.0);
    expect_close(energy.torsional_factor(undefined),0.0);
}

void test_pair_geometry(Energy& energy)
{
    Molecule first, second;
    ideal_dimer(energy,first,second);
    hbond contact;
    expect_close(energy.PairAttraction(first,second,&contact),-energy.bond_epsilon);
    assert(contact.arm1==0 && contact.arm2==0);
    expect_close(energy.PairRepulsion(first,second),0.0);
    expect_close(energy.PairEnergy(first,second),-energy.bond_epsilon);
    expect_close(energy.PairEnergy(first,second),energy.PairEnergy(second,first));

    Molecule tilted=second;
    tilted.orientation=quartermulti(angle_to_quarternion(0.15,0.0,0.0),tilted.orientation);
    tilted.UpdateVertices();
    assert(energy.PairAttraction(first,tilted)>energy.PairAttraction(first,second));

    // Twist about the contacting arm keeps its tip position and alignment fixed.
    XYZ axis=first.aorigin[0]/first.aorigin[0].norm();
    Molecule twisted=second;
    quarternion twist=angle_to_quarternion(energy.sigma_torsion,std::acos(axis.z),std::atan2(axis.y,axis.x));
    twisted.orientation=quartermulti(twist,twisted.orientation);
    twisted.UpdateVertices();
    expect_close(energy.PairAttraction(first,twisted),-energy.bond_epsilon*std::exp(-0.5));
    expect_close(energy.PairEnergy(first,twisted),energy.PairEnergy(twisted,first));

    Molecule compressed=second;
    compressed.centre=compressed.centre-XYZ(axis.x*0.05,axis.y*0.05,axis.z*0.05);
    compressed.UpdateVertices();
    double repulsion=energy.PairRepulsion(first,compressed);
    assert(repulsion>0.0);
    Energy different_angles=energy;
    different_angles.sigma_angle*=0.5;
    different_angles.sigma_torsion*=0.5;
    different_angles.alpha0=0.4;
    different_angles.xhi0=0.8;
    expect_close(different_angles.PairRepulsion(first,compressed),repulsion);
    assert(different_angles.PairAttraction(first,compressed)>energy.PairAttraction(first,compressed));

    Molecule wrapped=twisted;
    wrapped.centre=wrapped.centre+XYZ(4*energy.L,-3*energy.L,2*energy.L);
    wrapped.UpdateVertices();
    expect_close(energy.PairEnergy(first,wrapped),energy.PairEnergy(first,twisted));
    quarternion common_rotation=angle_to_quarternion(0.73,1.1,-0.4);
    Molecule rotated_first=first, rotated_second=twisted;
    rotate_molecule(rotated_first,common_rotation);
    rotate_molecule(rotated_second,common_rotation);
    expect_close(energy.PairEnergy(rotated_first,rotated_second),energy.PairEnergy(first,twisted));

    double original=energy.PairEnergy(first,second);
    first.hbond_list.push_back(hbond(0,1,0,0));
    second.hbond_list.push_back(hbond(1,0,0,0));
    first.nbonds=second.nbonds=1;
    first.vertype.assign(first.N_VER,'I');
    second.vertype.assign(second.N_VER,'I');
    expect_close(energy.PairEnergy(first,second),original);
    second.centre=XYZ(energy.interaction_cutoff+0.1,0.0,0.0);
    second.UpdateVertices();
    expect_close(energy.PairEnergy(first,second),0.0);
}
}

int main()
{
    System system=parameters();
    Energy energy;
    energy.Configure(system);
    expect_close(energy.L,system.L);
    expect_close(energy.interaction_cutoff,system.interaction_cutoff);
    expect_close(energy.bond_epsilon,system.bond_epsilon);
    expect_close(energy.bond_r0,system.bond_r0);
    expect_close(energy.bond_cutoff,system.bond_cutoff);
    expect_close(energy.sigma,system.core_sigma);
    expect_close(energy.epsilon,system.core_epsilon);
    expect_close(energy.feneepsilon,system.arm_epsilon);
    expect_close(energy.sigma_angle,system.sigma_angle);
    expect_close(energy.sigma_torsion,system.sigma_torsion);
    System preferred=system;
    preferred.angle0=0.13;
    preferred.torsion0=-0.4;
    Energy preferred_energy;
    preferred_energy.Configure(preferred);
    expect_close(preferred_energy.angular_factor(0.13,0.13),1.0);
    expect_close(preferred_energy.torsional_factor(-0.4),1.0);
    preferred_energy.sigma_angle=1e-200;
    preferred_energy.sigma_torsion=1e-200;
    expect_close(preferred_energy.angular_factor(0.13,0.13),1.0);
    expect_close(preferred_energy.angular_factor(0.14,0.13),0.0);
    expect_close(preferred_energy.torsional_factor(-0.4),1.0);
    expect_close(preferred_energy.torsional_factor(-0.3),0.0);
    test_radial_and_gates(energy);
    test_pair_geometry(energy);
    system.H.push_back(hbond(0,1,0,0));
    energy.Configure(system);
    test_pair_geometry(energy);
    system.bond_epsilon=0.0;
    energy.Configure(system);
    Molecule first, second;
    ideal_dimer(energy,first,second);
    expect_close(energy.radial_attraction(system.bond_r0),0.0);
    expect_close(energy.PairAttraction(first,second),0.0);
    std::cout << "Nonbonded energy tests passed.\n";
}

//NANOROD: mc.cpp MC Class Function Definitions (Revision Date: October 27,2023)
#include "mc.h" 
void MC::Sweep()
{
    Mnew=S.M;
    E.Configure(S);
    RebuildContacts();
    S.WriteMol2(0);
    S.WriteDump(0);
    S.WriteBond(0);
    double accept=0.0;
   
    int nsample=S.nsweep/10000;
    if(nsample<1)
      nsample=1;
    
    energy=TotalEnergy();
    WriteEnergy(0);
   
    
    WriteTemplate();
	LogProfile(0,accept);
    
    for(int i=1; i<=S.nsweep; i++)
    {
        time+=S.deltat;
        accept+=MoveMolecule();
        
        if(i%nsample==0 || i==S.nsweep)
        {
            
            // Periodic recomputation prevents accumulated floating-point energy drift.
            energy=TotalEnergy();
            int sampled_sweeps=i%nsample==0 ? nsample : i%nsample;
            LogProfile(i,accept/sampled_sweeps);
            WriteEnergy(i);
            S.WriteMol2(i);
            S.WriteDump(i);
            S.WriteBond(i);
            S.WriteGrid(i);
            accept=0.0;
        }
    }
	
   

}

void MC::WriteTemplate()
{
    ofstream out;
    out.open("_MC.log");
    out<<setprecision(16);
    out<<"# energy units: kBT; lengths: nm; angles: radians"<<endl;
    out<<"# core_sigma="<<S.core_sigma<<" core_epsilon="<<S.core_epsilon
       <<" bond_epsilon="<<S.bond_epsilon<<" bond_r0="<<S.bond_r0
       <<" bond_cutoff="<<S.bond_cutoff<<" arm_epsilon="<<S.arm_epsilon
       <<" sigma_angle="<<S.sigma_angle<<" sigma_torsion="<<S.sigma_torsion
       <<" angle0="<<S.angle0<<" torsion0="<<S.torsion0
       <<" single_partner="<<S.single_partner<<" contact_fraction="<<S.contact_fraction
       <<" MCstep="<<S.MCstep<<" rotation_step="<<S.MCstep_rotation
       <<" seed="<<S.GSL_SEED<<endl;
    out<<setw(12)<<"sweep"<<"\t"<<setw(12)<<"time"<<"\t"<<setw(12)<<"energy"<<"\t"<<setw(8)<<"accept"<<endl;
    out.close();
}

void MC::LogProfile(int i, double accept)
{
    ofstream out;
    out.open("_MC.log", ios::app);
    out<<setprecision(16)<<setw(12)<<i<<"\t"<<setw(12)<<time<<"\t"<<setw(12)<<energy<<"\t"<<setw(8)<<accept<<endl;
    
    out.close();
}



//Returns acceptance fraction
double MC::MoveMolecule()
{
    //S.UpdateGrid();
    //check if the total particle in the celllists add to N
    int total_particles=0;

    for(int i=0;i<S.NGRID3;i++)
    {
        if(S.G[i].n!=S.G[i].plist.size())
        {
            throw runtime_error("Cell-list particle count is inconsistent.");
        }
        total_particles+=S.G[i].n;
    }
    if(total_particles!=S.NMOL)
    {
            throw runtime_error("Cell list does not contain NMOL particles.");
    }
    ofstream out;
    out.open("events.txt",ios::app);

    //ofstream out2;
    //out2.open("bond_energy.txt",ios::app);
    double accept=0.0;//accept events
    
    int index;
    double delta;
    // Contacts now form and disappear through the nonbonded energy and geometry.
    // The old Arrhenius break loop (including its stray-semicolon bug) is obsolete.
    //diffusion rotation
    //diffusion translation
    //diffusion limited bond formation
    for(int i=0; i<S.NMOL; i++)
    {
        index=gsl_rng_uniform_int(S.gsl_r,S.NMOL);
        
        delta=0;//energy difference
        //int index=sequence_M[i];//index of the current trial molecule
        
        
        
       
        vector<hbond> old_hbondlist=S.M[index].hbond_list;
        Molecule newmolecule=S.M[index];
        newmolecule.centre=RandomTranslate(S.M[index].centre,S.MCstep,gsl_rng_uniform(S.gsl_r),gsl_rng_uniform(S.gsl_r));
        XYZ image_center=image(newmolecule.centre,S.L);
        int new_gID=GridIndex_xyz(image_center,S.NGRID,S.GRIDL,S.L);
        newmolecule.gID=new_gID;
        newmolecule.centre=image_center;
        
        newmolecule.orientation=RandomRotate(S.M[index].orientation,S.MCstep_rotation/(2*M_PI),gsl_rng_uniform(S.gsl_r),gsl_rng_uniform(S.gsl_r),gsl_rng_uniform(S.gsl_r));
        newmolecule.UpdateVertices();
  
        vector<hbond> new_hbondlist;
        bool valid_contacts=true;
        vector<int> contacts_per_arm(newmolecule.N_VER,0);
        
        //Evaluate the full nonbonded interaction before and after the trial.
        
        //Add trial pair energies, including previously unbound neighbors.
        for(size_t l=0;l<S.G[new_gID].nbr.size();l++)
        {
            //ID of new neighboring grids
            int temp_gID=S.G[new_gID].nbr[l];
            if(temp_gID<0 || temp_gID>=S.NGRID3)
            {
                throw runtime_error("Neighbor cell index is out of range.");
            }
            if(S.G[temp_gID].plist.empty())
            {
                continue;
            }
            //iterate about molecules in the neighborlist
            list<int>::iterator it;
            for(it=S.G[temp_gID].plist.begin();it!=S.G[temp_gID].plist.end();it++)
            {
                int j=*it;
                if(j!=index)
                {
                    hbond contact;
                    double attraction=E.PairAttraction(newmolecule,S.M[j],&contact);
                    delta+=E.PairRepulsion(newmolecule,S.M[j])+attraction;
                    if (IsContact(attraction))
                    {
                        new_hbondlist.push_back(contact);
                        if (++contacts_per_arm[contact.arm1] > 1 && S.single_partner)
                            valid_contacts=false;
                        if (S.single_partner)
                            for (const hbond& other : S.M[j].hbond_list)
                                if (other.M2 != index && other.arm1 == contact.arm2)
                                    valid_contacts=false;
                    }


                }
            }
        }
        //Subtract old pair energies over the old cell neighborhood.
        int old_gID=S.M[index].gID;
        for(size_t l=0;l<S.G[old_gID].nbr.size();l++)
        {
            //ID of new neighboring grids
            int temp_gID=S.G[old_gID].nbr[l];
            list<int>::iterator it;
            for(it=S.G[temp_gID].plist.begin();it!=S.G[temp_gID].plist.end();it++)
            {
                int j=*it;
                if(j!=index)
                {
                    //image distance
                    delta-=E.PairEnergy(S.M[index],S.M[j]);
                    


                }
            }
        }
        if(valid_contacts && Glauber(delta,gsl_rng_uniform(S.gsl_r)))
        {
            //Accept move
            S.M[index]=newmolecule;
    
            accept+=1.0;
            energy+=delta;

            //update grid list
            if(new_gID!=old_gID)
            {
                S.G[old_gID].n-=1;
                S.G[old_gID].plist.remove(S.M[index].MOL_ID);
                
                S.G[new_gID].n+=1;
                S.G[new_gID].plist.push_back(S.M[index].MOL_ID);
            }
        
            UpdateContacts(index,new_hbondlist);
            for (const hbond& previous : old_hbondlist)
            {
                bool retained=false;
                for (const hbond& current : new_hbondlist)
                    if (previous.M2==current.M2 && previous.arm1==current.arm1 && previous.arm2==current.arm2)
                        retained=true;
                if (!retained)
                    out<<"Break contact "<<index<<" "<<previous.M2<<" "<<previous.arm1<<" "<<previous.arm2<<endl;
            }
            for (const hbond& current : new_hbondlist)
            {
                bool existed=false;
                for (const hbond& previous : old_hbondlist)
                    if (previous.M2==current.M2 && previous.arm1==current.arm1 && previous.arm2==current.arm2)
                        existed=true;
                if (!existed)
                    out<<"Form contact "<<index<<" "<<current.M2<<" "<<current.arm1<<" "<<current.arm2<<endl;
            }
        }
        
        
    }
    
    return accept/double(S.NMOL);
}

bool MC::Glauber(double delta, double rand)
{
    if (isnan(delta)) return false;
    double weight=exp(-fabs(delta));
    double probability=delta>=0 ? weight/(1+weight) : 1/(1+weight);
    return rand < probability;
}
bool MC::Arrhenius(double A,double delta, double rand)
{
    
    if(A*(exp(-delta))>rand)
        {
        return true;}
    else
        return false;
    
    
}
double MC::WCAEnergy()
{
    double ewca=0;
    list<int>::iterator it;
    for(int i=0;i<S.NMOL;i++)
    {
        int old_gID=S.M[i].gID;
        for(size_t l=0;l<S.G[old_gID].nbr.size();l++)
        {
            //ID of new neighboring grids
            int temp_gID=S.G[old_gID].nbr[l];
            
            for(it=S.G[temp_gID].plist.begin();it!=S.G[temp_gID].plist.end();it++)
            {
                int j=*it;
                if(j>i)//avoid double counting
                {
                    //image distance
                    double r2=min_d2(S.M[j].centre,S.M[i].centre,S.L);
                    ewca+=E.WCA(r2);
                    


                }
            }
        }
    }
    return ewca;
}
// Legacy bonded-model diagnostics; excluded from the nonbonded Hamiltonian.
double MC::FENE_energy()
{
    double efene=0;
    list<hbond>::iterator it;
    for(it=S.H.begin();it!=S.H.end();it++)
    {
        hbond j=*it;
        efene+=E.hbonde_fene(S.M[j.M1],S.M[j.M2],j.arm1,j.arm2);
    }
    return efene;
}
double MC::Angle_energy()
{
    double eangle=0;
    list<hbond>::iterator it;
    for(it=S.H.begin();it!=S.H.end();it++)
    {
        hbond j=*it;
        eangle+=E.hbonde_angle(S.M[j.M1],S.M[j.M2],j.arm1,j.arm2);
    }
    return eangle;
}
double MC::Dihedral_energy()
{
    double edihedral=0;
    list<hbond>::iterator it;
    for(it=S.H.begin();it!=S.H.end();it++)
    {
        hbond j=*it;
        edihedral+=E.hbonde_dihedral(S.M[j.M1],S.M[j.M2],j.arm1,j.arm2);
    }
    return edihedral;   
}
double MC::bond_energy()
{
    double ebond;
    ebond=-static_cast<double>(S.H.size())*S.E_1;
    return ebond;
}
double MC::bond_freeze_freenerngy()
{
    double efreeze;
    int nfreeze=0;
    for(int i=0;i<S.NMOL;i++)
    {
        Molecule m=S.M[i];
        for(int l=0;l<m.N_VER;l++)
        {
            if(m.vertype[l]=='I'||m.vertype[neighborarm(l)]=='I')
            nfreeze+=1;

        }
    }
    // Loss of free-arm entropy is the opposite of the legacy free-arm contribution.
    efreeze=-nfreeze*S.free_bond_freeenergy;
    return efreeze;
}
double MC::TotalEnergy()
{
    double totalenergy=0.0;
    for (int i=0; i<S.NMOL; ++i)
        for (int cell : S.G[S.M[i].gID].nbr)
            for (int j : S.G[cell].plist)
                if (j>i)
                    totalenergy+=E.PairEnergy(S.M[i],S.M[j]);
    return totalenergy;
}

void MC::WriteEnergy(int timestep)
{
    double repulsion=0.0;
    double attraction=0.0;
    for (int i=0; i<S.NMOL; ++i)
        for (int cell : S.G[S.M[i].gID].nbr)
            for (int j : S.G[cell].plist)
                if (j>i)
                {
                    repulsion+=E.PairRepulsion(S.M[i],S.M[j]);
                    attraction+=E.PairAttraction(S.M[i],S.M[j]);
                }
    ofstream out("energy.txt",ios::app);
    out<<setprecision(16)<<"TIMESTEP"<<endl<<timestep<<endl;
    out<<"Repulsion\tAttraction\tTotal"<<endl;
    out<<repulsion<<'\t'<<attraction<<'\t'<<repulsion+attraction<<endl;
}

bool MC::IsContact(double attraction) const
{
    return S.bond_epsilon>0 && attraction < -S.contact_fraction*S.bond_epsilon;
}

void MC::RebuildContacts()
{
    S.H.clear();
    for (Molecule& molecule : S.M)
    {
        molecule.hbond_list.clear();
        molecule.nbonds=0;
        fill(molecule.vertype.begin(),molecule.vertype.end(),'A');
    }
    for (int i=0; i<S.NMOL; ++i)
        for (int cell : S.G[S.M[i].gID].nbr)
            for (int j : S.G[cell].plist)
                if (j>i)
                {
                    hbond contact;
                    if (!IsContact(E.PairAttraction(S.M[i],S.M[j],&contact))) continue;
                    if (S.single_partner && (S.M[i].vertype[contact.arm1]=='I' || S.M[j].vertype[contact.arm2]=='I'))
                        throw runtime_error("Initial configuration violates single_partner; use another seed or --single_partner 0.");
                    S.H.push_back(contact);
                    S.M[i].hbond_list.push_back(contact);
                    S.M[j].hbond_list.push_back(hbond(j,i,contact.arm2,contact.arm1));
                    S.M[i].vertype[contact.arm1]='I';
                    S.M[j].vertype[contact.arm2]='I';
                    ++S.M[i].nbonds;
                    ++S.M[j].nbonds;
                }
}

void MC::UpdateContacts(int index, const vector<hbond>& contacts)
{
    // All other pair geometries are unchanged by a single-molecule move.
    vector<int> affected(1,index);
    for (const hbond& old : S.M[index].hbond_list)
    {
        affected.push_back(old.M2);
        vector<hbond>& reciprocal=S.M[old.M2].hbond_list;
        reciprocal.erase(remove_if(reciprocal.begin(),reciprocal.end(),
                                   [index](const hbond& b){return b.M2==index;}),reciprocal.end());
    }
    S.H.remove_if([index](const hbond& b){return b.M1==index || b.M2==index;});
    S.M[index].hbond_list=contacts;
    for (const hbond& contact : contacts)
    {
        affected.push_back(contact.M2);
        S.M[contact.M2].hbond_list.push_back(hbond(contact.M2,index,contact.arm2,contact.arm1));
        S.H.push_back(index<contact.M2 ? contact : hbond(contact.M2,index,contact.arm2,contact.arm1));
    }
    for (int i : affected)
    {
        Molecule& molecule=S.M[i];
        molecule.nbonds=molecule.hbond_list.size();
        fill(molecule.vertype.begin(),molecule.vertype.end(),'A');
        for (const hbond& contact : molecule.hbond_list) molecule.vertype[contact.arm1]='I';
    }
}

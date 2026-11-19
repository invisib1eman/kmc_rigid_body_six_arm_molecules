//NANOROD: system.cpp System Class Function Definitions (Revision Date: Oct 27, 2023)
#include "system.h"

namespace
{
void read_initial_state_header(istream& input, int& count, double& length)
{
    string magic;
    int version;
    if (!(input>>magic>>version>>count>>length) || magic!="KMC_RIGID_SNAPSHOT" ||
        version!=1 || count<1 || !isfinite(length) || length<=0)
        throw invalid_argument("Invalid initial_state header; expected KMC_RIGID_SNAPSHOT 1, then NMOL and box length.");
}
}

void System::ReadInitialStateHeader(int& molecule_count, double& box_length) const
{
    ifstream input(initial_state_file.c_str());
    if (!input) throw runtime_error("Cannot open initial_state: "+initial_state_file);
    read_initial_state_header(input,molecule_count,box_length);
}

void System::LoadInitialState()
{
    ifstream input(initial_state_file.c_str());
    if (!input) throw runtime_error("Cannot open initial_state: "+initial_state_file);
    int count;
    double length;
    read_initial_state_header(input,count,length);
    if (count!=NMOL || length!=L)
        throw invalid_argument("initial_state header changed during initialization.");
    M.resize(NMOL);
    vector<bool> seen(NMOL,false);
    for (int row=0; row<NMOL; ++row)
    {
        Molecule molecule;
        if (!(input>>molecule.MOL_ID>>molecule.centre.x>>molecule.centre.y>>molecule.centre.z
              >>molecule.orientation.w>>molecule.orientation.x
              >>molecule.orientation.y>>molecule.orientation.z))
            throw invalid_argument("Invalid or truncated molecule row in initial_state.");
        int id=molecule.MOL_ID;
        if (id<0 || id>=NMOL || seen[id])
            throw invalid_argument("initial_state molecule IDs must uniquely cover 0..NMOL-1.");
        if (!isfinite(molecule.centre.x) || !isfinite(molecule.centre.y) || !isfinite(molecule.centre.z))
            throw invalid_argument("initial_state centers must be finite.");
        quarternion& q=molecule.orientation;
        double norm=sqrt(q.w*q.w+q.x*q.x+q.y*q.y+q.z*q.z);
        if (!isfinite(norm) || fabs(norm-1.0)>1e-6)
            throw invalid_argument("initial_state orientations must be unit quaternions in w,x,y,z order.");
        q.normalize();
        molecule.centre=image(molecule.centre,L);
        molecule.UpdateVertices();
        molecule.gID=GridIndex_xyz(molecule.centre,NGRID,GRIDL,L);
        G[molecule.gID].plist.push_back(id);
        ++G[molecule.gID].n;
        M[id]=molecule;
        seen[id]=true;
    }
    string extra;
    if (input>>extra)
        throw invalid_argument("Unexpected extra data after initial_state molecule rows.");
    cout<<"Loaded "<<NMOL<<" molecules from "<<initial_state_file<<endl;
}

void System::Create()
{
    
    cout<<"Creating System"<<endl;
    int i,j,k,l,n;
    //Allocate Grid
    G.clear();
    M.clear();
    Grid g;
    for(k=0; k<NGRID; k++)
    {
        for(j=0; j<NGRID; j++)
        {
          for(i=0; i<NGRID; i++)
          {
              g.g_index=GridIndex_index(i,j,k,NGRID);
              g.nbr.clear();
              g.plist.clear();
              g.cm.x=(double(i)+0.5)*GRIDL-0.5*L;
              g.cm.y=(double(j)+0.5)*GRIDL-0.5*L;
              g.cm.z=(double(k)+0.5)*GRIDL-0.5*L;
              g.n=0;//We fill this later now
              G.push_back(g);
             
          }
        }
    }
   
    UpdateGrid();
    //Fill particles Use random permutation
    bool flag;
    H.clear();//init hbond list
    if (!initial_state_file.empty())
    {
        LoadInitialState();
        return;
    }
    
    
    int count=0;
    /*Molecule m1;//initial test for 2 particles
    m1.MOL_ID=0;
    m1.centre=XYZ(0,0,0);
    m1.gID=GridIndex_xyz(m1.centre,NGRID,GRIDL,L);
    G[m1.gID].n+=1;
    G[m1.gID].plist.push_back(m1.MOL_ID);
    m1.UpdateVertices();
    m1.nbonds=1;
    m1.hbond_list.push_back(hbond(0,1,0,4));
    m1.vertype[0]='I';
    M.push_back(m1);
    Molecule m2;
    m2.MOL_ID=1;
    m2.centre=XYZ(2.26,0,1.06);
    m2.gID=GridIndex_xyz(m2.centre,NGRID,GRIDL,L);
    G[m2.gID].n+=1;
    G[m2.gID].plist.push_back(m2.MOL_ID);
    m2.orientation=angle_to_quarternion(M_PI/3,1,0);
    m2.UpdateVertices();
    m2.nbonds=1;
    m2.hbond_list.push_back(hbond(1,0,4,0));
    m2.vertype[4]='I';
    M.push_back(m2);
    NMOL=2;*/
    for(int i=0; i<NMOL; i++)
    {
        Molecule m;
        m.MOL_ID=i;
        m.centre=XYZ(gsl_rng_uniform(gsl_r)*L-0.5*L,gsl_rng_uniform(gsl_r)*L-0.5*L,gsl_rng_uniform(gsl_r)*L-0.5*L);
        m.gID=GridIndex_xyz(m.centre,NGRID,GRIDL,L);
        G[m.gID].n+=1;
        G[m.gID].plist.push_back(m.MOL_ID);
        m.orientation=RandomRotate(angle_to_quarternion(0,0,0), 1,gsl_rng_uniform(gsl_r),gsl_rng_uniform(gsl_r),gsl_rng_uniform(gsl_r));
        m.UpdateVertices();
        M.push_back(m);
    }
    //init a ring

}

void System::WriteMol2(int timestep)
{
    ofstream out;
    out.open("conf.mol2",ios::app);
    out<<timestep<<endl;
    int n_ver=M[0].N_VER;
    
    out<<"@<TRIPOS>MOLECULE"<<endl;
    out<<"Nanorod"<<endl;
    out<<NMOL*(n_ver+1)<<"\t"<<NMOL*n_ver+H.size()<<endl;
    out<<"SMALL"<<endl;
    out<<"NO_CHARGES"<<endl;

    out<<"@<TRIPOS>ATOM"<<endl;

    string name,type;
    
    int count=0;
    
    XYZ im,shift;
    
    for(int i=0; i<NMOL; i++)
    {
        shift=image(M[i].centre,L)-M[i].centre;
        im=M[i].centre+shift;
        out<<setw(6)<<++count<<"\t"<<"1"<<"\t"<<setw(8)<<im.x<<"\t"<<setw(8)<<im.y<<"\t"<<setw(8)<<im.z<<"\t"<<"C.3"<<endl;
        
        for(int j=0; j<n_ver; j++)
        {
            im=M[i].ver[j]+shift;
            out<<setw(6)<<++count<<"\t"<<"2"<<"\t"<<setw(8)<<im.x<<"\t"<<setw(8)<<im.y<<"\t"<<setw(8)<<im.z<<"\t"<<"N.3"<<endl;
        }
    }
          
    out<<"@<TRIPOS>BOND"<<endl;
    
    count=0;
    for(int i=0; i<NMOL; i++)
    {
        for(int j=0; j<n_ver; j++)
        {
            out<<setw(8)<<++count<<"\t"<<setw(8)<<(n_ver+1)*i+1<<"\t"<<setw(8)<<(n_ver+1)*i+j+2<<"\t"<<setw(2)<<"1"<<endl;
        }
    }
    list<hbond>::iterator it;
    for(it=H.begin();it!=H.end();it++)
    {
        hbond writebond=*it;
        out<<setw(8)<<++count<<"\t"<<setw(8)<<(n_ver+1)*writebond.M1+writebond.arm1+2<<"\t"<<setw(8)<<(n_ver+1)*writebond.M2+writebond.arm2+2<<"\t"<<setw(2)<<"1"<<endl;
            
        
        
    }
        
    

    out.close();
    //   cout<<"Mol2 input written in\t"<<FileName<<endl;
    return;
}
void System::WriteDump(int timestep)
{
    string FileName=Description+"_Dump.lammpstrj";
    ofstream out;
    out.open(FileName,ios::app);
    
    out<<"ITEM: TIMESTEP"<<endl;
    out<<timestep<<endl;
    out<<"ITEM: NUMBER OF ATOMS"<<endl;
    int NT=NMOL*7;
    out<<NT<<endl;
    out<<"ITEM: BOX BOUNDS pp pp pp"<<endl;
    out<<-0.5*L<<"\t"<<0.5*L<<endl;
    out<<-0.5*L<<"\t"<<0.5*L<<endl;
    out<<-0.5*L<<"\t"<<0.5*L<<endl;
    out<<"ITEM: ATOMS index type x y z"<<endl;
    /*int NT=NMOL*7;
    out<<NT<<endl;
    out<<"Time="<<timestep<<endl;*/
    XYZ im;
    XYZ im_centre;
    for(int i=0;i<NMOL;i++)
    {   
        im_centre=image(M[i].centre,L);
        out<<setw(6)<<i*7+1<<"\t"<<1<<"\t"<<setw(8)<<im_centre.x<<"\t"<<setw(8)<<im_centre.y<<"\t"<<setw(8)<<im_centre.z<<endl;
        for(int j=0;j<6;j++)
        {
            im=image(M[i].ver[j],L);
            out<<setw(6)<<i*7+j+2<<"\t"<<2<<"\t"<<setw(8)<<im.x<<"\t"<<setw(8)<<im.y<<"\t"<<setw(8)<<im.z<<endl;
        }
    }
    out.close();
}
void System::WriteBond(int timestep)
{
    ofstream out;
    out.open("Bondlist.txt",ios::app);
    out<<"TIMESTEP"<<endl;
    out<<timestep<<endl;
    out<<setw(12)<<"molecule1"<<"\t"<<setw(12)<<"molecule2"<<"\t"<<setw(12)<<"arm1"<<"\t"<<setw(8)<<"arm2"<<endl;
    list<hbond>::iterator it;
    for(it=H.begin();it!=H.end();it++)
    {
        hbond writebond=*it;
        out<<setw(12)<<writebond.M1<<setw(12)<<writebond.M2<<setw(12)<<writebond.arm1<<setw(12)<<writebond.arm2<<endl;
            
        
        
    }
    out.close();
}
void System::WriteOrientation(int timestep)
{
    ofstream out;
    out.open("Orientation.txt",ios::app);
    out<<"TIMESTEP"<<endl;
    out<<timestep<<endl;
    out<<setw(12)<<"molecule1"<<"\t"<<setw(12)<<"molecule2"<<"\t"<<setw(12)<<"arm1"<<"\t"<<setw(8)<<"arm2"<<endl;
    for(int j=0;j<NMOL;j++)
    {
        
        out<<setw(12)<<M[j].orientation.w<<setw(12)<<M[j].orientation.x<<setw(12)<<M[j].orientation.y<<setw(12)<<M[j].orientation.z<<endl;
            
        
    
    }
    out.close();

}
void System::WriteGrid(int timestep)
{
    ofstream out;
    out.open("Grid.txt",ios::app);
    out<<"TIMESTEP"<<endl;
    out<<timestep<<endl;
    for(int i=0;i<NGRID3;i++)
    {
     out<<setw(12)<<G[i].g_index<<setw(12)<<G[i].cm.x<<setw(12)<<G[i].cm.y<<setw(12)<<G[i].cm.z<<setw(12)<<G[i].n<<setw(8)<<endl;
              out<<"neighborlist"<<endl;
              for(int l=0;l<G[i].nbr.size();l++)
              {
                out<<G[i].nbr[l]<<endl;
              }
              
    }
    out.close();
}
void System::UpdateGrid()
{
    int neighbor_span=std::min(NGRID/2, int(ceil(interaction_cutoff/GRIDL)));
    for (int k=0; k<NGRID; ++k)
        for (int j=0; j<NGRID; ++j)
            for (int i=0; i<NGRID; ++i)
            {
                Grid& grid=G[GridIndex_index(i,j,k,NGRID)];
                grid.nbr.clear();
                for (int dz=-neighbor_span; dz<=neighbor_span; ++dz)
                    for (int dy=-neighbor_span; dy<=neighbor_span; ++dy)
                        for (int dx=-neighbor_span; dx<=neighbor_span; ++dx)
                            grid.nbr.push_back(GridIndex_index(i+dx,j+dy,k+dz,NGRID));
                // Small periodic grids can produce repeated wrapped cells.
                sort(grid.nbr.begin(),grid.nbr.end());
                grid.nbr.erase(unique(grid.nbr.begin(),grid.nbr.end()),grid.nbr.end());
            }
}

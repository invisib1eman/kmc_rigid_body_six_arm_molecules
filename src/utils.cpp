//NANOROD: utils.cpp Utilities Functions (Revision Date: Oct 27, 2023)
#include "utils.h"
#include <algorithm>
#include <limits>
#include <stdexcept>
//do not assume where the particle is (particle may have crossed the box twice or more)
XYZ image(XYZ p, double L)//note that every time we calculate vectors or distances we need to image XYZ
{
  XYZ xyz;
  xyz.x=myfmod(p.x+0.5*L, L)-0.5*L;
  xyz.y=myfmod(p.y+0.5*L, L)-0.5*L;
  xyz.z=myfmod(p.z+0.5*L, L)-0.5*L;
  return xyz;
}
double myfmod(double x, double y)
{
  double temp=fmod(x,y);
  if(temp>=0.0)
    return temp;
  else
    return temp+y;
}
//Minimum image distance squared: pass original coordinates
double min_d2(XYZ a, XYZ b, double L)
{
    a=image(a,L);
    b=image(b,L);
    XYZ d=a-b;
    d.my_abs();
    if(d.x>=0.5*L)
      d.x=L-d.x;
    
    if(d.y>=0.5*L)
      d.y=L-d.y;
    
    if(d.z>=0.5*L)
      d.z=L-d.z;
    
    return d.norm2();
}
XYZ real_vector(XYZ origin,double L)//minimum-image displacement, including multiple box crossings
{
  return image(origin,L);
}

XYZ RandomTranslate(XYZ old, double step,double u,double v)
{
    
    double theta=2.0*M_PI*u;
    double phi=acos(2.0*v-1.0);
    return XYZ(old.x+step*cos(theta)*sin(phi),old.y+step*sin(theta)*sin(phi),old.z+step*cos(phi));
}


quarternion RandomRotate(quarternion old, double step,double a,double b,double c)
{
    
    double theta=2.0*M_PI*a*step;
    double alpha=acos(2.0*b-1.0);
    double beta=2.0*M_PI*(c-0.5);
    quarternion rotate=angle_to_quarternion(theta,alpha,beta);
    quarternion neworientation=quartermulti(rotate,old);
    neworientation.normalize();
    return neworientation;
}
int getNum(vector<int>& v)
{
 
    // Size of the vector
    int n = v.size();
 
    // Generate a random number
    srand(time(NULL));
 
    // Make sure the number is within
    // the index range
    int index = rand() % n;
 
    // Get random number from the vector
    int num = v[index];
 
    // Remove the number from the vector
    swap(v[index], v[n - 1]);
    v.pop_back();
 
    // Return the removed number
    return num;
}
 
// Function to generate n non-repeating random numbers
vector<int> generateRandom(int n)
{
    vector<int> v(n);
 
    // Fill the vector with the values
    // 1, 2, 3, ..., n
    for (int i = 0; i < n; i++)
        v[i] = i + 1;
    vector<int> randomv;
    // While vector has elements
    // get a random number from the vector and print it
    while (v.size()) {
        randomv.push_back(getNum(v));
    }
    return randomv;
}
//inner product of two vectors
double inner_product(XYZ a,XYZ b)
{
  return a.x*b.x+a.y*b.y+a.z*b.z;
}
//cross product of two vectors
XYZ cross_product(XYZ a,XYZ b)
{
  return XYZ(a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x);
}
//calculate angle between two vectors
double angle_vectors(XYZ a,XYZ b)
{
  double length_a=a.norm();
  double length_b=b.norm();
  if(!(length_a>0.0)||!(length_b>0.0)||!isfinite(length_a)||!isfinite(length_b))
    return numeric_limits<double>::quiet_NaN();
  double cosine=inner_product(a/length_a,b/length_b);
  return acos(max(-1.0,min(1.0,cosine)));
}
double dihedral_vectors(XYZ a,XYZ b,XYZ c)
{
    XYZ normal1=cross_product(a,b);
    XYZ normal2=cross_product(b,c);
    double length_b=b.norm();
    double length_normal1=normal1.norm();
    double length_normal2=normal2.norm();
    if(!(length_b>0.0)||!(length_normal1>0.0)||!(length_normal2>0.0)
       ||!isfinite(length_b)||!isfinite(length_normal1)||!isfinite(length_normal2))
        return numeric_limits<double>::quiet_NaN();
    // Cancel the common positive denominator, preserving the original sign convention.
    return atan2(length_b*inner_product(a,normal2),inner_product(normal1,normal2));
}
int GridIndex_index(int i,int j,int k,int n)
{
  i=(i%n+n)%n;
  j=(j%n+n)%n;
  k=(k%n+n)%n;
  return n*n*k+n*j+i;
}
int GridIndex_xyz(const XYZ& p,int n,double dl,double L)
{
  XYZ wrapped=image(p,L);
  int i=int(floor((wrapped.x+0.5*L)/dl));
  int j=int(floor((wrapped.y+0.5*L)/dl));
  int k=int(floor((wrapped.z+0.5*L)/dl));
  return GridIndex_index(i,j,k,n);
}

void GridLoc(int& i,int& j,int& k,int n,int index)
{
  i=index%n;
  j=((index-i)/n)%n;
  k=(index-i-j*n)/(n*n);
}
int neighborarm(int n)
{
  return n+1-2*(n%2);
}
vector<Molecule> generate_unitcell(Molecule)
{
  throw logic_error("generate_unitcell is not implemented");
}
vector<Molecule> generate_newunit(vector<Molecule>,int,int,int)
{
  throw logic_error("generate_newunit is not implemented");
}

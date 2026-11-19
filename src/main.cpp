//NANOROD: main.cpp Main Program (Revision Date: Oct 27, 2023)
#include "header.h"
#include "system.h"
#include "mc.h"


int main(int argc, char *argv[])
{
    
    
	time_t start, end;
	time(&start);
	
    System sys;
    try
    {
    sys.ReadInput(argc,argv);
    sys.Create();
	
    MC mc;
    mc.S=sys;
    mc.Sweep();
    }
    catch (const exception& error)
    {
        cerr<<"Error: "<<error.what()<<endl;
        if (sys.gsl_r) gsl_rng_free(sys.gsl_r);
        return EXIT_FAILURE;
    }
    
	//Finalize Random number
	gsl_rng_free(sys.gsl_r);
	time(&end);
	cout<<"Time Elapsed="<<difftime(end,start)<<endl;
    return 0;
    /*
   XYZ a=quarterrotation(XYZ(1.0,0.0,-0.47),quarternion(0.899125,0.28284,0.290824,-0.164306));
   cout<<a.x<<endl;
   cout<<a.y<<endl;
   cout<<a.z<<endl;
   return 0;*/
}

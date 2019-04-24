/* WBL Crest 21 March 2009 $Revision: 1.2 $
 * Based on cuda/sdk/projects/quasirandomGenerator/quasirandomGenerator_gold.cpp
 * and pch.h W. Langdon cs.ucl.ac.uk 5 May 1994 (21 Feb 95 make it inline)
 */



#include <stdio.h>
#include <math.h>
#include "park-miller_common.h"



//Generate single Park-Miller psuedo random number for given index and dimension and normalize
extern "C" int parkmillerValue(INT64 i, int dim){
    unsigned int seed = i + dim;

// W. Langdon cs.ucl.ac.uk 5 May 1994

//inline int intrnd (int& seed) // 1<=seed<=m
{
#ifdef LONG_GE46BITS
opps
int const a    = 16807;      //ie 7**5
int const m    = 2147483647; //ie 2**31-1
	seed = (long(seed * a))%m;
	return seed;
#else
double const a    = 16807;      //ie 7**5
double const m    = 2147483647; //ie 2**31-1

	double temp = seed * a;
	return /*seed =*/ (int) (temp - m * floor ( temp / m ));
	//cout<<"seed "<<seed<<endl;
	//return seed;
#endif
}//end mycode

}




////////////////////////////////

#include "aux_functions.h"
#include <stdlib.h> // for dynamic mem mgmt.
#include <cstdio>	/* printf functionality */
#include <math.h>	/* pow() functionality */
#include <cmath>	/* fmod() functionality */
#define N 20000000

int main(){
	float *in=(float*)calloc(N,sizeof(float));
	float *out=(float*)calloc(N,sizeof(float));
	const float ref=0.5f;
	for(int i=0;i<N;++i){
		in[i]=scale(i,N);
	}

	distanceArray(out,in,ref,N);

	// For loo to print results for confirmation
	double x = pow(2, 18.253496664);
	for(int i=0;i<N;++i){
		if(std::fmod((float)i,(float)x) == 0){
			printf("%lf\n", out[i]);
		}
	}


	free(in);
	free(out);

	return 0;
}

//Generates Data in a data.txt

#include <math.h>
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <time.h>
#define N 1000

int main(){

	srand(time(NULL));

	int out[N]={0};

	std::ofstream fs("./data.txt");
	if(!fs){
		std::cerr<<"Cannot open the output file."<<std::endl;
	}
	//fs<<"Entry_ID"<<std::setw(10)<<std::right<<"Age"<<std::endl;
	for(int i=0;i<N;++i){
		out[i]=rand()%80 + 1;
		fs<<out[i]<<std::endl;
		//fs<<std::setw(4)<<i<<std::right<< ' ' << std::setw(14)<<out[i]<<std::left<<std::endl;
	}
	fs.close();

	return 0;
}

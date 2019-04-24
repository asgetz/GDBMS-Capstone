//Generates Data in a data.txt

#include <math.h>
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <time.h>
#define N 769240

int main(){

	srand(time(NULL));

	int out[N]={0};

	std::ofstream fs("./data6.txt");
	if(!fs){
		std::cerr<<"Cannot open the output file."<<std::endl;
	}
	//fs<<"Entry_ID"<<std::setw(10)<<std::right<<"Age"<<std::endl;
	for(int i=0;i<N;++i){
		out[i]=rand()%1000 + 1;
		fs<<i<<"    "<<out[i]<<std::endl;
		//fs<<std::setw(4)<<i<<std::right<< ' ' << std::setw(14)<<out[i]<<std::left<<std::endl;
	}
	fs.close();


	// std::string a = "769240    1000";
	// std::cout<<"Size of last string is: "<<a.length()<<std::endl;

	return 0;
}

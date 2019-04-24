#WBL 4 April 2009 

echo -n '#example.bat $Revision: 1.00 $'
date

#Generate 4 million random number streams.
#for each stream generate the first 100 PRNG.
#Ie 400 million pseudo random numbers on the nVidia GPU

#The 4 million sequences are run in parallel in 8192 blocks
#of 512 threads.

#x suppress debug output

#Note the a seed value of zero is illegal for Park-Miller and causes
#the first sequence to fail. Ie it always returns zero.
#Thus we also test a common failure mode.

../../../bin/linux/release/park-miller 100 8192 512 4194304 x

echo -n '#example.bat $Revision: 1.00 $'
date

#Same example. But with diagnostic output showing correct answers are
#being sent back from GPU to PC.

../../../bin/linux/release/park-miller 100 8192 512 4194304

echo -n '#example.bat $Revision: 1.00 $'
date
#


// ***********************************************************************
//
// Demo program for education in subject
// Computer Architectures and Paralel Systems
// Petr Olivka, dep. of Computer Science, FEI, VSB-TU Ostrava
// email:petr.olivka@vsb.cz
//
// Example of CUDA Technology Usage
// Multiplication of elements in float array
//
// ***********************************************************************

#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdint.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <time.h>
#include <sys/shm.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <stdlib.h>
#include <unistd.h>

// Function prototype from .cu file
void run_mult(char *pole, unsigned long long int L, uint8_t* result, unsigned long long int strlen );

#define N 200

int main()
{
	// Array initialization
	//int text_length = 3;
	long int first = time(NULL);
	unsigned long long int strlen = 4;
	char * vel;
	char velikost[strlen];
	//char * prvky = new char[N*sizeof(velikost)];
	uint8_t *result;
	uint8_t resultvel[16];
	//char * velikost;
	//char * prvky[N];
	//float *l_array;
	//printf( "test\n" );
	/*if ( cudaMallocManaged( &vel, N * sizeof( *velikost ) ) != cudaSuccess )
	{
		printf( "Unable to allocate Unified memory!\n" );
		return 1;
	} */
	if ( cudaMallocManaged( &result, N * sizeof( *resultvel ) ) != cudaSuccess )
	{
		printf( "Unable to allocate Unified memory!\n" );
		return 1;
	}


	int forknum = 1;
    unsigned long long int abecedaLen = 26;
    unsigned long long int allComb = 1;
    /*for(int c=0;c<forknum;c++)
    {
        int j=fork();
        if(j==0)
        {
            //char cha = c;
            //printf("%c",cha)
            paralellRecursion(ja, jo, ca, 'A');
            paralellRecursion(ja, jo, ca, 'A');
            //sleep(2);
            exit(17);
        }
    }*/
    for(int i = 0; i< strlen; i++){
        allComb = allComb *  abecedaLen;
    }

	printf("kombinace: %llu\n",allComb);
	//printf("sizeof: %ld\n",sizeof( resultvel[16] ));

	/*if ( cudaMallocManaged( &vel, allComb * strlen /*sizeof( *velikost ) ) != cudaSuccess )
	{
		printf( "Unable to allocate Unified memory!\n" );
		return 1;
	}

	if ( cudaMallocManaged( &result, allComb * 16 * sizeof( uint8_t ) /*sizeof( *resultvel )  ) != cudaSuccess )
	{
		printf( "Unable to allocate Unified memory!\n" );
		return 1;
	} */

    //printf("hej %d", allComb);


    /*fflush(stdout);
    int skok = allComb/ forknum;
    for(int c=0;c<forknum;c++)
    {
        //int j=fork();
        //if(j==0)
        //{
            int start = skok * c;
            int end = skok * (c+1);
            if(c == (forknum - 1)){
                end = allComb;
            }
            //printf("ITERACE:%d \n", end);
            //char *chs = malloc(strlen*sizeof(char*)); 
            for(int i = start; i < end; i++)
            {
                int help = i;
                for(int k = ((strlen-1) + i * strlen); k >=(0+i*strlen);k--){
					//printf("%d \n", k);
                    vel[k] = (help%abecedaLen) + 'a';
                    help = help / abecedaLen; // zajistím aby v další iteraci byl jiný char
                }
                //printf("%s \n", vel);
                fflush(stdout);
                //printf("\n");
            }
            
            //sleep(2);
            //exit(17);
        //}
    } */


    /*int status;
    int returncodes[forknum];
    int i = 0;
    while(waitpid(-1, &status, 0) > 0 ){
    returncodes[i] = WEXITSTATUS(status);
        i++;
    }; */
    //printf("Time je = %ld\n", second - first);

	//printf( "test\n" );
	//prvky = new char[N];
	//OCfloat prvky[ N ];

	/*for (int i = 0; i < N; i++ )
	{
		for (int j = 0; j < 3; j++){
			vel[i] = 'a';
		}
	} */

		//prvky[i] = (unsigned char *) i;
		//prvky[i] = "aaa";
	//printf( "test\n" );
	// Function calling 
	run_mult( vel, allComb, result, strlen);

	// Print result
	/*for (unsigned long long int i = allComb-1; i < allComb; i++ )
	{
		//printf( "%llu\n", i );
		for(unsigned long long int j = i * strlen; j < (i * strlen + strlen); j++){
			//printf( "%llu\n", j );
			printf( "%c", vel[j] );
		}
		printf( "\n" );
	} */
		//printf( "%8.2f\n", prvky[ i ] );
		//printf( "%s\n", prvky[ i ] );
	//printf( "\n" );
	/*for(unsigned long long int y = allComb-1; y < allComb; y++){
	for (unsigned long long int i = y * 16; i < (y * 16 + 16); i++){
        printf("%2.2x", result[i]);
		//printf("sizeoff %ld", sizeof(result[i]));
	}
    printf("\n");
	fflush(stdout);
	}*/
	long int second = time(NULL);
	printf("Time je = %ld\n", second - first);

	cudaFree( vel );
	cudaFree( result );

	return 0;
}


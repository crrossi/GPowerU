#include <iostream>
#include <math.h>
#include "nvml.h"
#include "helper_cuda.h"
#include "GPowerU.hpp"
     
#include <stdio.h>
//#include <time.h>
//#include <sys/time.h>

 
// CUDA kernel to add elements of two arrays with loops designed to stress the GPU
// and extend the kernel execution
 __global__
 void add(int n, float *x, float *y)
  	{
      int index = blockIdx.x * blockDim.x + threadIdx.x;
      int stride = blockDim.x * gridDim.x;
//      take_GPU_time(); //Checkpoint power measure __device__ function
      __syncthreads();
      
      
      for (int i = index; i < n; i += stride){
      	for(int k=0; k<1000; k++) y[i] = x[i] + y[i];
        }	
       
      __syncthreads();
      
  //    take_GPU_time(); //Checkpoint power measure __device__ function
      for (int i = index; i < n; i += stride){
      	for(int k=0; k<100000; k++) y[i] = x[i] + y[i];
      }
      __syncthreads();
      
    //  take_GPU_time(); //Checkpoint power measure __device__ function
      for (int i = index; i < n; i += stride){
      	for(int k=0; k<1000; k++) y[i] = x[i] + y[i];
        }
      __syncthreads();
		
		
     // take_GPU_time(true); //Checkpoint power measure __device__ function (last)
    } 
 


int blockSize = 256;
int numBlocks = (64 + blockSize - 1) / blockSize;

 
     
int main( int argc, char** argv)
    {	
//Initializations ==> enable the NVML library, starts CPU thread for the power monitoring,  
	if ( GPowerU_init() != 0 ) {
		fprintf ( stderr, "%s: error: initializing...\n", argv[0] );
			_exit (1);
	}
      	int N =1<<20;
      	float *x, *y;
      
      	// Allocate Unified Memory -- accessible from CPU or GPU
      	checkCudaErrors(cudaMallocManaged((void **)&x, N*sizeof(float)));
      	checkCudaErrors(cudaMallocManaged((void **)&y, N*sizeof(float)));
     	   	
      	// Initialize x and y arrays on the Host
      	for (int i = 0; i < N; i++) {
        	x[i] = 1.0f;
        	y[i] = 2.0f;
      	}		
      
      	//Launch the GPU kernel
      	add<<<numBlocks, blockSize>>>(N, x, y);
      
       	//Checkpoint power measure CPU function ==> it calls its own cudaDeviceSynchronize() 
     // 	GPowerU_checkpoints();
     	cudaDeviceSynchronize();
      	// Free memory
      	checkCudaErrors(cudaFree(x));
      	checkCudaErrors(cudaFree(y));
     	
     	//Ends power monitoring, returns data output files
      	if ( GPowerU_end(5) != 0 )
	{
		fprintf ( stderr, " error: terminating...\n" );
		_exit (1);
	}
     	
      

      return 0;
    }

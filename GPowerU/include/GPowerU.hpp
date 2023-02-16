// 2023 INFN APE Lab - Sezione di Roma
// cristian.rossi@roma1.infn.it

#include "GPowerU.h"

//CPU thread managing the parallel power data taking during the kernel execution
void *threadWork(void * arg) {
	unsigned int power;
	int i=0;
	bool not_enough=0;
   struct timeval tv_start, tv_aux;
   nvmlUtilization_t util;
	struct timeval *time = (struct timeval *) arg;
	
   tv_start=*time;

	while (!terminate_thread) {
		//GET POWER SAMPLES
		    	nvResult = nvmlDeviceGetPowerUsage(nvDevice, &power);
            nvResult = nvmlDeviceGetUtilizationRates (nvDevice, &util);

		if (NVML_SUCCESS != nvResult) {
			printf("Failed to get power usage: %s\n", nvmlErrorString(nvResult));
			pthread_exit(NULL);
		}
		
		
		if(i < SAMPLE_MAX_SIZE_DEFAULT && (power > POWER_THRESHOLD*1000.0  || thread_powers[0] > POWER_THRESHOLD*1000.0)) {
		
			if(i==0) printf("********STARTING GPU WORK********");
           		gettimeofday(&tv_aux,NULL);

            		thread_times[i] = (tv_aux.tv_sec-tv_start.tv_sec)*1000000;
            		thread_times[i] += (tv_aux.tv_usec-tv_start.tv_usec);
           
            		thread_powers[i] = power;
	    		i++;

		}
		else{
			if(i == SAMPLE_MAX_SIZE_DEFAULT) {
				printf("ERROR: POWER VECTOR SIZE EXCEEDED!\n");
				pthread_exit(NULL);
			}
			if(!not_enough){
				printf("NOT ENOUGH POWER!\n");
				not_enough=1;
			}
		}

		//i++;
		n_values = i;

		sleep(TIME_STEP);
	}
	
	pthread_exit(NULL);
}


//Generate the output samples files
float DataOutput() {
	int values_threshold=0;
	float acc0 = 0.0;
	float p_average;
	
	double interval;
   double interval_GPU;
   
   int begin_gpu=-1, end_gpu=n_values-1;
   power_peak=0;
	
	FILE  *fp2;
	fp2 = fopen("data/nvml_power_profile.csv", "w+");

	fprintf(fp2,"#sep=;\n#Timestamp [us];Power measure [W]");

	for(int i=0; i<n_values; i++) {
        fprintf(fp2, "\n%.6f;%.4f", (thread_times[i]-thread_times[0])/1000000, thread_powers[i]/1000.0);
		
        if (thread_powers[i] > power_peak) {
        		power_peak = thread_powers[i];
		  }

        if ( thread_powers[i]/1000.0 > 35 && begin_gpu == -1 ) begin_gpu = i;
        if ( thread_powers[i]/1000.0 < 35 && begin_gpu != -1 ) end_gpu = i;

        if (thread_powers[i]/1000.0 >= threshold) {
        		acc0 = acc0 + thread_powers[i];
         	values_threshold++;
        }
	}
   
   if (values_threshold>0) {
   	  p_average = acc0 / (values_threshold*1.0);
   }
   else {
    		printf("ERROR: DIVISION BY 0\n");
    		exit(-1);
   }
    

	interval = thread_times[n_values-1] - thread_times[0];
   interval_GPU = thread_times[end_gpu] - thread_times[begin_gpu];

   printf("\tAt current frequency (%d,%d) MHz:  Average Power: %.2f W;  Max Power: %.2f W;  Sampling Duration: %.2f ms;  GPU active duration: %.2f ms \n", mem_clock, core_clock, p_average/1000.0, power_peak/1000.0, (interval)/1000, interval_GPU/1000);
   
   fclose(fp2);

	return 0;
}

//Initializations ==> enable the NVML library, starts CPU thread for the power monitoring. It is synchronized with the start time of the program
int GPowerU_init() {
	//sleep(wait);
	unsigned int device_count;
	gettimeofday(&start_time,NULL);
	// unsigned int clock;
	int a;
   int major;
   int check = mkdir("data", 0777);
   
   for (int i = 0; i < MAX_CHECKPOINTS; i++) {
        kernel_checkpoints[i]= 0;
   }
     
    CUresult result;
    CUdevice device = 0;

    result = cuInit(0);
    if (result != CUDA_SUCCESS) {
        printf("Error code %d on cuInit\n", result);
        exit(-1);
    }
    result = cuDeviceGet(&device,0);
    if (result != CUDA_SUCCESS) {
        printf("Error code %d on cuDeviceGet\n", result);
        exit(-1);
    }

    result = cuDeviceGetAttribute (&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
    if (result != CUDA_SUCCESS) {
        printf("Error code %d on cuDeviceGetAttribute\n", result);
        exit(-1);
    }

	terminate_thread = 0;
	

	// NVML INITIALIZATIONS
	nvResult = nvmlInit();
	if (NVML_SUCCESS != nvResult)
    {
        printf("Failed to initialize NVML: %s\n", nvmlErrorString(nvResult));

        printf("Press ENTER to continue...\n");
        getchar();
        return -1;
    }

	nvResult = nvmlDeviceGetCount(&device_count);
    if (NVML_SUCCESS != nvResult)
    {
        printf("Failed to query device count: %s\n", nvmlErrorString(nvResult));
        return -1;
    }

	printf("Found %d device%s\n\n", device_count, device_count != 1 ? "s" : "");
    if (deviceID >= device_count) {
        printf("Device_id is out of range.\n");
        return -1;
    }
	nvResult = nvmlDeviceGetHandleByIndex(deviceID, &nvDevice);
	if (NVML_SUCCESS != nvResult)
	{
		printf("Failed to get handle for device 1: %s\n", nvmlErrorString(nvResult));
		 return -1;
	}
	nvmlDeviceGetApplicationsClock  ( nvDevice, NVML_CLOCK_GRAPHICS, &core_clock);


   nvmlDeviceGetApplicationsClock  ( nvDevice, NVML_CLOCK_MEM, &mem_clock);

	//LAUNCH POWER SAMPLER
	a = pthread_create(&thread_sampler, NULL, threadWork, &start_time);
	if(a) {
		fprintf(stderr,"Error - pthread_create() return code: %d\n",a);
		return -1;
	}

	return 0;
}


//ROOT graph making function
#if ROOT_ENABLED
void grapher(){
	auto c1 = new TCanvas("c1","PowerMeas",200,10,700,500);
   c1->SetGrid();
	TGraphErrors *gr1  = new TGraphErrors("data/nvml_power_profile.csv", "%lg;%lg");
	//gr1->SetMarkerStyle(1);
	//gr1->SetMarkerSize(1);
   gr1->Draw("AP");
   gr1->SetTitle("GPU Power Measurement (GPowerU) ;" "Time (s);" "Power (W)");
   
   
   TGraphErrors *gr2  = new TGraphErrors("data/power_checkpoints.csv", "%lg;%lg");
   gr2->SetMarkerColor(4);
   gr2->SetMarkerStyle(20);
	gr2->SetMarkerSize(1.5);
   gr2->Draw("P");
  
   
   
   c1->Print("data/gpu_graph.pdf");
}
#endif


//Checkpoint power measure __device__ function ==> last to be set =1 for the latest func call
__device__ void take_GPU_time(bool last = 0){
	static int i=0;
	if(kernel_checkpoints[i]==0){
		kernel_checkpoints[i]=1;
		i++;
		max_points++;
	}
	if(last) finish=1;
}


//Checkpoint power measure CPU function ==> it calls its own cudaDeviceSynchronize() 
void GPowerU_checkpoints(){
	unsigned int power;
 	FILE *fp2;
	//struct timespec time_aux;
   struct timeval tv_aux;
   
	fp2 = fopen("data/power_checkpoints.csv", "w");
   fprintf(fp2,"#sep=;\n Timestamp [s]; Power[W]");
   	
   	
   int n_saved_points = 0;
  	while (!finish && kernel_checkpoints[n_saved_points+1] == 0) {
  			for(int i = n_saved_points; i < max_points+1; i++){
  				if(kernel_checkpoints[i]==1) {
					gettimeofday(&tv_aux,NULL);
   				nvResult = nvmlDeviceGetPowerUsage(nvDevice, &power);
   				device_times[i] = (tv_aux.tv_sec - start_time.tv_sec )*1000000;
         		device_times[i] += (tv_aux.tv_usec - start_time.tv_usec)-thread_times[0];
         		device_powers[i] = power;
        			kernel_checkpoints[i]=0;
        			n_saved_points++;	
        	}
  		}
	}
	cudaDeviceSynchronize();
	
	
   for(int i = 0; i < max_points; i++) fprintf(fp2, "\n %.4f; %.4f", device_times[i]/1000000, device_powers[i]/1000);
   fclose(fp2);
   finish=0;
   
 }


//Ends power monitoring, returns data output files
int GPowerU_end(int zz=0) {
   sleep(zz);
	terminate_thread = 1;
	pthread_join(thread_sampler, NULL);
	DataOutput();
   //printf("File out_power_samples.csv created with all measured samples.\n");
   //printf("File out_power.txt average power consumption.\n");
   
	#if ROOT_ENABLED
	grapher();
	#endif
	//printf("Finished\n\n");

	return 0;
}

# GPowerU Tool (APE Lab, INFN Roma1)
__________________________________

GPowerU is a simple tool able to measure the power consumption of a CUDA kernel in specific points of the device code and to generate the complete power profile. 
To do so, we used the NVIDIA Management Library (NVML): a C-based programmatic interface for monitoring and managing various states within NVIDIA GPU devices. 

For power measuring in specific kernel points, since the NVML APIs can be called only by the host side, the idea behind the tool is to send a «message» to the CPU from the GPU in order to take the power value at specific locations of the CUDA kernel in during theexecution.


# Compile on the node
If the node on which the tool is tested has more then one GPU, it will be possible to set the variable MULTIGPU_DISABLED=0 on GPowerU.h. In this way, the power profiling tool and the test application will be compiled in order to manage a multiGPU execution.
Instead, in case of a single GPU node, it will be possible to set the variable MULTIGPU_DISABLED=1 on GPowerU.h

** ADD example [MULTIGPU_DISABLED=1]

In the example, the power profile of the device kernel __global__ void add(...) is obtained through the GPowerU_init() and GPowerU_end() functions, which both manage an additional CPU thread from which NVML APIs are called simultaneosly to the kernel execution.
Instead, the function GPowerU_checkpoints() can be called after kernel<<<...>>>() has launched and works on the same CPU thread of the main, exploting device asynchronus execution. This function has the task to call NVML APIs in correspondence of the take_GPU_time() functions' calls, which send a «message» to the CPU from the GPU in order to take the power value at specific instant.


** TRIAD_CU example (from LIKWID powermeter samples) [MULTIGPU_DISABLED=0]

In order to test the tool on a multi-GPU application, it is possible to use an executable from the likwid samples to launch multiple instances of the same simple CUDA kernel on all the GPUs of the node. In this way, through the GPowerU_init() and GPowerU_end() functions, we can profile the power consumption on all the GPUs of the node


Application's output consists in .csv files with power profile data (nvml_power_profile.csv) and, in case MULTIGPU_DISABLED=1, checkpoints power samples (power_checkpoints.csv), which are combined in the same graph.pdf through ROOT functions. (N.B.: in case ROOT is not installed on your machine, please set to 0 ROOT_ENABLED preprocessor variable).  


# Execution
Build: make

Exec: ./powmeas



GPowerU Tool (APE Lab, INFN Roma1)


GPowerU is a simple tool able to measure the power consumption of a CUDA kernel in specific points of the device code and to generate the power profile of a GPU application. 
To do so, we used the NVIDIA Management Library (NVML): a C-based programmatic interface for monitoring and managing various states within NVIDIA GPU devices. 

Power measurements are collected with a configurable time period and provided as a .csv file (nvml_power_profile.csv).
Furthermore the user can specify checkpoints (i.e. specific points in CUDA kernels) where power measurements are to be collected asynchronously.
Since the NVML APIs can be called only by host side, the idea behind the tool is to send a «message» to the CPU from the GPU in order to take the power value at specific locations of the CUDA kernel while it is running. Checkpoints measurements are provided in the power_checkpoints.csv file.

==> Powmeas example
In the example, the power profile of the device kernel __global__ void add(...) is obtained through the GPowerU_init() and GPowerU_end() functions, which both manage an additional CPU thread from which NVML APIs are called simultaneosly to the kernel execution.
Instead, the function GPowerU_checkpoints() can be called after kernel<<<...>>>() has launched and works on the same CPU thread of the main, exploting device asynchronus execution. This function has the task to call NVML APIs in correspondence of the take_GPU_time() functions' calls, which send a «message» to the CPU from the GPU in order to take the power value at specific instant.

The output consists in .csv files with power profile data (nvml_power_profile.csv) and checkpoints power samples (power_checkpoints.csv), which are combined in the same graph.pdf through ROOT functions. (N.B.: in case ROOT is not installed on your machine, please set to 0 ROOT_ENABLED preprocessor variable).  

==> Execution
Build: make
Exec: ./powmeas

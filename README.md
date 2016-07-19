# gpuSparse

Matlab mex wrappers to NVIDIA cuSPARSE (https://developer.nvidia.com/cusparse).


Uses int32 and single precision to save memory (Matlab uses int64 and double).


## Installation


1. Save in a folder called @gpuSparse on the matlab path

2. Inside Matlab, cd to the private directory

3. Type mex_all to compile the .cu files

4. Tested with Matlab R2016a and CUDA 7.5

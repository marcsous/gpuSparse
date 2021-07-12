# gpuSparse

Matlab mex wrappers to NVIDIA cuSPARSE (https://developer.nvidia.com/cusparse).


Uses int32 and single precision to save memory (Matlab sparse uses int64 and double).


## Installation


1. Save in a folder called @gpuSparse on the Matlab path

2. ```A = gpuSparse('recompile')``` to trigger compilation of mex

3. <b>Recommended:</b> CUDA-11 for <i>much</i> faster transpose-multiply


## Timings
<pre>
<b>Due to memory layout (row/col-major) multiply and transpose-multiply differ in performance.</b>

size(A) = 221,401 x 213,331
nnz(A)  = 23,609,791 (0.05%)
AT      = precomputed transpose of A

<b>CPU sparse</b>
A*x  (sparse)   : Elapsed time is 1.370207 seconds.
AT*y (sparse)   : Elapsed time is 1.347447 seconds.
A'*y (sparse)   : Elapsed time is 0.267259 seconds.

<b>GPU sparse</b>
A*x  (gpuArray) : Elapsed time is 0.137195 seconds.
AT*y (gpuArray) : Elapsed time is 0.106331 seconds.
A'*y (gpuArray) : Elapsed time is 0.232057 seconds. <i>(CUDA 11)</i>
<s>A'*y (gpuArray) : Elapsed time is 16.733638 seconds.</s>

<b>GPU gpuSparse</b>
A*x  (gpuSparse): Elapsed time is 0.068451 seconds.
At*y (gpuSparse): Elapsed time is 0.063651 seconds.
A'*y (gpuSparse): Elapsed time is 0.059236 seconds. <i>(CUDA 11)</i>
<s>A'*y (gpuSparse): Elapsed time is 3.094271 seconds.</s>
</pre>

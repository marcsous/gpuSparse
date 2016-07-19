% tested on R2016a with CUDA 7.5

delete csrgeam.mexa64
delete csrmv.mexa64
delete coo2csr.mexa64
delete csr2csc.mexa64
delete csr2csc_cpu.mexa64
delete csrmm.mexa64
delete csr2coo.mexa64
delete csrmv_v8.mexa64

mexcuda -v -I"/usr/local/cuda/samples/common/inc" -L"/usr/local/cuda/lib64" LDFLAGS='"$LDFLAGS -Wl,--no-as-needed"' -ldl -lcusparse -lcublas_static -lcusparse_static -lculibos csrgeam.cu

mexcuda -v -I"/usr/local/cuda/samples/common/inc" -L"/usr/local/cuda/lib64" LDFLAGS='"$LDFLAGS -Wl,--no-as-needed"' -ldl -lcusparse -lcublas_static -lcusparse_static -lculibos csrmv.cu

mexcuda -v -I"/usr/local/cuda/samples/common/inc" -L"/usr/local/cuda/lib64" LDFLAGS='"$LDFLAGS -Wl,--no-as-needed"' -ldl -lcusparse -lcublas_static -lcusparse_static -lculibos coo2csr.cu

mexcuda -v -I"/usr/local/cuda/samples/common/inc" -L"/usr/local/cuda/lib64" LDFLAGS='"$LDFLAGS -Wl,--no-as-needed"' -ldl -lcusparse -lcublas_static -lcusparse_static -lculibos csr2csc.cu

mexcuda -v -I"/usr/local/cuda/samples/common/inc" -L"/usr/local/cuda/lib64" LDFLAGS='"$LDFLAGS -Wl,--no-as-needed"' -ldl -lcusparse -lcublas_static -lcusparse_static -lculibos csr2csc_cpu.cu

mexcuda -v -I"/usr/local/cuda/samples/common/inc" -L"/usr/local/cuda/lib64" LDFLAGS='"$LDFLAGS -Wl,--no-as-needed"' -ldl -lcusparse -lcublas_static -lcusparse_static -lculibos csrmm.cu

mexcuda -v -I"/usr/local/cuda/samples/common/inc" -L"/usr/local/cuda/lib64" LDFLAGS='"$LDFLAGS -Wl,--no-as-needed"' -ldl -lcusparse -lcublas_static -lcusparse_static -lculibos csr2coo.cu


% try and compile with cuda 8 - not working
if false
    
    cmd = ['/usr/local/cuda-8.0/bin/nvcc -c --compiler-options=-D_GNU_SOURCE,-DMATLAB_MEX_FILE' ...
        ' -I"/usr/local/cuda-8.0/samples/common/inc"  -I"/usr/local/MATLAB/R2016a/extern/include"' ...
        ' -I"/usr/local/MATLAB/R2016a/simulink/include"' ...
        ' -I"/usr/local/MATLAB/R2016a/toolbox/distcomp/gpu/extern/include/"' ...
        ' -gencode=arch=compute_20,code=sm_20 -gencode=arch=compute_30,code=sm_30 -gencode=arch=compute_50,code=\"sm_50,compute_50\"' ...
        ' -std=c++11  --compiler-options=-ansi,-fexceptions,-fPIC,-fno-omit-frame-pointer,-pthread -O -DNDEBUG' ...
        ' csrmv.cu -o csrmv_v8.o'];
    system(cmd)
    
    cmd = ['/usr/bin/g++ -pthread -Wl,--no-undefined  -Wl,--no-as-needed -shared -O' ...
        ' -Wl,--version-script,"/usr/local/MATLAB/R2016a/extern/lib/glnxa64/mexFunction.map"' ...
        ' csrmv_v8.o   -ldl  -lcusparse  -lcublas_static  -lcusparse_static  -lculibos' ...
        ' -L/usr/local/cuda-8.0/lib64   -Wl,-rpath-link,/usr/local/MATLAB/R2016a/bin/glnxa64' ...
        ' -L"/usr/local/MATLAB/R2016a/bin/glnxa64" -lmx -lmex -lmat -lm -lstdc++ -lmwgpu' ...
        ' -lcudart -o csrmv_v8.mexa64']; %' /usr/local/MATLAB/R2016a/bin/glnxa64/libcudart.so.7.5 -o csrmv_v8.mexa64'];
    system(cmd)
    
    cmd = 'rm -f csrmv_v8.o';
    system(cmd)

end




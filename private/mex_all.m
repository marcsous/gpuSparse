function mex_all()

% checks
if ~exist('/usr/local/cuda','dir')
    warning('/usr/local/cuda directory not found. Try:\n%s','"sudo ln -s /usr/local/cuda-11 /usr/local/cuda"')
end

% override MATLAB's supplied version of nvcc - not sure what difference this makes
setenv('MW_ALLOW_ANY_CUDA','1')
setenv('MW_NVCC_PATH', '/usr/local/cuda/bin')

% need to be in the current directory for mexcuda
oldpath = pwd;
newpath = fileparts(mfilename('fullpath'));
cd(newpath);

% if the mexcuda fails, we are stuck - rethrow error
try
    mex_all_compile(); % see below in case of unsuppored gpu errors
    cd(oldpath)
catch ME
    cd(oldpath)
    rethrow(ME)
end

%% call mexcuda
function mex_all_compile()

% Note: if you run into errors like this:
%
%   nvcc fatal   : Unsupported gpu architecture 'compute_35':
%
% then
%
%   sudo vi /usr/local/MATLAB/R2023a/toolbox/parallel/gpu/extern/src/mex/glnxa64/nvcc_g++_dynamic.xml
%
% and remove -gencode=arch=compute_35,code=sm_35 from NVCCFLAGS, e.g.
%
%   NVCCFLAGS="-gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_50,code=sm_50 -gencode=arch=compute_60,code=sm_60...
%
%   NVCCFLAGS="-gencode=arch=compute_50,code=sm_50 -gencode=arch=compute_60,code=sm_60...
%
% Then it should compile.

mexcuda csrgeam.cu      -R2018a -I/usr/local/cuda/include -L/usr/local/cuda/lib64 NVCCFLAGS='"$NVCCFLAGS -w -Wno-deprecated-gpu-targets"' LDFLAGS='"$LDFLAGS -Wl,--no-as-needed"' -ldl -lcusparse -lcublas -lculibos -dynamic -v
mexcuda csrmv.cu        -R2018a -I/usr/local/cuda/include -L/usr/local/cuda/lib64 NVCCFLAGS='"$NVCCFLAGS -w -Wno-deprecated-gpu-targets"' LDFLAGS='"$LDFLAGS -Wl,--no-as-needed"' -ldl -lcusparse -lcublas -lculibos -dynamic
mexcuda coo2csr.cu      -R2018a -I/usr/local/cuda/include -L/usr/local/cuda/lib64 NVCCFLAGS='"$NVCCFLAGS -w -Wno-deprecated-gpu-targets"' LDFLAGS='"$LDFLAGS -Wl,--no-as-needed"' -ldl -lcusparse -lcublas -lculibos -dynamic
mexcuda csr2csc.cu      -R2018a -I/usr/local/cuda/include -L/usr/local/cuda/lib64 NVCCFLAGS='"$NVCCFLAGS -w -Wno-deprecated-gpu-targets"' LDFLAGS='"$LDFLAGS -Wl,--no-as-needed"' -ldl -lcusparse -lcublas -lculibos -dynamic
mexcuda csr2csc_cpu.cu  -R2018a -I/usr/local/cuda/include -L/usr/local/cuda/lib64 NVCCFLAGS='"$NVCCFLAGS -w -Wno-deprecated-gpu-targets"' LDFLAGS='"$LDFLAGS -Wl,--no-as-needed"' -ldl -lcusparse -lcublas -lculibos -dynamic
mexcuda csrmm.cu        -R2018a -I/usr/local/cuda/include -L/usr/local/cuda/lib64 NVCCFLAGS='"$NVCCFLAGS -w -Wno-deprecated-gpu-targets"' LDFLAGS='"$LDFLAGS -Wl,--no-as-needed"' -ldl -lcusparse -lcublas -lculibos -dynamic
mexcuda csr2coo.cu      -R2018a -I/usr/local/cuda/include -L/usr/local/cuda/lib64 NVCCFLAGS='"$NVCCFLAGS -w -Wno-deprecated-gpu-targets"' LDFLAGS='"$LDFLAGS -Wl,--no-as-needed"' -ldl -lcusparse -lcublas -lculibos -dynamic
mexcuda csrsort.cu      -R2018a -I/usr/local/cuda/include -L/usr/local/cuda/lib64 NVCCFLAGS='"$NVCCFLAGS -w -Wno-deprecated-gpu-targets"' LDFLAGS='"$LDFLAGS -Wl,--no-as-needed"' -ldl -lcusparse -lcublas -lculibos -dynamic
mexcuda coosortByRow.cu -R2018a -I/usr/local/cuda/include -L/usr/local/cuda/lib64 NVCCFLAGS='"$NVCCFLAGS -w -Wno-deprecated-gpu-targets"' LDFLAGS='"$LDFLAGS -Wl,--no-as-needed"' -ldl -lcusparse -lcublas -lculibos -dynamic

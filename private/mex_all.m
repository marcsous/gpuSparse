function mex_all()
% tested on Linux 64-bit with Matlab R2016a/CUDA-7.5, R2017a/CUDA-8, 2019b/CUDA-10.1, 2020a/CUDA-10.2, 2020b/CUDA11

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
    mex_all_compile();
    cd(oldpath)
catch ME
    cd(oldpath)
    rethrow(ME)
end


function mex_all_compile()

% clean
delete csrgeam.mex*
delete csrmv.mex*
delete coo2csr.mex*
delete csr2csc.mex*
delete csr2csc_cpu.mex*
delete csrmm.mex*
delete csr2coo.mex*
delete coosortByRow.mex*
delete csrsort.mex*

%% default mexcuda
mexcuda csrgeam.cu      -I/usr/local/cuda/include -L/usr/local/cuda/lib64 NVCCFLAGS='"$NVCCFLAGS -Wno-deprecated-gpu-targets"' LDFLAGS='"$LDFLAGS -Wl,--no-as-needed"' -ldl -lcusparse -lcublas -lculibos -dynamic -v
mexcuda csrmv.cu        -I/usr/local/cuda/include -L/usr/local/cuda/lib64 NVCCFLAGS='"$NVCCFLAGS -Wno-deprecated-gpu-targets"' LDFLAGS='"$LDFLAGS -Wl,--no-as-needed"' -ldl -lcusparse -lcublas -lculibos -dynamic
mexcuda coo2csr.cu      -I/usr/local/cuda/include -L/usr/local/cuda/lib64 NVCCFLAGS='"$NVCCFLAGS -Wno-deprecated-gpu-targets"' LDFLAGS='"$LDFLAGS -Wl,--no-as-needed"' -ldl -lcusparse -lcublas -lculibos -dynamic
mexcuda csr2csc.cu      -I/usr/local/cuda/include -L/usr/local/cuda/lib64 NVCCFLAGS='"$NVCCFLAGS -Wno-deprecated-gpu-targets"' LDFLAGS='"$LDFLAGS -Wl,--no-as-needed"' -ldl -lcusparse -lcublas -lculibos -dynamic
mexcuda csr2csc_cpu.cu  -I/usr/local/cuda/include -L/usr/local/cuda/lib64 NVCCFLAGS='"$NVCCFLAGS -Wno-deprecated-gpu-targets"' LDFLAGS='"$LDFLAGS -Wl,--no-as-needed"' -ldl -lcusparse -lcublas -lculibos -dynamic
mexcuda csrmm.cu        -I/usr/local/cuda/include -L/usr/local/cuda/lib64 NVCCFLAGS='"$NVCCFLAGS -Wno-deprecated-gpu-targets"' LDFLAGS='"$LDFLAGS -Wl,--no-as-needed"' -ldl -lcusparse -lcublas -lculibos -dynamic
mexcuda csr2coo.cu      -I/usr/local/cuda/include -L/usr/local/cuda/lib64 NVCCFLAGS='"$NVCCFLAGS -Wno-deprecated-gpu-targets"' LDFLAGS='"$LDFLAGS -Wl,--no-as-needed"' -ldl -lcusparse -lcublas -lculibos -dynamic
mexcuda coosortByRow.cu -I/usr/local/cuda/include -L/usr/local/cuda/lib64 NVCCFLAGS='"$NVCCFLAGS -Wno-deprecated-gpu-targets"' LDFLAGS='"$LDFLAGS -Wl,--no-as-needed"' -ldl -lcusparse -lcublas -lculibos -dynamic
mexcuda csrsort.cu      -I/usr/local/cuda/include -L/usr/local/cuda/lib64 NVCCFLAGS='"$NVCCFLAGS -Wno-deprecated-gpu-targets"' LDFLAGS='"$LDFLAGS -Wl,--no-as-needed"' -ldl -lcusparse -lcublas -lculibos -dynamic



// 
// Mex wrapper to CUSPARSE format converter (csr2coo).
//
// Inspired by cusparse samples (conugateGradient) and Matlab gcsparse.
//  http://docs.nvidia.com/cuda/cusparse/index.html#cusparse-lt-t-gt-csrmv
//  http://www.mathworks.com/matlabcentral/fileexchange/44423-gpu-sparse--accumarray--non-uniform-grid
//

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

/* Using updated (v2) interfaces to cublas */
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>

// MATLAB related
#include "mex.h"
#include "gpu/mxGPUArray.h"
#include "mxShowCriticalErrorMessage.c"

// Input Arguments
#define	ROW_CSR prhs[0]
#define	NROWS   prhs[1]

// Output Arguments
#define	ROW plhs[0]

void mexFunction(int nlhs, mxArray * plhs[], int nrhs, const mxArray * prhs[])
{
    // Checks
    if (nlhs > 1) mxShowCriticalErrorMessage("wrong number of output arguments",nlhs);
    if (nrhs != 2) mxShowCriticalErrorMessage("wrong number of input arguments",nrhs);

    // Initialize the MathWorks GPU API
    mxInitGPU();

    // Create Matlab pointers on the GPU
    mxGPUArray const *row_csr = mxGPUCreateFromMxArray(ROW_CSR);

    // Checks - note rows must be in CSR format
    if (!mxIsScalar(NROWS)) mxShowCriticalErrorMessage("NROWS argument must be a scalar");
    if (mxGPUGetClassID(row_csr) != mxINT32_CLASS) mxShowCriticalErrorMessage("ROW_CSR argument is not int32");
    mwSize nrows = mxGetScalar(NROWS);
    if (mxGPUGetNumberOfElements(row_csr) != nrows+1) mxShowCriticalErrorMessage("ROW_CSR argument is wrong size",mxGPUGetNumberOfElements(row_csr));

    // Get handle to the CUBLAS context
    cublasHandle_t cublasHandle = 0;
    cublasStatus_t cublasStatus;
    cublasStatus = cublasCreate(&cublasHandle);
    if (cublasStatus != CUBLAS_STATUS_SUCCESS) mxShowCriticalErrorMessage(cublasStatus);

    // Get handle to the CUSPARSE context
    cusparseStatus_t status;
    cusparseHandle_t cusparseHandle = 0;
    cusparseStatus_t cusparseStatus;
    cusparseStatus = cusparseCreate(&cusparseHandle);
    if (cusparseStatus != CUSPARSE_STATUS_SUCCESS) mxShowCriticalErrorMessage(cusparseStatus);
    cusparseMatDescr_t descr = 0;
    cusparseStatus = cusparseCreateMatDescr(&descr);
    if (cusparseStatus != CUSPARSE_STATUS_SUCCESS) mxShowCriticalErrorMessage(cusparseStatus);
    cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ONE);

    // Convert from matlab pointers to native pointers 
    const int * const d_row_csr = (int*)mxGPUGetDataReadOnly(row_csr);

    // Now we can access the arrays, we can do some checks
    int base;
    cudaMemcpy(&base, d_row_csr, sizeof(int), cudaMemcpyDeviceToHost);
    if (base != CUSPARSE_INDEX_BASE_ONE) mxShowCriticalErrorMessage("ROW_CSR not using 1-based indexing");

    int nnz;
    cudaMemcpy(&nnz, d_row_csr+nrows, sizeof(int), cudaMemcpyDeviceToHost);
    nnz -= CUSPARSE_INDEX_BASE_ONE;
    if (nnz < 0) mxShowCriticalErrorMessage("ROW_CSR returned negative nnz");

    // Create space for output vector
    const mwSize ndim = 1;
    mwSize dims[ndim] = {(mwSize)nnz}; // we checked that nnz is >=0 so cast is safe
    mxGPUArray *row = mxGPUCreateGPUArray(ndim, dims, mxINT32_CLASS, mxREAL, MX_GPU_INITIALIZE_VALUES);
    if (row==NULL) mxShowCriticalErrorMessage("mxGPUCreateGPUArray failed");

    // Convert from matlab pointers to native pointers 
    int *d_row = (int*)mxGPUGetData(row);

    // Call csr2coo
    status = cusparseXcsr2coo(cusparseHandle, d_row_csr, nnz, nrows, d_row, CUSPARSE_INDEX_BASE_ONE);

    if (status == CUSPARSE_STATUS_SUCCESS)
    {
    	// Return result
    	ROW = mxGPUCreateMxArrayOnGPU(row);

    	// Make sure operations are finished before deleting
    	//cudaDeviceSynchronize();
    }

    // Clean up
    cusparseDestroyMatDescr(descr);
    cusparseDestroy(cusparseHandle);
    cublasDestroy(cublasHandle);
    mxGPUDestroyGPUArray(row);
    mxGPUDestroyGPUArray(row_csr);

    // Failure
    if (status != CUSPARSE_STATUS_SUCCESS)
    {
	mxShowCriticalErrorMessage("Operation cusparseXcsr2coo failed",status);
    }

    return;
}

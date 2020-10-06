// 
// Mex wrapper to CUSPARSE format converter (coo2csr).
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
#define	ROW   prhs[0]
#define	NROWS prhs[1]

// Output Arguments
#define	ROW_CSR plhs[0]

void mexFunction(int nlhs, mxArray * plhs[], int nrhs, const mxArray * prhs[])
{
    // Checks
    if (nlhs > 1) mxShowCriticalErrorMessage("wrong number of output arguments",nlhs);
    if (nrhs != 2) mxShowCriticalErrorMessage("wrong number of input arguments",nrhs);

    // Initialize the MathWorks GPU API
    mxInitGPU();

    // Create Matlab pointers on the GPU
    mxGPUArray const *row = mxGPUCreateFromMxArray(ROW);

    // Checks - note rows must be in COO (uncompressed) format
    if (!mxIsScalar(NROWS)) mxShowCriticalErrorMessage("NROWS argument must be a scalar");
    if (mxGPUGetClassID(row) != mxINT32_CLASS) mxShowCriticalErrorMessage("ROW argument is not int32");

    mwSize nrows = mxGetScalar(NROWS);
    mwSize nnz = mxGPUGetNumberOfElements(row);

    // Create space for output vector
    const mwSize ndim = 1;
    mwSize dims[ndim] = {nrows+1};
    mxGPUArray *row_csr = mxGPUCreateGPUArray(ndim, dims, mxINT32_CLASS, mxREAL, MX_GPU_INITIALIZE_VALUES);
    if (row_csr==NULL) mxShowCriticalErrorMessage("mxGPUCreateGPUArray failed");

    // Get handle to the CUBLAS context
    cublasHandle_t cublasHandle = 0;
    cublasStatus_t cublasStatus;
    cublasStatus = cublasCreate(&cublasHandle);
    if (cublasStatus != CUBLAS_STATUS_SUCCESS) mxShowCriticalErrorMessage(cublasStatus);

    // Get handle to the CUSPARSE context
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
    const int * const d_row = (int*)mxGPUGetDataReadOnly(row);
    int *d_row_csr = (int*)mxGPUGetData(row_csr);
    char message[128] = {'\0'};
    int *buffer = NULL;

    // Call coo2csr - returns uninitialized when nnz==0 so need to handle separately
    if (nnz == 0)
    {
	buffer = (int *)mxMalloc((nrows+1)*sizeof(int));
	if (buffer == NULL) mxShowCriticalErrorMessage("mxMalloc failed");
    	for (int j=0; j<nrows+1; j++) buffer[j] = CUSPARSE_INDEX_BASE_ONE;
	cudaError_t status = cudaMemcpy((void *)d_row_csr, buffer, (nrows+1)*sizeof(int), cudaMemcpyHostToDevice);
	if (status != cudaSuccess) sprintf(message,"\nOperation cudaMemcpy failed with error code %i",status);
    }
    else
    {
    	cusparseStatus_t status = cusparseXcoo2csr(cusparseHandle, d_row, nnz, nrows, d_row_csr, CUSPARSE_INDEX_BASE_ONE);
    	if (status != CUSPARSE_STATUS_SUCCESS) sprintf(message,"\nOperation cusparseXcoo2csr failed with error code %i",status);
    }


    if (message[0] == '\0')
    {
    	// Return result
    	ROW_CSR = mxGPUCreateMxArrayOnGPU(row_csr);

    	// Make sure operations are finished before deleting
    	//cudaDeviceSynchronize();
    }

    // Clean up
    cusparseDestroyMatDescr(descr);
    cusparseDestroy(cusparseHandle);
    cublasDestroy(cublasHandle);
    mxGPUDestroyGPUArray(row);
    mxGPUDestroyGPUArray(row_csr);
    if (buffer) mxFree(buffer);

    // Failure
    if (message[0] != '\0') mxShowCriticalErrorMessage(message);

    return;
}

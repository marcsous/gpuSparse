// 
// Mex wrapper to CUSPARSE matrix-vector multiply (csrmv).
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

// Utilities and system includes
#include <helper_functions.h>  // helper for shared functions common to CUDA Samples
#include <helper_cuda.h>       // helper function CUDA error checking and initialization

// MATLAB related
#include "mex.h"
#include "gpu/mxGPUArray.h"
#include "mxShowCriticalErrorMessage.c"

// Input Arguments
#define	ROW_CSR prhs[0] // this in CSR format (returned from coo2csr.cu)
#define	COL     prhs[1]
#define	VAL     prhs[2]
#define	NROWS   prhs[3]
#define	NCOLS   prhs[4]
#define	TRANS   prhs[5]
#define	X       prhs[6]

// Output Arguments
#define	Y	plhs[0]

void mexFunction(int nlhs, mxArray * plhs[], int nrhs, const mxArray * prhs[])
{
    // Checks
    if (nlhs > 1) mxShowCriticalErrorMessage("wrong number of output arguments");
    if (nrhs != 7) mxShowCriticalErrorMessage("wrong number of input arguments");

    if(mxIsGPUArray(ROW_CSR) == 0) mxShowCriticalErrorMessage("ROW_CSR argument is not on GPU");
    if(mxIsGPUArray(COL) == 0) mxShowCriticalErrorMessage("COL argument is not on GPU");
    if(mxIsGPUArray(VAL) == 0) mxShowCriticalErrorMessage("VAL argument is not on GPU");

    // Initialize the MathWorks GPU API
    mxInitGPU();

    // Create Matlab pointers on the GPU
    mxGPUArray const *row_csr = mxGPUCreateFromMxArray(ROW_CSR);
    mxGPUArray const *col = mxGPUCreateFromMxArray(COL);
    mxGPUArray const *val = mxGPUCreateFromMxArray(VAL);
    mxGPUArray const *x = mxGPUCreateFromMxArray(X);

    // Check sizes - note rows are in CSR (compressed row) format
    int nnz = mxGPUGetNumberOfElements(val);
    int nrows = (int)mxGetScalar(NROWS);
    int ncols = (int)mxGetScalar(NCOLS);

    mwSize *xdims = (mwSize*)mxGPUGetDimensions(x); // xdims always has >= 2 elements
    if (xdims[1] != 1 || mxGPUGetNumberOfDimensions(x) > 2) mxShowCriticalErrorMessage("X has too many dimensions");
    int nx = xdims[0];

    if (mxGPUGetNumberOfElements(row_csr) != nrows+1) mxShowCriticalErrorMessage("ROW_CSR argument wrong size");
    if (mxGPUGetNumberOfElements(col) != nnz) mxShowCriticalErrorMessage("COL argument wrong size");

    cusparseOperation_t trans = (cusparseOperation_t)mxGetScalar(TRANS);
    if (trans == CUSPARSE_OPERATION_NON_TRANSPOSE)
    {
	if (nx != ncols) mxShowCriticalErrorMessage("X argument wrong size for multiply");
    }
    else
    {
	if (nx != nrows) mxShowCriticalErrorMessage("X argument wrong size for transpose multiply");
    }

    // Check types
    if (mxGPUGetClassID(row_csr) != mxINT32_CLASS) mxShowCriticalErrorMessage("ROW_CSR argument is not int32");
    if (mxGPUGetClassID(col) != mxINT32_CLASS) mxShowCriticalErrorMessage("COL argument is not int32");
    if (mxGPUGetClassID(val) != mxSINGLE_CLASS) mxShowCriticalErrorMessage("VAL argument is not single");
    if (mxGPUGetClassID(x) != mxSINGLE_CLASS) mxShowCriticalErrorMessage("X argument is not single");

    // Check real/complex
    mxComplexity ccx = mxGPUGetComplexity(x);
    if (mxGPUGetComplexity(val) != ccx) mxShowCriticalErrorMessage("VAL and X must have same complexity");

    // Create space for output vector
    const mwSize ndim = 1;
    mwSize dims[ndim] = {trans == CUSPARSE_OPERATION_NON_TRANSPOSE ? nrows : ncols};
    mxClassID cid = mxGPUGetClassID(x);
    
    mxGPUArray *y = mxGPUCreateGPUArray(ndim, dims, cid, ccx, MX_GPU_DO_NOT_INITIALIZE);
    if (y==NULL) mxShowCriticalErrorMessage("mxGPUCreateGPUArray failed.");

    // Get handle to the CUBLAS context
    cublasHandle_t cublasHandle = 0;
    cublasStatus_t cublasStatus;
    cublasStatus = cublasCreate(&cublasHandle);
    checkCudaErrors(cublasStatus);

    // Get handle to the CUSPARSE context
    cusparseHandle_t cusparseHandle = 0;
    cusparseStatus_t cusparseStatus;
    cusparseStatus = cusparseCreate(&cusparseHandle);
    checkCudaErrors(cusparseStatus);
    cusparseMatDescr_t descr = 0;
    cusparseStatus = cusparseCreateMatDescr(&descr);
    checkCudaErrors(cusparseStatus);
    cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ONE); // MATLAB unit offset

    // Convert from matlab pointers to native pointers 
    int *d_row_csr = (int*)mxGPUGetDataReadOnly(row_csr);
    int *d_col = (int*)mxGPUGetDataReadOnly(col);
    float *d_val = (float*)mxGPUGetDataReadOnly(val);
    float *d_x = (float*)mxGPUGetDataReadOnly(x);
    float *d_y = (float*)mxGPUGetData(y);

    // Now we can access the arrays, we can do some checks
    int base;
    cudaMemcpy(&base, d_row_csr, sizeof(int), cudaMemcpyDeviceToHost);
    if (base != CUSPARSE_INDEX_BASE_ONE) mxShowCriticalErrorMessage("ROW_CSR not using 1-based indexing");

    int nnz_check;
    cudaMemcpy(&nnz_check, d_row_csr+nrows, sizeof(int), cudaMemcpyDeviceToHost);
    nnz_check -= CUSPARSE_INDEX_BASE_ONE; // MATLAB unit offset
    if (nnz_check != nnz) mxShowCriticalErrorMessage("ROW_CSR argument last element != nnz");

    // Call cusparse multiply function in (S)ingle precision
    float alpha = 1.0;
    float beta = 0.0;
	char message[128];

#if CUDART_VERSION < 8000
    cusparseStatus_t status =
    cusparseScsrmv(cusparseHandle, trans, nrows, ncols, nnz, &alpha, descr, d_val, d_row_csr, d_col, d_x, &beta, d_y);
	sprintf(message,"\nOperation cusparseScsrmv failed with error code %i",status);
#else
    cusparseStatus_t status =
    cusparseScsrmv_mp(cusparseHandle, trans, nrows, ncols, nnz, &alpha, descr, d_val, d_row_csr, d_col, d_x, &beta, d_y);
	sprintf(message,"\nOperation cusparseScsrmv_mp failed with error code %i",status);
#endif

    if (status == CUSPARSE_STATUS_SUCCESS)
    {
    	// Return result
    	Y = mxGPUCreateMxArrayOnGPU(y);

    	// Make sure operations are finished before deleting
    	//cudaDeviceSynchronize();
    }

    // Clean up
    cusparseDestroy(cusparseHandle);
    cublasDestroy(cublasHandle);
    mxGPUDestroyGPUArray(row_csr);
    mxGPUDestroyGPUArray(col);
    mxGPUDestroyGPUArray(val);
    mxGPUDestroyGPUArray(x);
    mxGPUDestroyGPUArray(y);
    mxFree(xdims);

    // Failure
    if (status != CUSPARSE_STATUS_SUCCESS)
    {
		mxShowCriticalErrorMessage(message);
    }

    return;
}


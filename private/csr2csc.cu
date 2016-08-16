// 
// Mex wrapper to CUSPARSE format converter (csr2csc) to do transpose.
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
#define	ROW_CSR prhs[0] // CSR format
#define	COL     prhs[1]
#define	VAL     prhs[2]
#define	NROWS   prhs[3]
#define	NCOLS   prhs[4]

// Output Arguments
#define	ROW     plhs[0]
#define	COL_CSC plhs[1] // CSC format
#define	VAL_CSC plhs[2]

void mexFunction(int nlhs, mxArray * plhs[], int nrhs, const mxArray * prhs[])
{
    // Checks
    if (nlhs > 3) mxShowCriticalErrorMessage("wrong number of output arguments");
    if (nrhs != 5) mxShowCriticalErrorMessage("wrong number of input arguments");

    // Initialize the MathWorks GPU API
    mxInitGPU();

    // Create Matlab pointers on the GPU
    mxGPUArray const *row_csr = mxGPUCreateFromMxArray(ROW_CSR);
    mxGPUArray const *col = mxGPUCreateFromMxArray(COL);
    mxGPUArray const *val = mxGPUCreateFromMxArray(VAL);

    // Checks - note rows must be in CSR format
    int nnz = mxGPUGetNumberOfElements(val);
    int nrows = (int)mxGetScalar(NROWS);
    int ncols = (int)mxGetScalar(NCOLS);
    if (mxGPUGetClassID(row_csr) != mxINT32_CLASS) mxShowCriticalErrorMessage("ROW_CSR argument is not int32");
    if (mxGPUGetClassID(col) != mxINT32_CLASS) mxShowCriticalErrorMessage("COL argument is not int32");
    if (mxGPUGetClassID(val) != mxSINGLE_CLASS) mxShowCriticalErrorMessage("VAL argument is not single");

    // Create space for output vectors
    const mwSize ndim = 1;
    mwSize dims[ndim];

    dims[0] = ncols+1;
    mxGPUArray *col_csc = mxGPUCreateGPUArray(ndim, dims, mxINT32_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    if (col_csc==NULL) mxShowCriticalErrorMessage("mxGPUCreateGPUArray failed");

    dims[0] = nnz;
    mxGPUArray *row = mxGPUCreateGPUArray(ndim, dims, mxINT32_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    if (row==NULL) mxShowCriticalErrorMessage("mxGPUCreateGPUArray failed");

    mxComplexity ccx = mxGPUGetComplexity(val);
    mxGPUArray *val_csc = mxGPUCreateGPUArray(ndim, dims, mxSINGLE_CLASS, ccx, MX_GPU_DO_NOT_INITIALIZE);
    if (val_csc==NULL) mxShowCriticalErrorMessage("mxGPUCreateGPUArray failed");

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
    cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ONE);

    // Convert from matlab pointers to native pointers 
    int *d_row_csr = (int*)mxGPUGetDataReadOnly(row_csr);
    int *d_col = (int*)mxGPUGetDataReadOnly(col);

    int *d_row = (int*)mxGPUGetData(row);
    int *d_col_csc = (int*)mxGPUGetData(col_csc);

    // Now we can access row_csr[] array
    int base;
    cudaMemcpy(&base, d_row_csr, sizeof(int), cudaMemcpyDeviceToHost);
    if (base != CUSPARSE_INDEX_BASE_ONE) mxShowCriticalErrorMessage("ROW_CSR not using 1-based indexing");

    int nnz_check;
    cudaMemcpy(&nnz_check, d_row_csr+nrows, sizeof(int), cudaMemcpyDeviceToHost);
    nnz_check -= CUSPARSE_INDEX_BASE_ONE;
    if (nnz_check != nnz) mxShowCriticalErrorMessage("ROW_CSR argument last element != nnz");

    // Convert from CSR to CSC
    cusparseStatus_t status;

    if (ccx == mxREAL)
    {
    	float *d_val = (float*)mxGPUGetDataReadOnly(val);
    	float *d_val_csc = (float*)mxGPUGetData(val_csc);
	status =
  	cusparseScsr2csc(cusparseHandle, nrows, ncols, nnz, d_val, d_row_csr, d_col, d_val_csc, d_row, d_col_csc, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ONE);
    }
    else
    {
    	cuComplex *d_val = (cuComplex*)mxGPUGetDataReadOnly(val);
    	cuComplex *d_val_csc = (cuComplex*)mxGPUGetData(val_csc);
	status =
  	cusparseCcsr2csc(cusparseHandle, nrows, ncols, nnz, d_val, d_row_csr, d_col, d_val_csc, d_row, d_col_csc, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ONE);
    }

    if (status == CUSPARSE_STATUS_SUCCESS)
    {
	// Return result
    	ROW = mxGPUCreateMxArrayOnGPU(row);
    	COL_CSC = mxGPUCreateMxArrayOnGPU(col_csc);
    	VAL_CSC = mxGPUCreateMxArrayOnGPU(val_csc);

    	// Make sure operations are finished before deleting
    	//cudaDeviceSynchronize();
    }

    // Clean up
    cusparseDestroy(cusparseHandle);
    cublasDestroy(cublasHandle);
    mxGPUDestroyGPUArray(val);
    mxGPUDestroyGPUArray(col);
    mxGPUDestroyGPUArray(row_csr);
    mxGPUDestroyGPUArray(val_csc);
    mxGPUDestroyGPUArray(col_csc);
    mxGPUDestroyGPUArray(row);

    // Failure
    if (status != CUSPARSE_STATUS_SUCCESS)
    {
	char message[128];
	sprintf(message,"\nOperation cusparseScsr2csc failed with error code %i",status);
	mxShowCriticalErrorMessage(message);
    }

    return;
}



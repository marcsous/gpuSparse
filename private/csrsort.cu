// 
// Mex wrapper to CUSPARSE sort for CSR format (csrsort).
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
#define	COL     prhs[1]
#define	VAL     prhs[2]
#define	NROWS   prhs[3]
#define	NCOLS   prhs[4]

// Output Arguments
#define	COL_SORT plhs[0]
#define	VAL_SORT plhs[1]

void mexFunction(int nlhs, mxArray * plhs[], int nrhs, const mxArray * prhs[])
{
    // Checks
    if (nlhs > 2) mxShowCriticalErrorMessage("wrong number of output arguments",nlhs);
    if (nrhs != 5) mxShowCriticalErrorMessage("wrong number of input arguments",nrhs);

    // Initialize the MathWorks GPU API
    mxInitGPU();

    // Create Matlab pointers on the GPU
    mxGPUArray const *row_csr = mxGPUCreateFromMxArray(ROW_CSR);
    mxGPUArray const *col = mxGPUCreateFromMxArray(COL);
    mxGPUArray const *val = mxGPUCreateFromMxArray(VAL);

    // Checks - note vectors must be in CSR format
    int nnz = mxGPUGetNumberOfElements(val);
    if (mxGPUGetNumberOfElements(col) != nnz) mxShowCriticalErrorMessage("COL and VAL argument length mismatch");

    if (!mxIsScalar(NROWS)) mxShowCriticalErrorMessage("NROWS argument must be a scalar");
    if (!mxIsScalar(NCOLS)) mxShowCriticalErrorMessage("NCOLS argument must be a scalar");

    int ncols = (int)mxGetScalar(NCOLS);
    int nrows = (int)mxGetScalar(NROWS);
    if (mxGPUGetNumberOfElements(row_csr) != nrows+1) mxShowCriticalErrorMessage("ROW_CSR argument wrong size");

    if (mxGPUGetClassID(row_csr) != mxINT32_CLASS) mxShowCriticalErrorMessage("ROW_CSR argument is not int32");
    if (mxGPUGetClassID(col) != mxINT32_CLASS) mxShowCriticalErrorMessage("COL argument is not int32");
    if (mxGPUGetClassID(val) != mxSINGLE_CLASS) mxShowCriticalErrorMessage("VAL argument is not single");

    // Create space for output vectors
    const mwSize ndim = 1;
    mwSize dims[ndim];

    dims[0] = nnz;
    mxGPUArray *col_sort = mxGPUCreateGPUArray(ndim, dims, mxINT32_CLASS, mxREAL, MX_GPU_INITIALIZE_VALUES);
    if (col_sort==NULL) mxShowCriticalErrorMessage("mxGPUCreateGPUArray failed");

    mxComplexity ccx = mxGPUGetComplexity(val);
    mxGPUArray *val_sort = mxGPUCreateGPUArray(ndim, dims, mxSINGLE_CLASS, ccx, MX_GPU_INITIALIZE_VALUES);
    if (val_sort==NULL) mxShowCriticalErrorMessage("mxGPUCreateGPUArray failed");

    // Get handle to the CUBLAS context
    cublasHandle_t cublasHandle = 0;
    cublasStatus_t cublasStatus;
    cublasStatus = cublasCreate(&cublasHandle);
    if (cublasStatus != CUBLAS_STATUS_SUCCESS) mxShowCriticalErrorMessage(cublasStatus);

    // Get handle to the CUSPARSE context
    cudaError_t cudaStatus;
    cusparseStatus_t cusparseStatus;
    cusparseHandle_t cusparseHandle = 0;
    cusparseStatus = cusparseCreate(&cusparseHandle);
    if (cusparseStatus != CUSPARSE_STATUS_SUCCESS) mxShowCriticalErrorMessage(cusparseStatus);
    cusparseMatDescr_t descr = 0;
    cusparseStatus = cusparseCreateMatDescr(&descr);
    if (cusparseStatus != CUSPARSE_STATUS_SUCCESS) mxShowCriticalErrorMessage(cusparseStatus);
    cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ONE);

    // Convert from matlab pointers to native pointers 
    const int * const d_row_csr = (int*)mxGPUGetDataReadOnly(row_csr);
    const int * const d_col = (int*)mxGPUGetDataReadOnly(col);
    int *d_col_sort = (int*)mxGPUGetData(col_sort);

   // Now we can access the arrays, we can do some checks
    int base;
    cudaMemcpy(&base, d_row_csr, sizeof(int), cudaMemcpyDeviceToHost);
    if (base != CUSPARSE_INDEX_BASE_ONE) mxShowCriticalErrorMessage("A_ROW_CSR not using 1-based indexing");

    int nnz_check;
    cudaMemcpy(&nnz_check, d_row_csr+nrows, sizeof(int), cudaMemcpyDeviceToHost);
    nnz_check -= CUSPARSE_INDEX_BASE_ONE;
    if (nnz_check != nnz) mxShowCriticalErrorMessage("ROW_CSR argument last element != nnz",nnz_check);

    // Since sort is in-place, copy the read-only vectors to read-write ones
    cudaStatus = cudaMemcpy((void *)d_col_sort, d_col, nnz*sizeof(int), cudaMemcpyDeviceToDevice);
    if (cudaStatus != cudaSuccess) mxShowCriticalErrorMessage("Operation cudaMemcpy failed");

    if (ccx == mxREAL)
    {
    	const float * const d_val = (float*)mxGPUGetDataReadOnly(val);
    	float *d_val_sort = (float*)mxGPUGetData(val_sort);
    	cudaStatus = cudaMemcpy((void *)d_val_sort, d_val, nnz*sizeof(float), cudaMemcpyDeviceToDevice);
    }
    else
    {
    	const cuFloatComplex * const d_val = (cuFloatComplex*)mxGPUGetDataReadOnly(val);
    	cuFloatComplex *d_val_sort = (cuFloatComplex*)mxGPUGetData(val_sort);
    	cudaStatus = cudaMemcpy((void *)d_val_sort, d_val, nnz*sizeof(cuFloatComplex), cudaMemcpyDeviceToDevice);
    }
    if (cudaStatus != cudaSuccess) mxShowCriticalErrorMessage("Operation cudaMemcpy failed",cudaStatus);

    // Sort by rows
    int *P = NULL;
    void *pBuffer = NULL;
    size_t pBufferSizeInBytes = 0;

    if (nnz > 0)
    {
    	// step 1: allocate buffer
    	cusparseStatus = cusparseXcsrsort_bufferSizeExt(cusparseHandle, nrows, ncols, nnz, d_row_csr, d_col, &pBufferSizeInBytes);
    	if (cusparseStatus != CUSPARSE_STATUS_SUCCESS) mxShowCriticalErrorMessage("Operation cusparseXcoosort_bufferSizeExt failed",cusparseStatus);

    	cudaStatus = cudaMalloc( &pBuffer, sizeof(char)*pBufferSizeInBytes);
    	if (cudaStatus != cudaSuccess) mxShowCriticalErrorMessage("Operation cudaMalloc failed",cudaStatus);

    	// step 2: setup permutation vector P to identity
    	cudaStatus = cudaMalloc( &P, sizeof(int)*nnz);
    	if (cudaStatus != cudaSuccess) mxShowCriticalErrorMessage("Operation cudaMalloc failed",cudaStatus);

	cusparseStatus = cusparseCreateIdentityPermutation(cusparseHandle, nnz, P);
    	if (cusparseStatus != CUSPARSE_STATUS_SUCCESS) mxShowCriticalErrorMessage("Operation cusparseCreateIdentityPermutation failed",cusparseStatus);

    	// step 3: sort COO format by Row
    	cusparseStatus = cusparseXcsrsort(cusparseHandle, nrows, ncols, nnz, descr, d_row_csr, d_col_sort, P, pBuffer);
    	if (cusparseStatus != CUSPARSE_STATUS_SUCCESS) mxShowCriticalErrorMessage("Operation cusparseXcsrsort failed",cusparseStatus);

    	// step 4: gather sorted cooVals
    	if (ccx == mxREAL)
    	{
    	    const float * const d_val = (float*)mxGPUGetDataReadOnly(val);
    	    float *d_val_sort = (float*)mxGPUGetData(val_sort);
	    cusparseStatus = cusparseSgthr(cusparseHandle, nnz, d_val, d_val_sort, P, CUSPARSE_INDEX_BASE_ZERO); // MUST USE BASE_ZERO
	}
	else
	{
    	    const cuFloatComplex * const d_val = (cuFloatComplex*)mxGPUGetDataReadOnly(val);
    	    cuFloatComplex *d_val_sort = (cuFloatComplex*)mxGPUGetData(val_sort);
	    cusparseStatus = cusparseCgthr(cusparseHandle, nnz, d_val, d_val_sort, P, CUSPARSE_INDEX_BASE_ZERO); // MUST USE BASE_ZERO
	}
    	if (cusparseStatus != CUSPARSE_STATUS_SUCCESS) mxShowCriticalErrorMessage("Operation cusparseSgthr or cusparseCgthr failed",cusparseStatus);

    }

    // Return result
    COL_SORT = mxGPUCreateMxArrayOnGPU(col_sort);
    VAL_SORT = mxGPUCreateMxArrayOnGPU(val_sort);

    // Make sure operations are finished before deleting
    //cudaDeviceSynchronize();

    // Clean up
    cusparseDestroyMatDescr(descr);
    cusparseDestroy(cusparseHandle);
    cublasDestroy(cublasHandle);
    mxGPUDestroyGPUArray(row_csr);
    mxGPUDestroyGPUArray(col);
    mxGPUDestroyGPUArray(col_sort);
    mxGPUDestroyGPUArray(val);
    mxGPUDestroyGPUArray(val_sort);
    if (pBuffer) cudaFree(pBuffer);
    if (P) cudaFree(P);

    return;
}

// 
// Mex wrapper to CUSPARSE matrix-matrix addition (csrgeam).
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

#if CUDART_VERSION >= 11000
#include "wrappers_to_cuda_11.h"
#endif    
        
// MATLAB related
#include "mex.h"
#include "gpu/mxGPUArray.h"
#include "mxShowCriticalErrorMessage.c"

// Input Arguments
#define	A_ROW_CSR prhs[0] // this in CSR format (returned from coo2csr.cu)
#define	A_COL     prhs[1]
#define	A_VAL     prhs[2]
#define	NROWS     prhs[3]
#define	NCOLS     prhs[4]
#define	B_ROW_CSR prhs[5] // this in CSR format (returned from coo2csr.cu)
#define	B_COL     prhs[6]
#define	B_VAL     prhs[7]
#define	ALPHA     prhs[8] // scalar: C = ALPHA*A + BETA*B
#define	BETA      prhs[9] // scalar: C = ALPHA*A + BETA*B

// Output Arguments
#define	C_ROW_CSR plhs[0] // this in CSR format (returned from coo2csr.cu)
#define	C_COL     plhs[1]
#define	C_VAL     plhs[2]

void mexFunction(int nlhs, mxArray * plhs[], int nrhs, const mxArray * prhs[])
{
    // Checks
    if (nlhs > 3) mxShowCriticalErrorMessage("wrong number of output arguments",nlhs);
    if (nrhs != 10) mxShowCriticalErrorMessage("wrong number of input arguments",nrhs);

    if(!mxIsGPUArray(A_ROW_CSR)) mxShowCriticalErrorMessage("A_ROW_CSR argument is not on GPU");
    if(!mxIsGPUArray(A_COL)) mxShowCriticalErrorMessage("A_COL argument is not on GPU");
    if(!mxIsGPUArray(A_VAL)) mxShowCriticalErrorMessage("A_VAL argument is not on GPU");

    if (!mxIsScalar(ALPHA)) mxShowCriticalErrorMessage("ALPHA argument must be a scalar");
    if (!mxIsScalar(BETA)) mxShowCriticalErrorMessage("BETA argument must be a scalar");
    if (!mxIsScalar(NROWS)) mxShowCriticalErrorMessage("NROWS argument must be a scalar");
    if (!mxIsScalar(NCOLS)) mxShowCriticalErrorMessage("NCOLS argument must be a scalar");

    // Initialize the MathWorks GPU API
    mxInitGPU();

    // Create Matlab pointers on the GPU
    mxGPUArray const *a_row_csr = mxGPUCreateFromMxArray(A_ROW_CSR);
    mxGPUArray const *a_col = mxGPUCreateFromMxArray(A_COL);
    mxGPUArray const *a_val = mxGPUCreateFromMxArray(A_VAL);
    mxGPUArray const *b_row_csr = mxGPUCreateFromMxArray(B_ROW_CSR);
    mxGPUArray const *b_col = mxGPUCreateFromMxArray(B_COL);
    mxGPUArray const *b_val = mxGPUCreateFromMxArray(B_VAL);

    // Check sizes - note rows are in CSR (compressed row) format
    int a_nnz = mxGPUGetNumberOfElements(a_val);
    int b_nnz = mxGPUGetNumberOfElements(b_val);

    mwSize nrows = mxGetScalar(NROWS);
    mwSize ncols = mxGetScalar(NCOLS);

    if (mxGPUGetNumberOfElements(a_row_csr) != nrows+1) mxShowCriticalErrorMessage("A_ROW_CSR argument wrong size",mxGPUGetNumberOfElements(a_row_csr));
    if (mxGPUGetNumberOfElements(a_col) != a_nnz) mxShowCriticalErrorMessage("A_COL argument wrong size",mxGPUGetNumberOfElements(a_col));

    if (mxGPUGetNumberOfElements(b_row_csr) != nrows+1) mxShowCriticalErrorMessage("B_ROW_CSR argument wrong size",mxGPUGetNumberOfElements(b_row_csr));
    if (mxGPUGetNumberOfElements(b_col) != b_nnz) mxShowCriticalErrorMessage("B_COL argument wrong size",mxGPUGetNumberOfElements(b_col));

    if (mxGPUGetClassID(a_row_csr) != mxINT32_CLASS) mxShowCriticalErrorMessage("A_ROW_CSR argument is not int32");
    if (mxGPUGetClassID(a_col) != mxINT32_CLASS) mxShowCriticalErrorMessage("A_COL argument is not int32");
    if (mxGPUGetClassID(a_val) != mxSINGLE_CLASS) mxShowCriticalErrorMessage("A_VAL argument is not single");

    if (mxGPUGetClassID(b_row_csr) != mxINT32_CLASS) mxShowCriticalErrorMessage("B_ROW argument is not int32");
    if (mxGPUGetClassID(b_col) != mxINT32_CLASS) mxShowCriticalErrorMessage("B_COL argument is not int32");
    if (mxGPUGetClassID(b_val) != mxSINGLE_CLASS) mxShowCriticalErrorMessage("B_VAL argument is not single");

    // Allocate space for output row vector
    const mwSize ndim = 1;
    mwSize dims[ndim] = {nrows+1};
    mxGPUArray *c_row_csr = mxGPUCreateGPUArray(ndim, dims, mxINT32_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    if (c_row_csr==NULL) mxShowCriticalErrorMessage("mxGPUCreateGPUArray failed.");

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
    const int* const d_a_col = (int*)mxGPUGetDataReadOnly(a_col);
    const int* const d_b_col = (int*)mxGPUGetDataReadOnly(b_col);

    const float* const d_a_val = (float*)mxGPUGetDataReadOnly(a_val);
    const float* const d_b_val = (float*)mxGPUGetDataReadOnly(b_val);

    const int* const d_a_row_csr = (int*)mxGPUGetDataReadOnly(a_row_csr);
    const int* const d_b_row_csr = (int*)mxGPUGetDataReadOnly(b_row_csr);

    int *d_c_col = NULL;
    float *d_c_val = NULL;
    int *d_c_row_csr = (int*)mxGPUGetData(c_row_csr);
   
    const float alpha = (float)mxGetScalar(ALPHA);
    const float beta = (float)mxGetScalar(BETA);

    // Now we can access the arrays, we can do some checks
    int base;
    cudaMemcpy(&base, d_a_row_csr, sizeof(int), cudaMemcpyDeviceToHost);
    if (base != CUSPARSE_INDEX_BASE_ONE) mxShowCriticalErrorMessage("A_ROW_CSR not using 1-based indexing");

    int nnz_check;
    cudaMemcpy(&nnz_check, d_a_row_csr+nrows, sizeof(int), cudaMemcpyDeviceToHost);
    nnz_check -= CUSPARSE_INDEX_BASE_ONE;
    if (nnz_check != a_nnz) mxShowCriticalErrorMessage("A_ROW_CSR argument last element != nnz",nnz_check);

    cudaMemcpy(&base, d_b_row_csr, sizeof(int), cudaMemcpyDeviceToHost);
    if (base != CUSPARSE_INDEX_BASE_ONE) mxShowCriticalErrorMessage("B_ROW_CSR not using 1-based indexing");

    cudaMemcpy(&nnz_check, d_b_row_csr+nrows, sizeof(int), cudaMemcpyDeviceToHost);
    nnz_check -= CUSPARSE_INDEX_BASE_ONE;
    if (nnz_check != b_nnz) mxShowCriticalErrorMessage("B_ROW_CSR argument last element != nnz",nnz_check);

    // Get sparsity pattern and nnz of output matrix
    int c_nnz;
    int *nnzTotalDevHostPtr = &c_nnz;
    cusparseSetPointerMode(cusparseHandle, CUSPARSE_POINTER_MODE_HOST);

    char *buffer = NULL;            
    size_t bufferSizeInBytes;

#if CUDART_VERSION >= 11000
    cusparseScsrgeam2_bufferSizeExt(cusparseHandle, nrows, ncols,
        &alpha,
        descr, a_nnz, d_a_val, d_a_row_csr, d_a_col,
        &beta,
        descr, b_nnz, d_b_val, d_b_row_csr, d_b_col,
        descr,        d_c_val, d_c_row_csr, d_c_col,
        &bufferSizeInBytes);

    cudaError_t status0 = cudaMalloc((void**)&buffer, sizeof(char)*bufferSizeInBytes);
    if (status0 != cudaSuccess)
    {
        mxShowCriticalErrorMessage("Operation cudaMalloc failed",status0);
    }

    cusparseStatus_t status1 =
    cusparseXcsrgeam2Nnz(cusparseHandle, nrows, ncols,
        	descr, a_nnz, d_a_row_csr, d_a_col,
        	descr, b_nnz, d_b_row_csr, d_b_col,
        	descr, d_c_row_csr, nnzTotalDevHostPtr, buffer);
#else
    cusparseStatus_t status1 =
    cusparseXcsrgeamNnz(cusparseHandle, nrows, ncols,
        	descr, a_nnz, d_a_row_csr, d_a_col,
        	descr, b_nnz, d_b_row_csr, d_b_col,
        	descr, d_c_row_csr, nnzTotalDevHostPtr);
#endif

    // Failure
    if (status1 != CUSPARSE_STATUS_SUCCESS)
    {
        mxShowCriticalErrorMessage("Operation cusparseXcsrgeamNnz failed",status1);
    }

    if (NULL != nnzTotalDevHostPtr)
    {
        c_nnz = *nnzTotalDevHostPtr;
    }
    else
    {
    	int baseC = CUSPARSE_INDEX_BASE_ONE;
        cudaMemcpy(&c_nnz, d_c_row_csr+nrows, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&baseC, c_row_csr, sizeof(int), cudaMemcpyDeviceToHost);
        c_nnz -= baseC;
    }

    // Allocate space for output vectors
    dims[0] = {(mwSize)c_nnz};
    mxGPUArray *c_col = mxGPUCreateGPUArray(ndim, dims, mxINT32_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    if (c_col==NULL) mxShowCriticalErrorMessage("mxGPUCreateGPUArray failed");

    mxGPUArray *c_val = mxGPUCreateGPUArray(ndim, dims, mxSINGLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    if (c_val==NULL) mxShowCriticalErrorMessage("mxGPUCreateGPUArray failed");

    // Convert from matlab pointers to native pointers
    d_c_col = (int*)mxGPUGetData(c_col);
    d_c_val = (float*)mxGPUGetData(c_val);

    // Addition here
#if CUDART_VERSION >= 11000
    cusparseStatus_t status2 =
    cusparseScsrgeam2(cusparseHandle, nrows, ncols,
	        &alpha,
	        descr, a_nnz,
	        d_a_val, d_a_row_csr, d_a_col,
	        &beta,
	        descr, b_nnz,
	        d_b_val, d_b_row_csr, d_b_col,
	        descr,
	        d_c_val, d_c_row_csr, d_c_col, buffer);
#else
    cusparseStatus_t status2 =
    cusparseScsrgeam(cusparseHandle, nrows, ncols,
	        &alpha,
	        descr, a_nnz,
	        d_a_val, d_a_row_csr, d_a_col,
	        &beta,
	        descr, b_nnz,
	        d_b_val, d_b_row_csr, d_b_col,
	        descr,
	        d_c_val, d_c_row_csr, d_c_col);
#endif

    if (status2 == CUSPARSE_STATUS_SUCCESS)
    {
    	// Return results
    	C_ROW_CSR = mxGPUCreateMxArrayOnGPU(c_row_csr);
    	C_COL = mxGPUCreateMxArrayOnGPU(c_col);
    	C_VAL = mxGPUCreateMxArrayOnGPU(c_val);

    	// Make sure operations are finished before deleting
    	//cudaDeviceSynchronize();
    }

    // Clean up
    cusparseDestroyMatDescr(descr);
    cusparseDestroy(cusparseHandle);
    cublasDestroy(cublasHandle);
    if(buffer) cudaFree(buffer);
    mxGPUDestroyGPUArray(a_row_csr);
    mxGPUDestroyGPUArray(a_col);
    mxGPUDestroyGPUArray(a_val);
    mxGPUDestroyGPUArray(b_row_csr);
    mxGPUDestroyGPUArray(b_col);
    mxGPUDestroyGPUArray(b_val);
    mxGPUDestroyGPUArray(c_row_csr);
    mxGPUDestroyGPUArray(c_col);
    mxGPUDestroyGPUArray(c_val);

    // Failure
    if (status2 != CUSPARSE_STATUS_SUCCESS)
    {
	mxShowCriticalErrorMessage("Operation cusparseScsrgeam failed",status2);
    }

    return;
}

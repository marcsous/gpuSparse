// 
// Mex wrapper to CUSPARSE matrix-matrix multiply (csrmm).
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
#define	ROW_CSR prhs[0] // this in CSR format (returned from coo2csr.cu)
#define	COL     prhs[1]
#define	VAL     prhs[2]
#define	NROWS   prhs[3]
#define	NCOLS   prhs[4]
#define	TRANS   prhs[5]
#define	B       prhs[6] // dense matrix

// Output Arguments
#define	C	plhs[0] // C = op(A) * B (sparse A, dense B)

void mexFunction(int nlhs, mxArray * plhs[], int nrhs, const mxArray * prhs[])
{
    // Checks
    if (nlhs > 1) mxShowCriticalErrorMessage("wrong number of output arguments",nlhs);
    if (nrhs != 7) mxShowCriticalErrorMessage("wrong number of input arguments",nrhs);

    if(!mxIsGPUArray(ROW_CSR)) mxShowCriticalErrorMessage("ROW_CSR argument is not on GPU");
    if(!mxIsGPUArray(COL)) mxShowCriticalErrorMessage("COL argument is not on GPU");
    if(!mxIsGPUArray(VAL)) mxShowCriticalErrorMessage("VAL argument is not on GPU");
    //if(!mxIsGPUArray(B)) mxShowCriticalErrorMessage("B argument is not on GPU");

    // Initialize the MathWorks GPU API
    mxInitGPU();

    // Create Matlab pointers on the GPU
    mxGPUArray const *row_csr = mxGPUCreateFromMxArray(ROW_CSR);
    mxGPUArray const *col = mxGPUCreateFromMxArray(COL);
    mxGPUArray const *val = mxGPUCreateFromMxArray(VAL);
    mxGPUArray const *b = mxGPUCreateFromMxArray(B);

    // Check sizes of A - note rows are in CSR (compressed row) format
    int nnz = mxGPUGetNumberOfElements(val);

    if (!mxIsScalar(NROWS)) mxShowCriticalErrorMessage("NROWS argument must be a scalar");
    if (!mxIsScalar(NCOLS)) mxShowCriticalErrorMessage("NCOLS argument must be a scalar");
    if (!mxIsScalar(TRANS)) mxShowCriticalErrorMessage("TRANS argument must be a scalar");

    int m = (int)mxGetScalar(NROWS);
    int k = (int)mxGetScalar(NCOLS);

    if (mxGPUGetNumberOfElements(row_csr) != m+1) mxShowCriticalErrorMessage("ROW_CSR argument wrong size",mxGPUGetNumberOfElements(row_csr));
    if (mxGPUGetNumberOfElements(col) != nnz) mxShowCriticalErrorMessage("COL argument wrong size",mxGPUGetNumberOfElements(col));

    // Check sizes of B
    if (mxGPUGetNumberOfDimensions(b) > 2) mxShowCriticalErrorMessage("B has too many dimensions",mxGPUGetNumberOfDimensions(b));

    mwSize *bdims = (mwSize*)mxGPUGetDimensions(b); // dims always has >= 2 elements
    int ldb = bdims[0]; // leading dimension of B
    int n = bdims[1];

    cusparseOperation_t trans = (cusparseOperation_t)mxGetScalar(TRANS);
    if (trans == CUSPARSE_OPERATION_NON_TRANSPOSE)
    {
	if (ldb != k) mxShowCriticalErrorMessage("B argument wrong size for multiply",ldb);
    }
    else
    {
	if (ldb != m) mxShowCriticalErrorMessage("B argument wrong size for transpose multiply",ldb);
    }

    // Check types
    if (mxGPUGetClassID(row_csr) != mxINT32_CLASS) mxShowCriticalErrorMessage("ROW_CSR argument is not int32");
    if (mxGPUGetClassID(col) != mxINT32_CLASS) mxShowCriticalErrorMessage("COL argument is not int32");
    if (mxGPUGetClassID(val) != mxSINGLE_CLASS) mxShowCriticalErrorMessage("VAL argument is not single");
    if (mxGPUGetClassID(b) != mxSINGLE_CLASS) mxShowCriticalErrorMessage("B argument is not single");

    // Check real/complex
    mxComplexity ccb = mxGPUGetComplexity(b);
    if (mxGPUGetComplexity(val) != ccb) mxShowCriticalErrorMessage("VAL and B must have same complexity");

    // Create space for output vectors
    const mwSize ndim = 2;
    mwSize cdims[ndim] = {trans == CUSPARSE_OPERATION_NON_TRANSPOSE ? m : k, n};
    mxClassID cid = mxGPUGetClassID(b); // same class as B matrix
    int ldc = cdims[0]; // leading dimension of C
    
    mxGPUArray *c = mxGPUCreateGPUArray(ndim, cdims, cid, ccb, MX_GPU_INITIALIZE_VALUES); // initialize 0
    if (c==NULL) mxShowCriticalErrorMessage("mxGPUCreateGPUArray failed");

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
    const int * const d_row_csr = (int*)mxGPUGetDataReadOnly(row_csr);
    const int * const d_col = (int*)mxGPUGetDataReadOnly(col);
    const float * const d_val = (float*)mxGPUGetDataReadOnly(val);
    const float * const d_b = (float*)mxGPUGetDataReadOnly(b);
    float *d_c = (float*)mxGPUGetData(c);

    // Now we can access the arrays, we can do some checks
    int base;
    cudaMemcpy(&base, d_row_csr, sizeof(int), cudaMemcpyDeviceToHost);
    if (base != CUSPARSE_INDEX_BASE_ONE) mxShowCriticalErrorMessage("ROW_CSR not using 1-based indexing");

    int nnz_check;
    cudaMemcpy(&nnz_check, d_row_csr+m, sizeof(int), cudaMemcpyDeviceToHost);
    nnz_check -= CUSPARSE_INDEX_BASE_ONE;
    if (nnz_check != nnz) mxShowCriticalErrorMessage("ROW_CSR argument last element != nnz",nnz_check);

    // Call cusparse multiply function in (S)ingle precision
    const float alpha = 1.0;
    const float beta = 0.0;

    cusparseStatus = cusparseScsrmm(cusparseHandle, trans, m, n, k, nnz, &alpha, descr, d_val, d_row_csr, d_col, d_b, ldb, &beta, d_c, ldc);

    if (cusparseStatus == CUSPARSE_STATUS_SUCCESS)
    {
    	// Return result
    	C = mxGPUCreateMxArrayOnGPU(c);

    	// Make sure operations are finished before deleting
    	//cudaDeviceSynchronize();
    }

    // Clean up
    cusparseDestroyMatDescr(descr);
    cusparseDestroy(cusparseHandle);
    cublasDestroy(cublasHandle);
    mxGPUDestroyGPUArray(row_csr);
    mxGPUDestroyGPUArray(col);
    mxGPUDestroyGPUArray(val);
    mxGPUDestroyGPUArray(b);
    mxGPUDestroyGPUArray(c);
    mxFree(bdims);

    // Failure
    if (cusparseStatus != CUSPARSE_STATUS_SUCCESS)
    {
	mxShowCriticalErrorMessage("Operation cusparseScsrmm failed",cusparseStatus);
    }

    return;
}


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

#if CUDART_VERSION >= 11000
#include "wrappers_to_cuda_11.h"
#endif
  
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
    if(!mxIsGPUArray(B)) mxShowCriticalErrorMessage("B argument is not on GPU");

    // Initialize the MathWorks GPU API
    mxInitGPU();

    // Create Matlab pointers on the GPU
    mxGPUArray const *row_csr = mxGPUCreateFromMxArray(ROW_CSR);
    mxGPUArray const *col = mxGPUCreateFromMxArray(COL);
    mxGPUArray const *val = mxGPUCreateFromMxArray(VAL);
    mxGPUArray const *b = mxGPUCreateFromMxArray(B);

    // Check sizes of A - note rows are in CSR (compressed row) format
    mwSize nnz = mxGPUGetNumberOfElements(val);

    if (!mxIsScalar(NROWS)) mxShowCriticalErrorMessage("NROWS argument must be a scalar");
    if (!mxIsScalar(NCOLS)) mxShowCriticalErrorMessage("NCOLS argument must be a scalar");
    if (!mxIsScalar(TRANS)) mxShowCriticalErrorMessage("TRANS argument must be a scalar");

    mwSize m = mxGetScalar(NROWS);
    mwSize k = mxGetScalar(NCOLS);

    if (mxGPUGetNumberOfElements(row_csr) != m+1) mxShowCriticalErrorMessage("ROW_CSR argument wrong size",mxGPUGetNumberOfElements(row_csr));
    if (mxGPUGetNumberOfElements(col) != nnz) mxShowCriticalErrorMessage("COL argument wrong size",mxGPUGetNumberOfElements(col));

    // Check sizes of B
    if (mxGPUGetNumberOfDimensions(b) > 2) mxShowCriticalErrorMessage("B has too many dimensions",mxGPUGetNumberOfDimensions(b));

    mwSize *bdims = (mwSize*)mxGPUGetDimensions(b); // dims always has >= 2 elements
    mwSize ldb = bdims[0]; // leading dimension of B
    mwSize n = bdims[1];

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

    // Check real/complex - mixed is not supported except special case (real A / complex B)
    mxComplexity ccb = mxGPUGetComplexity(b);          
    mxComplexity ccv = mxGPUGetComplexity(val);
    mxComplexity ccc = (ccb==mxCOMPLEX || ccv==mxCOMPLEX) ? mxCOMPLEX : mxREAL;
    if(ccb==mxREAL && ccv==mxCOMPLEX) mxShowCriticalErrorMessage("Complex matrix and real vector not supported");

    // Create space for output vectors
    const mwSize ndim = 2;
    mwSize cdims[ndim] = {trans == CUSPARSE_OPERATION_NON_TRANSPOSE ? m : k, n};
    mxClassID cid = mxGPUGetClassID(b); // same class as B matrix
    int ldc = cdims[0]; // leading dimension of C
    mxGPUArray *c;

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

    // Now we can access the arrays, we can do some checks
    int base;
    cudaMemcpy(&base, d_row_csr, sizeof(int), cudaMemcpyDeviceToHost);
    if (base != CUSPARSE_INDEX_BASE_ONE) mxShowCriticalErrorMessage("ROW_CSR not using 1-based indexing");

    int nnz_check;
    cudaMemcpy(&nnz_check, d_row_csr+m, sizeof(int), cudaMemcpyDeviceToHost);
    nnz_check -= CUSPARSE_INDEX_BASE_ONE;
    if (nnz_check != nnz) mxShowCriticalErrorMessage("ROW_CSR argument last element != nnz",nnz_check);

    // Call cusparse multiply function in (S)ingle precision
    if (ccv==mxREAL && ccb==mxREAL)
    {
        const float alpha = 1.0;
        const float beta = 0.0;
        const float * const d_val = (float*)mxGPUGetDataReadOnly(val);
        const float * const d_b = (float*)mxGPUGetDataReadOnly(b);
       
        c = mxGPUCreateGPUArray(ndim, cdims, cid, ccc, MX_GPU_INITIALIZE_VALUES);
        if (c==NULL) mxShowCriticalErrorMessage("mxGPUCreateGPUArray failed");
        float *d_c = (float*)mxGPUGetData(c);

#if CUDART_VERSION >= 11000
        cusparseStatus = cusparseXcsrmm_wrapper(cusparseHandle, trans, m, k, n, nnz, &alpha, descr, d_val, d_row_csr, d_col, d_b, ldb, &beta, d_c, ldc);
#else
        cusparseStatus = cusparseScsrmm(cusparseHandle, trans, m, n, k, nnz, &alpha, descr, d_val, d_row_csr, d_col, d_b, ldb, &beta, d_c, ldc);
#endif
    }
    else if (ccv==mxREAL && ccb==mxCOMPLEX)
    {
        const float alpha = 1.0;
        const float beta = 0.0;
        const float * const d_val = (float*)mxGPUGetDataReadOnly(val);

        mxGPUArray* c_real = mxGPUCreateGPUArray(ndim, cdims, cid, mxREAL, MX_GPU_INITIALIZE_VALUES);
        mxGPUArray* c_imag = mxGPUCreateGPUArray(ndim, cdims, cid, mxREAL, MX_GPU_INITIALIZE_VALUES);
        if(!c_real || !c_imag) mxShowCriticalErrorMessage("mxGPUCreateGPUArray failed.");
        float* d_c_real = (float*)mxGPUGetDataReadOnly(c_real);
        float* d_c_imag = (float*)mxGPUGetDataReadOnly(c_imag);
        
        for(int i = 0; i<2; i++)
        {
            mxGPUArray const *b_tmp;
            if(i==0) b_tmp = mxGPUCopyReal(b);
            if(i==1) b_tmp = mxGPUCopyImag(b);
            const float* const d_b = (float*)mxGPUGetDataReadOnly(b_tmp);

#if CUDART_VERSION >= 11000
            if(i==0) cusparseStatus = cusparseXcsrmm_wrapper(cusparseHandle, trans, m, k, n, nnz, &alpha, descr, d_val, d_row_csr, d_col, d_b, ldb, &beta, d_c_real, ldc);
            if(i==1) cusparseStatus = cusparseXcsrmm_wrapper(cusparseHandle, trans, m, k, n, nnz, &alpha, descr, d_val, d_row_csr, d_col, d_b, ldb, &beta, d_c_imag, ldc);
#else
            if(i==0) cusparseStatus = cusparseScsrmm(cusparseHandle, trans, m, n, k, nnz, &alpha, descr, d_val, d_row_csr, d_col, d_b, ldb, &beta, d_c_real, ldc);
            if(i==1) cusparseStatus = cusparseScsrmm(cusparseHandle, trans, m, n, k, nnz, &alpha, descr, d_val, d_row_csr, d_col, d_b, ldb, &beta, d_c_imag, ldc);
#endif
            mxGPUDestroyGPUArray(b_tmp);
            if(cusparseStatus != CUSPARSE_STATUS_SUCCESS) mxShowCriticalErrorMessage("csrmm failed.");
        }
        c = mxGPUCreateComplexGPUArray(c_real,c_imag);
        if (c==NULL) mxShowCriticalErrorMessage("mxGPUCreateComplexGPUArray failed.");
        mxGPUDestroyGPUArray(c_real);
        mxGPUDestroyGPUArray(c_imag);
    }
    else
    {
        const cuFloatComplex alpha = make_cuFloatComplex(1.0, 0.0);
        const cuFloatComplex beta = make_cuFloatComplex(0.0, 0.0);
        const cuFloatComplex * const d_val = (cuFloatComplex*)mxGPUGetDataReadOnly(val);
        const cuFloatComplex * const d_b = (cuFloatComplex*)mxGPUGetDataReadOnly(b);

        c = mxGPUCreateGPUArray(ndim, cdims, cid, ccc, MX_GPU_INITIALIZE_VALUES);
        if (c==NULL) mxShowCriticalErrorMessage("mxGPUCreateGPUArray failed");
        cuFloatComplex *d_c = (cuFloatComplex*)mxGPUGetData(c);

#if CUDART_VERSION >= 11000
        cusparseStatus = cusparseXcsrmm_wrapper(cusparseHandle, trans, m, k, n, nnz, &alpha, descr, d_val, d_row_csr, d_col, d_b, ldb, &beta, d_c, ldc);
#else
        cusparseStatus = cusparseCcsrmm(cusparseHandle, trans, m, n, k, nnz, &alpha, descr, d_val, d_row_csr, d_col, d_b, ldb, &beta, d_c, ldc);
#endif
    }
   
 	// Return result
    if (cusparseStatus == CUSPARSE_STATUS_SUCCESS)
    {
    	C = mxGPUCreateMxArrayOnGPU(c);
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

    return;
}


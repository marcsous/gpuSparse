// wrappers to the new CUDA 11 interface for pre-11 code

#include <cuda_runtime.h> 
#include <cusparse.h> 
#include "mxShowCriticalErrorMessage.c"
#include <iostream>

#define CHECK_CUSPARSE(func)                                   \
{                                                              \
    cusparseStatus_t status = (func);                          \
    if (status != CUSPARSE_STATUS_SUCCESS)                     \
        mxShowCriticalErrorMessage("cusparseStatus_t",status); \
}

template<typename T> cudaDataType type_to_enum();
template<> cudaDataType type_to_enum<float>()           { return CUDA_R_32F; }
template<> cudaDataType type_to_enum<cuComplex>()       { return CUDA_C_32F; }

// -------------------------------------------------------------------------------//
template<typename T>
cusparseStatus_t
cusparseXcsrmv_wrapper(cusparseHandle_t         handle,
                       cusparseOperation_t      transA,
                       int                      A_num_rows,
                       int                      A_num_cols,
                       int                      A_num_nnz,
                       const T*                 alpha,
                       const cusparseMatDescr_t descrA,
                       const T*                 dA_values,
                       const int*               dA_csrOffsets,
                       const int*               dA_columns,
                       const T*                 dX,
                       const T*                 beta,
                       void*                    dY)
{
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;
    void*        buffer     = NULL;
    size_t       bufferSize = 0;
    cudaDataType typeA       = type_to_enum<T>();
    cudaDataType typeX       = type_to_enum<T>();
    cudaDataType typeY       = (typeA==CUDA_C_32F || typeX==CUDA_C_32F) ? CUDA_C_32F : CUDA_R_32F;
    
//std::cout << "typeA " << typeA << " typeX " << typeX << " typeY " << typeY << std::endl;
    
    // Create sparse matrix A in CSR format
    CHECK_CUSPARSE( cusparseCreateCsr(&matA, A_num_rows, A_num_cols, A_num_nnz,
                                      (void*)dA_csrOffsets, (void*)dA_columns, (void*)dA_values,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      cusparseGetMatIndexBase(descrA), typeA) )
    // Create dense vector X
    int X_rows = (transA==CUSPARSE_OPERATION_NON_TRANSPOSE) ? A_num_cols : A_num_rows;
    CHECK_CUSPARSE( cusparseCreateDnVec(&vecX, X_rows, (void*)dX, typeX) )
    // Create dense vector y
    int Y_rows = (transA==CUSPARSE_OPERATION_NON_TRANSPOSE) ? A_num_rows : A_num_cols;
    CHECK_CUSPARSE( cusparseCreateDnVec(&vecY, Y_rows, (void*)dY, typeY) )
    // allocate an external buffer if needed
    CHECK_CUSPARSE( cusparseSpMV_bufferSize(
                                 handle, transA,
                                 alpha, matA, vecX, beta, vecY, typeY,
                                 CUSPARSE_MV_ALG_DEFAULT, &bufferSize) )
    if (bufferSize > 0)
    {
        cudaError_t status = cudaMalloc(&buffer, bufferSize);
        if (status != cudaSuccess)
            return CUSPARSE_STATUS_ALLOC_FAILED;
    }
    
//std::cout << "bufferSize " << bufferSize << " CUSPARSE_STATUS_NOT_SUPPORTED " << CUSPARSE_STATUS_NOT_SUPPORTED << std::endl;     
    
    // execute SpMV
    CHECK_CUSPARSE( cusparseSpMV(handle, transA,
                                 alpha, matA, vecX, beta, vecY, typeY,
                                 CUSPARSE_MV_ALG_DEFAULT, buffer) )

                                 
    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
    CHECK_CUSPARSE( cusparseDestroyDnVec(vecX) )
    CHECK_CUSPARSE( cusparseDestroyDnVec(vecY) )
    if(buffer) cudaFree(buffer);
    return CUSPARSE_STATUS_SUCCESS;
}

// -------------------------------------------------------------------------------//
template<typename T, typename S>
cusparseStatus_t
cusparseXcsrmm_wrapper(cusparseHandle_t         handle,
                       cusparseOperation_t      transA,
                       int                      A_num_rows,
                       int                      A_num_cols,
                       int                      B_num_cols,
                       int                      A_num_nnz,
                       const T*                 alpha,
                       const cusparseMatDescr_t descrA,
                       const T*                 dA_values,
                       const int*               dA_csrOffsets,
                       const int*               dA_columns,
                       const S*                 dB,
                       int                      ldb,
                       const T*                 beta,
                       void*                    dC,
                       int                      ldc)
{
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;
    void*        buffer     = NULL;
    size_t       bufferSize = 0;
    cudaDataType typeA      = type_to_enum<T>();
    cudaDataType typeB      = type_to_enum<S>();  
    cudaDataType typeC      = (typeA==CUDA_C_32F || typeB==CUDA_C_32F) ? CUDA_C_32F : CUDA_R_32F;

    // handle some limited transpose functionality (A or A' only)
    cusparseOperation_t transB = CUSPARSE_OPERATION_NON_TRANSPOSE;   
    int B_num_rows = (transA==CUSPARSE_OPERATION_NON_TRANSPOSE) ? A_num_cols : A_num_rows;
    int C_num_rows = (transA==CUSPARSE_OPERATION_NON_TRANSPOSE) ? A_num_rows : A_num_cols;
    int C_num_cols = (transB==CUSPARSE_OPERATION_NON_TRANSPOSE) ? B_num_cols : B_num_rows;
    
    // Create sparse matrix A in CSR format
    CHECK_CUSPARSE( cusparseCreateCsr(&matA, A_num_rows, A_num_cols, A_num_nnz,
                                      (void*)dA_csrOffsets, (void*)dA_columns, (void*)dA_values,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      cusparseGetMatIndexBase(descrA), typeA) )
    // Create dense vector X
    CHECK_CUSPARSE( cusparseCreateDnMat(&matB, B_num_rows, B_num_cols, ldb, (void*)dB, typeB, CUSPARSE_ORDER_COL) )
    // Create dense vector y
    CHECK_CUSPARSE( cusparseCreateDnMat(&matC, C_num_rows, C_num_cols, ldc, (void*)dC, typeC, CUSPARSE_ORDER_COL) )
    // allocate an external buffer if needed
    CHECK_CUSPARSE( cusparseSpMM_bufferSize(
                                 handle, transA, transB,
                                 alpha, matA, matB, beta, matC, typeC,
                                 CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize) )
    if (bufferSize > 0)  {
        cudaError_t status = cudaMalloc(&buffer, bufferSize);
        if (status != cudaSuccess)
            return CUSPARSE_STATUS_ALLOC_FAILED;
    }
    // execute SpMM
    CHECK_CUSPARSE( cusparseSpMM(handle, transA, transB,
                                 alpha, matA, matB, beta, matC, typeC,
                                 CUSPARSE_SPMM_ALG_DEFAULT, buffer) )
    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matB) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matC) )
    if(buffer) cudaFree(buffer);
    return CUSPARSE_STATUS_SUCCESS;
}

// -------------------------------------------------------------------------------//
template<typename T>
cusparseStatus_t
cusparseXcsr2csc_wrapper(cusparseHandle_t       handle,
                       int                      m,
                       int                      n,
                       int                      nnz,
                       const T*                 csrVal,
                       const int*               csrRowPtr,
                       const int*               csrColInd,
                       T*                       cscVal,
                       int*                     cscRowInd,        
                       int*                     cscColPtr,
                       cusparseAction_t         copyValues,
                       cusparseIndexBase_t      idxBase)
{
    void*        buffer     = NULL;
    size_t       bufferSize = 0;
    cudaDataType valType    = type_to_enum<T>();
    cusparseCsr2CscAlg_t alg = CUSPARSE_CSR2CSC_ALG2;
    
    // fails if nnz==0
    if(nnz==0)
    {
        mxShowCriticalErrorMessage("BUG: cusparseCsr2cscEx2 fails when nnz=0");
        return CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED;
    }
    
    // make buffer
    CHECK_CUSPARSE( cusparseCsr2cscEx2_bufferSize(
                                handle,
                                m,
                                n,
                                nnz,
                                csrVal,
                                csrRowPtr,
                                csrColInd,
                                cscVal,
                                cscColPtr,
                                cscRowInd,
                                valType,
                                copyValues,
                                idxBase,
                                alg,
                                &bufferSize) )
    
    if (bufferSize > 0)
    {
        cudaError_t status = cudaMalloc(&buffer, bufferSize);
        if (status != cudaSuccess)
            return CUSPARSE_STATUS_ALLOC_FAILED;
    }                         
    
    CHECK_CUSPARSE( cusparseCsr2cscEx2(
                                handle,
                                m,
                                n,
                                nnz,
                                csrVal,
                                csrRowPtr,
                                csrColInd,
                                cscVal,
                                cscColPtr,
                                cscRowInd,
                                valType,
                                copyValues,
                                idxBase,
                                alg,
                                buffer) )

    if(buffer) cudaFree(buffer);
    return CUSPARSE_STATUS_SUCCESS;
}
                              
                              
    
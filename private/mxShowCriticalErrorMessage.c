// Beautified mex error message macro.
//
// Inspired by
//   http://www.advanpix.com/2016/02/14/short-and-informative-error-messages-from-mex/
//

// Use macro to expand __FILE__ and __LINE__ correctly
#define mxShowCriticalErrorMessage(...) error_function(basename(__FILE__),__LINE__,##__VA_ARGS__)

// Use overloads to handle __VA_ARGS__ correctly
void error_function(const char *function_name, int line_number, const char *error_message, int error_code)
{
    mxArray *err_args[5];
    err_args[0] = mxCreateString("%s(%i): %s (%i).");
    err_args[1] = mxCreateString(function_name);
    err_args[2] = mxCreateDoubleMatrix(1,1,mxREAL);
    err_args[3] = mxCreateString(error_message);
    err_args[4] = mxCreateDoubleMatrix(1,1,mxREAL);
    *mxGetPr(err_args[2]) = line_number;
    *mxGetPr(err_args[4]) = error_code;
    mexCallMATLAB(0,0,5,err_args,"error");
}

void error_function(const char *function_name, int line_number, const char *error_message)
{
    mxArray *err_args[4];
    err_args[0] = mxCreateString("%s(%i): %s.");
    err_args[1] = mxCreateString(function_name);
    err_args[2] = mxCreateDoubleMatrix(1,1,mxREAL);
    err_args[3] = mxCreateString(error_message);
    *mxGetPr(err_args[2]) = line_number;
    mexCallMATLAB(0,0,4,err_args,"error");
}

void error_function(const char *function_name, int line_number, int error_code)
{
    error_function(function_name, line_number, "Error occurred", error_code);
}

void error_function(const char *function_name, int line_number)
{
    error_function(function_name, line_number, "Error occurred");
}

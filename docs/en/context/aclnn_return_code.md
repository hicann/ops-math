# aclnn Return Codes

When calling aclnn APIs, common interface return codes are shown in [Table 1](#table1).
For abnormal status codes, you can use the aclGetRecentErrMsg interface ([Runtime API](https://hiascend.com/document/redirect/CannCommunityRuntimeApi)) to obtain exception information. You can troubleshoot the problem based on the error message or contact technical support.

**Table 1** Return Status Codes <a id="table1"></a>

| Status Code Name | Status Code Value | Status Code Description |
| ----- | ----- | ------ |
| ACLNN_SUCCESS | 0 | Success. |
| ACLNN_ERR_PARAM_NULLPTR | 161001 | Parameter validation error, illegal nullptr exists in parameters. |
| ACLNN_ERR_PARAM_INVALID | 161002 | Parameter validation error, such as two input data types not satisfying the input type deduction relationship. |
| ACLNN_ERR_RUNTIME_ERROR | 361001 | API internal call to npu runtime interface exception. |
| ACLNN_ERR_INNER_XXX | 561xxx | API internal exception. |

For more descriptions of ACLNN_ERR_INNER_XXX status codes, see [Table 2](#table2).

**Table 2** Exception Status Codes <a id="table2"></a>

| Status Code Name | Status Code Value | Status Code Description |
| ----- | ----- | ------ |
| ACLNN_ERR_INNER               |   561000     |    Internal exception: API internal exception occurred.     |
| ACLNN_ERR_INNER_INFERSHAPE_ERROR     |  561001      |  Internal exception: Error occurred during output shape deduction inside the API.       |
| ACLNN_ERR_INNER_TILING_ERROR     |   561002     |   Internal exception: Exception occurred during npu kernel tiling inside the API.     |
| ACLNN_ERR_INNER_FIND_KERNEL_ERROR   |   561003  |  Internal exception: Exception occurred when searching for npu kernel inside the API (possibly because the operator binary package is not installed).    |
| ACLNN_ERR_INNER_CREATE_EXECUTOR     |    561101    |   Internal exception: Failed to create aclOpExecutor inside the API (possibly due to operating system exception).      |
| ACLNN_ERR_INNER_NOT_TRANS_EXECUTOR     |  561102      |   Internal exception: uniqueExecutor ReleaseTo was not called inside the API.      |
| ACLNN_ERR_INNER_NULLPTR            |   561103     |    Internal exception: Exception occurred inside the aclnn API, nullptr exception appeared.     |
| ACLNN_ERR_INNER_WRONG_ATTR_INFO_SIZE     |   561104     |   Internal exception: Exception occurred inside the aclnn API, operator attribute count exception.      |
| ACLNN_ERR_INNER_KEY_CONFILICT (deprecated)     |   561105     |    **Deprecated, please use the latest ACLNN_ERR_INNER_KEY_CONFLICT.**     |
| ACLNN_ERR_INNER_KEY_CONFLICT             |   561105     |   Internal exception: Exception occurred inside the aclnn API, hash key conflict occurred in operator kernel matching. |
| ACLNN_ERR_INNER_INVALID_IMPL_MODE     |   561106     |   Internal exception: Exception occurred inside the aclnn API, operator implementation mode parameter error.    |
| ACLNN_ERR_INNER_OPP_PATH_NOT_FOUND     |  561107     |  Internal exception: Exception occurred inside the aclnn API, environment variable ASCEND_OPP_PATH not detected.       |
| ACLNN_ERR_INNER_LOAD_JSON_FAILED     |  561108      |   Internal exception: Exception occurred inside the aclnn API, failed to load operator information json file in operator kernel library.      |
| ACLNN_ERR_INNER_JSON_VALUE_NOT_FOUND     |   561109      |   Internal exception: Exception occurred inside the aclnn API, failed to load a field in operator information json file in operator kernel library.      |
| ACLNN_ERR_INNER_JSON_FORMAT_INVALID     |  561110     |     Internal exception: Exception occurred inside the aclnn API, format field in operator information json file in operator kernel library is filled with illegal value.    |
| ACLNN_ERR_INNER_JSON_DTYPE_INVALID     |    561111      |     Internal exception: Exception occurred inside the aclnn API, dtype field in operator information json file in operator kernel library is filled with illegal value.    |
| ACLNN_ERR_INNER_OPP_KERNEL_PKG_NOT_FOUND     |   561112    |    Internal exception: Exception occurred inside the aclnn API, operator binary kernel library not loaded.     |
| ACLNN_ERR_INNER_OP_FILE_INVALID     |  561113     |   Internal exception: Exception occurred inside the aclnn API when loading operator json file field.   |
| ACLNN_ERR_INNER_ATTR_NUM_OUT_OF_BOUND     |  561114      |  Internal exception: Exception occurred inside the aclnn API, operator attribute count inconsistent with operator information json, exceeding attr count specified in json.       |
| ACLNN_ERR_INNER_ATTR_LEN_NOT_ENOUGH     |   561115      |   Internal exception: Exception occurred inside the aclnn API, operator attribute count inconsistent with operator information json, less than attr count specified in json.      |
| ACLNN_ERR_INNER_INPUT_NUM_IN_JSON_TOO_LARGE     |   561116     |   Internal exception: Exception occurred inside the aclnn API, operator input count exceeds limit of 32.      |
| ACLNN_ERR_INNER_INPUT_JSON_IS_NULL     |   561117     |  Internal exception: Exception occurred inside the aclnn API, operator information json file description missing.       |
| ACLNN_ERR_INNER_STATIC_WORKSPACE_INVALID     |  561118     |    Internal exception: Exception occurred inside the aclnn API when parsing workspace information in static binary json file.     |
| ACLNN_ERR_INNER_STATIC_BLOCK_DIM_INVALID     |  561119      |    Internal exception: Exception occurred inside the aclnn API when parsing core usage information in static binary json file.  |

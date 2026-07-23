# Data Structure

This chapter provides basic data structures that CANN operator API invocation depends on. **Developers do not need to focus on their internal implementation, just use them directly**.

Note that this basic data structure can be created through "Public Interface" in opbase library, such as aclCreateTensor, etc. For details, please refer to [Public Interface](https://gitcode.com/cann/opbase/blob/master/docs/zh/api/nnopbase/aclnn/00_aclnn_api_list.md).

- **aclTensor**

  A structure defined by framework to manage and store tensor data (such as multi-dimensional data like vectors, matrices, etc.), can create this object through **aclCreateTensor** interface.

  ```bash
  typedef struct aclTensor aclTensor
  ```

- **aclScalar**

  A structure defined by framework to manage and store scalar data (i.e., single numerical value), can create this object through **aclCreateScalar** interface.

  ```bash
  typedef struct aclScalar aclScalar
  ```

- **aclIntArray**

  An array structure defined by framework to manage and store integer data, can create this object through **aclCreateIntArray** interface.

  ```bash
  typedef struct aclIntArray aclIntArray
  ```

- **aclFloatArray**

  An array structure defined by framework to manage and store float32 type data, can create this object through **aclCreateFloatArray** interface.

  ```bash
  typedef struct aclFloatArray aclFloatArray
  ```

- **aclBoolArray**

  An array structure defined by framework to manage and store boolean type data, can create this object through **aclCreateBoolArray** interface.

  ```bash
  typedef struct aclBoolArray aclBoolArray
  ```

- **aclTensorList**

  An array structure defined by framework to manage and store multiple tensor data, can create this object through **aclCreateTensorList** interface.

  ```bash
  typedef struct aclTensorList aclTensorList
  ```

- **aclScalarList**

  An array structure defined by framework to manage and store scalar data, can create this object through **aclCreateScalarList** interface.

  ```bash
  typedef struct aclScalarList aclScalarList
  ```

- **aclOpExecutor**

  An executor data structure defined by framework, a container used to execute operator computation.

  Usually when calling operator first-stage interface aclxxXxxGetWorkspaceSize, framework will automatically create aclOpExecutor; after calling second-stage interface aclxxXxx, this object will be automatically released.

  ```bash
  typedef struct aclOpExecutor aclOpExecutor
  ```

- **aclrtStream**

  A stream processing data structure defined by framework, used to manage and maintain execution order of some asynchronous operations.

  ```bash
  typedef void *aclrtStream
  ```

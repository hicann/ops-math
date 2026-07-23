# Mutual Conversion Relationship

When an API (such as aclnnAdd, aclnnMul, etc.) has **output aclTensor data type** inconsistent with **computation type derived from input data type**, the API internally converts computation result to data type corresponding to output type.

Data type conversion needs to satisfy the following rules. Conversions that do not satisfy the rules cannot be performed, and parameter validation will fail when calling API.

- Floating point types: ACL\_FLOAT16、ACL\_FLOAT、ACL\_DOUBLE、ACL\_BF16.
- Integer types: ACL\_INT8、ACL\_UINT8、ACL\_INT16、ACL\_UINT16、ACL\_INT32、ACL\_UINT32、ACL\_INT64、ACL\_UINT64.
- Complex types: ACL\_COMPLEX64、ACL\_COMPLEX128.
- Integer types can convert between each other, and also support conversion to floating point and complex types.
- Floating point types can convert between each other, and also support conversion to complex types.
- Complex types can convert between each other.
- BOOL supports conversion to integer, floating point and complex types.

Besides above scenarios, other conversion scenarios are not supported.

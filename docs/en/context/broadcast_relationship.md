# Broadcast Relationship

## Broadcast Concept

Broadcast describes how operators handle tensors (or arrays) of different shapes during computation. In most cases, tensors (or arrays) of different shapes are allowed to automatically expand their shapes during element operations to make their dimensions compatible. Usually, smaller tensors (or arrays) are "broadcast" to larger tensors (or arrays).

Currently, many CANN operator API parameter shapes support broadcasting, which can appropriately improve computational efficiency and reduce memory usage (especially in large-scale data scenarios). For more detailed broadcast technology introduction, refer to the [NumPy](https://numpy.org/doc/stable/user/basics.broadcasting.html) official website.

## Broadcast Rules

When performing broadcast calculations, you generally need to understand the following rules:

- Rule 1: If the number of dimensions between arrays is inconsistent, all arrays align to the array with the longest shape, and the insufficient parts of the shape are padded with 1 on the **left** until the number of dimensions is the same.
  
  > Note:
  > - Example 1: Number of Dimensions refers to the number of dimensions corresponding to the tensor (or array) shape. For example, x.shape=(1,1,2,4), the number of dimensions is 4.
  > - Example 2: For example, calculating a+b, where a.shape=(2, 2, 3), b.shape=(2, 3), then array b will be broadcast to b.shape=(1, 2, 3).
  
- Rule 2: If the number of dimensions between arrays is consistent, and one dimension of an array is 1, then the array with dimension 1 will be stretched to match the corresponding dimension shape of the other array.

  > Note:
  > In this scenario, you only need to ensure broadcasting in one dimension. For example, calculating a+b, where a.shape=(1, 3), b.shape=(3, 1), then both arrays will be broadcast to a.shape=(3, 3), b.shape=(3, 3).

- Rule 3: If the number of dimensions between arrays is inconsistent, and neither has a dimension equal to 1, an error will occur.

Based on the above rules, the broadcast process generally first expands dimensions according to **Rule 1**, then stretches shapes according to **Rule 2**. Specific examples are as follows:

```text
Assume a.shape=(2,2,3), values like:
[[[1 2 3],[4 5 6]],
 [[1 2 3],[4 5 6]]]
Assume b.shape=(2,3), values like:
[[1 2 3],
 [-1 -2 -3]]
According to Rule 1 expand dimensions, b.shape=(1,2,3), values as follows:
[[[1 2 3],
  [-1 -2 -3]]]
According to Rule 2 stretch shape, b.shape=(2,2,3), values as follows:
[[[1 2 3],[-1 -2 -3]],
 [[1 2 3],[-1 -2 -3]]]
Calculate a+b, actual result as follows:
 [[[2 4 6],[3 3 3]],
  [[2 4 6],[3 3 3]]]
```

## Limitations

When the data types of two inputs a and b satisfying the broadcast relationship or the deduced data types are in COMPLEX64, COMPLEX128, DOUBLE, INT16, UINT16, UINT64, in addition to satisfying the above broadcast rules, the following conditions must also be met, otherwise the broadcast will fail, causing the operator execution to report an error.
Condition: The dimension after merging continuous axes that need broadcasting and continuous axes that do not need broadcasting must be less than 6.
Examples:

- When a.shape=(5, 1, 5, 1, 5, 1), b.shape=(5, 5, 5, 5, 5, 5), there are no axes to merge, the final dimension is 6, broadcast error.
- When a.shape=(5, 1, 5, 5, 1, 1), b.shape=(5, 5, 5, 5, 5, 5), in the 2nd and 3rd dimensions no broadcasting is needed, in the 4th and 5th dimensions broadcasting is needed, merge separately and continuously, the merged dimension is 4, broadcast successful.

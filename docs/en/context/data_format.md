# Data Format

Data format (format) is used to describe the business semantics of a multi-dimensional Tensor's axes, representing the physical layout format of data, such as 1D, 2D, 3D, 4D, 5D, etc. Generally, CNN (Convolutional Neural Networks) class APIs need to describe specific formats.

For **full data format range** supported by aclTensor, please refer to [Runtime API](https://hiascend.com/document/redirect/CannCommunityRuntimeApi) "Data Type and Its Operation Interface > aclFormat".

For **data format layout principle** introduction, please refer to [Ascend C Operator Development Guide](https://hiascend.com/document/redirect/CannCommunityOpdevAscendC) "Concept Principles and Terminology > Neural Networks and Operators > Data Layout Format".

## Usage Instructions

Currently most operator APIs support ND data format. For example, aclnnAdd interface indicates supported data format is ND (i.e., multi-dimensional Tensor, low dimension priority continuous layout rule). For aclnnConvolution, it belongs to CNN class API, requiring input aclTensor to set format with business semantics, rather than ND format. Such operators need to know business semantics in Tensor during computation process. For example, in 2D convolution, need to know correspondence between Batch dimension, Channel dimension, Height dimension, Width dimension and Tensor dimensions.

>**Note:**
>
>- In two-stage interface parameter description, to simplify description, **original data format "ACL\_FORMAT\_XXXX_" is abbreviated as "\_XXXX_"**.
>- Data format dimension meanings: N (Batch) represents batch size, H (Height) represents feature map height, W (Width) represents feature map width, C (Channels) represents feature map channels, D (Depth) represents feature map depth, L (Length) represents feature map length.

## Common Data Formats

When creating aclTensor through **aclCreateTensor** interface, need to set data format according to API business requirements. Currently **supported data formats** are:

ACL\_FORMAT\_ND、ACL\_FORMAT\_NCHW、ACL\_FORMAT\_NHWC、ACL\_FORMAT\_HWCN、ACL\_FORMAT\_NDHWC、ACL\_FORMAT\_NCDHW、ACL\_FORMAT\_NC、ACL\_FORMAT\_NCL.

For non-ND Tensor, Tensor dimension requirements are consistent with format description. For example:

- 5D Tensor: Requires ACL\_FORMAT\_NCDHW, ACL\_FORMAT\_NDHWC or ACL\_FORMAT\_ND (if API parameter description does not indicate support for ND, setting ND format will cause API validation error).
- 4D Tensor: Requires ACL\_FORMAT\_NCHW, ACL\_FORMAT\_NHWC, ACL\_FORMAT\_HWCN or ACL\_FORMAT\_ND.
- 3D Tensor: Requires ACL\_FORMAT\_NCL or ACL\_FORMAT\_ND.
- 2D Tensor: Requires ACL\_FORMAT\_NC or ACL\_FORMAT\_ND.
- Other dimension Tensor: Requires ACL\_FORMAT\_ND.

## Private Data Formats

Besides above common data formats, there are other data formats, such as ACL\_FORMAT\_NC1HWC0、ACL\_FORMAT\_FRACTAL\_Z、ACL\_FORMAT\_NC1HWC0\_C04、ACL\_FORMAT\_FRACTAL\_NZ、ACL\_FORMAT\_NDC1HWC0、ACL\_FORMAT\_FRACTAL\_Z\_3D, etc.

These formats belong to some private formats of NPU. Currently most aclnn APIs do not support these formats. If individual API declares supported data formats, please refer to actual description of that API.

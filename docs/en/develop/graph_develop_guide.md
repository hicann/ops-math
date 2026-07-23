# Graph Mode Adaptation Guide

## Overview

This document introduces the graph mode adaptation method for custom operators. The overall process is consistent with the operator development guide ([AI Core Operator Development Guide](aicore_develop_guide.md)/[AI CPU Operator Development Guide](aicpu_develop_guide.md)). Notably, **aclnn adaptation is not required**, only the following deliverable adaptations are needed to achieve graph mode operator invocation.

```text
${op_name}                              # Replace with lowercase underscore form of actual operator name
├── op_host                             # Host-side implementation
│   └── ${op_name}_infershape.cpp       # InferShape implementation, implements operator shape deduction, deduces output shape at runtime
├── op_graph                            # Graph fusion related implementation
│   ├── CMakeLists.txt                  # op_graph side cmakelist file
│   ├── ${op_name}_graph_infer.cpp      # InferDataType file, implements operator type deduction, deduces output dataType at runtime
└── └── ${op_name}_proto.h              # Operator prototype definition, used for graph optimization and fusion phase to identify operator
```

This document will take `AddExample` operator (assuming AI Core operator) entering graph as an example to introduce the implementation of graph entry deliverables. AI CPU operator graph entry implementation is basically similar. For complete code, see `add_example` and `add_example_aicpu` under `examples` directory.

## Shape and DataType Deduction

Graph mode needs to complete two deliverables: ```${op_name}_infershape.cpp``` and ```${op_name}_graph_infer.cpp```

**Deliverable 1: ${op_name}_infershape.cpp**

The InferShape function's role is to deduce output shape based on input shape.

Example as follows, `AddExample` operator complete code please refer to [add_example_infershape.cpp](../../../examples/add_example/op_host/add_example_infershape.cpp) under `examples/add_example/op_host`.

```C++
// AddExample operator logic is two numbers added, so output shape is consistent with input shape
static ge::graphStatus InferShapeAddExample(gert::InferShapeContext* context)
{
    ....
    // Get input shape
    const gert::Shape* xShape = context->GetInputShape(IDX_0);
    // Get output shape
    gert::Shape* yShape = context->GetOutputShape(IDX_0);
    // Get input DimNum
    auto xShapeSize = xShape->GetDimNum();
    // Set output DimNum
    yShape->SetDimNum(xShapeSize);
    // Set input Dim values to output one by one
    for (size_t i = 0; i < xShapeSize; i++) {
        int64_t dim = xShape->GetDim(i);
        yShape->SetDim(i, dim);
    }
    ....
}
// inferShape registration
IMPL_OP_INFERSHAPE(AddExample).InferShape(InferShapeAddExample);
```

**Deliverable 2: ${op_name}_graph_infer.cpp**

The InferDataType function's role is to deduce output DataType based on input DataType.

Example as follows, `AddExample` operator complete code please refer to [add_example_graph_infer.cpp](../../../examples/add_example/op_graph/add_example_graph_infer.cpp) under `examples/add_example/op_graph`.

```C++
// AddExample operator logic is two numbers added, so output dataType is consistent with input dataType
static ge::graphStatus InferDataTypeAddExample(gert::InferDataTypeContext* context)
{
    ....
    // Get input dataType
    ge::DataType sizeDtype = context->GetInputDataType(IDX_0);
    // Set input dataType to output
    context->SetOutputDataType(IDX_0, sizeDtype);
    ....
}

// Register InferDataType
IMPL_OP(AddExample).InferDataType(InferDataTypeAddExample);
```

## Operator Prototype Configuration

**Deliverable: ${op_name}_proto.h**

Graph mode invocation requires registering operator prototype to [Graph Engine](https://www.hiascend.com/cann/graph-engine) (abbreviated as GE), so that GE can recognize the operator's input, output and attribute information. Registration is completed through `REG_OP` interface. Developers need to define basic information such as operator input, output tensor type and quantity.

### Common Data Types

Common tensor/attribute data type examples are as follows:

| Tensor Type | Attribute Type | Example |
|-----|------|-----|
| int64 | / | DT_INT64 |
| int32 | / | DT_INT32 |
| int16 | / | DT_INT16 |
| int8 | / | DT_INT8 |
| double | / | DT_DOUBLE |
| float32 | / | DT_FLOAT |
| float16 | / | DT_FLOAT16 |
| bfloat16 | / | DT_BF16 |
| complex128 | / | DT_COMPLEX128 |
| complex64 | / | DT_COMPLEX64 |
| complex32 | / | DT_COMPLEX32 |
| / | int | Int |
| / | bool | Bool |
| / | string | String |
| / | float | Float |
| / | list | ListInt |

### Input Output Definition

| Input/Output | Keyword | Example |
|-----|------|-----|
| Required input | INPUT | .INPUT(${name}, TensorType({input_dtype})) |
| Optional input | OPTIONAL_INPUT | .OPTIONAL_INPUT(${name}, TensorType({optional_input_dtype})) |
| Required attribute | REQUIRED_ATTR | .REQUIRED_ATTR(${name}, ${dtype}) |
| Optional attribute | ATTR | .ATTR(${name}, ${dtype}, ${default_value}) |
| Output | OUTPUT | .OUTPUT(${name}, TensorType({output_dtype})) |

### Code Example

Example code as follows, showing how to register `AddExample` operator:

```CPP
REG_OP(AddExample)
    .INPUT(x1, TensorType({DT_FLOAT}))
    .INPUT(x2, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(AddExample)
```

For complete code, please refer to [add_example_proto.h](../../../examples/add_example/op_graph/add_example_proto.h) under `examples/add_example/op_graph` directory.

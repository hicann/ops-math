import torch
import numpy as np

def gen_golden(case_info: dict):
    input_desc = case_info["input_desc"]
    input0 = input_desc[0]
    val_min = input_desc[1]["value"]
    val_max = input_desc[2]["value"]
    input0_t = torch.from_numpy(input0["value"])

    if input0["dtype"] == "float16":
        input0_fp32 = input0_t.to(torch.float32)
        out_fp32 = torch.nn.functional.hardtanh(input0_fp32, val_min, val_max)
        out = out_fp32.numpy().astype(np.float16)
        return out

    result = torch.nn.functional.hardtanh(input0_t, val_min, val_max)
    return result.numpy()
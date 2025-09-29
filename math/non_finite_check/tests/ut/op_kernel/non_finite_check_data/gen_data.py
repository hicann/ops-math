#!/usr/bin/python3
import sys
import os
import numpy as np
import re
import tensorflow as tf
import random


def parse_str_to_shape_list(shape_str):
    shape_list = []
    shape_str_arr = re.findall(r"\{([0-9 \,]+)\}\,", shape_str)
    for shape_str in shape_str_arr:
        single_shape = [int(x) for x in shape_str.split(",") if x.strip()]
        shape_list.append(single_shape)
    return shape_list


def gen_data_and_golden(shape_str, tiling_key="301"):
    add_ninf = True
    if tiling_key == "101":
        d_type = np.float16
    elif tiling_key == "201":
        add_ninf = False
        d_type = tf.bfloat16.as_numpy_dtype
    else:
        d_type = np.float32
    shape_list = parse_str_to_shape_list(shape_str)
    for index, shape in enumerate(shape_list):
        tmp_input = np.random.rand(*shape) * 100
        tmp_input = tmp_input.astype(d_type).flatten()
        if add_ninf:
            tmp_input[random.randrange(0, tmp_input.size)] = np.NINF
        tmp_input.reshape(*shape)
        tmp_input.astype(d_type).tofile(f"input_t{index}.bin")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Param num must be 3.")
        exit(1)
    # 清理bin文件
    os.system("rm -rf *.bin")
    gen_data_and_golden(sys.argv[1], sys.argv[2])
    exit(0)

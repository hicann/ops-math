#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

from enum import Enum
import math
import struct
from dataclasses import dataclass
import os
from typing import Any
import numpy as np


ONE_BYTE_BIT = 8
BYTE_ALIGN = 32

PRESET_ENCODES = {
    '00000000': float(0),
    '10000000': float('nan'),
    '01101111': float('inf'),
    '11101111': float('-inf')
}

UNCONVENTIONAL_MAP = {
    '11': 4,
    '10': 3,
    '01': 2,
    '001': 1,
    '0001': 0,
    '0000': 0,
}

MANTISSA_B2F_MAP = {'000': 0, '001': 0.125, '010': 0.25,
                    '011': 0.375, '100': 0.5, '101': 0.625, '110': 0.75, '111': 0.875,
                    '00': 0, '01': 0.25, '10': 0.5, '11': 0.75, '0': 0, '1': 0.5}

MANTISSA_F2B_MAP = {k: v for v, k in MANTISSA_B2F_MAP.items()}

HIF8_MANTISSA_V2B = {
    (0., 3): '000',
    (0.125, 3): '001',
    (0.25, 3): '010',
    (0.375, 3): '011',
    (0.5, 3): '100',
    (0.625, 3): '101',
    (0.75, 3): '110',
    (0.875, 3): '111',
    (0., 2): '00',
    (0.25, 2): '01',
    (0.5, 2): '10',
    (0.75, 2): '11',
    (0, 1): '0',
    (0.5, 1): '1',
}


def is_close(value_x: float, value_y: float, rel_tol: float = 1e-10, abs_tol: float = 0.0):
    """
    determines whether the values of two floating-point numbers
    are close or equal
    """
    return math.isclose(value_x, value_y, rel_tol=rel_tol, abs_tol=abs_tol)


@dataclass
class ExponentField:
    binary: str
    fixed_msb: bool
    value: int = None

    def __post_init__(self):
        if not self.binary:
            self.value = 0
            return
        symbol = 1 if self.binary[0] == '0' else -1
        revised_binary = self.binary[1:]
        if self.fixed_msb:
            revised_binary = '1' + self.binary[1:]
        self.value = symbol * int(revised_binary, 2)

    def get_val(self):
        return self.value


@dataclass
class MantissaField:
    binary: str
    value: float = None
    bit_width: int = None

    def __post_init__(self):
        self.bit_width = len(self.binary)
        self.value = int(self.binary, 2) / (2 ** self.bit_width)

    def get_val(self):
        return self.value


@dataclass
class HiFloat8:
    bits: str
    signal_bit: str = None
    dot_field: str = None
    d_val: int = None
    exponent_field: ExponentField = None
    mantissa_field: MantissaField = None
    s_v: int = None
    val: float = None

    def __post_init__(self):
        if self.bits in PRESET_ENCODES:
            self.val = PRESET_ENCODES[self.bits]
            return
        if len(self.bits) == 8:
            self.signal_bit = self.bits[0:1]
            if self.signal_bit == '0':
                self.s_v = 1
            else:
                self.s_v = -1
            for k, v in UNCONVENTIONAL_MAP.items():
                if self.bits[1:].startswith(k):
                    self.dot_field = k
                    self.d_val = v
                    break
            if not self.dot_field:
                raise RuntimeError(
                    "No dot field was found, {}".format(self.bits))
            # subnormal algo calculation
            if self.dot_field == '0000':
                self.m_v = int(self.bits[1 + len(self.dot_field):], 2)
                self.val = self.s_v * (2 ** (self.m_v - 23))
                return
            # normal algo calculation
            self.exponent_field = ExponentField(
                binary=self.bits[1 + len(self.dot_field):1 + len(self.dot_field) + self.d_val],
                fixed_msb=True
            )
            self.mantissa_field = MantissaField(
                self.bits[1 + len(self.dot_field) + self.d_val:])
            self.val = self.s_v * \
                (2 ** self.exponent_field.get_val()) * \
                (1 + self.mantissa_field.get_val())

    def get_val(self):
        return self.val


@dataclass
class FpaEbMc:
    """
    This is a father class representing Float point number with different format of exponential and mantissa
    Fp => Float Point
    a => total bits for this float
    E => Exponential
    b => number of bits for exponential field
    M => Mantissa
    c => number of bits for mantissa field
    ex: Fp8E5M2, Fp6E3M2, Fp4E1M2
    """
    bits: str
    exponent_bit_num: int
    mantissa_bit_num: int
    x_scale: str = ''
    exponent_bias: int = None
    symbol_bit: str = None
    exponent_bit: str = None
    mantissa_bit: str = None
    x_val: float = None

    is_normal: bool = True
    s_v: int = None
    e_v: int = None
    val: float = None

    def __post_init__(self):
        is_fp8_e8m0 = self.exponent_bit_num == 8 and self.mantissa_bit_num == 0
        symbol_bit_num = 0 if is_fp8_e8m0 else 1
        if self.exponent_bit_num + self.mantissa_bit_num + symbol_bit_num != len(self.bits):
            raise AttributeError(
                "The exponent bits and mantissa bits does not match with input bytes")
        self.exponent_bias = 2 ** (self.exponent_bit_num - 1) - 1
        if self.exponent_bit_num + self.mantissa_bit_num == 3:
            self.exponent_bias = 1  # fixed to 1 when fp 4
        self.symbol_bit = self.bits[0: symbol_bit_num]
        self.exponent_bit = self.bits[symbol_bit_num: symbol_bit_num + self.exponent_bit_num]
        self.mantissa_bit = self.bits[symbol_bit_num + self.exponent_bit_num:]
        if self.symbol_bit:
            self.s_v = int(self.symbol_bit, 2)
        if is_fp8_e8m0:
            return
        self.e_v = int(self.exponent_bit, 2)
        if self.x_scale == '11111111':
            raise RuntimeError("X Scale: 11111111 is invalid in Mx Data Type!")
        if self.x_scale:
            self.x_val = 2 ** (int(self.x_scale, 2) -
                               (2 ** (len(self.x_scale) - 1) - 1))
        else:
            self.x_val = 1
        if self.e_v == 0:
            self.is_normal = False
        self.__decode()

    def __decode(self):
        if self.is_normal:
            self.val = (-1) ** self.s_v * (2 ** (self.e_v - self.exponent_bias)) * (
                1 + MANTISSA_B2F_MAP[self.mantissa_bit]) * self.x_val
        else:
            self.val = (-1) ** self.s_v * (2 ** (self.e_v - self.exponent_bias + 1)) * MANTISSA_B2F_MAP[
                self.mantissa_bit] * self.x_val


@dataclass
class Fp8E5M2(FpaEbMc):
    def __init__(self, bits: str, x_scale: str = None):
        super().__init__(bits, 5, 2, x_scale)

    def __post_init__(self):
        super().__post_init__()
        self.__special_decode()

    def __special_decode(self):
        if self.exponent_bit == '11111' and self.mantissa_bit == '00':
            self.val = (-1) ** self.s_v * float('inf')
        elif self.exponent_bit == '11111':
            self.val = float('nan')


@dataclass
class Fp8E4M3(FpaEbMc):
    def __init__(self, bits: str, x_scale: str = None):
        super().__init__(bits, 4, 3, x_scale)

    def __post_init__(self):
        super().__post_init__()
        self.__special_decode()

    def __special_decode(self):
        if self.exponent_bit == '1111' and self.mantissa_bit == '111':
            self.val = float('nan')


@dataclass
class Fp8E8M0(FpaEbMc):
    def __init__(self, bits: str, x_scale: str = None):
        super().__init__(bits, 8, 0, x_scale)

    def __post_init__(self):
        super().__post_init__()
        self.__special_decode()

    def __special_decode(self):
        if self.exponent_bit == '11111111':
            raise RuntimeError("11111111 is not a valid coding in Fp8E8M0!")
        self.e_v = int(self.exponent_bit, 2)
        self.val = 2 ** (self.e_v - self.exponent_bias)


@dataclass
class Fp6E3M2(FpaEbMc):
    def __init__(self, bits: str, x_scale: str = None):
        super().__init__(bits, 3, 2, x_scale)

    def __post_init__(self):
        super().__post_init__()


@dataclass
class Fp6E2M3(FpaEbMc):
    def __init__(self, bits: str, x_scale: str = None):
        super().__init__(bits, 2, 3, x_scale)

    def __post_init__(self):
        super().__post_init__()


@dataclass
class Fp4E2M1(FpaEbMc):
    def __init__(self, bits: str, x_scale: str = None):
        super().__init__(bits, 2, 1, x_scale)

    def __post_init__(self):
        super().__post_init__()


@dataclass
class Fp4E1M2(FpaEbMc):
    def __init__(self, bits: str, x_scale: str = None):
        super().__init__(bits, 1, 2, x_scale)

    def __post_init__(self):
        super().__post_init__()


class FpType(Enum):
    HIFLOAT8 = "hifloat8"
    FLOAT8_E5M2 = "float8_e5m2"
    FLOAT8_E4M3 = "float8_e4m3fn"
    FLOAT6_E3M2 = "float6_e3m2"
    FLOAT6_E2M3 = "float6_e2m3"
    FLOAT4_E2M1 = "float4_e2m1"
    FLOAT4_E1M2 = "float4_e1m2"
    FLOAT8_E8M0 = "float8_e8m0"
    MX_FLOAT8_E5M2 = "mx_float8_e5m2"
    MX_FLOAT8_E4M3 = "mx_float8_e4m3fn"
    MX_FLOAT6_E3M2 = "mx_float6_e3m2"
    MX_FLOAT6_E2M3 = "mx_float6_e2m3"
    MX_FLOAT4_E2M1 = "mx_float4_e2m1"
    MX_FLOAT4_E1M2 = "mx_float4_e1m2"


def x_scale_convert(x_coding: str):
    return pow(2, int(x_coding, 2) - 127)


class FpNumConverter:
    def __init__(self, zero_symbol='1'):
        # to decide first symbol bit to be '1' or '0' when float number == 0
        self.zero_symbol = zero_symbol

    @staticmethod
    def fp16_ta_round_to_hif8(fraction16_int: int, hif8_bits_num: int, exponent: int):
        if exponent == -23:
            return True, 0
        # fp16 fraction is 10,keep hif8_bits_num + 1 bits
        hif8_value_tmp = fraction16_int >> (10 - (hif8_bits_num + 1))
        if hif8_value_tmp == pow(2, hif8_bits_num + 1) - 1:
            # carry exponent
            return True, 0
        elif hif8_value_tmp == 0:
            return False, 0
        elif hif8_value_tmp % 2 == 1:
            # carrys bits
            hif8_value_tmp += 1
            return False, hif8_value_tmp >> 1
        else:
            return False, hif8_value_tmp >> 1

    @classmethod
    def __encode_subnormal(cls, fp: float, e: int, m: int, x_scale: int = 1):
        exp_bias = 2 ** (e - 1) - 1
        if e + m == 3:
            exp_bias = 1
        e_v = 0

        m_v = fp / (2 ** (e_v - exp_bias + 1) * x_scale)
        for v in HIF8_MANTISSA_V2B.keys():
            if is_close(m_v, v[0]) and v[1] == m:
                return '0' * e + HIF8_MANTISSA_V2B[v]
        raise RuntimeError(
            f"Encode for subnormal F{e + m + 1}E{e}M{m} error, input:{fp}")

    @classmethod
    def __encode_normal(cls, fp: float, e: int, m: int, x_scale: int = 1):
        exp_bias = 2 ** (e - 1) - 1  # bias = 1 when fp4
        if e + m == 3:
            exp_bias = 1
        e_v = math.floor(math.log(fp / x_scale, 2)) + exp_bias

        exp_field = bin(e_v)[2:].rjust(e, '0')
        m_v = fp / 2 ** (e_v - exp_bias) - 1
        for v in HIF8_MANTISSA_V2B.keys():
            if is_close(m_v, v[0]) and v[1] == m:
                return exp_field + HIF8_MANTISSA_V2B[v]
        raise RuntimeError(
            f"Encode for normal F{e + m + 1}E{e}M{m} error, input:{fp}")

    @classmethod
    def __chk_special_encoding_hif8(cls, fp: float):
        # hif8
        value_boundary = 1.25 * pow(2.0, 15)
        fp_abs = math.fabs(fp)
        exceed_bound = fp_abs > value_boundary
        if is_close(fp, 0):
            return '00000000'
        elif math.isnan(fp):
            if exceed_bound:
                return '10000000'
            else:
                return '00000000'
        elif (math.isinf(fp) or exceed_bound) and fp > 0:
            return '01101111'
        elif (math.isinf(fp) or exceed_bound) and fp < 0:
            return '11101111'
        return ''

    @classmethod
    def __get_d_code(cls, e_v: float):
        if is_close(e_v, 0):
            return '0001'
        elif is_close(math.fabs(e_v), 1):
            return '001'
        elif 3 >= math.fabs(e_v) >= 2:
            return '01'
        elif 4 <= math.fabs(e_v) <= 7:
            return '10'
        else:
            return '11'

    @classmethod
    def __get_m_code(cls, m_v: float, m_bit_num: int):
        m_code = ''
        min_val = float('inf')
        for k, v in HIF8_MANTISSA_V2B.items():
            if k[1] == m_bit_num:
                if min_val > math.fabs(k[0] - m_v):
                    min_val = math.fabs(k[0] - m_v)
                    m_code = v
        return m_code

    @classmethod
    def __chk_hif8(cls, res: str):
        if len(res) != 8:
            raise ValueError(
                "Inference for HiFloat8 failed please check your float input!")

    def convert(self, fp: float, convert_type: FpType, x_scale: int = 1):
        if convert_type == FpType.HIFLOAT8:
            return self.__convert_hif8(fp)
        elif convert_type == FpType.FLOAT8_E8M0:
            return self.__convert(fp, 8, 0, x_scale)
        elif convert_type == FpType.FLOAT8_E5M2:
            return self.__convert(fp, 5, 2, x_scale)
        elif convert_type == FpType.FLOAT8_E4M3:
            return self.__convert(fp, 4, 3, x_scale)
        elif convert_type == FpType.FLOAT6_E3M2:
            return self.__convert(fp, 3, 2, x_scale)
        elif convert_type == FpType.FLOAT6_E2M3:
            return self.__convert(fp, 2, 3, x_scale)
        elif convert_type == FpType.FLOAT4_E1M2:
            return self.__convert(fp, 1, 2, x_scale)
        elif convert_type == FpType.FLOAT4_E2M1:
            return self.__convert(fp, 2, 1, x_scale)
        else:
            return ''

    def __chk_special_encoding(self, fp: float, e: int, m: int, x_scale: int = 1):
        if is_close(fp, 0):
            return self.zero_symbol + '0' * (e + m)
        if e == 5 and m == 2:
            value_boundary = pow(2, 15) * x_scale * \
                (1 + MANTISSA_B2F_MAP['1' * m])
            exceed_bound = math.fabs(fp) > value_boundary
            if math.isnan(fp):
                return self.zero_symbol + '1' * (m + e)
            if (math.isinf(fp) or exceed_bound) and fp < 0:
                return '11111100'
            if (math.isinf(fp) or exceed_bound) and fp > 0:
                return '01111100'
        if e == 4 and m == 3:
            value_boundary = pow(2, 8) * x_scale * \
                (1 + MANTISSA_B2F_MAP['1' * m])
            exceed_bound = math.fabs(fp) > value_boundary
            if math.isnan(fp):
                return self.zero_symbol + '1' * (m + e)
        if math.isinf(fp) or math.isinf(fp) or math.isnan(fp):
            raise IOError(f"The fp:{fp} is not valid")
        return ''

    def __convert(self, fp: float, e: int, m: int, x_scale: int = 1):
        special_encoding = self.__chk_special_encoding(fp, e, m, x_scale)
        if special_encoding:
            return special_encoding
        symbol_bit = '0'
        if fp < 0:
            symbol_bit = '1'
            fp *= -1
        exp_bias = 2 ** (e - 1) - 1
        if e + m == 3:
            exp_bias = 1
        # fp is normal floats
        if fp > MANTISSA_B2F_MAP['1' * m] * x_scale * pow(2, 1 - exp_bias):
            return symbol_bit + self.__encode_normal(fp, e, m, x_scale)
        else:
            return symbol_bit + self.__encode_subnormal(fp, e, m, x_scale)

    def __fp_round_to_hif8(self, fp: float, e_v: int):
        if e_v < 0:
            e_code = '1'

        elif e_v > 0:
            e_code = '0'
        else:
            e_code = ''
        if e_code:
            e_code = e_code + "{0:b}".format(abs(e_v))[1:]
        fraction_int = int(fp * pow(2, 10) * pow(2, -e_v) - pow(2, 10))
        d_code = self.__get_d_code(e_v)
        mantissa_bit_num = 7 - len(e_code) - len(d_code)
        carry_bit_flag, hif8_frac_value = self.fp16_ta_round_to_hif8(
            fraction_int, mantissa_bit_num, e_v)
        if carry_bit_flag:
            e_v += 1
            d_code = self.__get_d_code(e_v)
        return d_code, e_v

    def __convert_hif8(self, fp: float):
        special_encoding = self.__chk_special_encoding_hif8(fp)
        if special_encoding:
            return special_encoding
        symbol_bit = '0'
        if fp < 0:
            symbol_bit = '1'
            fp *= -1
        e_v = math.floor(math.log(fp, 2))

        d_code, e_v = self.__fp_round_to_hif8(fp, e_v)
        if -22 <= e_v < -15:
            # fp is subnormal floats
            m_v = e_v + 23
            result = symbol_bit + '0000' + "{0:b}".format(m_v).rjust(3, '0')
            self.__chk_hif8(result)
            return result
        m_v = (fp / (2 ** e_v)) - 1
        if e_v < 0:
            e_code = '1'
            e_v *= -1
        elif e_v > 0:
            e_code = '0'
        else:
            e_code = ''
        if e_code:
            e_code = e_code + "{0:b}".format(e_v)[1:]
        m_code = self.__get_m_code(m_v, 7 - len(e_code) - len(d_code))
        result = symbol_bit + d_code + e_code + m_code
        self.__chk_hif8(result)
        return result


def get_bits_by_bit_num(buffer: bytes, bit_num: int):
    bits_list = []
    byte_array = struct.unpack(f'{len(buffer)}B', buffer)
    for byte in byte_array:
        byte = bin(byte)[2:].rjust(ONE_BYTE_BIT, '0')
        for i in range(ONE_BYTE_BIT // bit_num):
            bits_list.append(byte[i * bit_num: i * bit_num + bit_num])
    return bits_list


def concat_bin_to_float(little_bits: list, big_bits: list, x_scale: str, func: Any):
    tensor_data = []
    big_iter = big_bits if big_bits else ('' for _ in little_bits)
    for pre_bits, little in zip(big_iter, little_bits):
        if func != HiFloat8:
            tensor_data.append(func(pre_bits + little, x_scale).val)
        else:
            tensor_data.append(func(little).val)
    return tensor_data


DAVID_TYPE_2_BITS = {'float8_e5m2': (8, 0, False, Fp8E5M2), 'float8_e4m3fn': (8, 0, False, Fp8E4M3),
                     "hifloat8": (8, 0, False, HiFloat8), "float8_e8m0": (8, 0, False, Fp8E8M0),
                     "float6_e3m2": (4, 2, False, Fp6E3M2), "float6_e2m3": (4, 2, False, Fp6E2M3),
                     "float4_e1m2": (4, 0, False, Fp4E1M2), "float4_e2m1": (4, 0, False, Fp4E2M1),
                     "mx_float6_e3m2": (4, 2, True, Fp6E3M2), 'mx_float8_e5m2': (8, 0, True, Fp8E5M2),
                     'mx_float8_e4m3fn': (8, 0, True, Fp8E4M3), "mx_float6_e2m3": (4, 2, True, Fp6E2M3),
                     "mx_float4_e1m2": (4, 0, True, Fp4E1M2), "mx_float4_e2m1": (4, 0, True, Fp4E2M1)
                     }


def get_david_data_offsets(little_bits: int, big_bits: int, is_mx: bool,
    bin_data_file: str, data_num: int):
    data_1_align_size = \
        math.ceil(little_bits * data_num / ONE_BYTE_BIT / BYTE_ALIGN) * BYTE_ALIGN
    data_2_align_size = \
        math.ceil(big_bits * data_num / ONE_BYTE_BIT / BYTE_ALIGN) * BYTE_ALIGN
    mx_align_size = math.ceil(data_num / 32) * BYTE_ALIGN
    expected_bin_size = data_1_align_size + data_2_align_size
    actual_file_size = os.path.getsize(bin_data_file)
    if is_mx:
        total_bits += mx_align_size
    if expected_bin_size != actual_file_size:
        raise RuntimeError(
            f"Tensor bin size does not match with file size :{file_size}, please check byte alignment")
    return data_1_align_size, data_2_align_size


def convert_dtype_bin(bin_data_file: str, dtype: str, shape: list):
    little_bits_num, big_bits_num, is_mx, func = DAVID_TYPE_2_BITS.get(
        dtype, (None, None, None, None))
    if not func:
        return None
    tensor_data = []
    total_ele_num = math.prod(shape)
    data_1_align_size, data_2_align_size = \
        get_david_data_offsets(little_bits_num, big_bits_num, is_mx, bin_data_file, total_ele_num)

    read_pos = 0
    read_big_bits_pos = data_1_align_size
    read_mx_bits_pos = data_1_align_size + data_2_align_size
    with open(bin_data_file, 'rb') as bin_file:
        while read_pos < data_1_align_size:
            read_little_length, read_big_length, read_mx_length = None, None, None
            x_scale = ''
            if is_mx:
                read_mx_length = 1
                read_little_length = 32 * little_bits_num // ONE_BYTE_BIT
                read_big_length = 32 * big_bits_num // ONE_BYTE_BIT
            elif big_bits_num:
                read_big_length = 1
                read_little_length = read_big_length * \
                    ONE_BYTE_BIT // big_bits_num * little_bits_num
            else:
                read_little_length = 1
            big_bits = []
            little_bits = []
            if read_little_length:
                bin_file.seek(read_pos)
                little_bits = get_bits_by_bit_num(
                    bin_file.read(read_little_length), little_bits_num)
                read_pos += read_little_length
            if read_big_length:
                bin_file.seek(read_big_bits_pos)
                big_bits = get_bits_by_bit_num(
                    bin_file.read(read_big_length), big_bits_num)
                read_big_bits_pos += read_big_length
            if read_mx_length:
                bin_file.seek(read_mx_bits_pos)
                mx_bits = get_bits_by_bit_num(
                    bin_file.read(read_mx_length), ONE_BYTE_BIT)
                read_mx_bits_pos += read_mx_length
                x_scale = mx_bits[0]
            tensor_data.extend(concat_bin_to_float(
                little_bits, big_bits, x_scale, func))
        return np.reshape(np.asarray(tensor_data, dtype=np.float32), shape)
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

import re
import glob
import logging
import subprocess
import json
import os
import shutil
import struct
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Any
from pathlib import Path
from utils import config, safe_check
from .dtype_convert import DAVID_TYPE_2_BITS, get_bits_by_bit_num, concat_bin_to_float, ONE_BYTE_BIT
BYTE_ALIGN = 32


logger = logging.getLogger(__name__)


@dataclass(repr=False)
class TLV:
    tag: int = 0
    length: int = 0
    value: bytes = None

    def __repr__(self):
        return f'TLV(tag={self.tag}, length={self.length})'

    @classmethod
    def get_tl_format(cls):
        return 'ii'

    @classmethod
    def get_tl_size(cls):
        return struct.calcsize(cls.get_tl_format())

    def get_size(self):
        return self.get_tl_size() + self.length

    def read(self, f):
        tl_fmt = self.get_tl_format()
        tl_size = self.get_tl_size()
        self.tag, self.length = struct.unpack(tl_fmt, f.read(tl_size))
        self.value = f.read(self.length)

    def write_to(self, buffer, offset):
        tl_fmt = self.get_tl_format()
        struct.pack_into(tl_fmt, buffer, offset, self.tag, self.length)
        value_offset = offset + self.get_tl_size()
        val_fmt = f'{self.length}s'
        struct.pack_into(val_fmt, buffer, value_offset, self.value)
        return self.get_tl_size() + self.length


@dataclass
class DumpMessageHeader:
    addr: int = 0
    data_type: int = 0
    desc: int = 0
    buffer_id: int = 0
    position: int = 0
    reserved: int = 0

    @classmethod
    def get_format(cls):
        return 'iiiiii'

    @classmethod
    def get_size(cls):
        fmt = cls.get_format()
        return struct.calcsize(fmt)

    def unpack(self, buffer):
        fmt = self.get_format()
        self.addr, self.data_type, self.desc, self.buffer_id, self.position, self.reserved = struct.unpack(
            fmt, buffer)

    def pack(self):
        fmt = self.get_format()
        return struct.pack(fmt, self.addr, self.data_type, self.desc, self.buffer_id, self.position, self.reserved)


@dataclass
class ShapeInfo:
    dim: int = 0
    shape: List[Any] = field(default_factory=list)
    rsv: int = 0
    total_ele_num = 0

    @classmethod
    def get_format(cls):
        # value format of ShapeInfo bin: dim, shape0, ..., shape7, rsv
        return 'iiiiiiiiii'

    @classmethod
    def get_size(cls):
        fmt = cls.get_format()
        return struct.calsize(fmt)

    def unpack(self, buffer):
        fmt = self.get_format()
        unpacked_data = struct.unpack(fmt, buffer)
        self.dim = unpacked_data[0]
        self.total_ele_num = 1
        for i in range(self.dim):
            self.shape.append(unpacked_data[i + 1])
            self.total_ele_num *= self.shape[-1]

    def parse_from(self, tlv: TLV):
        self.unpack(tlv.value)


dtype_to_fmt = {
    4: 'B',  # DT_UINT8
    2: 'b',  # DT_INT8
    6: 'h',  # DT_INT16
    7: 'H',  # DT_UINT16
    3: 'i',  # DT_INT32
    8: 'I',  # DT_UINT32
    10: 'Q',  # DT_UINT64
    9: 'q',  # DT_UINT64
    0: 'f',  # DT_FLOAT
    1: 'e',  # DT_FLOAT16
    42: ''  # DT_MAX
}

david_dtype_to_fmt = {
    34: 'hifloat8',  # DT_HIFLOAT8
    35: 'float8_e5m2',  # DT_FLOAT8_E5M2
    36: 'float8_e4m3fn',  # DT_FLOAT8_E4M3FN
    37: 'float8_e8m0',  # DT_FLOAT8_E8M0
    38: 'float6_e3m2',  # DT_FLOAT6_E3M2
    39: 'float6_e2m3',  # DT_FLOAT6_E2M3
    40: 'float4_e2m1',  # DT_FLOAT4_E2M1
    41: 'float4_e1m2',  # DT_FLOAT4_E1M2
}

dtype_to_data_type = {
    0: 'float32',  # DT_FLOAT
    1: 'float16',  # DT_FLOAT16
    2: 'int8',  # DT_INT8
    3: 'int32',  # DT_INT32
    4: 'uint8',  # DT_UINT8

    6: 'int16',  # DT_INT16
    7: 'uint16',  # DT_UINT16
    8: 'uint32',  # DT_UINT32
    9: 'int64',  # DT_INT64
    10: 'uint64',  # DT_UINT64

    42: ''  # DT_MAX
}


@dataclass(repr=False)
class DumpTensor:
    tag: int = 0
    length: int = 0
    dump_header: DumpMessageHeader = None
    dump_data: int = 0
    dump_value: List[Any] = field(default_factory=list)
    dump_shape: List[int] = field(default_factory=list)

    def __repr__(self):
        return f'DumpTensor(tag={self.tag}, length={self.length}, dump_header={self.dump_header},\
            tensor_shape={self.dump_shape}, dump_data_size={len(self.dump_data)})'

    def parse_from(self, tlv):
        self.tag = tlv.tag
        self.length = tlv.length

        head_size = DumpMessageHeader.get_size()
        head_buffer = tlv.value[0: head_size]
        self.dump_header = DumpMessageHeader()
        self.dump_header.unpack(head_buffer)
        self.dump_data = tlv.value[head_size:]
        self._parse_dump_data()

    def parse_to(self):
        tlv = TLV()
        tlv.tag = self.tag
        tlv.length = self.length
        head_buffer = self.dump_header.pack()

        tlv.value = head_buffer
        tlv.value += self.dump_data
        return tlv

    def __parse_mx_data(self, little_bits_num, big_bits_num, func):
        dump_data_num = self.dump_header.reserved
        if not dump_data_num:
            logger.error("Mx Data type does not specify dump number")
            return
        len_little_order = ((dump_data_num * little_bits_num / ONE_BYTE_BIT) + BYTE_ALIGN - 1) // BYTE_ALIGN
        len_big_order = ((dump_data_num * big_bits_num / ONE_BYTE_BIT) + BYTE_ALIGN - 1) // BYTE_ALIGN
        len_mx_order = ((dump_data_num / 32) + BYTE_ALIGN - 1) // BYTE_ALIGN
        if len_little_order + len_big_order + len_mx_order != len(self.dump_data):
            logger.error(f"Dump size {dump_data_num} does not match'\
                '{len_little_order} + {len_big_order} + {len_mx_order} != {len(self.dump_data)}")
            return
        little_bits = get_bits_by_bit_num(
            self.dump_data[:len_little_order], little_bits_num)
        big_bits = get_bits_by_bit_num(
            self.dump_data[len_little_order: len_little_order + len_big_order], big_bits_num)
        mx_bits = get_bits_by_bit_num(
            self.dump_data[len_little_order + len_big_order:], ONE_BYTE_BIT)
        tensor_datas = []
        for i, mx_bit in enumerate(mx_bits):
            data_num = 32 * i  # 如果后续没用到 data_num，其实可以删掉；但你保留它也没问题
            big_bits_ls = big_bits[i * 32: (i + 1) * 32] if big_bits else []
            tensor_datas.extend(concat_bin_to_float(
                little_bits[i * 32: (i + 1) * 32], big_bits_ls, mx_bit))
        self.dump_value.extend(tensor_datas[:dump_data_num])
        return

    def __parse_non_mx_data(self, little_bits_num, big_bits_num, func):
        len_little_order = len(self.dump_data) * \
                           little_bits_num // (little_bits_num + big_bits_num)
        dump_data_num = self.dump_header.reserved
        little_bits = get_bits_by_bit_num(
            self.dump_data[:len_little_order], little_bits_num)
        big_bits = get_bits_by_bit_num(
            self.dump_data[len_little_order:], big_bits_num)
        tensor_data = concat_bin_to_float(little_bits, big_bits, '', func)
        if dump_data_num:
            self.dump_value.extend(tensor_data[:dump_data_num])
        else:
            self.dump_value.extend(tensor_data)
        return

    def _parse_dump_data(self):
        fmt = dtype_to_fmt.get(self.dump_header.data_type, '')
        if fmt:
            self.dump_value.extend(value[0] for value in struct.iter_unpack(fmt, self.dump_data))
        elif self.dump_header.data_type in david_dtype_to_fmt.keys():
            dtype = david_dtype_to_fmt[self.dump_header.data_type]
            little_bits_num, big_bits_num, is_mx, func = DAVID_TYPE_2_BITS[dtype]
            if is_mx:
                self.__parse_mx_data(little_bits_num, big_bits_num, func)
            else:
                self.__parse_non_mx_data(little_bits_num, big_bits_num, func)
            return
            # need special parse way for
        else:
            logger.warning(
                f'data type {self.dump_header.data_type} is not supported')
            return


@dataclass(repr=False)
class PrintStruct:
    tag: int = 0
    length: int = 0
    fmt: str = ''
    args: List[Any] = field(default_factory=list)
    content: str = None

    @staticmethod
    def _read_arg_integer(buffer, offset):
        return struct.unpack('Q', buffer[offset: offset + 8])[0]

    @staticmethod
    def _read_arg_float(buffer, offset):
        return struct.unpack('f', buffer[offset: offset + 4])[0]

    @staticmethod
    def _read_arg_double(buffer, offset):
        return struct.unpack('d', buffer[offset: offset + 8])[0]

    @staticmethod
    def _read_arg_hex(buffer, offset):
        return struct.unpack('Q', buffer[offset: offset + 8])[0]

    @staticmethod
    def _read_arg_str(buffer, offset):
        relv_offset = struct.unpack('Q', buffer[offset: offset + 8])[0]
        abs_offset = offset + relv_offset
        return (PrintStruct._read_string(buffer, abs_offset), relv_offset)

    @staticmethod
    def _read_arg_point(buffer, offset):
        return struct.unpack('P', buffer[offset: offset + 8])[0]

    @staticmethod
    def _read_string(buffer: bytes, offset: int):
        max_length = len(buffer)
        if offset > max_length:
            raise RuntimeError(f'read str offset {offset} over max buffer length {max_length}')
        s: str = ''
        for i in range(offset, max_length):
            b = struct.unpack('1s', buffer[i: i + 1])[0]
            if b == b'\x00':
                return s
            s = s + b.decode('utf-8')
        return s

    def parse_from(self, tlv):
        self.tag = tlv.tag
        self.length = tlv.length

        args_start, args_end = self._read_fmt(tlv.value)  # fmt
        self._read_args(tlv.value, args_start, args_end)  # args

        self.fmt = self.fmt.replace('%p', '0x%x')  # python not support %p
        self.content = self.fmt % tuple(self.args)

    def _read_arg(self, buffer, offset, pld):
        if pld == '%d' or pld == '%i':
            arg = self._read_arg_integer(buffer, offset)
        elif pld == '%f' or pld == '%F':
            arg = self._read_arg_float(buffer, offset)
        elif pld == '%lf' or pld == '%LF':
            arg = self._read_arg_double(buffer, offset)
        elif pld == '%x' or pld == '%X':
            arg = self._read_arg_hex(buffer, offset)
        elif pld == '%s':
            arg, _ = self._read_arg_str(buffer, offset)
        elif pld == '%p':
            arg = self._read_arg_point(buffer, offset)
        elif pld == '%u':
            arg = self._read_arg_integer(buffer, offset)
        else:
            raise RuntimeError(f'not support pld({pld}), fmt: {self.fmt}')
        self.args.append(arg)

    def _all_fmt_placehold(self):
        pattern = r"%[a-zA-Z]{1,2}"
        matches = re.findall(pattern, self.fmt)
        matches = [x for x in matches if not (len(x) == 3 and '%l' not in x.lower())]
        return matches

    def _read_args(self, buffer, args_start, args_end):
        args_plds = self._all_fmt_placehold()
        args_num = len(args_plds)

        for i in range(0, args_num):
            offset = args_start + i * 8
            if offset + 8 > args_end:
                raise RuntimeError(f'arg {i} at [{offset}:{offset + 8}] over arg end({args_end}): \"{self.fmt}\"')
            pld = args_plds[i]
            self._read_arg(buffer, offset, pld)

    def _read_fmt(self, buffer):
        self.fmt, fmt_offset = self._read_arg_str(buffer, 0)
        return (8, fmt_offset)


@dataclass
class BlockInfo:
    total_size: int = 0
    block_id: int = 0
    block_num: int = 0
    remain_size: int = 0
    magic_num: int = 0
    reserved: int = 0
    dump_addr: int = 0  # 8 Byte

    @classmethod
    def get_format(cls):
        return 'iiiiiiQ'

    @classmethod
    def get_size(cls):
        fmt = cls.get_format()
        return struct.calcsize(fmt)

    def unpack(self, buffer):
        fmt = self.get_format()
        self.total_size, self.block_id, self.block_num, self.remain_size, \
            self.magic_num, self.reserved, self.dump_addr = struct.unpack(fmt, buffer)

    def pack_into(self, buffer, offset):
        fmt = self.get_format()
        struct.pack_into(fmt, buffer, offset, self.total_size, self.block_id, self.block_num,
                         self.remain_size, self.magic_num, self.reserved, self.dump_addr)
        return self.get_size()

    def is_valid(self):
        block_info_magic = 0x5aa5bccd
        return block_info_magic == self.magic_num


@dataclass
class DumpCoreContent:
    block_info: BlockInfo = None
    dump_tensor_map: Dict[int, List[DumpTensor]] = field(default_factory=dict)
    print_list: List[str] = field(default_factory=list)
    index_dtype_dt = {}
    shape: List[int] = field(default_factory=list)

    @staticmethod
    def _write_dump_tensor_data(dump_tensor, dump_data_path):
        with os.fdopen(os.open(dump_data_path, safe_check.OPEN_FLAGS, safe_check.SAVE_DATA_FILE_AUTHORITY), 'wb+') \
                       as f:
            f.write(dump_tensor.dump_data)

    @staticmethod
    def _format_by_shape(values, shape):
        shape = list(shape)
        ndim = len(shape)
        total = len(values)
        if total == 0:
            return ""

        for i in range(ndim - 2, -1, -1):
            shape[i] *= shape[i + 1]

        content = "[" * ndim
        for i in range(total):
            cnt = 0
            for s in shape:
                if (i + 1) % s == 0:
                    cnt += 1
            content += str(values[i])
            if cnt:
                content += "]" * cnt
                if i != total - 1:
                    content += ",\n" + "[" * cnt
            elif i != total - 1:
                content += ","
        return content

    @staticmethod
    def _format_flat(values):
        content = ""
        line_count = 0
        for val in values:
            content += str(val) + ","
            line_count += 1
            if line_count == 8:
                content += "\n"
                line_count = 0
        return content

    @staticmethod
    def _write_dump_tensor_value(dump_tensor, dump_value_path):
        total_ele_num = 0
        if dump_tensor.dump_shape:
            total_ele_num = 1
            for ele in dump_tensor.dump_shape:
                total_ele_num *= ele

        write_content = ""
        if total_ele_num != 0 and total_ele_num == len(dump_tensor.dump_value):
            write_content = DumpCoreContent._format_by_shape(dump_tensor.dump_value, dump_tensor.dump_shape)
        elif total_ele_num != 0:
            logger.warning(
                f'tensor shape {dump_tensor.dump_shape} '
                f'does not match with total length of {len(dump_tensor.dump_value)} '
                'will print data in default format.'
            )

        if not write_content:
            write_content = DumpCoreContent._format_flat(dump_tensor.dump_value)

        with os.fdopen(
                os.open(dump_value_path, safe_check.OPEN_FLAGS, safe_check.SAVE_DATA_FILE_AUTHORITY),
                'w+'
        ) as f:
            f.write(write_content)

    def add_tlv_data(self, tlv):
        if tlv.tag == DumpType.TENSOR_TYPE.value:
            dump_tensor = DumpTensor()
            dump_tensor.parse_from(tlv)
            index = dump_tensor.dump_header.desc
            data_type = dump_tensor.dump_header.data_type
            if self.shape:
                dump_tensor.dump_shape = self.shape.copy()
            self.shape.clear()
            if data_type in dtype_to_data_type.keys():
                self.index_dtype_dt[index] = dtype_to_data_type[data_type]
            else:
                self.index_dtype_dt[index] = david_dtype_to_fmt.get(data_type, '')
            self._add_dump_tensor(dump_tensor)
        elif tlv.tag == DumpType.SCALAR_TYPE.value or tlv.tag == DumpType.ASSERT_TYPE.value:
            print_struct = PrintStruct()
            print_struct.parse_from(tlv)
            self.print_list.append(print_struct.content)
        elif tlv.tag == DumpType.SHAPE_TYPE.value:
            shape_info = ShapeInfo()
            shape_info.parse_from(tlv)
            self.shape = shape_info.shape
        elif tlv.tag == DumpType.META_TYPE.value:
            logger.debug('ignore dump Type: %s', tlv.tag)
        else:
            logger.warning(f'Invalid dump Type: {tlv.tag}')

    def write_to_dir(self, output_dir):
        core_output_dir = os.path.join(
            output_dir, str(self.block_info.block_id))
        os.makedirs(core_output_dir, mode=safe_check.DATA_DIRECTORY_AUTHORITY, exist_ok=True)
        logger.debug(
            'write core %s dump data to dir: %s', self.block_info.block_id, core_output_dir)
        self._write_dump_tensor_by_index(core_output_dir)

    def show_print(self):
        if len(self.print_list) > 0:
            logger.info(f"================ block.{self.block_info.block_id} begin ==============")
            print(''.join(self.print_list), end='', flush=True)
            logger.info(f"================ block.{self.block_info.block_id} end ================")

    def _add_dump_tensor(self, dump_tensor):
        index = dump_tensor.dump_header.desc
        if index not in self.dump_tensor_map:
            self.dump_tensor_map[index] = []
        self.dump_tensor_map[index].append(dump_tensor)

    def _write_dump_tensor_by_loop(self, index, index_output_dir):
        loop_cnt = len(self.dump_tensor_map[index])
        for loop in range(0, loop_cnt):
            dump_tensor = self.dump_tensor_map[index][loop]
            dump_file_name = f'core_{self.block_info.block_id}_index_{index}_loop_{loop}.bin'
            dump_file_path = os.path.join(index_output_dir, dump_file_name)
            self._write_dump_tensor_data(dump_tensor, dump_file_path)

            parsed_dump_file_name = f'core_{self.block_info.block_id}_index_{index}_loop_{loop}.txt'
            parsed_dump_file_path = os.path.join(index_output_dir, parsed_dump_file_name)
            self._write_dump_tensor_value(dump_tensor, parsed_dump_file_path)

    def _write_dump_tensor_by_index(self, core_output_dir):
        for index in self.dump_tensor_map.keys():
            index_output_dir = os.path.join(core_output_dir, f'index_{index}')
            os.makedirs(index_output_dir, mode=safe_check.DATA_DIRECTORY_AUTHORITY, exist_ok=True)
            logger.debug(
                'write index %s tensor to dir: %s', index, index_output_dir)
            self._write_dump_tensor_by_loop(index, index_output_dir)


class DumpType(Enum):
    DEFAULT_TYPE = 0
    SCALAR_TYPE = 1
    TENSOR_TYPE = 2
    SHAPE_TYPE = 3
    ASSERT_TYPE = 4
    META_TYPE = 5


class DumpBinFile:
    def __init__(self, dump_bin):
        self.dump_bin = dump_bin
        self.dump_core_contents = []  # 每个core dump 内容
        self.index_dtype_dt = {}

    def get_dump_core_contents(self, bin_file, file_size):
        block_id = 0
        read_pos = 0
        total_block_size = 1024 * 1024
        block_info_size = BlockInfo.get_size()

        while read_pos + block_info_size < file_size:
            logger.debug('block.%s read from: %s', block_id, read_pos)
            core_content = DumpCoreContent()

            # read block info
            block_info_buffer = bin_file.read(block_info_size)
            block_info = BlockInfo()
            block_info.unpack(block_info_buffer)

            if block_info.reserved == 7:
                logger.warning(f"block.{block_id} remain space is NOT enough for last dump !!!")

            if not block_info.is_valid():
                logger.debug('block.%s block info is not valid, skip this block...', block_id)
                read_pos += total_block_size
                bin_file.seek(read_pos)
                block_id += 1
                continue

            logger.debug(block_info)
            core_content.block_info = block_info
            total_block_size = block_info.total_size

            # read tlv data
            tlv_offset = 0
            core_dump_size = block_info.total_size - \
                             block_info_size - block_info.remain_size
            while tlv_offset < core_dump_size:
                tlv = TLV()
                tlv.read(bin_file)
                core_content.add_tlv_data(tlv)
                tlv_offset += tlv.get_size()
                self.index_dtype_dt.update(core_content.index_dtype_dt)
            self.dump_core_contents.append(core_content)

            # read remain data
            read_pos += block_info.total_size
            bin_file.seek(read_pos)
            block_id += 1

    def parse(self):
        file_size = os.path.getsize(self.dump_bin)
        with open(self.dump_bin, 'rb') as bin_file:
            self.get_dump_core_contents(bin_file, file_size)

    def write_result(self, output_dir):
        if len(self.dump_core_contents) == 0:
            logger.warning('no dump data, exit...')
            return

        parse_output_dir = os.path.abspath(output_dir)
        logger.info(f'write dump workspace result: {parse_output_dir}')
        os.makedirs(parse_output_dir, mode=safe_check.DATA_DIRECTORY_AUTHORITY, exist_ok=True)
        for core_content in self.dump_core_contents:
            core_content.write_to_dir(parse_output_dir)

    def write_index_dtype(self, output_dir):
        if len(self.index_dtype_dt) == 0:
            logger.warning('no dump index, exit...')
            # remove index_dtype.json
            return
        json_path = os.path.join(output_dir, 'index_dtype.json')
        with os.fdopen(os.open(json_path, safe_check.OPEN_FLAGS, safe_check.SAVE_DATA_FILE_AUTHORITY), 'w') as f:
            json.dump(self.index_dtype_dt, f, indent=4)

    def show_print(self):
        logger.info('================ show print start ================')
        for core_content in self.dump_core_contents:
            core_content.show_print()
        logger.info('================ show print end ==================')


def get_data_type(data_type_json: str) -> dict:
    index_dtype_dt = {}
    with os.fdopen(os.open(data_type_json, safe_check.OPEN_FLAGS, safe_check.SAVE_DATA_FILE_AUTHORITY), 'rb') as f:
        index_dtype_dt = json.load(f)
    return index_dtype_dt


def parse_dump_bin(dumpbin_file) -> bool:
    if not os.path.isfile(dumpbin_file):
        return False
    try:
        dump_file = DumpBinFile(dumpbin_file)
        dump_file.parse()
        dump_file.write_result(os.path.dirname(os.path.abspath(dumpbin_file)))
        dump_file.write_index_dtype(os.path.dirname(os.path.abspath(dumpbin_file)))
        dump_file.show_print()
        return True
    except Exception:
        logger.info(f"parse workspace.bin occur exception.")
        import traceback
        traceback.print_exc()
        return False


def clear_dump_files(dump_data_path, workspace_path):
    if os.path.exists(dump_data_path):
        if not os.path.isdir(dump_data_path):
            raise FileExistsError(f'{dump_data_path} should be a directory which created by mskl,'
                                  f'please check if it contains important data and remove it manually')
        shutil.rmtree(dump_data_path)
    if os.path.exists(workspace_path):
        if not os.path.isfile(workspace_path):
            raise FileExistsError(f'{workspace_path} should be a binary file which created by mskl,'
                                  f'please check if it contains important data and remove it manually')
        os.remove(workspace_path)


def append_cpp_file(file_path, dumpbin_file):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        logging.error(f"Read file failed: {e}")
        return False

    content = _handle_include_fstream(content)
    if content is None:
        return False

    main_start_idx, main_end_idx, main_prefix, main_suffix = _extract_main_bounds(content)
    if main_start_idx is None:
        return False
    main_bounds = (main_start_idx, main_end_idx, main_prefix, main_suffix)
    new_content = _inject_dump_code(content, main_bounds, dumpbin_file)
    if new_content is None:
        return False

    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        return True
    except Exception as e:
        logging.error(f"Failed to write file: {e}")
        return False


def _handle_include_fstream(content):
    include_pattern = r'^\s*#include\s+["<].+[">]'
    include_matches = list(re.finditer(include_pattern, content, re.MULTILINE))

    if include_matches:
        last_include = include_matches[-1]
        insert_include_pos = last_include.end()
        return (
            content[:insert_include_pos] +
            '\n#include <fstream>' +
            content[insert_include_pos:]
        )
    else:
        content = '#include <fstream>\n' + content
        logging.warning("Failed to find any header file，No place for #include <fstream>")
        return content


def _extract_main_bounds(content):
    main_start_pattern = r'int\s+main\s*\(\s*\)\s*{'
    main_start_match = re.search(main_start_pattern, content)
    if not main_start_match:
        logging.error("Failed to find main function")
        return None, None, None, None
    main_start_idx = main_start_match.end()

    content_after_main = content[main_start_idx:]
    brace_count = 1
    main_end_idx = -1
    for idx, char in enumerate(content_after_main):
        if char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0:
                main_end_idx = main_start_idx + idx
                break

    if main_end_idx == -1:
        logging.error("Failed to find the } for main function")
        return None, None, None, None

    main_full_start = main_start_match.start()
    main_full_end = main_end_idx + 1
    main_prefix = content[main_full_start:main_full_start + len(main_start_match.group())]
    main_suffix = content[main_end_idx:main_full_end]

    return (main_start_idx, main_end_idx, main_prefix, main_suffix)


def _inject_dump_code(content, main_bounds, dumpbin_file):
    s, e, pre, suf = main_bounds
    mc = content[s:e]
    m = re.search(r'(\w+)GetWorkspaceSize\(([^)]+)\)', mc)
    if not m:
        logging.error("Failed to analyze parameter of GetWorkspaceSize")
        return None
    ps = [p.strip().lstrip('&') for p in m.group(2).split(',') if p.strip()]
    if len(ps) < 2:
        logging.error(f"The number of parameter for GetWorkspaceSize is insufficient.Only（{len(ps)}）")
        return None
    ws = ps[-2]
    ms = re.findall(r'aclrtMalloc\(\s*(&)?(\w+)', mc)
    if not ms:
        logging.error("Failed to find calling aclrtMalloc")
        return None
    wa = ms[-1][1]
    dm = re.search(r'\s*aclDestroyTensor', mc)
    if not dm:
        logging.error("Failed to find calling aclDestroyTensor")
        return None
    pos = dm.start()
    err_open = "Failed to open file: "
    err_write = "Error happened when write to file : "
    c = "\n    {\n"
    c += "        //dump bin file=========\n"
    c += f"        std::vector<int64_t> resultData({ws}, 0);\n"
    c += f"        ret = aclrtMemcpy(resultData.data(), {ws}, {wa}, {ws}, ACL_MEMCPY_DEVICE_TO_HOST);\n"
    c += f"        if (ret == ACL_SUCCESS) {{\n"
    c += f"            std::string filename = R\"({dumpbin_file})\";\n"
    c += f"            std::ofstream file(filename, std::ios::binary);\n"
    c += f"            if (!file) {{\n"
    c += f"                std::cerr << \"{err_open}\" << filename << std::endl;\n"
    c += f"                return -1;\n"
    c += f"            }}\n"
    c += f"            file.write(reinterpret_cast<const char*>(resultData.data()), {ws});\n"
    c += f"            if (!file) {{\n"
    c += f"                std::cerr << \"{err_write}\" << filename << std::endl;\n"
    c += f"                return -1;\n"
    c += f"            }}\n"
    c += f"            file.close();\n"
    c += f"        }}\n"
    c += f"    }}\n"
    c += f"    //dump bin file End=========\n"
    new_mc = mc[:pos] + c + mc[pos:]
    start = s - len(pre)
    end = e + 1
    return content[:start] + pre + new_mc + suf + content[end:]


def build_and_dump(run_cmd: str, op_name: str, dumpbin_file: str, output_folder: str):
    try:
        root_path = Path(__file__).parent.parent.parent.parent
        find_path = root_path / "math" / op_name / "examples" / "test_aclnn_*.cpp"
        file_pattern = str(find_path)
        files = glob.glob(file_pattern, recursive=True)

        if not files:
            logger.error(f"ERROR: {op_name} do not have eager examples\n")
            return

        for f in files:
            append_cpp_file(f, dumpbin_file)

        result = subprocess.run(run_cmd, cwd=str(root_path), capture_output=False, text=True) # noqa: B602
        if result.returncode != 0:
            logger.error(f"%s failed", run_cmd)
            return

        output_folder = os.path.join(os.path.dirname(dumpbin_file), "output")
        os.makedirs(output_folder, exist_ok=True)
        cmd = ["show_kernel_debug_data", dumpbin_file, output_folder]
        result = subprocess.run(cmd, cwd=str(root_path), capture_output=True, text=True) # noqa: B602
        if result.returncode != 0:
            logger.error(f"{run_cmd} failed,{result.stderr}")
            logger.info(f"Try to parse tensor data again \n")
            if not parse_dump_bin(dumpbin_file):
                logger.error("Failed to analyze dump file\n")
                return
        logger.info(f"Tensor data has been dumped to this path {output_folder} \n")
    except Exception as e:
        logger.error(f"Dump tensor %s failed： {e}", run_cmd)
        return
    finally:
        _rollback_modified_files(files)


def _rollback_modified_files(files):
    if not files:
        return
    for file_path in files:
        try:
            rollback_cmd = ["git checkout", "--", file_path]
            file_dir = os.path.dirname(file_path)
            rollback_result = subprocess.run(
                rollback_cmd,
                cwd=file_dir,
                capture_output=True,
                text=True
            )
            if rollback_result.returncode != 0:
                logger.error(f"rollback {file_path} failed：{rollback_result.stderr}")
        except Exception as e:
            logger.error(f"rollback {file_path} exception：{e}")
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import logging
from pathlib import Path


class LogManager:
    """统一的日志管理器"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LogManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.app_name = "assistant"
            self.log_dir = Path.home() / "log"
            self.log_file = self.log_dir / f"{self.app_name}.log"
            self._initialized = True
    
    def setup_logging(self, log_to_console=True, log_level=logging.INFO, app_name=None):
        """
        设置日志配置
        
        Args:
            log_to_console: 是否输出到控制台
            log_level: 日志级别
            app_name: 应用名称，用于日志文件名
        """
        if app_name:
            self.app_name = app_name
            self.log_file = self.log_dir / f"{app_name}.log"
        
        # 创建日志目录
        self.log_dir.mkdir(exist_ok=True)
        
        # 获取root logger
        logger = logging.getLogger()
        logger.setLevel(log_level)
        
        # 清除已有handler
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # 日志格式
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 文件handler（始终启用）
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)
        
        # 控制台handler（可选）
        if log_to_console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            console_handler.setLevel(log_level)
            logger.addHandler(console_handler)
        
        logging.info(f"日志系统初始化完成，日志文件: {self.log_file}")
        return logger

# 创建全局实例
log_manager = LogManager()


def get_logger(name=None):
    """获取logger的便捷函数"""
    return logging.getLogger(name)


def setup(app_name=None, log_to_console=True, log_level=logging.INFO):
    """设置日志的便捷函数"""
    return log_manager.setup_logging(
        app_name=app_name,
        log_to_console=log_to_console,
        log_level=log_level
    )
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

# AICPU built-in package manifest
# Default is OFF for all AICPU ops; only list ON entries here.
# Key is op name (e.g. "log", "cumsum").
# To enable all AICPU ops in built-in package, set FORCE_ALL to ON.

set(AICPU_BUILTIN_PACKAGE_FORCE_ALL OFF)

set(AICPU_BUILTIN_PACKAGE_MANIFEST
  # Add entries like "new_op=ON" when new AICPU ops should be packaged in built-in.
)

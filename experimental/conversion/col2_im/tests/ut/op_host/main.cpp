<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 70701343... cl compile
/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Qiu Zhuang <@qiu-zhuang>
 * - Su Tonghua <@sutonghua>
 *
 * This program is free software: you can redistribute it and/or modify it.
 * Licensed under the CANN Open Software License Agreement Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * See the LICENSE file at the root of the repository for the full text of the License.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

<<<<<<< HEAD
=======
>>>>>>> 0fad06aa... ut
=======
>>>>>>> 70701343... cl compile
#include <gtest/gtest.h>
#include <iostream>

int main(int argc, char **argv) {
    std::cout << "================================" << std::endl;
    std::cout << "Col2Im Tiling Test for Ascend 910B" << std::endl;
    std::cout << "================================" << std::endl;
    
    ::testing::InitGoogleTest(&argc, argv);
    
    // 可以添加全局设置
    testing::GTEST_FLAG(output) = "xml:test_report.xml";
    
    return RUN_ALL_TESTS();
}
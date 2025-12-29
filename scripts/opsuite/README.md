# 算子工具一站式脚本工具 opsuite

## 通过opsuite，可以完成如下用户神算子开发旅程：
以下旅程以开发算子pows为例，默认在项目根路径下执行：

### 构建算子：
- 普通构建：``` python scripts/opsuite/opsuite.py build ```
- 调试构建：``` python scripts/opsuite/opsuite.py build --debug ```
- 支持mssanitizer异常检测的构建：``` python scripts/opsuite/opsuite.py build --mssanitizer ```
注：以上命令是在项目根路径下完成，如果不是在根路径执行，需要通过"--script="参数指定build.sh的相对路径

### 算子执行
```
python scripts/opsuite/opsuite.py deploy_op ./build_out/cann-ops-math-custom_*.run
```
### 编译并执行example
```
python scripts/opsuite/opsuite.py run_example pows eager
```
生成可执行文件test_aclnn_pows，其中eager是控制模式，可选项有eager和graph，默认为eager

### 算子调试
```
python scripts/opsuite/opsuite.py debug build/test_aclnn_pows
```

### 算子性能指标采集
```
python scripts/opsuite/opsuite.py opprof type=onboard  --output=./output_data ./build/test_aclnn_pows
```
算子性能指标采集有上板(onboard)和仿真(simulator)两种运行模式，默认为onboard. output参数指定采集数据输出的路径。

### 算子异常检测
```
python scripts/opsuite/opsuite.py sanitizer ./build/test_aclnn_pows
```
默认执行内存检测

### opsuite的详细帮助信息可以通过执行如下命令获取：
```
python scripts/opsuite/opsuite.py -h
```

### 运行算子example并获取dump tensor
```
python opsuite.py dump pows eager
```
修改、编译并执行算子的example,保存kernel侧tensor数据到本地,解析并生成分析结果
其中pows是算子名称，必填项；eager则是控制模式，可选项有eager和graph，不填 默认为eager
依赖CANN包的show_kernel_debug_data工具,运行前先执行
```source {Ascend_install_path}/ascend-toolkit/latest/bin/setenv.bash```
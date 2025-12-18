# opsuite

算子工具一站式平台，命令行支持如下：

positional arguments:
  {debug,build,opprof,deploy_op,run_example}
                        操作命令
    debug               作用：对算子工程进行msdebug调试 命令行举例： python opsuite.py debug
    build               作用：调用算子的编译工程脚本入口build.sh(默认)或者通过--script指定的shell或者python脚本，目的是编译出算子的二进制文件: 指定编译工程脚本入口文件场景举例： python opsuite.py build --script=../build.sh --pkg
    opprof              作用：采集算子运行的关键性能指标，有上板(onboard)和仿真(simulator)两种运行模式: --type=onboard/simulator （默认为onboard） 命令行举例： python opsuite.py opprof
                        --type=simulator --output=./output_data ./build/test_aclnn_abs
    deploy_op           作用：执行算子安装包。 命令行举例： python opsuite.py deploy_op ./custom_*.run
    run_example         作用：编译并执行算子的调用者example。 命令行举例： python opsuite.py run_example abs eager，其中abs是算子名称，必填项；eager则是控制模式，可选项有eager和graph，不填 默认为eager

options:
  -h, --help            show this help message and exit
  --version, -v         显示算子工具一站式平台的版本号
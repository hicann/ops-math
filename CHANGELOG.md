# CHANGELOG

> 本文档记录各版本的重要变更，版本按时间倒序排列。

## v8.5.0-beta.1
发布日期：2025-12-30

ops-math 首个 Beta 版本 v8.5.0-beta.1 现已发布。
本版本引入了多项新增特性、问题修复及性能改进，目前仍处于测试阶段。
我们诚挚欢迎社区反馈，以进一步提升 ops-math 的稳定性和功能完备性。
使用方式请参阅[官方文档](https://gitcode.com/cann/ops-math/blob/master/README.md)。

### 🔗 版本地址
[CANN 8.5.0-beta 1](https://ascend.devcloud.huaweicloud.com/cann/run/software/8.5.0-beta.1/)

```
版本目录说明如下：
├── aarch64                 # CPU为ARM类型
│   ├── ops                  # ops算子包目录，用于归档算子子包
│   ├── ...
├── x86_64                   # CPU为X86类型
│   ├── ops                  # ops算子包目录，用于归档算子子包
│   ├── ...
```
### 📌 版本配套

**ops-math子包及相关组件与CANN版本配套关系**

| CANN子包版本 | 版本源码标签   | 配套CANN版本|
|--|--|--|
| [cann-ops-math   8.5.0-beta.1](https://ascend.devcloud.huaweicloud.com/cann/run/software/8.5.0-beta.1/) | [v8.5.0-beta.1](https://gitcode.com/cann/ops-math/tags/v8.5.0-beta.1) | CANN   8.5.0-beta.1 |
| [cann-ops-nn   8.5.0-beta.1](https://ascend.devcloud.huaweicloud.com/cann/run/software/8.5.0-beta.1/) | [v8.5.0-beta.1](https://gitcode.com/cann/ops-nn/tags/v8.5.0-beta.1) | CANN   8.5.0-beta.1 |
| [cann-ops-cv   8.5.0-beta.1](https://ascend.devcloud.huaweicloud.com/cann/run/software/8.5.0-beta.1/) | [v8.5.0-beta.1](https://gitcode.com/cann/ops-cv/tags/v8.5.0-beta.1) | CANN   8.5.0-beta.1 |
| [cann-ops-transformer   8.5.0-beta.1](https://ascend.devcloud.huaweicloud.com/cann/run/software/8.5.0-beta.1/) | [v8.5.0-beta.1](https://gitcode.com/cann/ops-transformer/tags/v8.5.0-beta.1) | CANN   8.5.0-beta.1 |
| [cann-hccl   8.5.0-beta.1](https://ascend.devcloud.huaweicloud.com/cann/run/software/8.5.0-beta.1/) | [v8.5.0-beta.1](https://gitcode.com/cann/hccl/tags/v8.5.0-beta.1) | CANN   8.5.0-beta.1 |
| [cann-hixl   8.5.0-beta.1](https://ascend.devcloud.huaweicloud.com/cann/run/software/8.5.0-beta.1/) | [v8.5.0-beta.1](https://gitcode.com/cann/hixl/tags/v8.5.0-beta.1) | CANN   8.5.0-beta.1 |

### 🚀 关键特性

- 【工程能力】math类onnx算子插件支持。([#431](https://gitcode.com/cann/ops-math/pull/431))
- 【算子实现】在多个类别中新支持算子concat([#573](https://gitcode.com/cann/ops-math/pull/573))、lerp([#519](https://gitcode.com/cann/ops-math/pull/519))、
drop_out_v3([#539](https://gitcode.com/cann/ops-math/pull/539))等。
- 【资料优化】新增快速入门指南，优化贡献指南中新算子贡献流程。([#472](https://gitcode.com/cann/ops-math/pull/472))
- 【易用性提升】kernel编译流程优化，开启info级别日志打屏。([#326](https://gitcode.com/cann/ops-math/pull/326))

### 🐛 问题修复
- tile算子aclnn接口与算子原型不符，内置算子调用失败。([Issue239](https://gitcode.com/cann/ops-math/issues/239))
- experimental目录算子使用代码仓自带op_api执行时报错。([Issue143](https://gitcode.com/cann/ops-math/issues/143))
- 算子部署路径与指定vendor_name不一致。([Issue86](https://gitcode.com/cann/ops-math/issues/86))
- 算子配置文件及文件夹无法自动创建。([Issue82](https://gitcode.com/cann/ops-math/issues/82))
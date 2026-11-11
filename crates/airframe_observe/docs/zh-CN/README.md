# airframe-observe 0.1.4 — 推理事实与观察者

**语言：** [English](../../README.md) · **简体中文** · [繁體中文](../zh-TW/README.md)

airframe-observe 为 Airframe 提供事实采集、观察者注册与宿主侧推理协调。它**不执行 GPU 矩阵运算，也不证明 logits 数值正确**。

## 组件职责

`ObservationSession` 注册观察者，接收引擎实际发出的层、logits、文本和调度时间事实，并执行饱和处理。ISF 在调用方提供 embedding、prefill、decode 和控制回调的条件下，通过 Dzero 的 `FactStore`、`ClosureProgram`、`RunBudget` 和 `SaturationFabric` 协调执行。

Airframe 依赖 airframe-observe（反方向**不依赖**）；其 Cargo 依赖要求 `dzero = "0.1"`，协调版本解析为 Dzero 0.1.0。

```toml
[dependencies]
airframe-observe = "0.1.4"
```

## 验证和性能边界

观察依赖于生产者实际发出的事实。观察者回调可能产生开销，不保证对观察者数量恒定成本，也不保证所有后端提供 GPU 时间戳。本组件不验证模型元数据、修复 GPU 驱动故障或保证延迟。数值认证必须对精确模型和配置另外运行引擎/FP32 对照测试。

libfse 0.1.4 的字节策略扫描与本组件职责不同；它具有单独的 [Apache-2.0 补充授权](https://github.com/Michael-A-Kuykendall/airframe/blob/8efb1e7a830b0822cfc5ac1b121e149948d5ffe3/crates/libfse/LICENSE-GRANT-0.1.4.md)，不能与本组件的 MIT 授权混为一谈。

## 许可及专利声明

完整职责、性能边界、专利声明与联系信息、许可证请参阅英文权威章节。

[Release responsibilities](../../README.md#release-responsibilities) · [Correctness and performance boundary](../../README.md#correctness-and-performance-boundary) · [Patent Notice](../../README.md#patent-notice) · [License](../../README.md#license)

使用 [MIT](../../LICENSE)。本组件描述的 FSE 推理观察架构涉及未决美国专利；其具体声明见[英文 README](../../README.md)。另请参阅 [Airframe 许可说明](https://github.com/Michael-A-Kuykendall/airframe/blob/366c8aa429116cf305e04c9dde182a842040b4b1/LICENSING.md)、[变更记录](../../CHANGELOG.md)和[路线图](https://github.com/Michael-A-Kuykendall/airframe/blob/main/ROADMAP.md)。
# Airframe — Rust/WGSL GGUF 推理引擎

候选版本 **0.4.6 尚未发布**，当前 crates.io 版本为 **0.4.5**。下方依赖示例用于 0.4.6 发布后；审核候选版本时请构建所提供的源码。

**语言：** [English](../../README.md) · **简体中文** · [繁體中文](../zh-TW/README.md)

Airframe **0.4.5** 是 Shimmy 的纯 Rust GPU 推理引擎，通过 wgpu/WGSL 在可用 GPU 后端执行限定范围的稠密模型。它不是 OpenAI HTTP 服务器；服务器接口由 [Shimmy](https://github.com/Michael-A-Kuykendall/shimmy) 提供。

## 已发布范围与后续候选

已发布的 0.4.5 包含 GGUF/张量边界检查、模型能力准入、Llama RoPE 因子、请求隔离状态及 Q/K 和 scratch 修正。已发布的数值比较主要来自 NVIDIA/WSL Vulkan；不同平台和模型不能据此自动获得认证。

**尚未发布**的 Apple Metal 0.4.6 候选源码 在 Apple M1（Virtual）/ Apple Paravirtual Metal / macOS 15.8.2 上完成 198/198 次完整 logits 比较，最大绝对误差 0.002768、RMS 0.000688，均在原有 0.1 / 0.03 阈值内，贪心 token 一致。这**不代表 0.4.5 的已发布包包含该更改**。实体 M4：NOT_RUN；间歇性 8K 停顿根因与 watchdog 清理缺陷未解决。

候选修复包括 Metal 大模型上传与输出头初始化之间的提交/等待边界，以及 macOS 外层 prefill 不超过 64 tokens；不修改神经网络数学或数值阈值。明确的覆盖细节见 [Shimmy 认证文档](https://github.com/Michael-A-Kuykendall/shimmy/blob/main/docs/CERTIFICATION.md)。

## 依赖与许可

Airframe 使用 MIT。直接依赖 [ShimmyTok 0.8.5](https://github.com/Michael-A-Kuykendall/shimmytok) 进行 GGUF tokenization、shimmyjinja 0.5.0 执行 chat_template，并使用工作区内的 libfse 0.1.4 和可选 airframe-observe 0.1.4。libfse 具有适用于现有 0.1.4 包的 [Apache-2.0 补充授权](https://github.com/Michael-A-Kuykendall/airframe/blob/8efb1e7a830b0822cfc5ac1b121e149948d5ffe3/crates/libfse/LICENSE-GRANT-0.1.4.md)。请保留第三方通知并阅读[完整许可说明](../../LICENSING.md)。

```toml
[dependencies]
airframe = "0.4.6"
```

0.4.5 的已发布存档不包含这些 Metal 修复。

## 执行范围与限制

只验证特定 Qwen3/Llama GGUF Q4_K_M 模型、上下文和 F32 KV/参考内核组合。解析元数据不等于可执行所有架构；INT4、实验内核、MoE、循环模型和多模态执行器均不能宣称已验证。没有跨 GPU/平台通用确定性、速度或稳定性保证。

## 开发者资料

[英文 README](../../README.md) · [更新记录](../../CHANGELOG.md) · [路线图](../../ROADMAP.md) · [架构与组件](../../Cargo.toml) · [贡献](../../CONTRIBUTING.md)。

所有运行命令、crate 名称和 API 标识均保留英文原样。

## 完整英文文档

本页为简明指南。未在此复写的完整 API 示例、计时开关、组件、开发与治理信息，请阅读以下英文权威章节。

[Release contract](../../README.md#release-contract) · [Library use](../../README.md#library-use) · [Components and precision](../../README.md#components-and-precision) · [Profiling and experimental code](../../README.md#profiling-and-experimental-code) · [Development](../../README.md#development) · [Ecosystem and licensing](../../README.md#ecosystem-and-licensing) · [Support](../../README.md#support) · [License](../../README.md#license)
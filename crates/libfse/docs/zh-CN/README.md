# libfse 0.1.4 — 字节模式策略扫描

**语言：** [English](../../README.md) · **简体中文** · [繁體中文](../zh-TW/README.md)

libfse 是字节模式匹配与动作扫描器，**不是 GPU 推理引擎，也不是通用语义安全系统**。它将规则编译为自动机，逐流处理字节，返回记录摘要或策略违规。

## 安装

```toml
[dependencies]
libfse = "0.1.4"
```

`FseMap::compile` 编译规则；`FseMap::scan_with_cursor` 使用流的独立 `ScanCursor` 扫描字节。每个逻辑流都应使用独立 cursor，请求之间应重置或丢弃状态。扫描是库函数，不提供命令行程序。

**调用方必须显式调用扫描器，并依据返回的 `Reject` / `Violation::PolicyReject` 阻止操作**。`Violation::IntegrityError` 是硬故障，不代表允许继续。`Record` 不自动生成防篡改审计日志；`Ignore` 不执行策略动作。

同一个自动机可以共享遍历，但实际开销取决于输入字节数、匹配动作次数及存储；不保证恒定成本或统一吞吐。模式匹配不等于完整 SQL 注入防护、权限控制或内容审核。

## 许可与专利

适用于已发布 **libfse 0.1.4** 的 [Apache-2.0 补充授权](../../LICENSE-GRANT-0.1.4.md) 明确允许商业使用、修改、再分发和产品集成，无须另行取得商业许可。历史 crates.io 包字节和历史元数据未改动；仍须遵守第三方条款、[NOTICE](../../NOTICE) 与 Apache-2.0 第 3 节适用的专利授权。参阅[完整许可证](../../LICENSE)。

## 与 Airframe Observe 的区别

libfse 负责策略扫描（字节、模式、动作及违规）；airframe-observe 0.1.4 负责推理事实、观察者和宿主回调协调，不会因此自动证明 GPU 数值正确性。两者职责及许可证不同。

[英文 README](../../README.md) · [变更记录](../../CHANGELOG.md) · [Airframe 路线图](https://github.com/Michael-A-Kuykendall/airframe/blob/main/ROADMAP.md)。

## 完整英文文档

本页为简明指南；完整代码示例、状态和错误边界、工作量限制及组件职责请参阅英文权威章节。

[What 0.1.4 provides](../../README.md#what-014-provides) · [Library example](../../README.md#library-example) · [State, errors and limits](../../README.md#state-errors-and-limits) · [Relationship to Airframe Observe](../../README.md#relationship-to-airframe-observe)
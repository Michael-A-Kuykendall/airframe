# Airframe — Rust/WGSL GGUF 推論引擎

**Airframe 0.4.6 已於 2026-11-12 正式發布**（[crates.io](https://crates.io/crates/airframe/0.4.6) · [GitHub](https://github.com/Michael-A-Kuykendall/airframe/releases/tag/v0.4.6)）。下方相依套件範例可直接安裝已發布版本。

**語言：** [English](../../README.md) · [简体中文](../zh-CN/README.md) · **繁體中文**

Airframe **0.4.6** 是 Shimmy 使用的純 Rust GPU 推論引擎，透過 wgpu/WGSL 在適當 GPU 後端執行有明確界限的稠密模型。它不是 OpenAI HTTP 伺服器；該介面由 [Shimmy](https://github.com/Michael-A-Kuykendall/shimmy) 提供。

## 已發布範圍與歷史版本

0.4.6 延續了歷史 0.4.5 版本中的 GGUF／張量邊界驗證、模型能力准入、Llama RoPE 因子、請求隔離狀態、Q/K 所有權及 scratch 修正。已發布版本的數值證據主要來自 NVIDIA/WSL Vulkan，不能據此推定其他平台已通過驗證。

**已發布的 Airframe 0.4.6** 在 Apple M1（Virtual）/ Apple Paravirtual Metal / macOS 15.8.2 完成 198/198 次完整 logits 比較，最大絕對誤差 0.002768、RMS 0.000688，符合原始 0.1 / 0.03 限制，且貪婪 token 決策一致。**歷史 0.4.5 不包含此 Metal 修正；0.4.6 已包含**。實體 M4：NOT_RUN；歷史 8K 間歇性停滯原因及 watchdog 清理問題尚未解決。

0.4.6 正式版修正新增 Metal 大型模型上傳與輸出頭配置之間的提交／等待界線，並將 macOS 外層 prefill 限制為最多 64 tokens；未更動數學運算或數值門檻。參閱 [Shimmy 認證文件](https://github.com/Michael-A-Kuykendall/shimmy/blob/main/docs/CERTIFICATION.md)。

## 依賴及授權

Airframe 使用 MIT。它直接依賴 [ShimmyTok 0.8.5](https://github.com/Michael-A-Kuykendall/shimmytok) 處理 GGUF tokenization、shimmyjinja 0.5.0 處理 chat_template，工作區內使用 libfse 0.1.4 和選用的 airframe-observe 0.1.4。libfse 另有針對既有版本的 [Apache-2.0 補充授權](https://github.com/Michael-A-Kuykendall/airframe/blob/8efb1e7a830b0822cfc5ac1b121e149948d5ffe3/crates/libfse/LICENSE-GRANT-0.1.4.md)。仍須保留第三方聲明，參閱[完整授權資訊](../../LICENSING.md)。

```toml
[dependencies]
airframe = "0.4.6"
```

**Airframe 0.4.6 已包含這些修正**；歷史 0.4.5 封存套件不包含。

## 執行界限

驗證僅涵蓋指定 Qwen3/Llama GGUF Q4_K_M 模型、上下文及 F32 KV／參考核心組合。能辨識中繼資料不代表所有架構可推論；INT4、實驗核心、MoE、循環模型與多模態執行器均不在認證範圍。沒有跨 GPU 或平台的通用確定性、效能與穩定性保證。

## 開發文件

[英文 README](../../README.md) · [變更記錄](../../CHANGELOG.md) · [路線圖](../../ROADMAP.md) · [元件資訊](../../Cargo.toml) · [貢獻](../../CONTRIBUTING.md)。

命令、crate 名稱及 API 識別字維持英文原文。

## 完整英文文件

本頁為簡明指南。未在此重述的完整 API 範例、計時開關、元件、開發與治理資訊，請閱讀下列英文權威章節。

[Release contract](../../README.md#release-contract) · [Library use](../../README.md#library-use) · [Components and precision](../../README.md#components-and-precision) · [Profiling and experimental code](../../README.md#profiling-and-experimental-code) · [Development](../../README.md#development) · [Ecosystem and licensing](../../README.md#ecosystem-and-licensing) · [Support](../../README.md#support) · [License](../../README.md#license)
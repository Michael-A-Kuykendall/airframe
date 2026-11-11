# Airframe — Rust/WGSL GGUF 推論引擎

**語言：** [English](../../README.md) · [简体中文](../zh-CN/README.md) · **繁體中文**

Airframe **0.4.5** 是 Shimmy 使用的純 Rust GPU 推論引擎，透過 wgpu/WGSL 在適當 GPU 後端執行有明確界限的稠密模型。它不是 OpenAI HTTP 伺服器；該介面由 [Shimmy](https://github.com/Michael-A-Kuykendall/shimmy) 提供。

## 已發布範圍與後續候選

現行 0.4.5 包含 GGUF／張量邊界驗證、模型能力准入、Llama RoPE 因子、請求隔離狀態、Q/K 所有權及 scratch 修正。已發布版本的數值證據主要來自 NVIDIA/WSL Vulkan，不能據此推定其他平台已通過驗證。

**尚未發布**的 Apple Metal 候選原始碼 [`68091da`](https://github.com/Michael-A-Kuykendall/airframe-private/commit/68091da8c55a2f96bc4309b46ece92630c996f71) 在 Apple M1（Virtual）/ Apple Paravirtual Metal / macOS 15.8.2 完成 198/198 次完整 logits 比較，最大絕對誤差 0.002768、RMS 0.000688，符合原始 0.1 / 0.03 限制，且貪婪 token 決策一致。**已發布的 0.4.5 不包含此候選修正**。實體 M4：NOT_RUN；歷史 8K 間歇性停滯原因及 watchdog 清理問題尚未解決。

候選修正新增 Metal 大型模型上傳與輸出頭配置之間的提交／等待界線，並將 macOS 外層 prefill 限制為最多 64 tokens；未更動數學運算或數值門檻。參閱 [Shimmy 認證文件](https://github.com/Michael-A-Kuykendall/shimmy/blob/main/docs/CERTIFICATION.md)。

## 依賴及授權

Airframe 使用 MIT。它直接依賴 [ShimmyTok 0.8.5](https://github.com/Michael-A-Kuykendall/shimmytok) 處理 GGUF tokenization、shimmyjinja 0.5.0 處理 chat_template，工作區內使用 libfse 0.1.4 和選用的 airframe-observe 0.1.4。libfse 另有針對既有版本的 [Apache-2.0 補充授權](../../crates/libfse/LICENSE-GRANT-0.1.4.md)。仍須保留第三方聲明，參閱[完整授權資訊](../../LICENSING.md)。

```toml
[dependencies]
airframe = "0.4.5"
```

這只指定**目前已發布**的套件，並不包含上述 Apple Metal 候選修正。

## 執行界限

驗證僅涵蓋指定 Qwen3/Llama GGUF Q4_K_M 模型、上下文及 F32 KV／參考核心組合。能辨識中繼資料不代表所有架構可推論；INT4、實驗核心、MoE、循環模型與多模態執行器均不在認證範圍。沒有跨 GPU 或平台的通用確定性、效能與穩定性保證。

## 開發文件

[英文 README](../../README.md) · [變更記錄](../../CHANGELOG.md) · [路線圖](../../ROADMAP.md) · [元件資訊](../../Cargo.toml) · [貢獻](../../CONTRIBUTING.md)。

命令、crate 名稱及 API 識別字維持英文原文。
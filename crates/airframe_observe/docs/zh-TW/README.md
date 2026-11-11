# airframe-observe 0.1.4 — 推論事實與觀察者

**語言：** [English](../../README.md) · [简体中文](../zh-CN/README.md) · **繁體中文**

airframe-observe 為 Airframe 提供事實收集、觀察者註冊與主機端推論協調。它**不執行 GPU 矩陣運算，也不能證明 logits 數值正確**。

## 元件職責

`ObservationSession` 註冊觀察者、接收引擎實際發出的層、logits、文字及排程時間事實，並執行飽和處理。ISF 在呼叫端提供 embedding、prefill、decode 和控制回呼時，透過 Dzero 的 `FactStore`、`ClosureProgram`、`RunBudget` 和 `SaturationFabric` 進行協調。

Airframe 依賴 airframe-observe，而非反向依賴。Cargo 需求為 `dzero = "0.1"`，協調版本解析為 Dzero 0.1.0。

```toml
[dependencies]
airframe-observe = "0.1.4"
```

## 正確性與效能界限

觀察功能僅能處理生產者真正發出的事實。回呼會產生成本，不保證觀察者數量與成本無關，亦不保證所有後端提供 GPU 時間戳。本套件不驗證模型中繼資料、不修復 GPU 驅動故障，也不保證延遲。數值認證必須另以精確模型、設定及 FP32 參考進行實際測試。

libfse 0.1.4 的位元組政策掃描與此元件不同；其獨立的 [Apache-2.0 補充授權](../../../libfse/LICENSE-GRANT-0.1.4.md) 不等同於本元件的 MIT 授權。

## 授權及專利聲明

採 [MIT](../../LICENSE)。所述 FSE 推論觀察架構涉及申請中的美國專利，請參閱[英文 README](../../README.md) 的完整聲明，以及 [Airframe 授權說明](../../../../LICENSING.md)、[變更記錄](../../CHANGELOG.md)和[路線圖](https://github.com/Michael-A-Kuykendall/airframe/blob/main/ROADMAP.md)。
# libfse 0.1.4 — 位元組模式政策掃描

**語言：** [English](../../README.md) · [简体中文](../zh-CN/README.md) · **繁體中文**

libfse 是位元組模式比對與動作掃描函式庫，**不是 GPU 推論引擎，也不是通用語意安全系統**。它將規則編譯成自動機，逐個資料流掃描位元組，回傳紀錄摘要或政策違規。

## 安裝

```toml
[dependencies]
libfse = "0.1.4"
```

`FseMap::compile` 編譯規則；`FseMap::scan_with_cursor` 使用獨立的 `ScanCursor` 處理資料流。每個邏輯資料流應使用各自的 cursor，並於請求界線重設或捨棄狀態。本套件是函式庫，沒有命令列執行檔。

**呼叫端必須實際執行掃描，並根據 `Reject`／`Violation::PolicyReject` 結果阻止操作**。`Violation::IntegrityError` 是不可忽略的完整性錯誤。`Record` 不會自動建立防竄改稽核紀錄；`Ignore` 不執行政策動作。

共享自動機走訪不等於恆定成本；效能取決於輸入長度、實際執行動作及記憶體。位元組比對並非完整 SQL 注入防護、授權或內容審核系統。

## 授權與專利

針對既有 **libfse 0.1.4** 套件的 [Apache-2.0 補充授權](../../LICENSE-GRANT-0.1.4.md) 明確允許商業使用、修改、再散布及嵌入產品，不需另行取得商業授權。歷史 crates.io 壓縮檔與原始登錄資訊未變；仍須保留第三方條款、[NOTICE](../../NOTICE) 及 Apache-2.0 第 3 節適用專利授權。詳見[完整授權](../../LICENSE)。

## 與 Airframe Observe 的差異

libfse 專責政策掃描（位元組、模式、動作及違規）；airframe-observe 0.1.4 協調推論事實、觀察者與主機回呼，不因此驗證 GPU 數值正確性。兩套函式庫的職責與授權不同。

[英文 README](../../README.md) · [變更記錄](../../CHANGELOG.md) · [Airframe 路線圖](https://github.com/Michael-A-Kuykendall/airframe/blob/main/ROADMAP.md)。

## 完整英文文件

本頁為簡明指南；完整程式碼範例、狀態與錯誤界線、工作量限制及元件職責請參閱英文權威章節。

[What 0.1.4 provides](../../README.md#what-014-provides) · [Library example](../../README.md#library-example) · [State, errors and limits](../../README.md#state-errors-and-limits) · [Relationship to Airframe Observe](../../README.md#relationship-to-airframe-observe)
# Signal & Alert Policy | 訊號與警報政策

*Multi-Asset AI Trading Assistant | 多資產AI交易助手*

---

## 🎯 Signal Card Standard | 訊號卡標準

### Common Fields | 通用欄位

All trading signals must include the following standardized fields:

所有交易訊號必須包含以下標準化欄位：

| Field | 欄位 | Type | Description | 描述 |
|-------|------|------|-------------|------|
| **Symbol** | 代號 | String | Trading instrument | 交易標的 |
| **Asset Class** | 資產類別 | Enum | crypto/fx/gold/us_equity/hk_equity | 加密貨幣/外匯/黃金/美股/港股 |
| **Direction** | 方向 | String | "long" or "short" | 做多或做空 |
| **Entry Range** | 入場區間 | Object | {"low": X, "high": Y} | 進場價格範圍 |
| **Targets** | 目標價 | Object | {"t1": X, "t2": Y} | 部分獲利目標 |
| **Stop Loss** | 止損 | Float | Hard stop price | 硬止損價格 |
| **Horizon** | 持有期 | String | "intraday"/"swing"/"position" | 當日/擺盪/持倉交易 |
| **Conviction** | 置信度 | Integer | 1-5 scale | 1-5分信心等級 |
| **Expected Alpha** | 預期超額 | Float | Basis points, net of costs | 基點，扣除成本後 |
| **Reasoning** | 理由 | Array | 3 key reasons | 三大關鍵原因 |
| **Size %** | 建議倉位 | Float | % of portfolio | 組合占比建議 |

---

## 🏛️ Asset-Specific Fields | 資產特有欄位

### 🪙 Crypto | 加密貨幣
- **Maker/Taker Fee Tier** | 掛單/吃單費率檔位
- **Funding Rate** | 資金費率 (8h or 1h)
- **24/7 Venue Liquidity** | 全天候交易所流動性
- **Volatility Regime** | 波動性狀態

### 💱 FX | 外匯
- **Pip Entry/Stop/Targets** | 點子計價進場/止損/目標
- **Swap Rate Estimate** | 隔夜利息預估
- **Trading Session** | 交易時段 (Asia/Europe/US)
- **Macro Event Risk** | 宏觀事件風險

### 🥇 Gold | 黃金 (XAUUSD/COMEX)
- **Contract/Spot** | 期貨/現貨
- **Contract Multiplier** | 合約乘數
- **Roll Window** | 換倉窗口
- **Dollar Per Tick** | 每跳點值

### 🇺🇸 US Stocks | 美股
- **Earnings Date** | 財報日期
- **Pre/Post Market Note** | 盤前盤後流動性提醒
- **Borrow Cost** | 融券成本 (if short)

### 🇭🇰 HK Stocks | 港股
- **HKEX Schedule Note** | 港交所時間提醒 (午休/半日市)
- **Stamp Duty/Levy** | 印花稅/交易徵費 (in cost model)
- **Borrow Availability** | 融券可用性

---

## 🚨 Alert Trigger Rules | 警報觸發規則

### Non-Negotiable Filters | 不可妥協的過濾條件

#### 1. Edge Requirement | 邊際要求
```
Alert ONLY if: Expected Alpha ≥ 2× (fees + slippage + borrow/funding/swap)
僅在以下條件下警報：預期超額 ≥ 2×（手續費＋滑點＋融券/資金費/隔夜息）
```

#### 2. Position Limits | 倉位限制
- **Single Position**: ≤ 6% of portfolio | 單一持倉 ≤ 組合6%
- **Daily Turnover**: ≤ 15% of portfolio | 日換手率 ≤ 組合15%
- **Portfolio Target Vol**: 10-12% annually | 組合目標波動率 10-12%年化

#### 3. Quality Gates | 質量門檻
- **CPCV** (Combinatorially Purged Cross-Validation) passed
- **Deflated Sharpe** ≥ 0
- **2-week Paper-Live Parity** for IS & turnover

#### 4. Kill-Switch Triggers | 熔斷觸發
Automatic alert pause when:
自動暫停警報條件：

- **Intraday Drawdown** > 3% | 日內回撤 > 3%
- **Data Freshness** > 30 seconds | 數據延遲 > 30秒
- **Reject Rate Spike** > 20% | 拒單率飆升 > 20%

When triggered: Stop new alerts + Red `/status` | 觸發時：停止新警報＋紅色狀態

---

## 🕐 Time Zones & Trading Hours | 時區與交易時間

### Default Display | 預設顯示
- **Primary**: Hong Kong Time (HKT, UTC+8) | 主要：香港時間
- **Tooltips**: Local exchange time | 提示：交易所當地時間

### Asset Class Schedules | 資產類別時間表

| Asset | 資產 | Hours | 時間 | Special Notes | 特殊說明 |
|-------|------|-------|------|---------------|----------|
| **Crypto** | 加密貨幣 | 24/7 | 全天候 | Continuous trading | 連續交易 |
| **FX** | 外匯 | 24/5 | 24/5 | Weekend gaps, 5PM NY roll | 週末間隔，紐約5點換日 |
| **Gold** | 黃金 | ~23/5 | ~23/5 | Contract rolls monthly | 合約月度換倉 |
| **US Equity** | 美股 | 9:30-16:00 EST | 美東時間 | + Pre/post market | +盤前盤後 |
| **HK Equity** | 港股 | 9:30-12:00, 13:00-16:00 HKT | 港時 | Lunch break, typhoon/rain halt | 午休，颱風/暴雨停市 |

---

## 🎚️ Signal Priority & Ranking | 訊號優先級與排序

### Ranking Formula | 排序公式
```
Signal Score = (Expected Alpha × Conviction × Liquidity Score) - Risk Penalty
訊號分數 = (預期超額 × 置信度 × 流動性分數) - 風險懲罰
```

### Alert Frequency | 警報頻率
- **High Priority**: Immediate push | 高優先級：立即推送
- **Medium Priority**: Batched every 30 minutes | 中優先級：每30分鐘批次
- **Low Priority**: Daily digest only | 低優先級：僅日報

### Quiet Hours | 靜默時間
Configurable per region:
按地區可配置：

- **Asia**: 23:00-07:00 HKT | 亞洲：港時23:00-07:00
- **US**: 22:00-06:00 EST | 美國：美東時間22:00-06:00

---

## 📊 Performance Tracking | 績效追蹤

### Signal KPIs | 訊號關鍵指標
- **T1 Hit Rate**: % reaching first target | T1命中率：到達首個目標百分比
- **Stop Out Rate**: % hitting stop loss | 止損率：觸及止損百分比
- **Average Hold Time**: By horizon type | 平均持有時間：按期間類型
- **Implementation Shortfall**: p50/p95 | 執行差異：中位數/95分位

### Attribution | 歸因分析
- **By Asset Class** | 按資產類別
- **By Agent Type** | 按代理類型
- **By Cost Bucket** | 按成本類別
- **By Market Regime** | 按市場狀態

---

## ⚖️ Legal & Compliance | 法律與合規

### Disclaimers | 免責聲明

**English**: 
*This system provides educational trading signals for paper trading by default. Signals are not investment advice. All trading involves substantial risk of loss. Past performance does not guarantee future results. Users must comply with local financial regulations.*

**繁體中文**：
*本系統預設提供教育性交易訊號供紙上交易使用。訊號非投資建議。所有交易均涉及重大虧損風險。過往表現不保證未來結果。用戶必須遵守當地金融法規。*

### Paper Mode Default | 預設紙上模式
- All signals default to paper trading | 所有訊號預設紙上交易
- Live mode requires explicit opt-in + 2-week validation | 實盤模式需明確選擇＋2週驗證
- Clear "PAPER MODE" labeling on all outputs | 所有輸出清楚標示「紙上模式」

---

## 🔄 Implementation Checklist | 實施檢查清單

### Phase 1: Foundation | 第一階段：基礎
- [ ] Lock signal card format (bilingual) | 鎖定訊號卡格式（雙語）
- [ ] Implement edge ≥ 2× cost filter | 實施邊際≥2×成本過濾
- [ ] Set up HKT + local time display | 設置港時＋當地時間顯示
- [ ] Create asset-specific cost models | 建立資產特定成本模型

### Phase 2: Quality Gates | 第二階段：質量門檻
- [ ] CPCV validation framework | CPCV驗證框架
- [ ] Deflated Sharpe calculation | 通縮夏普比計算
- [ ] Paper-live parity monitoring | 紙上-實盤一致性監控
- [ ] Kill-switch implementation | 熔斷機制實施

### Phase 3: Operations | 第三階段：運營
- [ ] Multi-asset signal ranking | 多資產訊號排序
- [ ] Quiet hours by region | 按地區靜默時間
- [ ] Performance attribution | 績效歸因
- [ ] Bilingual alert system | 雙語警報系統

---

*Last Updated: September 2025 | 最後更新：2025年9月*
*Status: Draft for Review | 狀態：草案待審*

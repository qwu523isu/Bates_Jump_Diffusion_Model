"""
To formalize this, we report the Mincer-Zarnowitz volatility prediction regressions:
RVi = $\alpha + \beta_1 \times \sigma^Q_{\text{FOMC}, i} + \beta_2 \times V_{\text{ol}_i} + \varepsilon_i$
where RVi is the ith event’s realized volatility, 
$\sigma^Q_{\text{FOMC}, i}$ is the ex-ante option-implied event risk, 
and $V_{\text{ol}_i}$ is a measure of diffusive spot volatility. 
We include two alternative controls for non-event “background” volatility $V_{\text{ol}_i}$: 
a rolling window GARCH estimates of the daily volatility on the event day 
and also the calibrated spot volatility state variable from the SV model prior to the event, $\sqrt{v_{\tau_i^-}}$.
为了正式化这一分析，我们报告了 Mincer-Zarnowitz 波动率预测回归：
RVi = $\alpha + \beta_1 \times \sigma^Q_{\text{FOMC}, i} + \beta_2 \times V_{\text{ol}_i} + \varepsilon_i$
其中，RVi 是第 i 次事件的实现波动率，$\sigma^Q_{\text{FOMC}, i}$ 是事前期权隐含的 FOMC 事件风险，
$V_{\text{ol}_i}$ 是扩散现货波动率的衡量指标。我们纳入了两种替代控制非事件“背景”波动率 $V_{\text{ol}_i}$ 的方法：
事件每日波动率的滚动窗口 GARCH 估计，以及事件前 SV 模型校准的现货波动率状态变量，$\sqrt{v_{\tau_i^-}}$。
"""
import os
import glob
import numpy as np
import pandas as pd
import statsmodels.api as sm

# ============================================================
# 配置 / CONFIG
# ============================================================
OUTPUT_BASE_DIR = r"E:\Output"
DATE_START = "20210516"
DATE_END   = "20210521"

# 用于事件实现波动率的窗口（与之前一致）
PRE_MIN  = 5    # 事件前 5 分钟
POST_MIN = 30   # 事件后 30 分钟
RET_FREQ = "5min"

# HAC 最大滞后阶数
HAC_LAGS = 1

# ============================================================
# 帮助函数：计算给定事件的实现波动率 RV_i
# ============================================================
def compute_event_realized_vol(df_prices, event_ts):
    """
    df_prices: 带有 local_timestamp, underlying_price 的 DataFrame
    event_ts: pandas.Timestamp 事件时间

    返回: 实现波动率 (sqrt(realized variance))，单位与 log return 一致
    """
    start = event_ts - pd.Timedelta(minutes=PRE_MIN)
    end   = event_ts + pd.Timedelta(minutes=POST_MIN)

    window = df_prices[(df_prices["local_timestamp"] >= start) &
                       (df_prices["local_timestamp"] <= end)].copy()
    if len(window) < 3:
        return np.nan

    window = window.set_index("local_timestamp")
    px = window["underlying_price"].resample(RET_FREQ).last().dropna()
    if len(px) < 3:
        return np.nan

    logret = np.log(px).diff().dropna()
    rv = np.sum(logret**2)
    return np.sqrt(rv)


# 帮助函数：计算当天“背景日波动率” (简化版 daily realized vol)
def compute_daily_realized_vol(df_prices, date_str):
    """
    用当天全部 5 分钟收益的 realized volatility 作为背景波动率 proxy
    """
    df_day = df_prices[
        df_prices["local_timestamp"].dt.strftime("%Y%m%d") == date_str
    ].copy()
    if len(df_day) < 10:
        return np.nan

    df_day = df_day.set_index("local_timestamp")
    px = df_day["underlying_price"].resample(RET_FREQ).last().dropna()
    if len(px) < 3:
        return np.nan
    logret = np.log(px).diff().dropna()
    rv = np.sum(logret**2)
    return np.sqrt(rv)


# ============================================================
# 主循环：构造回归数据集
# ============================================================
rows = []

# 找所有日期的文件夹
pattern = os.path.join(OUTPUT_BASE_DIR, "BTC_event_windows_*")
for day_dir in sorted(glob.glob(pattern)):
    date_str = os.path.basename(day_dir).split("_")[-1]
    if date_str < DATE_START or date_str > DATE_END:
        continue

    print(f"\nProcessing date {date_str} ...")

    # step2 价格数据
    step2_file = os.path.join(
        day_dir, "step2_data_reconstruction",
        f"BTC_event_windows_{date_str}_step2_reconstructed_BTC.csv"
    )
    # 对应的事件时间文件（step3）
    event_time_file = os.path.join(
        day_dir, "step3_event_time_information",
        f"event_time_{date_str}.csv"
    )

    if not (os.path.exists(step2_file) and os.path.exists(event_time_file)):
        print(f"  Missing step2 or step3 for {date_str}, skip.")
        continue

    df_prices = pd.read_csv(step2_file)
    df_prices["local_timestamp"] = pd.to_datetime(df_prices["local_timestamp"])

    df_events = pd.read_csv(event_time_file)

    # 计算当天的背景日波动率（简化 GARCH proxy）
    daily_vol = compute_daily_realized_vol(df_prices, date_str)

    # 对每个事件
    for idx, erow in df_events.iterrows():
        event_time_str = erow["event_time"]
        event_ts = pd.to_datetime(event_time_str)

        # 对应的 step4 事件目录
        event_dir_pattern = os.path.join(
            day_dir, "step4_Bates_Model_Calibration",
            f"event_{idx+1}_*"
        )
        matches = glob.glob(event_dir_pattern)
        if len(matches) == 0:
            print(f"  No step4 event folder for event {idx+1} on {date_str}")
            continue
        event_dir = matches[0]

        # 1) 实现波动率 RV_i
        RV_i = compute_event_realized_vol(df_prices, event_ts)

        # 2) 期权隐含事件风险 sigmaQ_i (from variance_decomposition.csv, pre-row)
        var_file = os.path.join(event_dir, "variance_decomposition.csv")
        if not os.path.exists(var_file):
            print(f"  Missing variance_decomposition for {event_dir}")
            continue
        var_df = pd.read_csv(var_file)
        # 我们定义 pre 行
        pre_row = var_df[var_df["window"] == "pre"].iloc[0]
        tau_pre = pre_row["tau"]
        V_event_pre = pre_row["V_event"]
        if tau_pre <= 0 or V_event_pre <= 0:
            sigmaQ_i = np.nan
        else:
            sigmaQ_i = np.sqrt(V_event_pre / tau_pre)

        # 3) SV 模型 spot 波动率 sqrt(v0_pre)
        pre_param_file = os.path.join(event_dir, "pre_event_calibration_params.csv")
        if not os.path.exists(pre_param_file):
            print(f"  Missing pre_event_calibration_params for {event_dir}")
            continue
        pre_params = pd.read_csv(pre_param_file).iloc[0]
        v0_pre = pre_params["v0"]
        spot_vol_sv = np.sqrt(v0_pre) if v0_pre > 0 else np.nan

        rows.append({
            "date": date_str,
            "event_index": idx+1,
            "event_time": event_time_str,
            "RV": RV_i,
            "sigmaQ_event": sigmaQ_i,
            "daily_vol": daily_vol,
            "spot_vol_sv": spot_vol_sv
        })

# 构造 DataFrame
reg_df = pd.DataFrame(rows)
print("\nRegression dataset:")
print(reg_df)

# 丢掉有缺失的行
reg_df = reg_df.dropna(subset=["RV", "sigmaQ_event"])

# ============================================================
# 回归函数：带 HAC 标准误的 OLS
# ============================================================
def hac_reg(y, X, maxlags=1):
    Xc = sm.add_constant(X)  # 加截距
    model = sm.OLS(y, Xc).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return model

results = {}

# 回归 (1): 控制 daily_vol
if reg_df["daily_vol"].notna().sum() >= 3:
    model1 = hac_reg(reg_df["RV"], reg_df[["sigmaQ_event", "daily_vol"]], HAC_LAGS)
    results["GARCH_proxy"] = model1
    print("\n=== Regression (1): RV ~ sigmaQ_event + daily_vol ===")
    print(model1.summary())

# 回归 (2): 控制 spot_vol_sv
if reg_df["spot_vol_sv"].notna().sum() >= 3:
    model2 = hac_reg(reg_df["RV"], reg_df[["sigmaQ_event", "spot_vol_sv"]], HAC_LAGS)
    results["SV_state"] = model2
    print("\n=== Regression (2): RV ~ sigmaQ_event + spot_vol_sv ===")
    print(model2.summary())

# ============================================================
# 保存回归数据与简单结果
# ============================================================
out_dir = os.path.join(OUTPUT_BASE_DIR, "step54_mincer_zarnowitz")
os.makedirs(out_dir, exist_ok=True)

reg_df.to_csv(os.path.join(out_dir, "mz_regression_data.csv"), index=False)

# 也可以把系数/ t 值整理成表
rows_out = []
for name, m in results.items():
    for param, coef, tval in zip(m.params.index, m.params.values, m.tvalues.values):
        rows_out.append({
            "spec": name,
            "param": param,
            "coef": coef,
            "t_HAC": tval
        })
summary_table = pd.DataFrame(rows_out)
summary_table.to_csv(os.path.join(out_dir, "mz_regression_results.csv"), index=False)

print(f"\nSaved regression data and results to: {out_dir}")

"""
总结波动率估计，基于step4风险模型，包括：
- \textbf{样本 (Sample)}：样本期间。
- \textbf{均值和 t 统计量 (Mean and t-stat)}：平均 FOMC 事件波动率及其 HAC 校正 t 统计量。
- \textbf{中位数 (Median)}：估计的中位数。
- \textbf{5\\% 和 95\\% 分位数 (Perc(5) and Perc(95))}：估计的第 5 和第 95 分位数。
- \textbf{\\% $\\geq$ 1bps}：估计值超过 1 个基点的百分比。
- \textbf{观测数 (Obs)}：观测数量。
"""
import pandas as pd
import os
import numpy as np
import statsmodels.api as sm
import glob

# ===========================================================================
# CONFIG
# ===========================================================================

OUTPUT_BASE_DIR = r"E:\Output"
DATE_START = "20210516"
DATE_END   = "20210521"

# ===========================================================================
# Collect IV jumps from all events
# ===========================================================================

iv_jumps = []
event_dates = []

pattern = os.path.join(OUTPUT_BASE_DIR, "BTC_event_windows_*", "step4_Bates_Model_Calibration", "event_*")
event_dirs = glob.glob(pattern)

for ed in event_dirs:
    try:
        # Load pre/post events
        pre_file = os.path.join(ed, "pre_event_filtered_options.csv")
        post_file = os.path.join(ed, "post_event_filtered_options.csv")

        if not (os.path.exists(pre_file) and os.path.exists(post_file)):
            continue

        pre_df = pd.read_csv(pre_file)
        post_df = pd.read_csv(post_file)

        if len(pre_df) == 0 or len(post_df) == 0:
            continue

        # Compute mean IV (decimal)
        iv_pre  = pre_df["mark_iv"].mean() / 100.0
        iv_post = post_df["mark_iv"].mean() / 100.0

        iv_jump = iv_post - iv_pre   # this is our volatility estimator

        iv_jumps.append(iv_jump)

        # extract event date
        d = ed.split("BTC_event_windows_")[1][:8]
        event_dates.append(d)

    except Exception as e:
        print(f"Failed reading event at {ed}: {e}")

iv_jumps = np.array(iv_jumps)
N = len(iv_jumps)

print(f"Collected {N} volatility estimates. Range dates = {DATE_START}–{DATE_END}")

# ===========================================================================
# HAC Newey-West t-stat
# ===========================================================================

y = iv_jumps
X = np.ones(len(y))
ols = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={'maxlags':1})

mean_iv = y.mean()
tstat = ols.tvalues[0]

# ===========================================================================
# Summary statistics
# ===========================================================================
median_iv = np.median(y)
p5 = np.percentile(y, 5)
p95 = np.percentile(y, 95)
pct_1bps = np.mean(y >= 0.0001) * 100   # 1 bps
obs = len(y)

# ===========================================================================
# Output summary table
# ===========================================================================
summary = pd.DataFrame([{
    "Sample": f"{DATE_START}–{DATE_END}",
    "Mean": mean_iv,
    "t-stat (HAC)": tstat,
    "Median": median_iv,
    "Perc(5)": p5,
    "Perc(95)": p95,
    "% >= 1bps": pct_1bps,
    "Obs": obs
}])

save_path = os.path.join(OUTPUT_BASE_DIR, "step51_volatility_estimates_for_the_event_risk_model", "volatility_summary_table.csv")
summary.to_csv(save_path, index=False)
print(f"Saved summary table to: {save_path}")

print(summary)

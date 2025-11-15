"""
Figure provides scatter plots of event risk and measures of realized volatility/variance. 
图提供了事件风险与实现波动率/方差测量的散点图。
事件风险波动率是从事件前 5（15）分钟到事件后 30（90）分钟的 5 分钟对数（百分比）underlying price回报实现方差的平方根。
每个标记表示一个事件波动率/实现波动率观测值
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from statsmodels.nonparametric.smoothers_lowess import lowess
import statsmodels.api as sm

# =====================================================================
# CONFIG
# =====================================================================
OUTPUT_BASE_DIR = r"E:\Output"
STEP52_OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, "step52_time_series_of_event_risk")

# Realized window settings (minutes)
PRE_WINDOW  = 5      # before event
POST_WINDOW = 30     # after event
RET_FREQ = '5min'    # 5-minute returns

# =====================================================================
# Helper to compute realized volatility for one event
# =====================================================================
def compute_realized_vol(df, event_time):
    event_dt = pd.to_datetime(event_time)

    # event window
    start = event_dt - pd.Timedelta(minutes=PRE_WINDOW)
    end   = event_dt + pd.Timedelta(minutes=POST_WINDOW)

    window_df = df[(df['local_timestamp'] >= start) & 
                   (df['local_timestamp'] <= end)].copy()

    if len(window_df) < 3:
        return np.nan

    # resample to 5-minute log returns
    window_df = window_df.set_index('local_timestamp')
    px = window_df['underlying_price'].resample(RET_FREQ).last().dropna()

    logret = np.log(px).diff().dropna()

    # realized variance = sum(r^2)
    realized_var = np.sum(logret**2)

    # realized volatility = sqrt(realized_var)
    realized_vol = np.sqrt(realized_var)

    return realized_vol

# =====================================================================
# Collect all event-level (iv_jump, realized_vol)
# =====================================================================
data = []

step4_summary_file = os.path.join(STEP52_OUTPUT_DIR, "step4_results_summary.csv")
if not os.path.exists(step4_summary_file):
    print(f"Error: step4_results_summary.csv not found at: {step4_summary_file}")
    exit(1)
step4_df = pd.read_csv(step4_summary_file)

# add datetime
step4_df['datetime'] = pd.to_datetime(step4_df['date'].astype(str) + " " + step4_df['event_time'])

# loop through dates
for date_str in step4_df['date'].astype(str).unique():

    # step2 file
    step2_file = os.path.join(
        OUTPUT_BASE_DIR, 
        f"BTC_event_windows_{date_str}",
        "step2_data_reconstruction",
        f"BTC_event_windows_{date_str}_step2_reconstructed_BTC.csv"
    )

    # check file
    if not os.path.exists(step2_file):
        print(f"Missing step2 file: {step2_file}")
        continue

    df2 = pd.read_csv(step2_file)
    df2['local_timestamp'] = pd.to_datetime(df2['local_timestamp'])

    # process each event on that date
    day_events = step4_df[step4_df['date'].astype(str) == date_str]

    for _, row in day_events.iterrows():
        event_time = row['datetime']
        iv_jump = row['iv_jump']

        realized_vol = compute_realized_vol(df2, event_time)

        if not np.isnan(realized_vol):
            data.append({
                'date': date_str,
                'event_time': row['event_time'],
                'iv_jump': iv_jump,
                'realized_vol': realized_vol,
                'model_type_pre': row.get('model_type_pre', 'Unknown'),
                'model_type_post': row.get('model_type_post', 'Unknown')
            })

# convert to dataframe
res_df = pd.DataFrame(data)

if len(res_df) == 0:
    print("Error: No data points found. Cannot create plot.")
    exit(1)

print(f"\nTotal events processed: {len(res_df)}")
print(res_df.head(10))

# =====================================================================
# Save raw data
# =====================================================================
out_dir = os.path.join(OUTPUT_BASE_DIR, "step54_event_risk_vs_realized_vol")
os.makedirs(out_dir, exist_ok=True)
raw_data_path = os.path.join(out_dir, "event_risk_vs_realized_vol_raw.csv")
res_df.to_csv(raw_data_path, index=False)
print(f"Saved raw data to: {raw_data_path}")

# =====================================================================
# Enhanced plot: IV jump vs realized volatility
# =====================================================================
df = res_df.copy()

# Use post-event model type for coloring
if "model_type_post" in df.columns:
    df["model"] = df["model_type_post"]
else:
    df["model"] = "Unknown"

# Winsorize realized volatility to improve axis scale
df["realized_vol"] = df["realized_vol"].clip(
    lower=df["realized_vol"].quantile(0.05),
    upper=df["realized_vol"].quantile(0.95)
)

# Prepare figure
plt.figure(figsize=(10, 7))

# Color by model type
palette = {"Bates": "red", "Heston": "blue", "Unknown": "gray"}

# Scatter plot
sns.scatterplot(data=df,
                x="iv_jump",
                y="realized_vol",
                hue="model",
                palette=palette,
                s=120, alpha=0.7, edgecolor="black")

# Add LOWESS smoothing curve
lowess_curve = lowess(df["realized_vol"], df["iv_jump"], frac=0.6)
plt.plot(lowess_curve[:, 0], lowess_curve[:, 1],
         color="black", linewidth=2.5, label="LOWESS smoother")

# Add linear regression line + confidence interval
X = sm.add_constant(df["iv_jump"])
model = sm.OLS(df["realized_vol"], X).fit()
xvals = np.linspace(df["iv_jump"].min(), df["iv_jump"].max(), 100)
pred = model.get_prediction(sm.add_constant(xvals))
pred_mean = pred.predicted_mean
pred_ci = pred.conf_int(alpha=0.05)

plt.plot(xvals, pred_mean, color="green", linewidth=2, label="OLS regression")
plt.fill_between(xvals, pred_ci[:, 0], pred_ci[:, 1],
                 color="green", alpha=0.2)

# Labels and titles
plt.title("Event Risk vs Realized Volatility", fontsize=16)
plt.xlabel("Event Risk (IV Jump)", fontsize=13)
plt.ylabel("Realized Volatility (5-minute RV)", fontsize=13)

plt.grid(alpha=0.25)
plt.legend(title="Model Type")

plt.tight_layout()

# Save
plot_path = os.path.join(out_dir, "event_risk_vs_realized_vol.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')

print("Saved plot to:", plot_path)
plt.show()


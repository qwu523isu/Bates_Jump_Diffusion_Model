"""
图展示了事件风险的时间序列。
非事件期间的事件风险以黑色标记，事件窗口以灰色标记。
图中的红色曲线表示滚动窗口估计，其基于最近若干事件风险的简单平均。
阴影区域是基于 ±1.96 倍 HAC 校正标准误差 的置信区间（95%），用于衡量滚动估计的不确定性。
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import os
import re

OUTPUT_BASE_DIR = r"E:\Output"
STEP52_OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, "step52_time_series_of_event_risk")

# Function: extract_date_from_path
def extract_date_from_path(path):
    match = re.search(r'BTC_event_windows_(\d{8})', path)
    return match.group(1) if match else None

# Function: extract_event_time_from_dir
def extract_event_time_from_dir(dir_name):
    match = re.search(r'event_\d+_(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})-(\d{2})', dir_name)
    if match:
        return f"{match.group(4)}:{match.group(5)}"
    return None

# Function: load_event_times_map
def load_event_times_map(date_str):
    event_file = os.path.join(
        OUTPUT_BASE_DIR,
        f"BTC_event_windows_{date_str}",
        "step3_event_time_information",
        f"event_time_{date_str}.csv"
    )
    if not os.path.exists(event_file):
        return {}
    event_df = pd.read_csv(event_file)
    event_map = {}
    for idx, row in event_df.iterrows():
        dt = pd.to_datetime(row['event_time'])
        event_map[idx + 1] = dt.strftime('%H:%M')
    return event_map

# Function: process_event_directory
def process_event_directory(event_dir, date_str, event_index, event_times_map):
    variance_file = os.path.join(event_dir, "variance_decomposition.csv")
    pre_params_file = os.path.join(event_dir, "pre_event_calibration_params.csv")
    post_params_file = os.path.join(event_dir, "post_event_calibration_params.csv")
    if not os.path.exists(variance_file):
        return None
    variance_df = pd.read_csv(variance_file)
    pre_row = variance_df[variance_df['window'] == 'pre']
    post_row = variance_df[variance_df['window'] == 'post']
    if len(pre_row) == 0 or len(post_row) == 0:
        return None
    iv_pre = float(pre_row['iv'].iloc[0])
    iv_post = float(post_row['iv'].iloc[0])
    iv_jump = iv_post - iv_pre
    model_type_pre = 'fail'
    model_type_post = 'fail'
    if os.path.exists(pre_params_file):
        pre_params_df = pd.read_csv(pre_params_file)
        if 'model_type' in pre_params_df.columns:
            model_type_pre = str(pre_params_df['model_type'].iloc[0])
    if os.path.exists(post_params_file):
        post_params_df = pd.read_csv(post_params_file)
        if 'model_type' in post_params_df.columns:
            model_type_post = str(post_params_df['model_type'].iloc[0])
    event_time = event_times_map.get(event_index)
    if not event_time:
        event_time = extract_event_time_from_dir(os.path.basename(event_dir))
    if not event_time:
        return None
    return {
        'date': date_str,
        'event_time': event_time,
        'iv_pre': iv_pre,
        'iv_post': iv_post,
        'iv_jump': iv_jump,
        'model_type_pre': model_type_pre,
        'model_type_post': model_type_post
    }

# Function: generate_step4_summary
def generate_step4_summary():
    print("=" * 70)
    print("Step4 Results Summary Generator")
    print("=" * 70)
    summary_rows = []
    if not os.path.exists(OUTPUT_BASE_DIR):
        print(f"Error: Output directory not found: {OUTPUT_BASE_DIR}")
        return None
    for item in os.listdir(OUTPUT_BASE_DIR):
        item_path = os.path.join(OUTPUT_BASE_DIR, item)
        if not os.path.isdir(item_path):
            continue
        date_str = extract_date_from_path(item)
        if not date_str:
            continue
        step4_dir = os.path.join(item_path, "step4_Bates_Model_Calibration")
        if not os.path.exists(step4_dir):
            continue
        print(f"\nProcessing date: {date_str}")
        event_times_map = load_event_times_map(date_str)
        for event_dir_name in os.listdir(step4_dir):
            event_dir = os.path.join(step4_dir, event_dir_name)
            if not os.path.isdir(event_dir):
                continue
            if not event_dir_name.startswith('event_'):
                continue
            match = re.search(r'event_(\d+)_', event_dir_name)
            if not match:
                continue
            event_index = int(match.group(1))
            result = process_event_directory(event_dir, date_str, event_index, event_times_map)
            if result:
                summary_rows.append(result)
                print(f"  Processed: {event_dir_name}")
    if not summary_rows:
        print("\nNo events found to process.")
        return None
    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values(['date', 'event_time'])
    os.makedirs(STEP52_OUTPUT_DIR, exist_ok=True)
    output_file = os.path.join(STEP52_OUTPUT_DIR, "step4_results_summary.csv")
    summary_df.to_csv(output_file, index=False)
    print(f"\n{'=' * 70}")
    print(f"Summary saved to: {output_file}")
    print(f"Total events: {len(summary_df)}")
    print(f"{'=' * 70}\n")
    print("First few rows:")
    print(summary_df.head(10).to_string(index=False))
    return summary_df

# Function: hac_se
def hac_se(y):
    if len(y) < 2:
        return 0
    X = np.ones(len(y))
    model = sm.OLS(y, X)
    res = model.fit(cov_type='HAC', cov_kwds={'maxlags':1})
    return res.bse[0]

# Function: analyze_time_series
def analyze_time_series(df, window=3):
    print("\n" + "=" * 70)
    print("Time Series Event Risk Analysis")
    print("=" * 70)
    df['date'] = df['date'].astype(str)
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['event_time'])
    df = df.sort_values('datetime').reset_index(drop=True)
    risk = df['iv_jump'].values
    dates = df['datetime']
    rolling_mean = []
    rolling_se = []
    for t in range(len(risk)):
        left = max(0, t - window + 1)
        window_data = risk[left:t+1]
        mu = window_data.mean()
        se = hac_se(window_data)
        rolling_mean.append(mu)
        rolling_se.append(se)
    rolling_mean = np.array(rolling_mean)
    rolling_se = np.array(rolling_se)
    upper = rolling_mean + 1.96 * rolling_se
    lower = rolling_mean - 1.96 * rolling_se
    plt.figure(figsize=(14, 6))
    plt.scatter(dates, risk, c='gray', s=50, label='Event risk')
    plt.plot(dates, rolling_mean, color='red', linewidth=2, label='Rolling estimator')
    plt.fill_between(dates, lower, upper, color='red', alpha=0.2, label='95% HAC CI')
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Event Risk (IV Jump)", fontsize=12)
    plt.title("Time Series of Event Risk with Rolling HAC Estimates", fontsize=14)
    plt.grid(alpha=0.3)
    plt.legend()
    os.makedirs(STEP52_OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(STEP52_OUTPUT_DIR, "event_risk_timeseries.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved figure to: {save_path}")
    print("=" * 70)
    plt.show()

# Function: main
def main():
    df = generate_step4_summary()
    if df is not None and len(df) > 0:
        analyze_time_series(df, window=3)

if __name__ == "__main__":
    main()

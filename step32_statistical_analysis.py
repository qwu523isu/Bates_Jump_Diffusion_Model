"""
Step 3: Statistical Analysis of BTC_Event_Timeline
Step 3: BTC事件时间线统计分析
Analyze event distribution, importance, sentiment, and temporal patterns.
分析事件分布、重要性、情感和时间模式。
"""

import pandas as pd
import os

# Function: load_data
# Load and preprocess event timeline data / 加载并预处理事件时间线数据
def load_data(path):
    df = pd.read_csv(path)
    df['newsDatetime'] = pd.to_datetime(df['newsDatetime'])
    df['date'] = df['newsDatetime'].dt.date
    df['hour'] = df['newsDatetime'].dt.hour
    df['dayofweek'] = df['newsDatetime'].dt.dayofweek
    df['net_sentiment'] = df['positive'] - df['negative']
    return df

# Function: compute_summary
# Compute basic summary statistics / 计算基本汇总统计
def compute_summary(df):
    duration = (df['newsDatetime'].max() - df['newsDatetime'].min()).days
    return pd.DataFrame([{
        'total_events': len(df),
        'date_start': df['newsDatetime'].min(),
        'date_end': df['newsDatetime'].max(),
        'duration_days': duration,
        'avg_events_per_day': len(df) / (duration + 1)
    }])

# Function: compute_importance
# Compute importance level distribution and statistics / 计算重要性级别分布和统计
def compute_importance(df):
    imp_dist = df['important'].value_counts().sort_index().reset_index()
    imp_dist.columns = ['importance_level', 'count']
    imp_dist['percentage'] = imp_dist['count'] / len(df) * 100
    stats = pd.DataFrame([{
        'high_importance_count': len(df[df['important'] >= 3]),
        'critical_count': len(df[df['important'] >= 10]),
        'mean': df['important'].mean(),
        'median': df['important'].median(),
        'std': df['important'].std()
    }])
    return imp_dist, stats

# Function: compute_sentiment
# Compute sentiment statistics / 计算情感统计
def compute_sentiment(df):
    return pd.DataFrame([{
        'mean_positive': df['positive'].mean(),
        'mean_negative': df['negative'].mean(),
        'mean_net_sentiment': df['net_sentiment'].mean(),
        'positive_events': len(df[df['net_sentiment'] > 0]),
        'negative_events': len(df[df['net_sentiment'] < 0]),
        'neutral_events': len(df[df['net_sentiment'] == 0])
    }])

# Function: compute_temporal
# Compute temporal distribution by hour and day of week / 按小时和星期计算时间分布
def compute_temporal(df):
    hour_dist = df['hour'].value_counts().sort_index().reset_index()
    hour_dist.columns = ['hour', 'count']
    dow_dist = df['dayofweek'].value_counts().sort_index().reset_index()
    dow_dist.columns = ['day_of_week', 'count']
    dow_dist['day_name'] = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    return hour_dist, dow_dist

# Function: compute_high_importance
# Compute top dates with high importance events / 计算高重要性事件最多的日期
def compute_high_importance(df):
    high_imp = df[df['important'] >= 3]
    top_dates = high_imp['date'].value_counts().head(20).reset_index()
    top_dates.columns = ['date', 'event_count']
    return top_dates

# Configuration / 配置
TIMELINE_PATH = r"E:\Data\BTC\analysis_by_importance\BTC_Event_Timeline.csv"
OUTPUT_DIR = r"E:\Bates_Jump_Diffusion_Model\statistics"

# Main execution / 主执行部分
os.makedirs(OUTPUT_DIR, exist_ok=True)
df = load_data(TIMELINE_PATH)

# Generate and save statistics / 生成并保存统计结果
compute_summary(df).to_csv(f"{OUTPUT_DIR}/summary.csv", index=False)
imp_dist, imp_stats = compute_importance(df)
imp_dist.to_csv(f"{OUTPUT_DIR}/importance_distribution.csv", index=False)
imp_stats.to_csv(f"{OUTPUT_DIR}/importance_stats.csv", index=False)
compute_sentiment(df).to_csv(f"{OUTPUT_DIR}/sentiment_stats.csv", index=False)
hour_dist, dow_dist = compute_temporal(df)
hour_dist.to_csv(f"{OUTPUT_DIR}/hourly_distribution.csv", index=False)
dow_dist.to_csv(f"{OUTPUT_DIR}/day_of_week_distribution.csv", index=False)
compute_high_importance(df).to_csv(f"{OUTPUT_DIR}/top_dates_high_importance.csv", index=False)


"""
Step 3: Event Time Information Script
Load BTC_Event_Timeline and fetch event time by date.
加载BTC_Event_Timeline并按日期获取事件时间。
"""

import pandas as pd
import os

# Main execution / 主执行部分
TIMELINE_PATH = r"E:\Data\BTC\analysis_by_importance\BTC_Event_Timeline.csv"
OUTPUT_BASE_DIR = r"E:\Output"
DATE_START = "20210516"  # Start date in YYYYMMDD format
DATE_END = "20210521"  # End date in YYYYMMDD format

# Load event timeline / 加载事件时间线
event_timeline_df = pd.read_csv(TIMELINE_PATH)
event_timeline_df['newsDatetime'] = pd.to_datetime(event_timeline_df['newsDatetime'])
event_timeline_df['date'] = event_timeline_df['newsDatetime'].dt.date

# Generate date range / 生成日期范围
date_start_obj = pd.to_datetime(DATE_START, format='%Y%m%d').date()
date_end_obj = pd.to_datetime(DATE_END, format='%Y%m%d').date()
dates = pd.date_range(date_start_obj, date_end_obj, freq='D').date.tolist()

# Process event times / 处理事件时间
total_dates = len(dates)
print(f"\nProcessing {total_dates} date(s)\n")

successful = 0
failed = 0
for idx, date_obj in enumerate(dates, 1):
    date_str = date_obj.strftime('%Y%m%d')
    print(f"[{idx}/{total_dates}] {date_str}")
    try:
        events = event_timeline_df[(event_timeline_df['date'] == date_obj) & (event_timeline_df['important'] >= 3)]
        
        if len(events) == 0:
            print(f"  No event found (importance >= 3)")
            failed += 1
            continue
        
        events_sorted = events.sort_values('newsDatetime')
        event_records = []
        for _, row in events_sorted.iterrows():
            event_records.append({
                'date': date_str,
                'event_time': row['newsDatetime'].strftime('%Y-%m-%d %H:%M:%S'),
                'important': int(row['important']),
                'positive': int(row['positive']),
                'negative': int(row['negative'])
            })
        
        date_folder = f"BTC_event_windows_{date_str}"
        output_dir = os.path.join(OUTPUT_BASE_DIR, date_folder, "step3_event_time_information")
        os.makedirs(output_dir, exist_ok=True)
        
        pd.DataFrame(event_records).to_csv(os.path.join(output_dir, f"event_time_{date_str}.csv"), index=False)
        print(f"  {len(event_records)} event(s) found")
        successful += 1
    except Exception as e:
        print(f"  Error: {e}")
        failed += 1

print(f"\nSummary: {successful} successful, {failed} failed\n")


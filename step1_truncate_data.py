"""
Step 1: Load Data Script
This script reads a large CSV file in chunks and processes all data.
该脚本以分块方式读取大型CSV文件，并处理所有数据。

Purpose / 目的:
- Process large CSV files that may not fit in memory / 处理可能无法完全加载到内存的大型CSV文件
- Load all data using chunked reading / 使用分块读取加载所有数据
- Save the complete data to a new CSV file / 将完整数据保存到新的CSV文件
"""

import pandas as pd
import os
import re

# Configuration / 配置参数
CHUNK_SIZE = 50000  
# Number of rows to read per chunk / 每次读取的行数 

# Function: extract_date_folder
def extract_date_folder(file_path):
    """Extract date from filename and convert to folder format (YYYY_MM_DD) / 从文件名提取日期并转换为文件夹格式"""
    filename = os.path.basename(file_path)
    # Try YYYY-MM-DD format first / 先尝试YYYY-MM-DD格式
    date_match = re.search(r'(\d{4}-\d{2}-\d{2})', filename)
    if date_match:
        date_str = date_match.group(1)
        return date_str.replace('-', '_')
    # Try YYYYMMDD format / 尝试YYYYMMDD格式
    date_match = re.search(r'(\d{4})(\d{2})(\d{2})', filename)
    if date_match:
        year, month, day = date_match.groups()
        return f"{year}_{month}_{day}"
    raise ValueError(f"Could not extract date from filename: {filename}")

# Function: extract_base_name
def extract_base_name(file_path):
    """Extract base filename without extension / 提取不带扩展名的文件名"""
    filename = os.path.basename(file_path)
    return os.path.splitext(filename)[0]

# Function: find_csv_files
def find_csv_files(base_dir, year_subfolder=None, file_pattern=None, date_start=None, date_end=None):
    """Find CSV files in directory structure / 在目录结构中查找CSV文件"""
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Base directory not found: {base_dir}")
    
    csv_files = []
    
    if year_subfolder:
        search_dir = os.path.join(base_dir, year_subfolder)
        if not os.path.exists(search_dir):
            raise FileNotFoundError(f"Directory not found: {search_dir}")
        files = [f for f in os.listdir(search_dir) if f.endswith('.csv')]
        if file_pattern:
            files = [f for f in files if file_pattern in f]
        if date_start or date_end:
            files = filter_by_date_range(files, date_start, date_end)
        csv_files.extend([os.path.join(search_dir, f) for f in files])
    else:
        subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        for subdir in subdirs:
            search_dir = os.path.join(base_dir, subdir)
            files = [f for f in os.listdir(search_dir) if f.endswith('.csv')]
            if file_pattern:
                files = [f for f in files if file_pattern in f]
            if date_start or date_end:
                files = filter_by_date_range(files, date_start, date_end)
            csv_files.extend([os.path.join(search_dir, f) for f in files])
    
    return sorted(csv_files)

# Function: filter_by_date_range
def filter_by_date_range(files, date_start=None, date_end=None):
    """Filter files by date range (YYYYMMDD format) / 按日期范围过滤文件"""
    filtered = []
    for f in files:
        date_match = re.search(r'(\d{4})(\d{2})(\d{2})', f)
        if date_match:
            file_date = int(date_match.group(1) + date_match.group(2) + date_match.group(3))
            if date_start and file_date < int(date_start):
                continue
            if date_end and file_date > int(date_end):
                continue
            filtered.append(f)
    return filtered

# Function: build_file_path
def build_file_path(base_dir, year_subfolder=None, filename=None, date_str=None):
    """Build file path from components / 从组件构建文件路径"""
    if filename:
        if year_subfolder:
            return os.path.join(base_dir, year_subfolder, filename)
        return os.path.join(base_dir, filename)
    
    if date_str:
        year = date_str[:4]
        year_subfolder = f"BTC_event_windows_{year}"
        filename = f"BTC_event_windows_{date_str}.csv"
        return os.path.join(base_dir, year_subfolder, filename)
    
    raise ValueError("Either filename or date_str must be provided")

def read_csv_in_chunks(file_path=None, 
                       chunk_size=CHUNK_SIZE,
                       output_dir=None):
    """
    Read a CSV file in chunks and save all data.
    以分块方式读取CSV文件，并保存所有数据。
    
    Parameters / 参数:
    ----------
    file_path : str, optional
        Path to the input CSV file / 输入CSV文件的路径
    chunk_size : int, default=CHUNK_SIZE
        Number of rows to read per chunk / 每个分块读取的行数
    output_dir : str, optional
        Directory to save the output file / 保存输出文件的目录
    
    Returns / 返回值:
    -------
    tuple : (chunk_count, total_rows)
        Number of chunks processed and total rows read / 处理的分块数和读取的总行数
    """
    
    # Store the file path to process / 存储要处理的文件路径
    file_to_process = file_path
    
    # Initialize counters / 初始化计数器
    chunk_count = 0
    total_rows = 0
    all_chunks = []
    
    # Read CSV file in chunks to handle large files / 以分块方式读取CSV文件以处理大文件
    if not os.path.exists(file_to_process):
        # Check if directory exists / 检查目录是否存在
        file_dir = os.path.dirname(file_to_process)
        if not os.path.exists(file_dir):
            raise FileNotFoundError(f"Directory not found: {file_dir}\nPlease check the file path: {file_to_process}")
        # Check if file exists with different case or extension / 检查文件是否存在（不同大小写或扩展名）
        if os.path.exists(file_dir):
            files_in_dir = os.listdir(file_dir)
            similar_files = [f for f in files_in_dir if os.path.basename(file_to_process).lower() in f.lower()]
            if similar_files:
                raise FileNotFoundError(f"Input file not found: {file_to_process}\nSimilar files found in directory: {similar_files}")
        raise FileNotFoundError(f"Input file not found: {file_to_process}\nDirectory exists: {file_dir}")
    
    for chunk in pd.read_csv(file_to_process, chunksize=chunk_size):
        chunk_count += 1
        total_rows += len(chunk)
        all_chunks.append(chunk)
        print(f"Processed chunk {chunk_count}, rows: {len(chunk)}, total rows: {total_rows}")

    if len(all_chunks) == 0:
        raise ValueError("No data read from file")
    
    # Ensure output directory exists / 确保输出目录存在
    output_subdir = os.path.join(output_dir, "step1_truncate_data")
    os.makedirs(output_subdir, exist_ok=True)
    
    # Generate output file path / 生成输出文件路径
    out_file = os.path.join(output_subdir, os.path.basename(file_path).replace(".csv", "_processed.csv"))
    
    # Combine all chunks / 合并所有分块
    df_all = pd.concat(all_chunks, ignore_index=True)
    
    # Convert price columns to numeric / 将价格列转换为数值类型
    price_columns = ['mark_price', 'bid_price', 'ask_price', 'last_price', 'underlying_price', 'strike_price', 'open_interest']
    for col in [c for c in price_columns if c in df_all.columns]:
        df_all[col] = pd.to_numeric(df_all[col], errors='coerce')
    
    # Save all data to CSV file / 将所有数据保存到CSV文件
    df_all.to_csv(out_file, index=False)
    print(f"Saved {total_rows} rows to {out_file}")
    
    return chunk_count, total_rows

# Main execution / 主执行部分

# Configuration / 配置参数
BASE_DIR = r"E:\Data\BTC\analysis_by_importance\event_windows"
YEAR_SUBFOLDER = "BTC_event_windows_2021"  # e.g., "BTC_event_windows_2021" or None for all years
FILE_PATTERN = None  # e.g., "20210517" or None for all files
DATE_START = "20210516"  # Start date in YYYYMMDD format or None
DATE_END = "20210521"  # End date in YYYYMMDD format or None

# Build file path(s) / 构建文件路径
file_paths = find_csv_files(BASE_DIR, year_subfolder=YEAR_SUBFOLDER, file_pattern=FILE_PATTERN, 
                             date_start=DATE_START, date_end=DATE_END)
if not file_paths:
    pattern_info = f" with pattern '{FILE_PATTERN}'" if FILE_PATTERN else ""
    folder_info = f" in '{YEAR_SUBFOLDER}'" if YEAR_SUBFOLDER else ""
    date_info = f" from {DATE_START} to {DATE_END}" if DATE_START or DATE_END else ""
    raise FileNotFoundError(f"No CSV files found in {BASE_DIR}{folder_info}{pattern_info}{date_info}")

# Process each file / 处理每个文件
total_files = len(file_paths)
print(f"\nProcessing {total_files} file(s)\n")

successful = 0
failed = 0
for idx, file_path in enumerate(file_paths, 1):
    print(f"[{idx}/{total_files}] {os.path.basename(file_path)}")
    try:
        BASE_NAME = extract_base_name(file_path)
        output_dir = os.path.join(r"E:\Output", BASE_NAME)
        read_csv_in_chunks(file_path, output_dir=output_dir)
        successful += 1
    except Exception as e:
        print(f"  Error: {e}")
        failed += 1

print(f"\nSummary: {successful} successful, {failed} failed\n")
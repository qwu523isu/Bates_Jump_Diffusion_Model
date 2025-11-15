"""
Step 2: Data Reconstruction Script
We process the raw data as follows. 
First, we discard options where the bid price is not smaller than the ask price. 
We then calculate the mid-price using the average of the bid and ask quotes and discard options that have a negative time to maturity recorded. 
Second, for each timestamp and option maturity on each trading day in the sample, 
we calculate the at-the-money strike price by choosing the strike for which the put and call option mid prices are closest to each other. 

我们按照以下方式处理原始数据。
首先，我们丢弃买价不小于卖价的期权。然后，使用买卖报价的平均值计算中间价，并丢弃记录的到期时间为负的期权。
其次，对于样本中每个交易日的每个时间戳和期权到期日，我们通过选择看跌期权和看涨期权中间价最接近的行权价来计算平值行权价。
"""

import pandas as pd
import numpy as np
import os
import re

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

# Function: format_price_columns
def format_price_columns(df):
    """Format price columns to numeric with 2 decimal places before saving."""
    df = df.copy()
    price_cols = ['mark_price', 'bid_price', 'ask_price', 'last_price', 'underlying_price']
    for col in price_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').round(2)
    if 'mid_price' in df.columns:
        df['mid_price'] = pd.to_numeric(df['mid_price'], errors='coerce').round(4)
    return df

# Function: step0_preprocess_data
def step0_preprocess_data(df):
    """
    Step 0: Preprocess data - remove Greeks, convert timestamps, and separate by symbol (BTC/ETH).
    步骤0: 预处理数据 - 移除Greeks信息，转换时间戳，并按符号（BTC/ETH）分开。
    
    Steps / 步骤:
    1. Remove Greeks columns (delta, gamma, vega, theta, rho) / 移除Greeks列（delta, gamma, vega, theta, rho）
    2. Remove unnecessary columns (exchange, underlying_index, bid_amount, ask_amount, timestamp, bid_iv, ask_iv) / 移除不必要的列（exchange, underlying_index, bid_amount, ask_amount, timestamp, bid_iv, ask_iv）
    3. Extract symbol (BTC/ETH) from symbol column / 从symbol列提取符号（BTC/ETH）
    4. Convert timestamps from microseconds to normal datetime format (YYYY-MM-DD HH:MM, rounded to minutes) / 将时间戳从微秒转换为正常日期时间格式（YYYY-MM-DD HH:MM，四舍五入到分钟）
    5. Calculate time to maturity in days and filter negative values / 计算到期时间（天）并过滤负值
    6. Normalize price columns (mark_price, bid_price, ask_price, last_price) by multiplying with underlying_price / 通过乘以underlying_price归一化价格列（mark_price、bid_price、ask_price、last_price）
    7. Note: strike_price and open_interest are NOT normalized (kept as absolute values) / 注意：strike_price和open_interest不归一化（保持绝对值）
    8. Aggregate data: for each minute, same time_to_maturity, type, and strike, take average / 聚合数据：对于每一分钟，相同time_to_maturity、type和strike，取平均
    9. Separate data by asset symbol / 按资产符号分开数据
    
    Parameters / 参数:
    ----------
    df : pd.DataFrame
        Input dataframe with raw options data / 包含原始期权数据的输入数据框
    
    Returns / 返回值:
    -------
    dict
        Dictionary with asset_symbol as keys and preprocessed dataframes as values / 以asset_symbol为键、预处理后数据框为值的字典
    """
    df = df.copy()
    
    # Remove columns / 移除列
    cols_to_drop = (['delta', 'gamma', 'vega', 'theta', 'rho'] + 
                    ['exchange', 'underlying_index', 'bid_amount', 'ask_amount', 'timestamp', 'bid_iv', 'ask_iv'])
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    
    # Extract symbol (BTC/ETH) from symbol column / 从symbol列提取符号（BTC/ETH）
    df['asset_symbol'] = df['symbol'].str.split('-').str[0]
    
    # Convert timestamps to datetime (rounded to minutes) / 将时间戳转换为日期时间（四舍五入到分钟）
    # 
    # ERROR ANALYSIS / 错误分析:
    # The data can contain timestamps in multiple formats / 数据可能包含多种格式的时间戳:
    # 1. Numeric microseconds: 1553990196013709 (Deribit format) / 数值微秒格式（Deribit格式）
    # 2. Numeric strings: '1640937600000000' (read as string from CSV) / 数值字符串（从CSV读取为字符串）
    # 3. ISO strings with microseconds: '2021-05-17 04:01:25.000122+00:00' / 带微秒的ISO字符串
    # 4. ISO strings without microseconds: '2021-05-17 05:46:56+00:00' / 不带微秒的ISO字符串
    #
    # PROBLEM / 问题:
    # - If we use format='ISO8601' on numeric strings like '1640937600000000', we get:
    #    ValueError: Time data 1640937600000000 is not ISO8601 format
    # - If we use unit='us' on ISO strings, we get:
    #    ValueError: non convertible value 2021-05-17 04:01:25.000122+00:00 with the unit 'us'
    #
    # SOLUTION / 解决方案:
    # - First try to convert entire column to numeric using pd.to_numeric() with errors='coerce'
    # - This handles both numeric types AND numeric strings (like '1640937600000000')
    # - If conversion succeeds (no NaN introduced), treat as microseconds and use unit='us'
    # - Otherwise, treat as ISO datetime strings and use format='ISO8601'
    # - The format='ISO8601' parameter handles variable precision (with/without microseconds)
    #
    # 首先尝试将整列转换为数值，使用pd.to_numeric()和errors='coerce'
    # 这可以处理数值类型和数值字符串（如'1640937600000000'）
    # 如果转换成功（没有引入NaN），则视为微秒并使用unit='us'
    # 否则，视为ISO日期时间字符串并使用format='ISO8601'
    # format='ISO8601'参数可以处理不同精度（带/不带微秒）
    for col, dt_col in [('local_timestamp', 'local_timestamp_dt'), ('expiration', 'expiration_dt')]:
        if col in df.columns:
            # Try to convert to numeric first (handles both numeric and numeric strings) / 首先尝试转换为数值（处理数值和数值字符串）
            numeric_col = pd.to_numeric(df[col], errors='coerce')
            # If all values converted successfully (no NaN introduced), use numeric conversion / 如果所有值都成功转换（没有引入NaN），使用数值转换
            if numeric_col.notna().all() or (numeric_col.notna().sum() >= df[col].notna().sum()):
                # Numeric format (microseconds) / 数值格式（微秒）
                df[dt_col] = pd.to_datetime(numeric_col, unit='us', utc=True).dt.floor('min')
            else:
                # String format (ISO datetime with variable precision) / 字符串格式（不同精度的ISO日期时间）
                df[dt_col] = pd.to_datetime(df[col], format='ISO8601', utc=True).dt.floor('min')
    
    # Calculate time to maturity in days and filter negative values / 计算到期时间（天）并过滤负值
    if 'local_timestamp_dt' in df.columns and 'expiration_dt' in df.columns:
        time_to_maturity_seconds = (df['expiration_dt'] - df['local_timestamp_dt']).dt.total_seconds()
        df = df[time_to_maturity_seconds > 0]
        df['time_to_maturity_days'] = ((df['expiration_dt'] - df['local_timestamp_dt']).dt.total_seconds() / 86400).round(0).astype(int)
    
    # Convert datetime to string format / 将日期时间转换为字符串格式
    for col, dt_col in [('local_timestamp', 'local_timestamp_dt'), ('expiration', 'expiration_dt')]:
        if dt_col in df.columns:
            df[col] = df[dt_col].dt.strftime('%Y-%m-%d %H:%M')
            df = df.drop(columns=[dt_col])
    
    # Normalize price columns by multiplying with underlying_price / 通过乘以underlying_price归一化价格列
    # mark_price, bid_price, ask_price, and last_price need to be multiplied by underlying_price / mark_price、bid_price、ask_price和last_price需要乘以underlying_price
    if 'underlying_price' in df.columns:
        df['underlying_price'] = pd.to_numeric(df['underlying_price'], errors='coerce').round(2)
        price_columns = ['mark_price', 'bid_price', 'ask_price', 'last_price']
        for col in [c for c in price_columns if c in df.columns]:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = (df[col] * df['underlying_price']).round(2)
    
    # Note: strike_price and open_interest should NOT be normalized / 注意：strike_price和open_interest不应该归一化
    
    # Aggregate data: for each minute, same time_to_maturity, type, and strike, take average / 聚合数据：对于每一分钟，相同time_to_maturity、type和strike，取平均
    # This creates aggregated data grouped by: local_timestamp (minute), time_to_maturity_days, type, strike_price / 这创建了按以下列分组聚合的数据：local_timestamp（分钟）、time_to_maturity_days、type、strike_price
    # Result can be referred to as: {asset_symbol}_type_strike_time_to_maturity (e.g., BTC_type_strike_time_to_maturity) / 结果可以称为：{asset_symbol}_type_strike_time_to_maturity（例如：BTC_type_strike_time_to_maturity）
    groupby_cols = ['local_timestamp', 'time_to_maturity_days', 'type', 'strike_price']
    if all(col in df.columns for col in groupby_cols):
        numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in groupby_cols]
        non_numeric_cols = [c for c in df.select_dtypes(exclude=[np.number]).columns if c not in groupby_cols]
        agg_dict = {col: 'mean' for col in numeric_cols}
        agg_dict.update({col: 'first' for col in non_numeric_cols})
        df = df.groupby(groupby_cols, as_index=False).agg(agg_dict)
        
        # Round numeric columns to appropriate precision / 将数值列四舍五入到适当精度
        price_cols_2dp = ['strike_price', 'open_interest', 'underlying_price', 'mark_price', 'bid_price', 'ask_price', 'last_price']
        for col in price_cols_2dp:
            if col in df.columns:
                df[col] = df[col].round(2)
        if 'mid_price' in df.columns:
            df['mid_price'] = df['mid_price'].round(4)
        
        # Create naming column: {asset_symbol}_{type}_{strike}_{time_to_maturity} / 创建命名列：{asset_symbol}_{type}_{strike}_{time_to_maturity}
        # Example: BTC_call_0.79_128 / 示例：BTC_call_0.79_128
        required_cols = ['asset_symbol', 'type', 'strike_price', 'time_to_maturity_days']
        if all(col in df.columns for col in required_cols):
            df['contract'] = df[required_cols].astype(str).agg('_'.join, axis=1)
    
    # Remove symbol column before output / 在输出前移除symbol列
    if 'symbol' in df.columns:
        df = df.drop(columns=['symbol'])
    
    # Define column order for output / 定义输出列顺序
    column_order = ['local_timestamp', 'asset_symbol', 'contract', 'type', 'time_to_maturity_days', 
                    'strike_price', 'mark_price', 'mark_iv', 'open_interest', 'ask_price', 
                    'bid_price', 'last_price', 'underlying_price', 'expiration']
    
    # Separate data by asset symbol / 按资产符号分开数据
    results = {}
    for asset_symbol in df['asset_symbol'].unique():
        if pd.isna(asset_symbol):
            continue
        df_symbol = df[df['asset_symbol'] == asset_symbol].copy()
        existing_ordered_cols = [col for col in column_order if col in df_symbol.columns]
        remaining_cols = [col for col in df_symbol.columns if col not in column_order]
        df_symbol = df_symbol[existing_ordered_cols + remaining_cols]
        results[asset_symbol] = df_symbol
    
    return results

# Function: step1_filter_and_calculate_midprice
def step1_filter_and_calculate_midprice(df):
    """
    Step 1: Filter invalid data and calculate mid-price.
    步骤1: 过滤无效数据并计算中间价。
    
    Steps / 步骤:
    1. Filter out options where bid >= ask / 过滤掉买价>=卖价的期权
    2. Calculate mid-price from bid and ask / 从买卖价计算中间价
    
    Parameters / 参数:
    ----------
    df : pd.DataFrame
        Input dataframe with preprocessed options data (already separated by symbol, timestamps as strings) / 包含预处理期权数据的输入数据框（已按符号分开，时间戳为字符串）
    
    Returns / 返回值:
    -------
    pd.DataFrame
        Processed dataframe with filtered data and mid-price / 包含过滤数据和中间价的处理后数据框
    """
    df = df.copy()
    # Filter out options where bid >= ask / 过滤掉买价>=卖价的期权
    df = df[(df['bid_price'] < df['ask_price']) & df['bid_price'].notna() & df['ask_price'].notna()]
    # Calculate mid-price using average of bid and ask quotes / 使用买卖报价的平均值计算中间价
    df['mid_price'] = ((df['bid_price'] + df['ask_price']) / 2.0).round(4)
    return df

# Function: step2_calculate_atm_strikes
def step2_calculate_atm_strikes(df):
    """
    Step 2: Calculate ATM strikes for each timestamp and maturity.
    步骤2: 为每个时间戳和到期日计算平值行权价。
    
    For each timestamp and option maturity, calculate the at-the-money strike price
    by choosing the strike for which the put and call option mid prices are closest to each other.
    对于每个时间戳和期权到期日，通过选择看跌期权和看涨期权中间价最接近的行权价来计算平值行权价。
    
    Methodology / 方法说明:
    --------------------
    The method is based on put-call parity theory. In an efficient market without arbitrage,
    at-the-money options should have call and put prices that are closest to each other.
    This is because ATM options have similar intrinsic values and time values.
    该方法基于看涨-看跌平价理论。在无套利的有效市场中，平值期权的看涨和看跌价格应该最接近。
    这是因为平值期权具有相似的内在价值和时间价值。
    
    Note: strike_price in the dataframe is NOT normalized (kept as absolute value in step0).
    Price columns (mark_price, bid_price, ask_price, last_price) are normalized by multiplying
    with underlying_price. We use market prices to determine the actual ATM strike.
    注意：数据框中的strike_price未归一化（在step0中保持绝对值）。
    价格列（mark_price、bid_price、ask_price、last_price）通过乘以underlying_price归一化。
    我们使用市场价格来确定实际的平值行权价。
    
    Parameters / 参数:
    ----------
    df : pd.DataFrame
        Input dataframe from step1 with mid_price calculated (already separated by symbol) / 来自步骤1的输入数据框，已计算中间价（已按符号分开）
        Required columns: local_timestamp, expiration, type, strike_price, mid_price / 必需列：local_timestamp, expiration, type, strike_price, mid_price
    
    Returns / 返回值:
    -------
    pd.DataFrame
        Processed dataframe with ATM strikes calculated / 包含已计算平值行权价的处理后数据框
        Added columns: atm_strike, is_atm / 新增列：atm_strike, is_atm
    """
    df = df.copy()
    
    # Group by timestamp and expiration (maturity) to find ATM strike / 按时间戳和到期日（到期时间）分组以找到平值行权价
    # ATM strike is calculated independently for each (timestamp, expiration) combination / 平值行权价针对每个（时间戳，到期日）组合独立计算
    atm_results = []
    
    for (timestamp, expiration), group in df.groupby(['local_timestamp', 'expiration']):
        calls = group[group['type'] == 'call'].copy()
        puts = group[group['type'] == 'put'].copy()
        
        if len(calls) == 0 or len(puts) == 0:
            # Skip if we don't have both calls and puts / 如果没有看涨和看跌期权则跳过
            # Cannot determine ATM strike without both option types / 没有两种期权类型无法确定平值行权价
            continue
        
        # Find common strikes between calls and puts / 找到看涨和看跌期权的共同行权价
        # Only strikes that exist for both call and put options can be considered for ATM / 只有同时存在看涨和看跌期权的行权价才能被考虑为平值
        common_strikes = set(calls['strike_price']).intersection(set(puts['strike_price']))
        
        if len(common_strikes) == 0:
            # No common strikes, cannot determine ATM for this timestamp/maturity / 没有共同行权价，无法确定该时间戳/到期日的平值
            continue
        
        # For each common strike, calculate the absolute difference between call and put mid prices / 对于每个共同行权价，计算看涨和看跌期权中间价的绝对差值
        # The strike with the minimum price difference is considered the ATM strike / 价格差值最小的行权价被认为是平值行权价
        min_diff = np.inf
        atm_strike = None
        
        for strike in common_strikes:
            call_mid = calls[calls['strike_price'] == strike]['mid_price'].values
            put_mid = puts[puts['strike_price'] == strike]['mid_price'].values
            
            if len(call_mid) > 0 and len(put_mid) > 0:
                # Calculate absolute difference between call and put mid prices / 计算看涨和看跌期权中间价的绝对差值
                # According to put-call parity, ATM options should have the smallest price difference / 根据看涨-看跌平价，平值期权应该具有最小的价格差值
                diff = abs(call_mid[0] - put_mid[0])
                if diff < min_diff:
                    min_diff = diff
                    atm_strike = strike
        
        if atm_strike is not None:
            # Store ATM strike information for this timestamp and expiration / 存储该时间戳和到期日的平值行权价信息
            # This includes the strike price, call/put mid prices, and their difference / 这包括行权价、看涨/看跌中间价及其差值
            atm_results.append({
                'local_timestamp': timestamp,
                'expiration': expiration,
                'atm_strike': atm_strike,
                'call_mid_price': calls[calls['strike_price'] == atm_strike]['mid_price'].values[0],
                'put_mid_price': puts[puts['strike_price'] == atm_strike]['mid_price'].values[0],
                'mid_price_diff': min_diff
            })
    
    # Create ATM strikes dataframe from collected results / 从收集的结果创建平值行权价数据框
    atm_df = pd.DataFrame(atm_results)
    
    # Merge ATM strike information back to original dataframe / 将平值行权价信息合并回原始数据框
    # This adds the atm_strike column to each row, matching by timestamp and expiration / 这为每一行添加atm_strike列，通过时间戳和到期日匹配
    if len(atm_df) > 0:
        df = df.merge(atm_df[['local_timestamp', 'expiration', 'atm_strike']], 
                      on=['local_timestamp', 'expiration'], how='left')
    else:
        # If no ATM strikes found, add empty column / 如果未找到平值行权价，添加空列
        # This can happen if there are no common strikes or insufficient data / 如果没有共同行权价或数据不足，可能会发生这种情况
        df['atm_strike'] = np.nan
    
    # Add flag to indicate if this option is at-the-money / 添加标志以指示该期权是否为平值
    # is_atm = True if the option's strike_price equals the calculated atm_strike / 如果期权的strike_price等于计算的atm_strike，则is_atm = True
    # Handle NaN values properly to avoid comparison errors / 正确处理NaN值以避免比较错误
    df['is_atm'] = (df['strike_price'] == df['atm_strike']) & (df['atm_strike'].notna())
    
    return df

# Function: reconstruct_data
def reconstruct_data(input_file_path, output_file_path=None):
    """
    Process raw options data according to the reconstruction steps.
    根据重构步骤处理原始期权数据。
    
    This function orchestrates the three-step process:
    0. Step 0: Remove Greeks and round timestamps / 步骤0: 移除Greeks并将时间戳四舍五入
    1. Step 1: Filter invalid data and calculate mid-price / 步骤1: 过滤无效数据并计算中间价
    2. Step 2: Calculate ATM strikes / 步骤2: 计算平值行权价
    
    Parameters / 参数:
    ----------
    input_file_path : str
        Path to the input CSV file from step1 / 来自步骤1的输入CSV文件路径
    output_file_path : str, optional
        Path to save the processed data / 保存处理后数据的路径
        If None, saves to same directory with '_reconstructed' suffix / 如果为None，保存到同目录并添加'_reconstructed'后缀
    
    Returns / 返回值:
    -------
    pd.DataFrame
        Processed dataframe with reconstructed data / 包含重构数据的处理后的数据框
    """
    # Read input data / 读取输入数据
    df = pd.read_csv(input_file_path)
    
    # Convert underlying_price to numeric and round to 2 decimal places immediately after reading / 读取后立即将underlying_price转换为数值类型并四舍五入到小数点后两位
    # Handle string format like '4094.2200000000003' / 处理像'4094.2200000000003'这样的字符串格式
    if 'underlying_price' in df.columns:
        df['underlying_price'] = pd.to_numeric(df['underlying_price'], errors='coerce').round(2)
    
    # Determine output directory / 确定输出目录
    if output_file_path is None:
        filename = os.path.basename(input_file_path)
        base_name = os.path.splitext(filename)[0].replace("_processed", "")
        # Extract date folder from input path / 从输入路径提取日期文件夹
        # Input path format: E:\Output\{BASE_NAME}\step1_truncate_data\{BASE_NAME}_processed.csv
        input_dir = os.path.dirname(input_file_path)
        date_folder = os.path.basename(os.path.dirname(input_dir))
        output_base_dir = os.path.join(r"E:\Output", date_folder)
        output_subdir = os.path.join(output_base_dir, "step2_data_reconstruction")
    else:
        output_dir = os.path.dirname(output_file_path)
        output_subdir = output_dir if output_dir else "step2_data_reconstruction"
        base_name = os.path.splitext(os.path.basename(output_file_path if output_dir else input_file_path))[0]
    os.makedirs(output_subdir, exist_ok=True)
    
    # Step 0: Preprocess - remove Greeks, convert timestamps, and separate by symbol / 步骤0: 预处理 - 移除Greeks，转换时间戳，并按符号分开
    step0_results = step0_preprocess_data(df)
    
    # Process each symbol separately through Step 1 and Step 2 / 分别处理每个符号的步骤1和步骤2
    final_results = {}
    for asset_symbol, df_step0 in step0_results.items():
        # Save Step 0 output / 保存步骤0的输出
        df_step0_formatted = format_price_columns(df_step0)
        df_step0_formatted.to_csv(os.path.join(output_subdir, f"{base_name}_step0_preprocessed_{asset_symbol}.csv"), index=False)
        
        # Step 1: Filter and calculate mid-price / 步骤1: 过滤并计算中间价
        df_step1 = step1_filter_and_calculate_midprice(df_step0)
        df_step1_formatted = format_price_columns(df_step1)
        df_step1_formatted.to_csv(os.path.join(output_subdir, f"{base_name}_step1_filtered_{asset_symbol}.csv"), index=False)
        
        # Step 2: Calculate ATM strikes / 步骤2: 计算平值行权价
        df_step2 = step2_calculate_atm_strikes(df_step1)
        df_step2_atm = df_step2[df_step2['is_atm'] == True].copy()
        df_step2_atm_formatted = format_price_columns(df_step2_atm)
        df_step2_atm_formatted.to_csv(os.path.join(output_subdir, f"{base_name}_step2_reconstructed_{asset_symbol}.csv"), index=False)
        
        final_results[asset_symbol] = df_step2_atm
    
    # Return combined dataframe (only ATM options) / 返回合并的数据框（仅平值期权）
    return pd.concat(final_results.values(), ignore_index=True) if final_results else pd.DataFrame()

# Function: find_step1_output_files
def find_step1_output_files(output_base_dir, date_start=None, date_end=None):
    """Find all step1 output files / 查找所有步骤1输出文件"""
    if not os.path.exists(output_base_dir):
        raise FileNotFoundError(f"Output directory not found: {output_base_dir}")
    
    step1_files = []
    for item in os.listdir(output_base_dir):
        item_path = os.path.join(output_base_dir, item)
        if os.path.isdir(item_path):
            step1_dir = os.path.join(item_path, "step1_truncate_data")
            if os.path.exists(step1_dir):
                for file in os.listdir(step1_dir):
                    if file.endswith("_processed.csv"):
                        file_path = os.path.join(step1_dir, file)
                        if date_start or date_end:
                            date_match = re.search(r'(\d{4})(\d{2})(\d{2})', file)
                            if date_match:
                                file_date = int(date_match.group(1) + date_match.group(2) + date_match.group(3))
                                if date_start and file_date < int(date_start):
                                    continue
                                if date_end and file_date > int(date_end):
                                    continue
                        step1_files.append(file_path)
    
    return sorted(step1_files)

# Main execution / 主执行部分
OUTPUT_BASE_DIR = r"E:\Output"
DATE_START = "20210516"  # Start date in YYYYMMDD format or None
DATE_END = "20210521"  # End date in YYYYMMDD format or None

# Find all step1 output files / 查找所有步骤1输出文件
step1_files = find_step1_output_files(OUTPUT_BASE_DIR, date_start=DATE_START, date_end=DATE_END)
if not step1_files:
    date_info = f" from {DATE_START} to {DATE_END}" if DATE_START or DATE_END else ""
    raise FileNotFoundError(f"No step1 output files found in {OUTPUT_BASE_DIR}{date_info}")

# Process each file / 处理每个文件
total_files = len(step1_files)
print(f"\nProcessing {total_files} file(s)\n")

successful = 0
failed = 0
for idx, input_file in enumerate(step1_files, 1):
    print(f"[{idx}/{total_files}] {os.path.basename(input_file)}")
    try:
        reconstruct_data(input_file)
        successful += 1
    except Exception as e:
        print(f"  Error: {e}")
        failed += 1

print(f"\nSummary: {successful} successful, {failed} failed\n")

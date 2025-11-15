"""
Bates Model Event Study - Enhanced V3 with Adaptive Model Selection
Bates模型事件研究 - 增强V3版（自适应 Bates / Heston 切换）

Features:
- Auto-switch between Bates (8 params) and Heston (5 params) based on data richness.
- Full pre/post event calibration, variance decomposition, and rich visualization.
- Cross-date summary with model-type heatmap.

Usage / 使用说明:
1. 安装 QuantLib-Python:
   pip install QuantLib-Python

2. 确保 E:\Output\ 下的 step2 / step3 输出已经生成:
   - step2_data_reconstruction\BTC_event_windows_YYYYMMDD_step2_reconstructed_BTC.csv
   - step3_event_time_information\event_time_YYYYMMDD.csv

3. 在命令行中运行:
   python step4_Bates_Model_Calibration.py
"""

import pandas as pd
import numpy as np
import QuantLib as ql
from datetime import timedelta
import os
import re
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ----------------------------------------------------------------------
# Global settings / 全局参数设置
# ----------------------------------------------------------------------

OUTPUT_BASE_DIR = r"E:\Output"

# Date range for processing / 处理的日期范围（包含端点）
DATE_START = "20210516"  # Start date in YYYYMMDD format
DATE_END   = "20210521"  # End date in YYYYMMDD format

# Event window size (minutes) / 事件窗口（分钟）
WINDOW_MINUTES = 15

# Risk-free rate / 无风险利率
RISK_FREE_RATE = 0.0

DAYS_PER_YEAR = 365.0

# Maturity and moneyness filters / 到期与虚实度过滤
MIN_MATURITY_DAYS = 0.5
MAX_MATURITY_DAYS = 7
MONEYNESS_RANGE   = 0.10  # ATM ±10%

# Model selection thresholds
# 模型选择阈值：基于有效 helper 数量
MIN_HELPERS_HESTON = 5    # ≥5 → 可以用 Heston
MIN_HELPERS_BATES  = 8    # ≥8 → 可以用 Bates


# ----------------------------------------------------------------------
# Utility: find data + event_time files
# 查找 step2 数据文件 和 step3 事件时间文件
# ----------------------------------------------------------------------
def find_data_and_event_files(output_base_dir, date_start=None, date_end=None):
    """Find matching step2 data and step3 event time files / 查找匹配的步骤2数据和步骤3事件时间文件"""
    if not os.path.exists(output_base_dir):
        raise FileNotFoundError(f"Output directory not found: {output_base_dir}")
    
    file_pairs = []
    for item in os.listdir(output_base_dir):
        item_path = os.path.join(output_base_dir, item)
        if os.path.isdir(item_path):
            date_match = re.search(r'(\d{4})(\d{2})(\d{2})', item)
            if date_match:
                date_str = "".join(date_match.groups())
                if date_start and date_str < date_start:
                    continue
                if date_end and date_str > date_end:
                    continue
                
                data_file = os.path.join(
                    item_path,
                    "step2_data_reconstruction",
                    f"BTC_event_windows_{date_str}_step2_reconstructed_BTC.csv"
                )
                event_file = os.path.join(
                    item_path,
                    "step3_event_time_information",
                    f"event_time_{date_str}.csv"
                )
                
                if os.path.exists(data_file) and os.path.exists(event_file):
                    file_pairs.append((date_str, data_file, event_file))
    
    return sorted(file_pairs)


# ----------------------------------------------------------------------
# Utility: choose Bates pricing engine
# 为 Bates 模型选择定价引擎
# ----------------------------------------------------------------------
def get_bates_engine(model):
    engines = [
        ('BatesEngine',          lambda m: ql.BatesEngine(m, 64)),
        ('AnalyticBatesEngine',  lambda m: ql.AnalyticBatesEngine(m)),
        ('BatesDoubleExpEngine', lambda m: ql.BatesDoubleExpEngine(m)),
        ('FdBatesVanillaEngine', lambda m: ql.FdBatesVanillaEngine(m, 100, 100, 50))
    ]
    for name, factory in engines:
        if hasattr(ql, name):
            print(f"  Using {name}")
            return factory(model)
    raise RuntimeError("No compatible Bates engine found")


# ----------------------------------------------------------------------
# Core: Calibrate model (Bates or Heston) + build IV/Price comparison table
# 核心：根据数据量选择 Bates / Heston 校准 + 构建 IV/价格对比表
# ----------------------------------------------------------------------
def calibrate_bates(options_df, S0, r=0.0):
    """
    Adaptive calibration:
    - If #helpers >= MIN_HELPERS_BATES → Bates model
    - If MIN_HELPERS_HESTON <= #helpers < MIN_HELPERS_BATES → Heston model
    - Else → raise ValueError

    options_df: aggregated options with columns
        ['time_to_maturity_days', 'strike_price', 'mark_iv', 'underlying_price', 'mark_price', ...]
    S0: spot price (underlying)
    r: risk-free rate
    """
    if len(options_df) < 3:
        raise ValueError(f"Insufficient data: {len(options_df)} options (raw)")

    # Set evaluation date
    today = ql.Date.todaysDate()
    ql.Settings.instance().evaluationDate = today
    calendar = ql.NullCalendar()
    dc = ql.Actual365Fixed()

    # Initial variance from average IV
    v0 = (options_df['mark_iv'].mean() / 100.0) ** 2
    print(f"  Market IV: {np.sqrt(v0):.2%}, Spot: ${S0:.2f}, Options (raw): {len(options_df)}")

    riskfree_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, r, dc))
    dividend_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, 0.0, dc))
    spot_handle = ql.QuoteHandle(ql.SimpleQuote(S0))

    # ------------------------------------------------------------------
    # 1) Build calibration helpers (filtering extreme IV/moneyness)
    # 构建 helper（过滤极端 IV 和虚实度），暂不指定引擎
    # ------------------------------------------------------------------
    helpers = []
    helper_meta = []
    filtered_count = 0

    for _, r_row in options_df.iterrows():
        iv = float(r_row['mark_iv']) / 100.0  # decimal
        moneyness = r_row['strike_price'] / r_row['underlying_price']

        # Filter extreme vols / 虚高或极端波动率
        if not (0.01 < iv <= 3.0):  # up to 300% IV allowed
            filtered_count += 1
            continue

        # Extra loose moneyness guard (in case upstream filters change)
        if not (0.5 < moneyness < 2.0):
            filtered_count += 1
            continue

        helper = ql.HestonModelHelper(
            ql.Period(int(r_row['time_to_maturity_days']), ql.Days),
            calendar,
            S0,
            float(r_row['strike_price']),
            ql.QuoteHandle(ql.SimpleQuote(iv)),
            riskfree_ts,
            dividend_ts,
            ql.BlackCalibrationHelper.ImpliedVolError
        )

        helpers.append(helper)
        helper_meta.append({
            'time_to_maturity_days': float(r_row['time_to_maturity_days']),
            'strike_price': float(r_row['strike_price']),
            'underlying_price': float(r_row['underlying_price']),
            'moneyness': float(r_row.get('moneyness', moneyness)),
            'market_iv': iv,
            'market_price': float(r_row.get('mark_price', 0.0))
        })

    if filtered_count > 0:
        print(f"  Filtered out {filtered_count} extreme options")

    n_helpers = len(helpers)
    if n_helpers < MIN_HELPERS_HESTON:
        raise ValueError(
            f"Only {n_helpers} valid helpers after filtering "
            f"(need ≥{MIN_HELPERS_HESTON} for Heston or ≥{MIN_HELPERS_BATES} for Bates)"
        )

    # Decide model type based on helper count
    if n_helpers >= MIN_HELPERS_BATES:
        use_heston = False
        model_type = "Bates"
    else:
        use_heston = True
        model_type = "Heston"

    print(f"  Using {model_type} model with {n_helpers} helpers")

    # ------------------------------------------------------------------
    # 2) Construct model & engine based on chosen type
    # 根据选择构造 Bates / Heston 模型和引擎
    # ------------------------------------------------------------------
    if use_heston:
        # Heston: 5 parameters
        process = ql.HestonProcess(
            riskfree_ts,
            dividend_ts,
            spot_handle,
            v0,    # v0
            1.5,   # kappa
            v0,    # theta
            0.8,   # sigma
            -0.6   # rho
        )
        model = ql.HestonModel(process)
        engine = ql.AnalyticHestonEngine(model)
    else:
        # Bates: Heston + jumps
        process = ql.BatesProcess(
            riskfree_ts,
            dividend_ts,
            spot_handle,
            v0,    # v0
            1.5,   # kappa
            v0,    # theta
            0.8,   # sigma
            -0.6,  # rho
            1.0,   # lambda
            -0.02, # nu
            0.15   # delta
        )
        model = ql.BatesModel(process)
        engine = get_bates_engine(model)

    # Attach engine to all helpers
    for helper in helpers:
        helper.setPricingEngine(engine)

    # ------------------------------------------------------------------
    # 3) Calibrate
    # ------------------------------------------------------------------
    model.calibrate(
        helpers,
        ql.LevenbergMarquardt(),
        ql.EndCriteria(1000, 100, 1e-8, 1e-8, 1e-8)
    )

    # Helper to safely fetch attributes
    def _get_attr(obj, names):
        if obj is None:
            return None
        for name in names:
            if hasattr(obj, name):
                attr = getattr(obj, name)
                return attr() if callable(attr) else attr
        return None

    # Core Heston parameters
    params = {
        'v0':    model.v0(),
        'kappa': model.kappa(),
        'theta': model.theta(),
        'sigma': model.sigma(),
        'rho':   model.rho(),
        'model_type': model_type
    }

    # Jump-related parameters
    lam = 0.0
    nu  = 0.0
    delta = 0.0

    if not use_heston and isinstance(model, ql.BatesModel):
        proc = _get_attr(model, ["process"])
        lam = (
            _get_attr(model, ["lambda", "lambda_"]) or
            _get_attr(proc, ["lambda", "lambda_", "jumpIntensity"]) or
            0.0
        )
        nu = _get_attr(model, ["nu"]) or 0.0
        delta = _get_attr(model, ["delta"]) or 0.0

    params.update({'lambda': lam, 'nu': nu, 'delta': delta})

    # Calibration errors
    errors = [h.calibrationError() for h in helpers]
    rmse = np.sqrt(np.mean([e ** 2 for e in errors]))
    max_err = max(abs(e) for e in errors)
    params.update({'rmse': rmse, 'max_error': max_err, 'n_helpers': n_helpers})

    print(f"  RMSE: {params['rmse']:.4%}, Max Error: {params['max_error']:.4%}")

    # ------------------------------------------------------------------
    # 4) Build IV & price comparison table
    # IV & 价格对比表：市场 vs 模型
    # ------------------------------------------------------------------
    smile_rows = []
    failed_iv_count = 0

    for helper, meta in zip(helpers, helper_meta):
        # true market price from data
        market_price = meta['market_price']
        # model price from chosen model
        model_price = helper.modelValue()

        # implied vol that reproduces model_price under Black
        try:
            model_iv = helper.impliedVolatility(
                model_price, 1e-6, 100, 1e-4, 5.0
            )
        except Exception:
            model_iv = np.nan
            failed_iv_count += 1

        smile_rows.append({
            'time_to_maturity_days': meta['time_to_maturity_days'],
            'strike_price': meta['strike_price'],
            'underlying_price': meta['underlying_price'],
            'moneyness': meta['moneyness'],
            'market_iv': meta['market_iv'],
            'model_iv': model_iv,
            'iv_error': model_iv - meta['market_iv'],
            'market_price': market_price,
            'model_price': model_price,
            'price_error': model_price - market_price,
            'price_error_pct': ((model_price - market_price) / market_price * 100.0)
                               if market_price != 0 else np.nan
        })

    if failed_iv_count > 0:
        print(f"  Warning: {failed_iv_count}/{n_helpers} IV calculations failed")

    smile_df = pd.DataFrame(smile_rows)

    # IV error statistics
    mae_iv = np.nanmean(np.abs(smile_df['iv_error']))
    valid_ivs = smile_df.dropna(subset=['model_iv'])
    if len(valid_ivs) > 0:
        ss_res = np.sum((valid_ivs['market_iv'] - valid_ivs['model_iv']) ** 2)
        ss_tot = np.sum((valid_ivs['market_iv'] - np.mean(valid_ivs['market_iv'])) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    else:
        r_squared = 0.0

    params.update({'mae': mae_iv, 'r_squared': r_squared})
    print(f"  MAE: {mae_iv:.4%}, R²: {r_squared:.4f}")

    return params, smile_df


# ----------------------------------------------------------------------
# Extract event windows
# ----------------------------------------------------------------------
def extract_event_windows(df, event_time, window_minutes=1):
    event_dt = pd.to_datetime(event_time)
    pre_df = df[
        (df['local_timestamp'] >= event_dt - timedelta(minutes=window_minutes)) &
        (df['local_timestamp'] < event_dt)
    ].copy()
    post_df = df[
        (df['local_timestamp'] >= event_dt) &
        (df['local_timestamp'] <= event_dt + timedelta(minutes=window_minutes))
    ].copy()
    print(f"\nEvent Window: {event_time}")
    print(f"  Pre: {len(pre_df)} records, Post: {len(post_df)} records")
    return pre_df, post_df


# ----------------------------------------------------------------------
# Filter & aggregate options
# ----------------------------------------------------------------------
def filter_options(df, min_days=0.5, max_days=7, moneyness_range=0.10):
    df = df[
        (df['time_to_maturity_days'] >= min_days) &
        (df['time_to_maturity_days'] <= max_days)
    ].copy()
    if len(df) == 0:
        return df

    df['moneyness'] = df['strike_price'] / df['underlying_price']
    df = df[
        (df['moneyness'] >= 1 - moneyness_range) &
        (df['moneyness'] <= 1 + moneyness_range)
    ].copy()
    if len(df) == 0:
        return df

    numeric_cols = [
        'time_to_maturity_days', 'strike_price', 'mark_price',
        'mark_iv', 'underlying_price', 'moneyness'
    ]
    agg_dict = {col: 'mean' for col in numeric_cols if col in df.columns}
    agg_dict['type'] = 'first'

    df_agg = df.groupby('contract').agg(agg_dict).reset_index()
    print(
        f"  Filtered: {len(df_agg)} contracts, "
        f"IV: {df_agg['mark_iv'].min():.2f}%-{df_agg['mark_iv'].max():.2f}%"
    )
    return df_agg


# ----------------------------------------------------------------------
# Parameter shift analysis
# ----------------------------------------------------------------------
def analyze_parameter_shifts(pre_params, post_params):
    print("\n" + "=" * 70)
    print("Parameter Shift Analysis / 参数变化分析")
    print("=" * 70)

    shifts = {}
    for param in ['v0', 'kappa', 'theta', 'sigma', 'rho', 'lambda', 'nu', 'delta']:
        pre_val = pre_params[param]
        post_val = post_params[param]
        pct_change = ((post_val - pre_val) / pre_val * 100.0) if pre_val != 0 else 0.0
        shifts[param] = {
            'pre': pre_val,
            'post': post_val,
            'change': post_val - pre_val,
            'pct_change': pct_change
        }
        flag = "**" if abs(pct_change) > 20 else "  "
        print(f"{flag} {param:8s}: {pre_val:8.5f} -> {post_val:8.5f} ({pct_change:+7.2f}%)")
    return shifts


# ----------------------------------------------------------------------
# Variance decomposition
# ----------------------------------------------------------------------
def variance_decomposition(pre_opts, post_opts, pre_params, post_params):
    print("\n" + "=" * 70)
    print("Variance Decomposition / 方差分解")
    print("=" * 70)

    def calc_variance(opts, params):
        tau = opts['time_to_maturity_days'].mean() / DAYS_PER_YEAR
        iv = opts['mark_iv'].mean() / 100.0
        V_total = (iv ** 2) * tau
        V_base = params['theta'] * tau
        V_event = max(V_total - V_base, 0.0)
        return {'tau': tau, 'iv': iv, 'V_total': V_total, 'V_base': V_base, 'V_event': V_event}

    pre = calc_variance(pre_opts, pre_params)
    post = calc_variance(post_opts, post_params)

    print(
        f"\nPre-Event: IV={pre['iv']:.2%}, V_total={pre['V_total']:.6f}, "
        f"V_base={pre['V_base']:.6f} ({pre['V_base']/pre['V_total']*100:.1f}%), "
        f"V_event={pre['V_event']:.6f} ({pre['V_event']/pre['V_total']*100:.1f}%)"
    )
    print(
        f"Post-Event: IV={post['iv']:.2%}, V_total={post['V_total']:.6f}, "
        f"V_base={post['V_base']:.6f} ({post['V_base']/post['V_total']*100:.1f}%), "
        f"V_event={post['V_event']:.6f} ({post['V_event']/post['V_total']*100:.1f}%)"
    )
    print(
        f"\nChange: DeltaIV={(post['iv']-pre['iv'])*100:+.2f}pp, "
        f"DeltaV_event={post['V_event']-pre['V_event']:+.6f}"
    )

    return {'pre': pre, 'post': post}


# ----------------------------------------------------------------------
# Comprehensive plots for a single event
# ----------------------------------------------------------------------
def plot_comprehensive_comparison(pre_smile, post_smile, pre_params, post_params, outdir):
    print("\n" + "=" * 70)
    print("Generating Comprehensive Plots / 生成综合图表")
    print("=" * 70)

    pre_model = pre_params.get("model_type", "Model")
    post_model = post_params.get("model_type", "Model")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    pre_sorted = pre_smile.sort_values('moneyness')
    post_sorted = post_smile.sort_values('moneyness')
    
    # Price scatter: pre
    axes[0, 0].scatter(
        pre_smile['market_price'], pre_smile['model_price'],
        alpha=0.7, s=80, edgecolors='black', linewidths=1
    )
    min_p = min(pre_smile['market_price'].min(), pre_smile['model_price'].min())
    max_p = max(pre_smile['market_price'].max(), pre_smile['model_price'].max())
    axes[0, 0].plot([min_p, max_p], [min_p, max_p], 'r--', linewidth=2, label='45° line')
    axes[0, 0].set_xlabel('Market Price ($)', fontsize=11)
    axes[0, 0].set_ylabel(f'{pre_model} Model Price ($)', fontsize=11)
    axes[0, 0].set_title(
        f'Pre-Event ({pre_model}): Price Comparison (R²={pre_params.get("r_squared", 0):.3f})',
        fontsize=12, fontweight='bold'
    )
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Price scatter: post
    axes[0, 1].scatter(
        post_smile['market_price'], post_smile['model_price'],
        alpha=0.7, s=80, color='orange', edgecolors='black', linewidths=1
    )
    min_p = min(post_smile['market_price'].min(), post_smile['model_price'].min())
    max_p = max(post_smile['market_price'].max(), post_smile['model_price'].max())
    axes[0, 1].plot([min_p, max_p], [min_p, max_p], 'r--', linewidth=2, label='45° line')
    axes[0, 1].set_xlabel('Market Price ($)', fontsize=11)
    axes[0, 1].set_ylabel(f'{post_model} Model Price ($)', fontsize=11)
    axes[0, 1].set_title(
        f'Post-Event ({post_model}): Price Comparison (R²={post_params.get("r_squared", 0):.3f})',
        fontsize=12, fontweight='bold'
    )
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # IV smile: pre
    axes[1, 0].plot(
        pre_sorted['moneyness'], pre_sorted['market_iv'] * 100,
        'o-', label='Market IV', markersize=8, linewidth=2.5, color='blue', alpha=0.8
    )
    valid_pre = pre_sorted[~pre_sorted['model_iv'].isna()]
    if len(valid_pre) > 0:
        axes[1, 0].plot(
            valid_pre['moneyness'], valid_pre['model_iv'] * 100,
            's--', label=f'{pre_model} IV', markersize=8, linewidth=2.5, color='red', alpha=0.8
        )
        axes[1, 0].fill_between(
            valid_pre['moneyness'],
            valid_pre['market_iv'] * 100,
            valid_pre['model_iv'] * 100,
            alpha=0.2, color='gray', label='Error band'
        )
    axes[1, 0].set_xlabel('Moneyness (K/S)', fontsize=11)
    axes[1, 0].set_ylabel('Implied Volatility (%)', fontsize=11)
    axes[1, 0].set_title(
        f'Pre-Event ({pre_model}): IV Smile (RMSE={pre_params["rmse"]:.4%})',
        fontsize=12, fontweight='bold'
    )
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # IV smile: post
    axes[1, 1].plot(
        post_sorted['moneyness'], post_sorted['market_iv'] * 100,
        'o-', label='Market IV', markersize=8, linewidth=2.5, color='blue', alpha=0.8
    )
    valid_post = post_sorted[~post_sorted['model_iv'].isna()]
    if len(valid_post) > 0:
        axes[1, 1].plot(
            valid_post['moneyness'], valid_post['model_iv'] * 100,
            's--', label=f'{post_model} IV', markersize=8, linewidth=2.5, color='red', alpha=0.8
        )
        axes[1, 1].fill_between(
            valid_post['moneyness'],
            valid_post['market_iv'] * 100,
            valid_post['model_iv'] * 100,
            alpha=0.2, color='gray', label='Error band'
        )
    axes[1, 1].set_xlabel('Moneyness (K/S)', fontsize=11)
    axes[1, 1].set_ylabel('Implied Volatility (%)', fontsize=11)
    axes[1, 1].set_title(
        f'Post-Event ({post_model}): IV Smile (RMSE={post_params["rmse"]:.4%})',
        fontsize=12, fontweight='bold'
    )
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(outdir, 'comprehensive_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: comprehensive_comparison.png")
    plt.close()
    
    # Error analysis plots
    fig2, axes2 = plt.subplots(2, 2, figsize=(16, 10))
    
    pre_valid_iv = pre_smile[~pre_smile['iv_error'].isna()]
    if len(pre_valid_iv) > 0:
        axes2[0, 0].bar(
            range(len(pre_valid_iv)),
            pre_valid_iv['iv_error'] * 100,
            alpha=0.7, color='steelblue', edgecolor='black'
        )
        axes2[0, 0].axhline(y=0, color='red', linestyle='--', linewidth=2)
        axes2[0, 0].set_xlabel('Option Index', fontsize=11)
        axes2[0, 0].set_ylabel('IV Error (vol pts)', fontsize=11)
        axes2[0, 0].set_title(
            f'Pre-Event ({pre_model}): IV Errors (MAE={pre_params.get("mae", 0):.4%})',
            fontsize=12, fontweight='bold'
        )
        axes2[0, 0].grid(True, alpha=0.3, axis='y')
    
    post_valid_iv = post_smile[~post_smile['iv_error'].isna()]
    if len(post_valid_iv) > 0:
        axes2[0, 1].bar(
            range(len(post_valid_iv)),
            post_valid_iv['iv_error'] * 100,
            alpha=0.7, color='orange', edgecolor='black'
        )
        axes2[0, 1].axhline(y=0, color='red', linestyle='--', linewidth=2)
        axes2[0, 1].set_xlabel('Option Index', fontsize=11)
        axes2[0, 1].set_ylabel('IV Error (vol pts)', fontsize=11)
        axes2[0, 1].set_title(
            f'Post-Event ({post_model}): IV Errors (MAE={post_params.get("mae", 0):.4%})',
            fontsize=12, fontweight='bold'
        )
        axes2[0, 1].grid(True, alpha=0.3, axis='y')
    
    pre_valid_price = pre_smile[~pre_smile['price_error_pct'].isna()]
    if len(pre_valid_price) > 0:
        axes2[1, 0].bar(
            range(len(pre_valid_price)),
            pre_valid_price['price_error_pct'],
            alpha=0.7, color='steelblue', edgecolor='black'
        )
        axes2[1, 0].axhline(y=0, color='red', linestyle='--', linewidth=2)
        axes2[1, 0].set_xlabel('Option Index', fontsize=11)
        axes2[1, 0].set_ylabel('Price Error (%)', fontsize=11)
        axes2[1, 0].set_title(
            f'Pre-Event ({pre_model}): Price Errors',
            fontsize=12, fontweight='bold'
        )
        axes2[1, 0].grid(True, alpha=0.3, axis='y')
    
    post_valid_price = post_smile[~post_smile['price_error_pct'].isna()]
    if len(post_valid_price) > 0:
        axes2[1, 1].bar(
            range(len(post_valid_price)),
            post_valid_price['price_error_pct'],
            alpha=0.7, color='orange', edgecolor='black'
        )
        axes2[1, 1].axhline(y=0, color='red', linestyle='--', linewidth=2)
        axes2[1, 1].set_xlabel('Option Index', fontsize=11)
        axes2[1, 1].set_ylabel('Price Error (%)', fontsize=11)
        axes2[1, 1].set_title(
            f'Post-Event ({post_model}): Price Errors',
            fontsize=12, fontweight='bold'
        )
        axes2[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    error_plot_path = os.path.join(outdir, 'error_analysis.png')
    plt.savefig(error_plot_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: error_analysis.png")
    plt.close()


# ----------------------------------------------------------------------
# Process a single event; return (success_flag, summary_row)
# 处理单个事件；返回 (是否成功, 汇总行)
# ----------------------------------------------------------------------
def process_single_event(df, event_time, output_subdir, date_str, event_index):
    print(f"\nEvent time: {event_time}")
    print(f"Output: {output_subdir}")
    
    os.makedirs(output_subdir, exist_ok=True)
    
    pre_df, post_df = extract_event_windows(df, event_time, WINDOW_MINUTES)
    
    summary_row = {
        'date': date_str,
        'event_index': event_index,
        'event_time': event_time,
        'n_records_pre': len(pre_df),
        'n_records_post': len(post_df),
        'n_opts_pre': 0,
        'n_opts_post': 0,
        'model_type_pre': 'NA',
        'model_type_post': 'NA',
        'rmse_pre': np.nan,
        'rmse_post': np.nan,
        'succ_pre': 0,
        'succ_post': 0
    }
    
    if len(pre_df) == 0 or len(post_df) == 0:
        print("  Error: Empty event windows")
        with open(os.path.join(output_subdir, "error_log.txt"), 'w') as f:
            f.write("Empty event windows\n")
        return False, summary_row
    
    pre_opts = filter_options(pre_df, MIN_MATURITY_DAYS, MAX_MATURITY_DAYS, MONEYNESS_RANGE)
    post_opts = filter_options(post_df, MIN_MATURITY_DAYS, MAX_MATURITY_DAYS, MONEYNESS_RANGE)
    
    summary_row['n_opts_pre'] = len(pre_opts)
    summary_row['n_opts_post'] = len(post_opts)
    
    if len(pre_opts) < 3 or len(post_opts) < 3:
        print(f"  Error: Insufficient data (Pre: {len(pre_opts)}, Post: {len(post_opts)})")
        pre_opts.to_csv(os.path.join(output_subdir, "pre_event_filtered_options.csv"), index=False)
        post_opts.to_csv(os.path.join(output_subdir, "post_event_filtered_options.csv"), index=False)
        with open(os.path.join(output_subdir, "error_log.txt"), 'w') as f:
            f.write(f"Insufficient data: Pre={len(pre_opts)}, Post={len(post_opts)}\n")
        return False, summary_row
    
    S0_pre = pre_opts['underlying_price'].mean()
    S0_post = post_opts['underlying_price'].mean()
    
    try:
        pre_params, pre_smile = calibrate_bates(pre_opts, S0_pre, RISK_FREE_RATE)
        post_params, post_smile = calibrate_bates(post_opts, S0_post, RISK_FREE_RATE)
    except Exception as e:
        print(f"  Calibration failed: {e}")
        pre_opts.to_csv(os.path.join(output_subdir, "pre_event_filtered_options.csv"), index=False)
        post_opts.to_csv(os.path.join(output_subdir, "post_event_filtered_options.csv"), index=False)
        with open(os.path.join(output_subdir, "error_log.txt"), 'w') as f:
            f.write(f"Calibration failed: {e}\n")
            f.write(f"Pre options: {len(pre_opts)}, Post options: {len(post_opts)}\n")
        return False, summary_row
    
    # Fill summary info
    summary_row['model_type_pre'] = pre_params.get('model_type', 'NA')
    summary_row['model_type_post'] = post_params.get('model_type', 'NA')
    summary_row['rmse_pre'] = pre_params.get('rmse', np.nan)
    summary_row['rmse_post'] = post_params.get('rmse', np.nan)
    summary_row['succ_pre'] = 1
    summary_row['succ_post'] = 1
    
    shifts = analyze_parameter_shifts(pre_params, post_params)
    variance_results = variance_decomposition(pre_opts, post_opts, pre_params, post_params)
    
    # Plots
    plot_comprehensive_comparison(pre_smile, post_smile, pre_params, post_params, output_subdir)
    
    # Save CSVs
    pd.DataFrame([pre_params]).to_csv(os.path.join(output_subdir, "pre_event_calibration_params.csv"), index=False)
    pd.DataFrame([post_params]).to_csv(os.path.join(output_subdir, "post_event_calibration_params.csv"), index=False)
    pd.DataFrame(shifts).T.to_csv(os.path.join(output_subdir, "parameter_shifts.csv"))
    
    variance_df = pd.DataFrame({
        'window': ['pre', 'post'],
        'tau': [variance_results['pre']['tau'], variance_results['post']['tau']],
        'iv': [variance_results['pre']['iv'], variance_results['post']['iv']],
        'V_total': [variance_results['pre']['V_total'], variance_results['post']['V_total']],
        'V_base': [variance_results['pre']['V_base'], variance_results['post']['V_base']],
        'V_event': [variance_results['pre']['V_event'], variance_results['post']['V_event']]
    })
    variance_df.to_csv(os.path.join(output_subdir, "variance_decomposition.csv"), index=False)
    
    pre_opts.to_csv(os.path.join(output_subdir, "pre_event_filtered_options.csv"), index=False)
    post_opts.to_csv(os.path.join(output_subdir, "post_event_filtered_options.csv"), index=False)
    pre_smile.to_csv(os.path.join(output_subdir, "pre_event_iv_price_comparison.csv"), index=False)
    post_smile.to_csv(os.path.join(output_subdir, "post_event_iv_price_comparison.csv"), index=False)
    
    print(f"  Success: RMSE={pre_params['rmse']:.4%}/{post_params['rmse']:.4%}")
    return True, summary_row


# ----------------------------------------------------------------------
# Summary heatmap across dates (Style B)
# ----------------------------------------------------------------------
def plot_model_type_heatmaps(summary_df, outdir):
    """
    Create heatmap-style summary of model types across dates & events.
    生成跨日期/事件的模型类型 heatmap（风格B）.
    """
    print("\n" + "=" * 70)
    print("Generating Model-Type Summary Heatmap / 生成模型类型汇总热图")
    print("=" * 70)

    # Encode model types: 0 = NA/fail, 1 = Heston, 2 = Bates
    def encode(m):
        if m == 'Bates':
            return 2
        elif m == 'Heston':
            return 1
        else:
            return 0

    summary_df = summary_df.copy()
    summary_df['model_code_pre'] = summary_df['model_type_pre'].apply(encode)
    summary_df['model_code_post'] = summary_df['model_type_post'].apply(encode)

    # Sort by date then event_index
    summary_df = summary_df.sort_values(['date', 'event_index'])

    # Build pivot tables (date x event_index)
    pre_pivot = summary_df.pivot(index='date', columns='event_index', values='model_code_pre')
    post_pivot = summary_df.pivot(index='date', columns='event_index', values='model_code_post')

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, pivot, title in zip(
        axes,
        [pre_pivot, post_pivot],
        ['Pre-Event Model Type', 'Post-Event Model Type']
    ):
        im = ax.imshow(pivot.values, aspect='auto', interpolation='nearest')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Event Index', fontsize=11)
        ax.set_ylabel('Date', fontsize=11)
        ax.set_xticks(np.arange(pivot.shape[1]))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(np.arange(pivot.shape[0]))
        ax.set_yticklabels(pivot.index)
        # colorbar with discrete ticks
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_ticks([0, 1, 2])
        cbar.set_ticklabels(['NA/Fail', 'Heston', 'Bates'])

    plt.tight_layout()
    out_path = os.path.join(outdir, "model_type_summary_heatmap.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved heatmap: {out_path}")
    plt.close()


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    print("=" * 70)
    print("Bates Model Event Study - Enhanced V3 / Bates模型事件研究 - 增强V3版")
    print("=" * 70)
    print(f"\nQuantLib version: {getattr(ql, '__version__', 'Unknown')}")
    
    file_pairs = find_data_and_event_files(OUTPUT_BASE_DIR, date_start=DATE_START, date_end=DATE_END)
    if not file_pairs:
        date_info = f" from {DATE_START} to {DATE_END}" if DATE_START or DATE_END else ""
        raise FileNotFoundError(f"No data/event files found{date_info}")
    
    total_files = len(file_pairs)
    print(f"\nProcessing {total_files} date(s)\n")
    
    successful = 0
    failed = 0
    summary_rows = []
    
    for idx, (date_str, data_file, event_file) in enumerate(file_pairs, 1):
        print(f"\n{'='*70}")
        print(f"[{idx}/{total_files}] Date: {date_str}")
        print(f"{'='*70}")
        
        try:
            df = pd.read_csv(data_file)
            df['local_timestamp'] = pd.to_datetime(df['local_timestamp'])
            print(f"Data records: {len(df):,}")
            
            event_times_df = pd.read_csv(event_file)
            print(f"Events: {len(event_times_df)}")
            
            date_folder = f"BTC_event_windows_{date_str}"
            base_output_dir = os.path.join(OUTPUT_BASE_DIR, date_folder, "step4_Bates_Model_Calibration")
            
            for event_idx, event_row in event_times_df.iterrows():
                event_time = event_row['event_time']
                event_output_dir = os.path.join(
                    base_output_dir,
                    f"event_{event_idx+1}_{event_time.replace(' ', '_').replace(':', '-')}"
                )
                
                ok, row = process_single_event(df, event_time, event_output_dir, date_str, event_idx+1)
                summary_rows.append(row)
                if ok:
                    successful += 1
                else:
                    failed += 1
        
        except Exception as e:
            print(f"Error processing {date_str}: {e}")
            failed += 1
    
    print(f"\n{'='*70}")
    print(f"Summary: {successful} successful, {failed} failed events")
    print(f"{'='*70}\n")

    # --------------------------------------------------------------
    # Cross-date summary & heatmap
    # --------------------------------------------------------------
    if summary_rows:
        summary_dir = os.path.join(OUTPUT_BASE_DIR, "summary_across_dates")
        os.makedirs(summary_dir, exist_ok=True)
        summary_df = pd.DataFrame(summary_rows)
        
        # Save summary table
        summary_csv = os.path.join(summary_dir, "event_model_type_summary.csv")
        summary_df.to_csv(summary_csv, index=False)
        print(f"Saved event-level summary: {summary_csv}")

        # Quick text stats
        print("\nModel usage (pre-event):")
        print(summary_df['model_type_pre'].value_counts(dropna=False))
        print("\nModel usage (post-event):")
        print(summary_df['model_type_post'].value_counts(dropna=False))

        # Heatmap (style B)
        plot_model_type_heatmaps(summary_df, summary_dir)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Detect season boundaries by comparing each candidate season against a rolling reference window.
def set_season_start_week(data, epi, window_years = 3, season_years=1, half_year_weeks = 26):
    results = []

    WEEKS_IN_YEAR = 52
    window_weeks = window_years * WEEKS_IN_YEAR
    season_weeks = season_years * WEEKS_IN_YEAR
    first_start_date = data["Date"].min() + pd.DateOffset(years=window_years)
    start_idx = data[data["Date"] >= first_start_date].index[0]

    season_id = 1

    while True:
        season_start = data.loc[start_idx, "Date"]
        season_end = season_start + pd.Timedelta(weeks=season_weeks)

        ref_start = season_start - pd.Timedelta(weeks=window_weeks)
        ref_data = data[(data["Date"] >= ref_start) & (data["Date"] < season_start)]
        if ref_data["Date"].dt.year.nunique() < window_years:
            break

        threshold = ref_data[epi].mean()

        season_data = data[(data["Date"] >= season_start) & (data["Date"] < season_end)]
        if season_data.empty:
            break

        season_data = season_data.copy()
        season_data["exceed"] = season_data[epi] > threshold

        epidemic_data = season_data[season_data["exceed"]]

        if epidemic_data.empty:
            next_idx = data[data["Date"] >= season_end].index
            if len(next_idx) == 0:
                break

            start_idx = next_idx[0]
            season_id += 1
            year_id +=1
            continue

        peak = epidemic_data.loc[epidemic_data[epi].idxmax()]
        peak_date = peak["Date"]

        new_start = peak_date - pd.Timedelta(weeks=26)
        year_id = data[data["Date"] >= new_start].index[0]

        results.append({
            "season":  data.loc[year_id, 'Year'],
            "season_start": season_start,
            "threshold": threshold,
            "peak_date": peak_date,
            "start_week": data.loc[start_idx, "Week"],
            "peak_week": peak["Week"],
            "peak_epi": peak[epi],
            "epi_dates": season_data.loc[season_data["exceed"] == 1, "Date"].tolist(),
            "epi_weeks": season_data.loc[season_data["exceed"] == 1, "Week"].tolist(),
            "threshold": threshold
        })

        epi_end = peak_date + pd.Timedelta(weeks=half_year_weeks)

        next_idx = data[data["Date"] >= epi_end].index
        if len(next_idx) == 0:
            break

        start_idx = next_idx[0]
        season_id += 1
        year_id +=1
    results_df = pd.DataFrame(results)
    return results_df

def set_season_start_week_adaptive(data, epi, window_years=3, season_years=1, half_year_weeks=26):
    df = data.copy().sort_values('Date').reset_index(drop=True)
    min_date = df['Date'].min().normalize()
    max_date = df['Date'].max().normalize()

    if max_date < (min_date + pd.DateOffset(years=1)):
        return pd.DataFrame(), {
            'mode': 'insufficient',
            'reason': 'At least 1 year of data is required.'
        }

    if max_date >= (min_date + pd.DateOffset(years=4)):
        return set_season_start_week(
            df,
            epi,
            window_years=window_years,
            season_years=season_years,
            half_year_weeks=half_year_weeks
        ), {'mode': 'standard'}

    mode = 'short_single' if max_date < (min_date + pd.DateOffset(years=2)) else 'short_multi'
    results_df = set_short_history_season_start_week(
        df,
        epi,
        mode=mode,
        half_year_weeks=half_year_weeks
    )
    start_week = None
    if not results_df.empty and 'start_week' in results_df.columns:
        start_week = int(results_df['start_week'].dropna().iloc[0])

    return results_df, {'mode': mode, 'start_week': start_week}

def set_short_history_season_start_week(data, epi, mode='short_multi', half_year_weeks=26):
    df = data.copy().sort_values('Date').reset_index(drop=True)
    first_year_end = df['Date'].min() + pd.DateOffset(years=1)
    first_year_data = df[(df['Date'] >= df['Date'].min()) & (df['Date'] < first_year_end)]

    if first_year_data.empty:
        return pd.DataFrame()

    first_peak = first_year_data.loc[first_year_data[epi].idxmax()]
    accepted_peaks = [first_peak]

    if mode == 'short_multi':
        candidate_start = first_peak['Date'] + pd.Timedelta(weeks=half_year_weeks)
        while candidate_start <= df['Date'].max():
            candidate_end = candidate_start + pd.Timedelta(weeks=52)
            candidate_data = df[(df['Date'] >= candidate_start) & (df['Date'] < candidate_end)]
            if candidate_data.empty:
                break

            previous_peak_dates = [peak['Date'] for peak in accepted_peaks]
            previous_windows = []
            for peak_date in previous_peak_dates:
                win_start = peak_date - pd.Timedelta(weeks=half_year_weeks)
                win_end = peak_date + pd.Timedelta(weeks=half_year_weeks)
                previous_windows.append(df[(df['Date'] >= win_start) & (df['Date'] < win_end)])
            previous_data = pd.concat(previous_windows, ignore_index=True)
            threshold = previous_data[epi].mean()

            candidate_peak = candidate_data.loc[candidate_data[epi].idxmax()]
            if candidate_peak[epi] > threshold:
                accepted_peaks.append(candidate_peak)
                candidate_start = candidate_peak['Date'] + pd.Timedelta(weeks=half_year_weeks)
            else:
                candidate_start = candidate_end

    peak_weeks = [int(peak['Week']) for peak in accepted_peaks]
    median_peak_week = int(np.median(peak_weeks))
    start_week = median_peak_week - half_year_weeks
    while start_week <= 0:
        start_week += 52
    while start_week > 52:
        start_week -= 52

    anchor_start = first_peak['Date'] - pd.Timedelta(weeks=half_year_weeks)
    while anchor_start > df['Date'].min():
        anchor_start -= pd.Timedelta(weeks=52)
    while (anchor_start + pd.Timedelta(weeks=52)) <= df['Date'].min():
        anchor_start += pd.Timedelta(weeks=52)

    boundaries = []
    boundary = anchor_start
    while boundary <= (df['Date'].max() + pd.Timedelta(weeks=52)):
        boundaries.append(boundary)
        boundary += pd.Timedelta(weeks=52)

    assigned = df.copy()
    assigned['Season'] = np.nan
    for idx, boundary_start in enumerate(boundaries[:-1]):
        boundary_end = boundaries[idx + 1]
        season_label = int(boundary_start.year)
        mask = (assigned['Date'] >= boundary_start) & (assigned['Date'] < boundary_end)
        assigned.loc[mask, 'Season'] = season_label

    assigned = assigned.dropna(subset=['Season']).copy()
    assigned['Season'] = assigned['Season'].astype(int)

    results = []
    previous_season_means = []
    for season in sorted(assigned['Season'].dropna().unique()):
        season_data = assigned[assigned['Season'] == season].copy()
        if season_data.empty:
            continue

        peak = season_data.loc[season_data[epi].idxmax()]
        threshold = np.mean(previous_season_means) if previous_season_means else first_year_data[epi].mean()
        exceed_data = season_data[season_data[epi] > threshold]

        results.append({
            'season': int(season),
            'season_start': season_data['Date'].min(),
            'boundary_start': season_data['Date'].min(),
            'threshold': threshold,
            'peak_date': peak['Date'],
            'start_week': int(start_week),
            'peak_week': int(peak['Week']),
            'peak_epi': peak[epi],
            'epi_dates': exceed_data['Date'].tolist(),
            'epi_weeks': exceed_data['Week'].tolist(),
        })
        previous_season_means.append(season_data[epi].mean())

    return pd.DataFrame(results)

def filter_data(data, seasons, start_week=24):
    df = data.copy()

    # 시즌 데이터만 사용
    df['Season'] = np.where(df['Week'] >= start_week, df['Year'], df['Year'] - 1)
    df = df[df['Season'].isin(seasons)]

    # train / test 구분
    season_counts = df.groupby('Season')['Week'].nunique()
    full_seasons = season_counts[season_counts >= 52].index
    df['set'] = np.where(df['Season'].isin(full_seasons), 'train', 'test')

    return df

def extend_standard_seasons_with_leading_history(data, detected_seasons, start_week, epi, season_results):
    if not detected_seasons:
        return []

    threshold_values = pd.to_numeric(season_results.get('threshold'), errors='coerce').dropna()
    if threshold_values.empty:
        return sorted({int(season) for season in detected_seasons})

    leading_threshold = threshold_values.mean()
    df = data.copy().sort_values('Date').reset_index(drop=True)
    df['Season'] = np.where(df['Week'] >= start_week, df['Year'], df['Year'] - 1)

    detected_set = {int(season) for season in detected_seasons}
    first_detected_season = min(detected_set)
    leading_candidates = sorted(
        int(season)
        for season in df.loc[df['Season'] < first_detected_season, 'Season'].dropna().unique()
    )

    leading_valid_seasons = []
    for season in leading_candidates:
        season_data = df[df['Season'] == season]
        if season_data.empty:
            continue
        if (season_data[epi] > leading_threshold).any():
            leading_valid_seasons.append(season)

    return sorted(set(leading_valid_seasons).union(detected_set))

def assign_retrospective_period(data, seasons=None, start_week=24, season_starts=None):
    df = data.copy()
    df = df.sort_values('Date').reset_index(drop=True)

    if season_starts is not None and not season_starts.empty:
        starts = season_starts[['season', 'season_start']].dropna().copy()
        starts['season_start'] = pd.to_datetime(starts['season_start'])
        starts = starts.sort_values('season_start').reset_index(drop=True)

        df['Season'] = np.nan
        for idx, row in starts.iterrows():
            season_start = row['season_start']
            season_end = starts.loc[idx + 1, 'season_start'] if idx + 1 < len(starts) else df['Date'].max() + pd.Timedelta(days=1)
            mask = (df['Date'] >= season_start) & (df['Date'] < season_end)
            df.loc[mask, 'Season'] = int(row['season'])
        df = df.dropna(subset=['Season']).copy()
        df['Season'] = df['Season'].astype(int)
    else:
        df['Season'] = np.where(df['Week'] >= start_week, df['Year'], df['Year'] - 1)

    if seasons is not None:
        df = df[df['Season'].isin(seasons)].copy()
    df = df.sort_values('Date').reset_index(drop=True)
    df['set'] = 'analysis'

    count_col = 'Date' if 'Date' in df.columns else 'Week'
    season_counts = df.groupby('Season')[count_col].nunique()
    complete_seasons = season_counts[season_counts >= 52].index.tolist()
    incomplete_seasons = season_counts[season_counts < 52].index.tolist()
    analysis_seasons = sorted(df['Season'].unique().tolist())
    warmup_seasons = []
    if analysis_seasons and analysis_seasons[0] in incomplete_seasons:
        warmup_seasons = [analysis_seasons[0]]

    complete_seasons_for_window = complete_seasons

    meta = {
        'analysis_start': df['Date'].min(),
        'analysis_end': df['Date'].max(),
        'analysis_seasons': analysis_seasons,
        'complete_seasons': sorted(complete_seasons),
        'incomplete_seasons': sorted(incomplete_seasons),
        'window_eval_seasons': sorted(complete_seasons_for_window),
        'warmup_seasons': warmup_seasons,
    }

    return df.reset_index(drop=True), meta

# Split the dataset into fitting and real-time periods while respecting the learned season start.
def assign_analysis_periods(data, seasons, start_week, fit_end_date):
    df = data.copy()
    df = df.sort_values('Date').reset_index(drop=True)

    df['Season'] = np.where(df['Week'] >= start_week, df['Year'], df['Year'] - 1)
    df = df[df['Season'].isin(seasons)].copy()
    df = df.sort_values('Date').reset_index(drop=True)

    fit_end_date = pd.to_datetime(fit_end_date)
    available_dates = df.loc[df['Date'] <= fit_end_date, 'Date']
    if available_dates.empty:
        raise ValueError("No available data on or before the selected fitting end date.")

    effective_fit_input = available_dates.max()
    simulation_start_candidates = df.loc[df['Date'] > effective_fit_input, 'Date']
    if simulation_start_candidates.empty:
        raise ValueError("No data available after the selected fitting end date.")

    simulation_start = simulation_start_candidates.min()

    df['set'] = 'train'
    df.loc[df['Date'] >= simulation_start, 'set'] = 'test'

    train_seasons = sorted(df.loc[df['set'] == 'train', 'Season'].unique().tolist())
    test_seasons = sorted(df.loc[df['set'] == 'test', 'Season'].unique().tolist())
    overlap_seasons = sorted(set(train_seasons).intersection(test_seasons))

    meta = {
        'fit_end_date_user': effective_fit_input,
        'simulation_start': simulation_start,
        'train_seasons': train_seasons,
        'test_seasons': test_seasons,
        'overlap_seasons': overlap_seasons,
    }

    return df.reset_index(drop=True), meta

def Observation_period(outbreak_result, year, peak_week, len_period=40):
    df_sorted = outbreak_result.sort_values(['Year', 'Week']).reset_index(drop=True)
    target_idx = df_sorted[(df_sorted['Season'] == year) & (df_sorted['Week'] == peak_week)].index
    
    if len(target_idx) == 0:
        print(f"Warning: {year}-{year+1}시즌 {peak_week}주차 데이터를 찾을 수 없습니다.")
        return None
    
    idx = target_idx[0]
    start_idx = max(0, idx - len_period+1)
    d = df_sorted.iloc[start_idx : idx + 1].copy()
    
    return d.reset_index(drop=True)

# Fit a single-break hockey-stick curve to the observation window around a seasonal peak.
def Module_HockeyStick(x, y, k, ty='linear'):
    if ty == 'linear':
        y_obs = y
    elif ty == 'exponential':
        eps = 1e-8
        y_obs = np.log(np.maximum(y, eps))
    else:
        warnings.warn(
            f"[Module_HockeyStick] Unknown type='{ty}'. "
            "Supported types are 'linear' or 'exponential'. "
            "Defaulting to 'linear'.",
            UserWarning
        )
        ty = 'linear'
        y_obs = y
    
    hinge = np.maximum(0.0, x - k)
    X = np.column_stack([np.ones_like(x), hinge])
    beta, *_ = np.linalg.lstsq(X, y_obs, rcond=None)
    
    yhat_obs = X @ beta
    sse = float(np.sum((y_obs - yhat_obs) ** 2))

    yhat = np.exp(yhat_obs) if ty == 'exponential' else yhat_obs
    beta[0] = np.exp(beta[0]) if ty == 'exponential' else beta[0]
    
    return beta, yhat, sse

def Run_HockeyStick(d, epi, min_seg_len=4, ty='linear'):
    x = d.index.to_numpy(dtype=float)
    y = d[epi].to_numpy(dtype=float)
    n = len(x)
    
    if n < 2 * min_seg_len + 1:
        return np.nan, None
    k_candidates = x[min_seg_len: n - min_seg_len]

    best = {'k': None, 'sse': np.inf, 'beta': None, 'yhat': None}
    for k in k_candidates:
        beta, yhat, sse = Module_HockeyStick(x, y, k, ty)
        if sse < best['sse']:
            best.update({'k': int(k), 'sse': sse, 'beta': beta, 'yhat': yhat})

    b0, b1 = best['beta']
    best['b0'] = float(b0)
    best['b1'] = float(b1)
    best['slope_before'] = 0.0
    best['slope_after']  = float(b1)

    return best['k'], best

def assign_season(row, base_idx=30):
    year = int(row['Year'])
    week = int(row['Week'])
    return year if week >= base_idx else year - 1

# Estimate one hockey-stick breakpoint per season from the detected seasonal peak window.
def hockey_stick_regression(data, epi, HockeyStick_type, years):
    from scipy.signal import find_peaks
    rows = []
    info_by_year = {}

    peak_period = []
    peak_date = []
    for season in years:
        season_data = data[data['Season'] == season].reset_index(drop=True)
        peaks = find_peaks(season_data[epi], distance=30)
        if len(peaks[0]) > 0:
            peak_idx = peaks[0][0]
        else:
            peak_idx = int(season_data[epi].idxmax())
        peak_period.append(peak_idx + 1)
        peak_date.append(season_data.iloc[peak_idx]['Date'])
    peak_df = data[data['Date'].isin(peak_date)].sort_values('Year').reset_index(drop=True)

    for i, yr in enumerate(years):
        row = peak_df.loc[peak_df['Season'] == yr, ['Season', 'Week']].iloc[0]
        target_year = int(row['Season'])
        target_week = int(row['Week'])
    
        d = Observation_period(data, target_year, target_week, peak_period[i])

        k, info = Run_HockeyStick(d, epi, min_seg_len=4, ty=HockeyStick_type)
        if pd.isna(k):
            continue
            
        rows.append({
            'Season':               yr,
            'peak_year':            target_year,
            'peak_week':            target_week,
            'hockey_break_year': int(d.iloc[int(k)]['Year']),
            'hockey_break_week': int(d.iloc[int(k)]['Week']),
            'b0':                   None if info is None else info['b0'],
            'b1':                   None if info is None else info['b1'],
            'SSE':                  None if info is None else info['sse'],
            'n_points':             len(d)
        })
        info_by_year[yr] = (d, info)

        info_by_year[yr] = (d, info)
    hockey_peak_df = pd.DataFrame(rows)

    break_dates_by_season = {}

    for index, row in hockey_peak_df.iterrows():
        year = row['hockey_break_year']
        week = row['hockey_break_week']
        
        match = data[(data['Year'] == year) & (data['Week'] == week)]['Date']
        if not match.empty:
            break_dates_by_season[int(row['Season'])] = pd.to_datetime(match.values[0])

    break_dates = [break_dates_by_season.get(int(year), pd.NaT) for year in years]

    return break_dates, hockey_peak_df

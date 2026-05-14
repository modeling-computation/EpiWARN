
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

# Compute season-wise cumulative deviation from the within-season mean.
def cumulative_sum(data, epi, season_start_week = 24):
    data["cusum"] = (data[epi] - data.groupby("Season")[epi].transform("mean")).groupby(data["Season"]).cumsum()
    return data

# Compute cumulative deviation from a rolling 3-season baseline.
def cumulative_sum_3years(data, epi, season_start_week = 24, n_years = 3):
    season_mean = data.groupby("Season")[epi].mean()
    season_baseline = (season_mean.shift(1).rolling(window=n_years, min_periods=n_years).mean())

    data = data.merge(season_baseline.rename("baseline"), left_on="Season", right_index=True, how="left")
    data["diff"] = data[epi] - data["baseline"]
    data["cusum"] = (data.groupby("Season")["diff"].cumsum())
    return data

# Use within-season CUSUM for complete seasons and a rolling baseline for incomplete seasons.
def cumulative_sum_hybrid(data, epi, season_start_week=24, n_years=3, full_season_weeks=52):
    result = data.copy()
    result["cusum"] = np.nan

    count_col = "Date" if "Date" in result.columns else "Week"
    season_counts = result.groupby("Season")[count_col].nunique()
    full_seasons = season_counts[season_counts >= full_season_weeks].index.tolist()
    partial_seasons = season_counts[season_counts < full_season_weeks].index.tolist()

    if full_seasons:
        full_mask = result["Season"].isin(full_seasons)
        full_result = cumulative_sum(result.loc[full_mask].copy(), epi, season_start_week=season_start_week)
        result.loc[full_mask, "cusum"] = full_result["cusum"].values

    if partial_seasons:
        partial_result = cumulative_sum_3years(
            result.copy(),
            epi,
            season_start_week=season_start_week,
            n_years=n_years
        )
        partial_mask = result["Season"].isin(partial_seasons)
        result.loc[partial_mask, "cusum"] = partial_result.loc[partial_mask, "cusum"].values

    return result

def cumulative_sum_adaptive(data, epi, mode='standard', season_start_week=24, n_years=3, full_season_weeks=52):
    if mode == 'standard':
        return cumulative_sum_hybrid(
            data,
            epi,
            season_start_week=season_start_week,
            n_years=n_years,
            full_season_weeks=full_season_weeks
        )

    result = data.copy()
    result['cusum'] = np.nan
    result['baseline'] = np.nan
    result['diff'] = np.nan

    count_col = "Date" if "Date" in result.columns else "Week"
    season_counts = result.groupby('Season')[count_col].nunique()
    season_means = result.groupby('Season')[epi].mean()
    seasons = sorted(result['Season'].dropna().unique())

    for season in seasons:
        season_mask = result['Season'] == season
        is_complete = season_counts.get(season, 0) >= full_season_weeks

        if is_complete:
            baseline = season_means.loc[season]
        else:
            previous_seasons = [prev for prev in seasons if prev < season]
            previous_mean = season_means.loc[previous_seasons[-1]] if previous_seasons else np.nan
            baseline = previous_mean if pd.notna(previous_mean) else season_means.loc[season]

        result.loc[season_mask, 'baseline'] = baseline
        result.loc[season_mask, 'diff'] = result.loc[season_mask, epi] - baseline
        result.loc[season_mask, 'cusum'] = result.loc[season_mask, 'diff'].cumsum()

    return result

def window_sample(data, itv, epi):
    scaler = MinMaxScaler()
    temp_df = {}
    if len(data) < itv:
        return {}
        
    for i in range(len(data) - itv + 1):
        temp = data.iloc[i:i+itv].copy()
        temp.reset_index(drop=True, inplace=True)
        # 기울기용 Scaled Data
        temp['N_total'] = scaler.fit_transform(temp[[epi]])
        key = i
        temp_df[key] = temp
    return temp_df

# Fit a linear trend inside each window and summarize the window-level CUSUM signal.
def window_sample_feature(df_dict):
    import numpy as np
    from sklearn.linear_model import LinearRegression

    temp_lr = {}
    for key, temp in df_dict.items():
        temp_lr[key] = {}

        lr = LinearRegression()
        X_data = temp.num.values.reshape(-1, 1)
        y_data = temp.N_total.values
        lr.fit(X_data, y_data)
        lr_pred = lr.predict(X_data)
        temp_lr[key]['LR'] = lr
        temp_lr[key]['LR_val'] = lr_pred

        temp_lr[key]['CS'] = temp['cusum'].values
        temp_lr[key]['CS_mean'] = np.mean(temp['cusum'].values)

    return temp_lr

# Convert each rolling window into the feature set used by clustering.
def make_df(data, itv, epi): 
    dic1 = window_sample(data, itv, epi)
    dic2 = window_sample_feature(dic1)
    
    cols = ['data_num','mean', 'slope', 'CS_mean']
    rows = []
    
    for key in dic1.keys():
        data_tmp = dic1[key]
        add = [
            key,
            data_tmp[epi].mean(),
            dic2[key]['LR'].coef_[0],
            dic2[key]['CS_mean'],
        ]
        rows.append(add)
        
    features = pd.DataFrame(rows, columns=cols)
    return features
# Build feature windows for either the fitting period or a season-specific test period.
def make_raw(data, set_type, sample_window, epi, target_season=None):
    if set_type in ['train', 'analysis', 'all']:
        if set_type in ['analysis', 'all']:
            data_df = data.copy()
        else:
            data_df = data[data['set'] == set_type].copy()
    elif set_type == 'test':
        if target_season is None:
            test_idx = data.index[data['set'] == 'test']
        else:
            test_idx = data.index[(data['set'] == 'test') & (data['Season'] == target_season)]

        if len(test_idx) == 0:
            return pd.DataFrame(), pd.DataFrame()

        start_idx = max(0, test_idx.min() - sample_window + 1)
        end_idx = test_idx.max()
        data_df = data.iloc[start_idx:end_idx + 1].copy()
    else:
        raise ValueError("set_type must be one of ['train', 'test', 'analysis', 'all']")
    n = sample_window

    if 2020 in np.unique(data['Season']):
        df = make_df(data_df,n,epi)
        data_all = data_df[n-1:].reset_index(drop=True)
    else:
        data_bp = data_df[data_df['Season']<=2020]
        data_ap = data_df[data_df['Season']>2020]

        df1 = make_df(data_bp,n,epi)
        df2 = make_df(data_ap,n,epi)

        valid_data_frames = [frame for frame in [data_bp[n-1:], data_ap[n-1:]] if not frame.empty]
        if valid_data_frames:
            data_all = pd.concat(valid_data_frames, ignore_index=True)
        else:
            data_all = pd.DataFrame(columns=data_df.columns)

        valid_feature_frames = [frame for frame in [df1, df2] if not frame.empty]
        if valid_feature_frames:
            df = pd.concat(valid_feature_frames, ignore_index=True)
        else:
            df = pd.DataFrame(columns=['data_num', 'mean', 'slope', 'CS_mean'])
    
    df.reset_index(drop=True, inplace=True)
    df['data_num'] = df.index

    return df, data_all

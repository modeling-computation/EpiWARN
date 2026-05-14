
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
import joblib

def _warning_run_detection_indices(result_data, warning_label=1, min_run_length=3):
    if result_data.empty:
        return []

    labels = result_data['label'].astype(str).tolist()
    warning_label = str(warning_label)
    detection_indices = []
    run_start = None
    run_length = 0

    for idx, label in enumerate(labels):
        if label == warning_label:
            if run_start is None:
                run_start = idx
                run_length = 1
            else:
                run_length += 1
        elif run_start is not None:
            if run_length >= min_run_length:
                detection_indices.append(run_start + min_run_length - 1)
            run_start = None
            run_length = 0

    if run_start is not None and run_length >= min_run_length:
        detection_indices.append(run_start + min_run_length - 1)

    return detection_indices

def extract_seasonal_detection_dates(result_data, data_all, warning_label=1, min_run_length=3):
    if result_data.empty or data_all.empty:
        return pd.DataFrame(columns=['Season', 'detect_date'])

    limit = min(len(result_data), len(data_all))
    aligned_result = result_data.iloc[:limit].reset_index(drop=True).copy()
    aligned_data = data_all.iloc[:limit].reset_index(drop=True).copy()

    detection_indices = _warning_run_detection_indices(
        aligned_result,
        warning_label=warning_label,
        min_run_length=min_run_length
    )
    if not detection_indices:
        return pd.DataFrame(columns=['Season', 'detect_date'])

    detected_rows = aligned_data.loc[detection_indices, ['Date', 'Season']].copy()
    detected_rows['detect_date'] = pd.to_datetime(detected_rows['Date'], errors='coerce').dt.normalize()
    detected_rows = detected_rows.dropna(subset=['detect_date', 'Season'])

    if detected_rows.empty:
        return pd.DataFrame(columns=['Season', 'detect_date'])

    detected_rows['Season'] = detected_rows['Season'].astype(int)
    seasonal_dates = (
        detected_rows.groupby('Season', as_index=False)['detect_date']
        .min()
        .sort_values('Season')
        .reset_index(drop=True)
    )
    return seasonal_dates

# Find the first sustained high-risk run and convert it to season-specific warning dates.
def find_warning_periods(result_data, data_all_train, outbreak_season, warning_label=1):
    seasonal_dates = extract_seasonal_detection_dates(
        result_data,
        data_all_train,
        warning_label=warning_label
    )
    return seasonal_dates['detect_date'].tolist()

def summarize_detection_progression(detect_dates, monitoring_start=None, orange_threshold=0.05, red_threshold=0.10):
    detect_series = pd.to_datetime(pd.Series(detect_dates), errors='coerce').dt.normalize()
    total_models = len(detect_series)

    empty_timeline = pd.DataFrame(columns=['Date', 'Cumulative_Count', 'Cumulative_Ratio', 'Level'])
    empty_summary = {
        'blue_date': None,
        'orange_date': None,
        'red_date': None,
        'final_ratio': 0.0,
        'final_count': 0,
        'total_models': total_models,
    }

    if total_models == 0:
        return empty_timeline, empty_summary

    monitoring_start = pd.to_datetime(monitoring_start).normalize() if monitoring_start is not None else None
    valid_dates = detect_series.dropna().sort_values()
    initial_count = 0
    post_start_dates = valid_dates

    if monitoring_start is not None:
        initial_count = int((valid_dates < monitoring_start).sum())
        post_start_dates = valid_dates[valid_dates >= monitoring_start]

    counts = post_start_dates.value_counts().sort_index()
    rows = []
    cumulative_count = initial_count

    if monitoring_start is not None and initial_count > 0 and monitoring_start not in counts.index:
        initial_ratio = cumulative_count / total_models
        rows.append({
            'Date': monitoring_start,
            'Cumulative_Count': cumulative_count,
            'Cumulative_Ratio': initial_ratio,
            'Level': _classify_alert_level(initial_ratio, orange_threshold, red_threshold)
        })

    for date, count in counts.items():
        cumulative_count += int(count)
        cumulative_ratio = cumulative_count / total_models
        rows.append({
            'Date': pd.to_datetime(date).normalize(),
            'Cumulative_Count': cumulative_count,
            'Cumulative_Ratio': cumulative_ratio,
            'Level': _classify_alert_level(cumulative_ratio, orange_threshold, red_threshold)
        })

    if not rows:
        return empty_timeline, empty_summary

    timeline = pd.DataFrame(rows).sort_values('Date').reset_index(drop=True)

    blue_rows = timeline[timeline['Level'] == 'blue']
    blue_date = blue_rows.iloc[0]['Date'] if not blue_rows.empty else None

    orange_rows = timeline[timeline['Level'] == 'orange']
    orange_date = orange_rows.iloc[0]['Date'] if not orange_rows.empty else None

    red_rows = timeline[timeline['Level'] == 'red']
    red_date = red_rows.iloc[0]['Date'] if not red_rows.empty else None

    summary = {
        'blue_date': blue_date,
        'orange_date': orange_date,
        'red_date': red_date,
        'final_ratio': float(timeline.iloc[-1]['Cumulative_Ratio']),
        'final_count': int(timeline.iloc[-1]['Cumulative_Count']),
        'total_models': total_models,
    }
    return timeline, summary

def _classify_alert_level(ratio, orange_threshold=0.05, red_threshold=0.10):
    if ratio >= red_threshold:
        return 'red'
    if ratio >= orange_threshold:
        return 'orange'
    if ratio > 0:
        return 'blue'
    return 'none'

# Fit K-means on the training features and reorder labels by risk intensity.
def K_means_clustering(df):
    X_train = df[['slope', 'mean', 'CS_mean']].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    model = KMeans(random_state=727, n_init=10)
    
    visualizer = KElbowVisualizer(model, k=(2, 10), metric='silhouette', timings=True)
    visualizer.fit(X_scaled)
    visualizer.finalize()
    k = visualizer.elbow_value_
    plt.close()
    
    kmeans = KMeans(n_clusters=k, random_state=727, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    df['label'] = labels
    
    RI_means = df.groupby('label')['slope'].mean()
    label_order = RI_means.sort_values().index
    custom_order = {old_label: new_label for new_label, old_label in enumerate(label_order)}
    df['label'] = df['label'].map(custom_order)
    
    result_data = df.sort_values('data_num')
    result_data.reset_index(drop=True, inplace=True)
    result_data['label'] = result_data['label'].astype(str)

    return result_data, kmeans, k, scaler

def train_bootstrap_ensemble(df_train, scaler, feature_cols, B=300, k_best=2, types='random'):
    print(f"Training Ensemble of {B} models ({types})...")

    T = len(df_train)
    indices = np.arange(T)
    boot_ensemble = []

    if types == 'random':
        sampler = [(np.random.choice(indices, T, replace=True),) for _ in range(B)]
    elif types == 'sb':
        ### acf 코드 추가 필요
        sb = StationaryBootstrap(11, indices)
        sampler = (sample[0] for sample in sb.bootstrap(B))
    elif types == 'cbb':
        ### acf 코드 추가 필요
        cbb = CircularBlockBootstrap(12, indices)
        sampler = (sample[0] for sample in cbb.bootstrap(B))
    else:
        raise ValueError("types must be one of ['random', 'sb', 'cbb']")

    for b, boot_idx in enumerate(sampler):

        resampled_df = df_train.iloc[boot_idx].copy()
        X_resamp = scaler.transform(resampled_df[feature_cols])
        kmeans = KMeans(n_clusters=k_best, random_state=b, n_init=10)
        resampled_df['temp_label'] = kmeans.fit_predict(X_resamp)
        RI_means = resampled_df.groupby('temp_label')['slope'].mean()
        label_order = RI_means.sort_values().index
        custom_order = {old: new for new, old in enumerate(label_order)}
        boot_ensemble.append({
            'model': kmeans,
            'order': custom_order
        })

    return boot_ensemble, scaler

# Apply the bootstrap ensemble to the training set to summarize alert-date variability.
def analyze_train_distribution(df_train, data_all_train, feature_cols, boot_ensemble, scaler, outbreak_season, ED_date):
    X_orig_scaled = scaler.transform(df_train[feature_cols])
    detection_rows = []
    label_df = pd.DataFrame()
    
    for item in boot_ensemble:
        model = item['model']
        order = item['order']
        preds = model.predict(X_orig_scaled)
        temp_df = df_train.copy()
        temp_df['label'] = pd.Series(preds).map(order).values
        seasonal_dates = extract_seasonal_detection_dates(
            temp_df,
            data_all_train,
            warning_label=temp_df['label'].max()
        )
        row = {int(season): detect_date for season, detect_date in seasonal_dates[['Season', 'detect_date']].itertuples(index=False)}
        detection_rows.append(row)

        label_df = pd.concat([label_df, temp_df['label'].to_frame().T], ignore_index=True)

    date_df = pd.DataFrame(detection_rows)
    if not date_df.empty:
        ordered_cols = sorted(date_df.columns)
        date_df = date_df.reindex(columns=ordered_cols)
        for col in date_df.columns:
            date_df[col] = pd.to_datetime(date_df[col], errors='coerce').dt.normalize()

    return date_df, label_df

# def analyze_distribution_with_bootstrap(df_input, data_all_input, feature_cols, boot_ensemble, scaler, outbreak_season):
#     X_input_scaled = scaler.transform(df_input[feature_cols])
#     date_df = pd.DataFrame()
#     label_df = pd.DataFrame()
#
#     for item in boot_ensemble:
#         model = item['model']
#         order = item['order']
#         preds = model.predict(X_input_scaled)
#         temp_df = df_input.copy()
#         temp_df['label'] = pd.Series(preds).map(order).values
#         ed_dates = find_warning_periods(temp_df, data_all_input, outbreak_season, warning_label=temp_df['label'].max())
#
#         tmp = pd.DataFrame({"date": pd.to_datetime(ed_dates, errors='coerce')}).dropna(subset=["date"])
#         if not tmp.empty:
#             tmp["year"] = tmp["date"].dt.year.astype(str)
#             tmp["idx"] = tmp.groupby(tmp["date"].dt.year).cumcount() + 1
#             tmp["col"] = tmp["year"] + "_" + tmp["idx"].astype(str)
#             row = tmp.set_index("col")["date"].to_frame().T
#             date_df = pd.concat([date_df, row], ignore_index=True)
#
#         label_df = pd.concat([label_df, temp_df['label'].to_frame().T], ignore_index=True)
#
#     return date_df, label_df

# Apply the trained bootstrap ensemble to any retrospective period without retraining.
def analyze_distribution_with_bootstrap(df_input, data_all_input, feature_cols, boot_ensemble, scaler, outbreak_season):
    X_input_scaled = scaler.transform(df_input[feature_cols])
    detection_rows = []
    label_df = pd.DataFrame()

    for item in boot_ensemble:
        model = item['model']
        order = item['order']
        preds = model.predict(X_input_scaled)
        temp_df = df_input.copy()
        temp_df['label'] = pd.Series(preds).map(order).values
        seasonal_dates = extract_seasonal_detection_dates(
            temp_df,
            data_all_input,
            warning_label=temp_df['label'].max()
        )
        row = {int(season): detect_date for season, detect_date in seasonal_dates[['Season', 'detect_date']].itertuples(index=False)}
        detection_rows.append(row)

        label_df = pd.concat([label_df, temp_df['label'].to_frame().T], ignore_index=True)

    date_df = pd.DataFrame(detection_rows)
    if not date_df.empty:
        ordered_cols = sorted(date_df.columns)
        date_df = date_df.reindex(columns=ordered_cols)
        for col in date_df.columns:
            date_df[col] = pd.to_datetime(date_df[col], errors='coerce').dt.normalize()

    return date_df, label_df

# Apply the trained ensemble to each incremental real-time step.
def predict_new_data_probability(df_test, data_all_test, boot_ensemble, scaler, outbreak_season, step, initial_detection_dates=None):
    incremental_detection_results = {}
    incremental_prob_results = {}
    feature_cols = ['slope', 'mean', 'CS_mean']

    X_test = scaler.transform(df_test[feature_cols])
    test_dates_objects = pd.to_datetime(data_all_test['Date'].values).normalize()
    target_season = int(pd.Series(data_all_test['Season']).dropna().mode().iloc[0]) if 'Season' in data_all_test.columns and not data_all_test.empty else None

    total_models = len(boot_ensemble)
    initial_series = pd.Series([pd.NaT] * total_models, dtype='datetime64[ns]')
    if initial_detection_dates is not None:
        init_values = pd.to_datetime(pd.Series(initial_detection_dates), errors='coerce').dt.normalize()
        usable_len = min(total_models, len(init_values))
        initial_series.iloc[:usable_len] = init_values.iloc[:usable_len].values

    binary_prediction_rows = []
    for item in boot_ensemble:
        model = item['model']
        order = item['order']

        raw_preds = model.predict(X_test)
        mapped_preds = pd.Series(raw_preds).map(order).values
        binary_preds = (mapped_preds == (len(order) - 1)).astype(int)
        binary_prediction_rows.append(binary_preds)

    prob_rows = []
    final_detection_dates = []
    
    max_idx = len(df_test)
    for end_idx in range(step, max_idx + 1, step):
        current_df = df_test.iloc[:end_idx].reset_index(drop=True)
        current_data = data_all_test.iloc[:end_idx].reset_index(drop=True)
        
        last_date_val = test_dates_objects[end_idx - 1]
        last_date_str = last_date_val.strftime('%Y-%m-%d')
        
        detection_dates = []
        
        for model_idx, binary_preds in enumerate(binary_prediction_rows):
            temp_df = current_df.copy()
            temp_df['label'] = binary_preds[:end_idx]

            seasonal_dates = extract_seasonal_detection_dates(
                temp_df,
                current_data,
                warning_label=1
            )
            sim_date = pd.NaT
            if target_season is not None and not seasonal_dates.empty:
                matched = seasonal_dates.loc[seasonal_dates['Season'] == target_season, 'detect_date']
                if not matched.empty:
                    sim_date = matched.iloc[0]

            initial_date = initial_series.iloc[model_idx]
            if pd.notna(initial_date) and (pd.isna(sim_date) or initial_date <= sim_date):
                detection_dates.append(initial_date)
            else:
                detection_dates.append(sim_date)
        
        incremental_detection_results[last_date_str] = detection_dates

        cumulative_count = int(sum(pd.notna(date) and date <= last_date_val for date in detection_dates))
        cumulative_ratio = (cumulative_count / total_models) if total_models > 0 else 0.0
        prob_rows.append({
            'Date': last_date_val,
            'Warning_Probability': cumulative_ratio
        })

        timeline_df, _ = summarize_detection_progression(
            detection_dates,
            monitoring_start=test_dates_objects[0] if len(test_dates_objects) > 0 else None
        )
        incremental_prob_results[last_date_str] = timeline_df

        if end_idx == max_idx:
            final_detection_dates = detection_dates
    
    prob_df = pd.DataFrame(prob_rows)
    date_df = pd.DataFrame({'Detect_date': final_detection_dates})
    iteration_results = pd.DataFrame(incremental_detection_results)
    
    return prob_df, date_df, iteration_results, incremental_prob_results

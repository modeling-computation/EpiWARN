import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

# Import the modules used across preprocessing, modeling, and visualization.
try:
    from config import Config
    from src.preprocessing import (
        make_raw,
        cumulative_sum_adaptive,
        load_data,
        format_alert_date,
        drop_warmup_detection_columns,
    )
    from src.clustering import (
        K_means_clustering, 
        find_warning_periods,
        train_bootstrap_ensemble, 
        analyze_train_distribution,
        summarize_detection_progression,
        optimize_window_size,
    )
    from src.visualization import (
        visualization_season,
        render_analysis_report,
    )
    from src.season_setting import (
        set_season_start_week_adaptive,
        hockey_stick_regression,
        assign_retrospective_period,
        extend_standard_seasons_with_leading_history,
    )
except ImportError as e:
    st.error(f"Failed to import modules from src folder.\nError: {e}")
    st.stop()

st.set_page_config(
    page_title="EpiWARN",
    layout="wide",
    initial_sidebar_state="expanded"
)

with st.sidebar:
    st.title("Settings")
    
    st.header("1. Data Upload")
    uploaded_file = st.file_uploader("Upload Excel File (.xlsx)", type=["xlsx", "xls"])
    
    raw_data = load_data(uploaded_file, getattr(Config, 'DATA_PATH', None))
    
    if raw_data is not None:
        all_cols = raw_data.columns.tolist()
        
        # Infer a sensible default target column from common disease-related column names.
        disease_keywords = ['ili', 'noro', 'hfmd', 'hrsv', 'covid', 'flu', 'patient', 'cases']
        
        target_default_idx = 0
        found = False
        
        for i, col in enumerate(all_cols):
            col_lower = str(col).lower()
            if any(keyword in col_lower for keyword in disease_keywords):
                target_default_idx = i
                found = True
                break
        
        if not found and len(all_cols) > 1:
            target_default_idx = 1
                
        target_col = st.selectbox("Target Column (EPI_COL)", all_cols, index=target_default_idx, key="target_col_select")
        
        date_default_idx = 0
        if 'Date' in all_cols:          
            date_default_idx = all_cols.index('Date')
        elif 'date' in all_cols:        
            date_default_idx = all_cols.index('date')
        elif '일자' in all_cols:        
            date_default_idx = all_cols.index('일자')
        else:
            candidates = [c for c in all_cols if 'date' in str(c).lower()]
            if candidates:
                date_default_idx = all_cols.index(candidates[0])

        date_col = st.selectbox("Date Column", all_cols, index=date_default_idx, key="date_col_select")        
        temp_dates = pd.to_datetime(raw_data[date_col], errors='coerce')
        if temp_dates.notna().sum() == 0:
            st.warning("No valid date data found in the selected column.")
            st.stop()
            
        raw_data[date_col] = temp_dates
        raw_data = raw_data.dropna(subset=[date_col])
        
        if raw_data.empty:
            st.error("No valid data available.")
            st.stop()
    else:
        st.warning("No data loaded.")
        st.stop()
    manual_start_week = 1

    # st.markdown("---")
    # st.header("2. Reference Dates(Optional)")

    # reference_file = st.file_uploader(
    #     "Upload Reference Date File (.xlsx)",
    #     type=["xlsx", "xls"],
    #     key="reference_date_file"
    # )
    reference_file = None
    reference_dates = None
    reference_label = None

    if reference_file is not None:
        reference_data = pd.read_excel(reference_file)
        reference_cols = reference_data.columns.tolist()

        reference_date_default_idx = 0
        if 'Date' in reference_cols:
            reference_date_default_idx = reference_cols.index('Date')
        elif 'date' in reference_cols:
            reference_date_default_idx = reference_cols.index('date')
        else:
            reference_candidates = [c for c in reference_cols if 'date' in str(c).lower()]
            if reference_candidates:
                reference_date_default_idx = reference_cols.index(reference_candidates[0])

        reference_date_col = st.selectbox(
            "Reference Date Column",
            reference_cols,
            index=reference_date_default_idx,
            key="reference_date_col_select"
        )

        reference_dates = pd.to_datetime(reference_data[reference_date_col], errors='coerce').dropna().tolist()
        reference_label = f"User Input Date ({Path(reference_file.name).stem})"

        if len(reference_dates) == 0:
            st.warning("No valid dates were found in the uploaded reference date file.")

    st.markdown("---")
    st.header("2. Parameters")
    
    boot_num = st.number_input("Repeat Runs", 50, 2000, 200, step=50, key="boot_num_input")
    HockeyStick_type = "linear"
    
    st.markdown("---")
    run_btn = st.button("Run Analysis", type="primary")

st.markdown("""
<div style="margin: 6px 0 18px 0; line-height: 1.08;">
    <div style="font-size: 52px; font-weight: 900; color: #2c3e50; letter-spacing: 0;">
        <span style="color: #1f77b4; font-weight: 900;">Epi</span><span style="color: #ff7f0e; font-weight: 900;">WA</span><span style="color: #b91c1c; font-weight: 900;">RN</span>
    </div>
    <div style="font-size: 34px; font-weight: 500; color: #2c3e50; letter-spacing: 0; margin-top: 6px;">
        : <span style="font-weight: 900;">Epi</span>demic
        <span style="font-size: 1.12em; font-weight: 900;">W</span>arning
        simul<span style="font-size: 1.12em; font-weight: 900;">A</span>tion
        for <span style="font-size: 1.12em; font-weight: 900;">R</span>eal-time
        transmission dy<span style="font-size: 1.12em; font-weight: 900;">N</span>amics
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="padding: 20px 24px; margin-bottom: 12px; color: #000000; background-color: #f0f2f6; border-radius: 10px; border-top: 1px solid #d9d9d9;">
    <div style="font-size: 24px; font-weight: 800; line-height: 1.35;">[Analysis Report]</div>
    <div style="font-size: 20px; line-height: 1.6; margin-top: 6px;">
        Upload your data, configure the settings, and run the analysis to view the results.<br>
        This tab shows early warning signals in seasonal time-series data and helps you check when seasonal activity begins and how patterns change over time.
    </div>
    <br>
    <div style="font-size: 24px; font-weight: 800; line-height: 1.35;">[Setup Guide]</div>
    <div style="font-size: 20px; line-height: 1.6; margin-top: 6px;">
        If this is your first time using the dashboard, start here.<br>
        This tab explains what the dashboard is for, how to run the analysis, and how to read the results.
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<style>
button[data-baseweb="tab"] {
    padding-top: 0.85rem !important;
    padding-bottom: 0.85rem !important;
}
button[data-baseweb="tab"] > div[data-testid="stMarkdownContainer"] > p {
    font-size: 1.7rem !important;
    font-weight: 800 !important;
    line-height: 1.2 !important;
}
button[data-baseweb="tab"][aria-selected="true"] > div[data-testid="stMarkdownContainer"] > p {
    font-size: 1.75rem !important;
    font-weight: 800 !important;
}
[data-testid="stTabs"] button[data-baseweb="tab"] {
    padding-top: 0.75rem !important;
    padding-bottom: 0.75rem !important;
}
</style>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Analysis Report", "Setup Guide"])
with tab1:
    if run_btn:
        status = st.status("Preprocessing...", expanded=True)
        
        try:
            proc_data = raw_data.copy()
            proc_data['Date'] = proc_data[date_col]
            if 'Year' not in proc_data.columns:
                proc_data['Year'] = proc_data['Date'].dt.year
            if 'Week' not in proc_data.columns:
                proc_data['Week'] = proc_data['Date'].dt.isocalendar().week
            full_period_data = proc_data.copy()
            status.update(label="Preprocessing...", state="running", expanded=True)
            season_df, season_meta = set_season_start_week_adaptive(proc_data, target_col)
            if season_meta.get('mode') == 'insufficient':
                st.error(season_meta.get('reason', 'At least 1 year of data is required.'))
                st.stop()

            if season_df.empty:
                st.error("No valid seasons were detected from the selected data.")
                st.stop()

            detected_seasons = season_df['season'].to_list()
            visual_peak_start, peak_len = visualization_season(proc_data, season_df, start_week=manual_start_week)
            if season_meta.get('mode') == 'standard':
                peak_start = visual_peak_start
                seasons = extend_standard_seasons_with_leading_history(
                    proc_data,
                    detected_seasons,
                    peak_start,
                    target_col,
                    season_df
                )
                season_starts = None
            else:
                peak_start = season_meta.get('start_week')
                if peak_start is None:
                    st.error("No valid short-history season start week was detected.")
                    st.stop()
                seasons = detected_seasons
                season_starts = season_df

            data, period_meta = assign_retrospective_period(
                proc_data,
                seasons,
                start_week=peak_start,
                season_starts=season_starts
            )
            window_eval_seasons = period_meta['window_eval_seasons']
            if window_eval_seasons:
                hockey_date, hockey_df = hockey_stick_regression(data, target_col, HockeyStick_type, window_eval_seasons)
            else:
                hockey_date, hockey_df = [], pd.DataFrame()
                st.warning("No complete season was available for window-size optimization. A default 12-week window was used.")
            cusum_result = cumulative_sum_adaptive(
                data.copy(),
                target_col,
                mode=season_meta.get('mode', 'standard'),
                season_start_week=peak_start
            )
            data['cusum'] = cusum_result['cusum'].values
            
            data.reset_index(drop=True, inplace=True)
            data['num'] = data.index
            
            proc_data = data.copy()

            status.update(label="Analyzing...", state="running", expanded=True)
            if window_eval_seasons:
                best_window, best_score = optimize_window_size(
                    proc_data,
                    target_col,
                    hockey_date,
                    window_eval_seasons,
                    peak_start
                )
            else:
                best_window, best_score = 12, np.inf
            st.toast(f"Optimal Window Size Auto-selected: {best_window} Weeks")

            feature_col = ['slope', 'mean', 'CS_mean']
            df_analysis, data_all_analysis = make_raw(proc_data, 'analysis', best_window, target_col)
            valid_analysis_mask = df_analysis[feature_col].notna().all(axis=1)
            df_analysis = df_analysis.loc[valid_analysis_mask].reset_index(drop=True)
            data_all_analysis = data_all_analysis.loc[valid_analysis_mask].reset_index(drop=True)

            if df_analysis.empty or data_all_analysis.empty:
                st.error("No valid analysis windows were available after feature calculation.")
                st.stop()

            status.update(label="Analyzing...", state="running", expanded=True)

            result_data_t, kmeans, best_k, scaler = K_means_clustering(df_analysis)
            warning_label_t = result_data_t['label'].max()
            ED_date = find_warning_periods(result_data_t, data_all_analysis, peak_start, warning_label_t)

            boot_ensemble, scaler = train_bootstrap_ensemble(df_analysis, scaler, feature_col, B=boot_num, k_best=best_k, types='random')
            status.update(label="Analyzing...", state="running", expanded=True)

            status.update(label="Visualizing...", state="running", expanded=True)

            date_df, label_df = analyze_train_distribution(
                df_analysis,
                data_all_analysis,
                feature_col,
                boot_ensemble,
                scaler,
                peak_start,
                ED_date
            )
            warmup_seasons = period_meta.get('warmup_seasons', [])
            date_df_display = drop_warmup_detection_columns(date_df, warmup_seasons)

            season_summary_items = []
            for season in date_df_display.columns:
                _, summary = summarize_detection_progression(date_df_display[season])
                d_blue = format_alert_date(summary.get('blue_date'), fallback="-")
                d_orange = format_alert_date(summary.get('orange_date'), fallback="-")
                d_red = format_alert_date(summary.get('red_date'), fallback="-")
                season_summary_items.append({
                    'season': int(season),
                    'summary': summary,
                    'blue': d_blue,
                    'orange': d_orange,
                    'red': d_red,
                })

            analysis_period_text = (
                f"{period_meta['analysis_start'].strftime('%Y/%m/%d')} ~ "
                f"{period_meta['analysis_end'].strftime('%Y/%m/%d')}"
            )
            season_count_text = (
                f"{len(period_meta['analysis_seasons'])} seasons detected, "
                f"{len(window_eval_seasons)} complete seasons used for window optimization "
                f"({season_meta.get('mode', 'standard')} mode)"
            )
            warmup_text = ""
            if warmup_seasons:
                warmup_text = (
                    f"{len(warmup_seasons)} leading incomplete season "
                    "is used for modeling context but hidden from detection markers."
                )

            other_dates = None
            if reference_dates is not None and len(reference_dates) > 0:
                other_dates = {reference_label or 'User Input Date': reference_dates}

            st.session_state.pop("season_summary_multiselect_v3", None)
            st.session_state["analysis_result"] = {
                "full_period_data": full_period_data,
                "proc_data": proc_data,
                "data_all_analysis": data_all_analysis,
                "target_col": target_col,
                "other_dates": other_dates,
                "hockey_date": hockey_date,
                "date_df_display": date_df_display,
                "best_window": best_window,
                "season_summary_items": season_summary_items,
                "analysis_period_text": analysis_period_text,
                "season_count_text": season_count_text,
                "warmup_text": warmup_text,
            }
            status.update(label="Analysis Complete", state="complete", expanded=False)
            st.rerun()
        except Exception as e:
            status.update(label="Analysis Error", state="error", expanded=True)
            st.error(f"Details: {e}")
            st.exception(e)

    result = st.session_state.get("analysis_result")
    if result is None:
        st.markdown("---")
        st.header("Analysis Report")
        st.info("Please review the settings in the left sidebar, then click Run Analysis to view the Analysis Report.")
    else:
        render_analysis_report(result)

with tab2:
    st.markdown("<h2 style='font-size: 32px; font-weight: 800; color: #2c3e50; margin-bottom: 20px;'>Getting Started & Setup</h2>", unsafe_allow_html=True)
    st.markdown("""
    <style>
    div[data-testid="stExpander"] details summary p,
    div[data-testid="stExpander"] details summary span,
    [data-testid="stExpander"] summary p,
    [data-testid="stExpander"] summary span {
        font-size: 24px !important; 
        font-weight: 800 !important;
        color: #2c3e50 !important; /* Dark navy tone that matches the main heading */
    }
    /* Slightly enlarge the chevron icon to match the heading text size */
    [data-testid="stExpander"] summary svg {
        width: 24px !important;
        height: 24px !important;
        color: #2c3e50 !important;
    }
    </style>
    """, unsafe_allow_html=True)
    with st.expander("0. Introduction", expanded=True):
        col1, col2 = st.columns([0.7, 2.3], gap="large")
        
        with col1:
            import os
            import base64
            
            current_dir = os.path.dirname(os.path.abspath(__file__))
            img_path = os.path.join(current_dir, "images", "intro_example.png")
            
            try:
                with open(img_path, "rb") as image_file:
                    encoded_string = base64.b64encode(image_file.read()).decode()
                st.markdown(f"""
                <div style="
                    min-height: 500px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    padding: 5px 0 40px 0;
                ">
                    <img src="data:image/png;base64,{encoded_string}" style="
                        max-width: 88%;
                        max-height: 360px;
                        object-fit: contain;
                        border-radius: 8px;
                        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                    ">
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Image load error: {e}")

        with col2:
            st.markdown("""
            <div style="min-height: 500px; border-left: 5px solid #1f77b4; background-color: #f8fbff; padding: 20px 25px; border-radius: 0 8px 8px 0; margin-top: 5px; margin-bottom: 40px; display: flex; flex-direction: column; justify-content: center;">
                <div style="color: #000000; line-height: 1.6; font-size: 20px;">
                    <div style="font-size: 28px; font-weight: 800; margin-bottom: 5px;">
                        What this dashboard does
                    </div>
                    <div style="margin-left: 18px; margin-bottom: 26px;">
                        - Detects early warning signals from seasonal time-series data<br>
                        - Identifies when signal activity begins within a season
                    </div>
                    <div style="font-size: 28px; font-weight: 800; margin-bottom: 5px;">
                        How the analysis works
                    </div>
                    <div style="margin-left: 18px; margin-bottom: 26px;">
                        - The full retrospective period is used to learn a stable baseline pattern<br>
                        - <strong>At least 1 year</strong> of data is required, <strong>4 years</strong> or more uses the standard season algorithm<br>
                    </div>
                    <div style="font-size: 28px; font-weight: 800; margin-bottom: 5px;">
                        How to set it up
                    </div>
                    <div style="margin-left: 18px;">
                        - Upload your data, configure the settings, and click <strong>Run Analysis</strong><br>
                        - Review retrospective signal detection results in the <strong>Analysis Report</strong>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    with st.expander("1. Setup Guide", expanded=False):
        import os
        import base64
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        setting_img_path = os.path.join(current_dir, "images", "Setting_guide.png")
        
        try:
            with open(setting_img_path, "rb") as image_file:
                encoded_setting_img = base64.b64encode(image_file.read()).decode()
            st.markdown(f"""
                <div style="display: flex; justify-content: center; padding: 20px 0;">
                    <img src="data:image/jpeg;base64,{encoded_setting_img}" style="max-width: 100%; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Image load error: {e}")
            st.info("💡 'images' 폴더 안에 'Setting_guide.jpg' 파일이 있는지, 파일명 대소문자가 정확한지 확인해주세요!")

    with st.expander("2. Dashboard Analysis", expanded=False):
        st.markdown("""
        <div style="border: 2px dashed #ccc; border-radius: 8px; padding: 80px 20px; text-align: center; margin-bottom: 25px; margin-top: 15px; background-color: #fafafa;">
            <span style="font-size: 26px; color: #555; font-weight: bold;">Plot 1</span><br>
            <span style="font-size: 16px; color: #999;">(Insert your first plot code here later)</span>
        </div>
        
        <div style="border: 2px dashed #ccc; border-radius: 8px; padding: 80px 20px; text-align: center; margin-bottom: 15px; background-color: #fafafa;">
            <span style="font-size: 26px; color: #555; font-weight: bold;">Plot 2</span><br>
            <span style="font-size: 16px; color: #999;">(Insert your second plot code here later)</span>
        </div>
        """, unsafe_allow_html=True)

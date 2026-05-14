import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import ListedColormap
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def _level_dates_from_timeline(timeline_df):
    if timeline_df is None or timeline_df.empty:
        return {'blue': None, 'orange': None, 'red': None}

    blue_rows = timeline_df[timeline_df['Level'] == 'blue']
    orange_rows = timeline_df[timeline_df['Level'] == 'orange']
    red_rows = timeline_df[timeline_df['Level'] == 'red']

    return {
        'blue': blue_rows.iloc[0]['Date'] if not blue_rows.empty else None,
        'orange': orange_rows.iloc[0]['Date'] if not orange_rows.empty else None,
        'red': red_rows.iloc[0]['Date'] if not red_rows.empty else None,
    }

def _add_cumulative_detection_overlay(fig, timeline_df, secondary_y=True, legend_seen=None, showlegend=True):
    if timeline_df is None or timeline_df.empty:
        return {'blue': None, 'orange': None, 'red': None}

    thin_width = 3 * 24 * 60 * 60 * 1000
    legend_seen = legend_seen if legend_seen is not None else set()
    level_dates = _level_dates_from_timeline(timeline_df)

    level_specs = [
        ('blue', 'Caution', 'blue'),
        ('orange', 'Alert', 'orange'),
        ('red', 'Severe', 'red'),
    ]

    for level, name, color in level_specs:
        level_df = timeline_df[timeline_df['Level'] == level]
        if level_df.empty:
            continue

        fig.add_trace(
            go.Bar(
                x=level_df['Date'],
                y=level_df['Cumulative_Count'],
                name=name,
                marker_color=color,
                opacity=0.6,
                width=thin_width,
                showlegend=showlegend and (name not in legend_seen)
            ),
            secondary_y=secondary_y,
        )
        legend_seen.add(name)

    for level, color in [('blue', 'blue'), ('orange', 'orange'), ('red', 'red')]:
        date_val = level_dates[level]
        if pd.notnull(date_val):
            fig.add_vline(
                x=date_val,
                line_dash="dash",
                line_color=color,
                opacity=0.45,
                line_width=2,
                secondary_y=secondary_y
            )

    return level_dates

# Visualize the automatically detected seasonal windows and recover the aligned season start week.
def visualization_season(data, results_df, start_week = 1, half_year_weeks = 26):
    from matplotlib.colors import ListedColormap

    season_pattern = list(range(start_week, 54)) + list(range(1, start_week))
    season_weeks = season_pattern * 2
    TOTAL_WEEKS = len(season_weeks)

    seasons = sorted(results_df['season'])
    heatmap_rows = []
    for s in seasons:
        heatmap_rows.append(s)
        heatmap_rows.append(s)
        heatmap_rows.append(None)
    heatmap_df = pd.DataFrame(
        0,
        index=range(len(heatmap_rows)),
        columns=range(TOTAL_WEEKS)
    )

    week_to_idx_map = {w: i for i, w in enumerate(season_pattern)}
    def week_to_index(week, week_to_idx_map):
        return week_to_idx_map[week]

    peak_list = []
    old_epi_start = week_to_index(results_df.iloc[0,:].start_week, week_to_idx_map)
    
    for i, r in enumerate(results_df.itertuples()):
        s_row1 = i*3
        s_row2 = i*3 + 1
        
        peak_idx = week_to_index(r.peak_week, week_to_idx_map)
        old_epi_end = old_epi_start + 52
        epi_start = peak_idx - half_year_weeks-1
        if r.peak_week==53:
            epi_start = epi_start-1
        if epi_start < 0:
            epi_start += 53
            peak_idx += 53
        epi_end = min(TOTAL_WEEKS - 1, peak_idx + half_year_weeks-1)
        peak_list.append(peak_idx+1)

        heatmap_df.loc[s_row1, old_epi_start:old_epi_end] = 3
        if hasattr(r, 'epi_weeks') and r.epi_weeks is not None:
            for w in r.epi_weeks:
                w_idx = week_to_index(w, week_to_idx_map)
                if not (old_epi_start <= w_idx <= old_epi_end):
                    w_idx += 53
                heatmap_df.loc[s_row1, w_idx] = 4
        heatmap_df.loc[s_row1, peak_idx] = 2

        heatmap_df.loc[s_row2, epi_start:epi_end] = 1
        heatmap_df.loc[s_row2, peak_idx] = 2

        old_epi_start = epi_start

    cmap = ListedColormap([
        (1, 1, 1, 1.0),        # 0: white
        (0.4, 0.6, 0.9, 0.6),  # 1: steelblue
        (0.9, 0.3, 0.3, 0.8),  # 2: crimson (peak)
        (1.0, 0.7, 0.8, 0.5),  # 3: pink (epi_start~end)
        (1.0, 0.85, 0.0, 0.8)  # 4: gold (epi_weeks)
    ])
    fig, ax = plt.subplots(figsize=(20, 0.5 * len(heatmap_df)), dpi=300)
    ax.grid(False)
    ax.imshow(
        heatmap_df.values,
        aspect='auto',
        cmap=cmap,
        interpolation='none',
        alpha=1
    )

    xticks = list(range(TOTAL_WEEKS))
    xtick_labels = season_pattern * 2
    ax.set_xticks(range(0, len(xticks), 2))
    ax.set_xticklabels([str(w) for w in xtick_labels[::2]], fontsize=8)

    yticks = [i*3 + 0.5 for i in range(len(seasons))]
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{s}-{s+1} season" for s in seasons], fontsize=8)

    ax.set_yticks(np.arange(-0.5, len(heatmap_df), 1), minor=True)
    ax.set_xticks(np.arange(-0.5, TOTAL_WEEKS, 1), minor=True)
    ax.grid(which='minor', axis='both', linestyle='-', linewidth=0.5)

    special_y = (3 * np.arange(len(seasons))[:, None] + np.array([1.5, 2.5])).ravel()
    for y in special_y:
        ax.hlines(y, -0.5, heatmap_df.shape[1]-0.5, color='black', linewidth=1.0)
    
    ax.set_xlabel("Week")

    SEASON_WEEKS = 52

    peak_min = np.min(peak_list)
    peak_max = np.max(peak_list)
    peak_range = peak_max - peak_min
    tmp = np.ceil((SEASON_WEEKS - peak_range)/2)
    peak_start_idx = int(peak_min - tmp)

    peak_start_week = season_pattern[peak_start_idx % SEASON_WEEKS] - 1

    peak_value = [int(x - peak_start_idx + 1) for x in peak_list]
    peak_len = dict(zip(seasons, peak_value))

    ax.axvline(peak_start_idx-1, color='red', linestyle='--', alpha=0.7)
    ax.axvline(peak_start_idx+51, color='red', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    return int(peak_start_week), peak_len

# Plot the feature distributions for the K-means clusters.
def K_means_visualization(result_data, input_var, var_name): 
    result_df = result_data.copy()
    result_df['label'] = result_df['label'].astype(str)
    palette = {'0': "#348ABD", '1': "#A6D854", '2': "#D62728", '3': "#D62728"}
    order = np.sort(result_df['label'].unique())
    sns.set_style("whitegrid")

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    axes = axes.ravel()
    for i, col in enumerate(input_var):
        ax = axes[i]
        ax.grid(False)
        sns.boxplot(data=result_df, x='label', y=col, order=order, palette=palette, width=0.6, ax=ax, showfliers=True)
        ax.set_title(f'{var_name[i]}')
        ax.set_xlabel('Cluster')
        ax.set_ylabel(var_name[i])
        ax.set_xticklabels([f'C{c}' for c in order])
    for j in range(len(input_var), len(axes)):
        axes[j].remove()
    plt.tight_layout()
    plt.show()

# Compare training-period alerts from clustering, hockey-stick breakpoints, and optional reference dates.
def early_warning_visualization(data, data_all, epi, result_data, ED_date, Hockey_date, other_dates, sample_window):
    fig, ax = plt.subplots(figsize=(10, 2), dpi=400)

    x=data_all['Date']
    y=data_all[epi]
    max_y = y.max()

    ax.bar(data['Date'],data[epi],color='gray',alpha=0.5, label=epi, width=5.5)
    ax.axhline(y=max_y*1.3, color='black', linewidth=1)

    ax.set_xlabel('Week')
    ax.set_ylabel(epi)
    ax.grid(False)

    colors = ['orange', 'red']
    markers = ['^', '*']
    line_style = ['--', '-']
    if other_dates is not None:
        for j, key in enumerate(other_dates.keys()):
            for i, date in enumerate(other_dates[key]):
                ax.scatter(date, max_y*(1.4+0.1*j), color=colors[j], marker=markers[j], s=50, label=key if i==0 else "")
                ax.vlines(date, ymin=max_y*1.3, ymax=max_y*1.6, color=colors[j], linestyle=line_style[j], linewidth=2)

    for i, date in enumerate(Hockey_date):
        label = 'Hockey' if i == 0 else ""
        ax.scatter(date, max_y*1.2, color='green', marker='D', s=50, label=label)
        ax.vlines(x=date, ymin=0, ymax=max_y*1.3, color='green', linestyle='--', linewidth=2, alpha=0.7)

    for i, date in enumerate(ED_date):
        plt.scatter(date, max_y, color='blue', marker='o', s=50, label=f'Baseline clustering ($M_0$)' if i==0 else "")
        ax.vlines(date, ymin=0, ymax=max_y*1.3, color='blue', linestyle='-')

    ax.set_ylabel(epi, fontdict={'fontweight':'bold', 'fontsize':'12'})
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    ticks = ax.get_yticks()
    lower = max_y * 1.3
    upper = max_y * 1.6
    labels = ['' if (lower <= t <= upper) else f'{int(t)}' for t in ticks]
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    ax.set_ylim(0,max_y*1.6)
    ax.set_xlim(x.min() - pd.Timedelta(weeks = sample_window-1), x.max())
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.47), ncol=5, frameon = False)
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 3), dpi=400)

    ax.bar(data['Date'], data[epi], color='gray', alpha=0.5, label=epi, width=5.5)

    for i, date in enumerate(ED_date):
        plt.scatter(date, max_y*1.3, color='blue', marker='o', s=50, label=f'Baseline clustering ($M_0$)' if i==0 else "")
        ax.axvline(date, color='blue', linestyle='-')

    ax.set_xlabel('Week')
    ax.set_ylabel(epi)
    ax.grid(False)
    ax.set_xlim(x.min() - pd.Timedelta(weeks = sample_window-1), x.max())

    ax2 = ax.twinx()

    plot_dt =  pd.date_range(start=x.min(), end=x.max(), freq='W-SUN')
    y_series = pd.Series(data=result_data['label'].values, index=x.values)
    y_plot = y_series.reindex(plot_dt, fill_value=np.nan).astype(float)
    ax2.plot(plot_dt, y_plot.values, color='black', label='Label',  linewidth=3)
    
    ax2.grid(False)
    ax2.set_ylabel('Cluster', rotation=270, labelpad=15)
    order = np.sort(result_data['label'].astype(int).unique())
    ax2.set_yticks(order)
    ax2.set_yticklabels([f'C{c}' for c in order])

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc='lower center', bbox_to_anchor=(0.5, -0.47),
            ncol=5, frameon=False, fontsize=12)

    plt.tight_layout()
    plt.show()

    return 
#         '0': "#348ABD", '1': "#A6D854", '2': "#D62728", '3': "#D62728"
#     }
#     order = np.sort(result_data['label'].unique())
#     sns.set_style("whitegrid")

#     fig, axes = plt.subplots(1, 3, figsize=(9, 3))
#     axes = axes.ravel()
#     for i, col in enumerate(input_var):
#         ax = axes[i]
#         ax.grid(False)
#         # [수정됨] hue=x 변수 할당 및 legend=False 추가
#         sns.boxplot(
#             data=result_data, x='label', y=col, hue='label',
#             order=order, palette=palette,
#             width=0.6, ax=ax, showfliers=True, legend=False
#         )
#         ax.set_title(f'{var_name[i]}')
#         ax.set_xlabel('Cluster')
#         ax.set_ylabel(var_name[i])
        
#         # [수정됨] set_ticks 먼저 호출하여 경고 해결
#         ax.set_xticks(range(len(order)))
#         ax.set_xticklabels([f'C{c}' for c in order])
        
#     for j in range(len(input_var), len(axes)):
# def early_warning_visualization_bootstrap(data, data_all, epi, other_dates, Hockey_date, date_df, sample_window):
#         if epi != 'ILI':
#             other_dates = None
#         train_seasons = [int(str(col)[:-2]) for col in date_df.columns]
#         data = data[data['Season'].isin(train_seasons)].reset_index(drop=True)
#         x = data['Date']
#         y = data[epi]
#         max_y = y.max()
#         train_len = len(train_seasons)
#
#         fig1 = go.Figure()
#         fig1.add_trace(go.Bar(
#             x=data['Date'], y=data[epi],
#             marker_color='gray', opacity=0.5, name=epi,
#             hoverinfo='x+y'
#         ))
#
#         if other_dates is not None:
#             fig1.add_hline(y=max_y*1.3, line_color='black', line_width=1)
#
#         if other_dates is not None:
#             colors = ['orange', 'red']
#             markers = ['triangle-up', 'star']
#             dash_styles = ['dash', 'solid']
#             for j, key in enumerate(other_dates.keys()):
#                 for i, date in enumerate(other_dates[key][:train_len]):
#                     show_leg = True if i == 0 else False
#                     fig1.add_trace(go.Scatter(
#                         x=[date], y=[max_y*(1.4+0.1*j)],
#                         mode='markers', marker=dict(color=colors[j], symbol=markers[j], size=10),
#                         name=key, showlegend=show_leg, hoverinfo='name+x'
#                     ))
#                     fig1.add_shape(type='line', x0=date, x1=date, y0=max_y*1.3, y1=max_y*1.6,
#                                    line=dict(color=colors[j], dash=dash_styles[j], width=2))
#
#         for i, date in enumerate(Hockey_date[:train_len]):
#             show_leg = True if i == 0 else False
#             fig1.add_trace(go.Scatter(
#                 x=[date], y=[max_y*1.1],
#                 mode='markers', marker=dict(color='green', symbol='diamond', size=10),
#                 name='Hockey', showlegend=show_leg, hoverinfo='name+x'
#             ))
#             fig1.add_shape(type='line', x0=date, x1=date, y0=0, y1=max_y*1.3,
#                            line=dict(color='green', dash='dash', width=2), opacity=0.7)
#
#         rank_cols = date_df.columns
#         for i, col in enumerate(rank_cols):
#             iter_results = pd.to_datetime(date_df[col]).dt.normalize().dropna()
#             if not iter_results.empty:
#                 mode_date = iter_results.mode().iloc[0]
#                 low = iter_results.min()
#                 high = iter_results.max()
#                 show_leg = True if i == 0 else False
#                 fig1.add_trace(go.Scatter(
#                     x=[mode_date], y=[max_y],
#                     mode='markers', marker=dict(color='blue', symbol='circle', size=10),
#                     name='Bootstrap', showlegend=show_leg, hoverinfo='name+x'
#                 ))
#                 fig1.add_shape(type='line', x0=mode_date, x1=mode_date, y0=0, y1=max_y*1.3,
#                                line=dict(color='blue', width=2))
#                 fig1.add_vrect(x0=low, x1=high, fillcolor='blue', opacity=0.3, line_width=0)
#
#         y_max_limit = max_y * 1.6 if other_dates is not None else max_y * 1.25
#         fig1.update_layout(
#             yaxis_title=epi,
#             yaxis=dict(range=[0, y_max_limit], fixedrange=False),
#             xaxis=dict(
#                 range=[x.min() - pd.Timedelta(weeks=sample_window-1), x.max()],
#                 rangeslider=dict(visible=True)
#             ),
#             legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5),
#             margin=dict(l=20, r=20, t=60, b=20),
#             hovermode="x unified",
#             plot_bgcolor='white'
#         )
#         fig1.update_xaxes(showline=True, linewidth=1, linecolor='lightgray', gridcolor='whitesmoke')
#         fig1.update_yaxes(showline=True, linewidth=1, linecolor='lightgray', gridcolor='whitesmoke')
#         return fig1

# Build a bootstrap detection timeline that can be reused for overall-period and fitting-period views.
def _build_bootstrap_detection_timeline(
    data,
    epi,
    other_dates,
    hockey_dates,
    date_df,
    sample_window,
    fitting_end_date=None,
    split_labels=False,
    show_blue_band=True,
    show_season_boundaries=False,
):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    x = data['Date']
    y = data[epi]
    max_y = y.max()
    top_line_y = max_y * 1.28

    fig.add_trace(
        go.Bar(
            x=x,
            y=y,
            name=f"{epi} Patients",
            marker_color='gray',
            opacity=0.35
        ),
        secondary_y=False,
    )

    if fitting_end_date is not None and pd.notnull(fitting_end_date):
        fig.add_vline(
            x=fitting_end_date,
            line_color='#7c3aed',
            line_width=3,
            opacity=0.8
        )
        fig.add_annotation(
            x=fitting_end_date,
            y=1.08,
            yref='paper',
            text='Fitting end',
            showarrow=False,
            font=dict(size=12, color='#6d28d9')
        )

    if split_labels:
        pass

    if other_dates is not None:
        fig.add_hline(y=top_line_y, line_color='black', line_width=1)
        ref_colors = ['#ef4444', '#f97316', '#a855f7']
        ref_markers = ['star', 'triangle-up', 'diamond']
        ref_dashes = ['solid', 'dash', 'dot']

        for j, (key, dates) in enumerate(other_dates.items()):
            color = ref_colors[j % len(ref_colors)]
            marker = ref_markers[j % len(ref_markers)]
            dash = ref_dashes[j % len(ref_dashes)]

            for i, date in enumerate(pd.to_datetime(dates, errors='coerce')):
                if pd.isna(date):
                    continue
                if date < x.min() or date > x.max():
                    continue
                fig.add_trace(
                    go.Scatter(
                        x=[date],
                        y=[max_y * (1.38 + 0.08 * j)],
                        mode='markers',
                        marker=dict(color=color, symbol=marker, size=10),
                        name=key,
                        showlegend=(i == 0),
                        hoverinfo='name+x'
                    ),
                    secondary_y=False,
                )
                fig.add_shape(
                    type='line',
                    x0=date,
                    x1=date,
                    y0=top_line_y,
                    y1=max_y * (1.34 + 0.08 * j),
                    line=dict(color=color, dash=dash, width=2)
                )

    max_count = 0
    legend_seen = set()
    season_columns = sorted(date_df.columns) if len(date_df.columns) > 0 else []
    for col in season_columns:
        detect_dates = pd.to_datetime(date_df[col], errors='coerce').dt.normalize()
        valid_dates = detect_dates.dropna()
        if valid_dates.empty:
            continue

        counts = valid_dates.value_counts().sort_index()
        counts = counts[(counts.index >= x.min()) & (counts.index <= x.max())]
        if counts.empty:
            continue

        cumulative_counts = counts.cumsum()
        timeline_df = pd.DataFrame({
            'Date': cumulative_counts.index,
            'Cumulative_Count': cumulative_counts.values,
            'Cumulative_Ratio': cumulative_counts.values / len(detect_dates),
        })
        timeline_df['Level'] = np.select(
            [
                timeline_df['Cumulative_Ratio'] >= 0.10,
                timeline_df['Cumulative_Ratio'] >= 0.05,
                timeline_df['Cumulative_Ratio'] > 0,
            ],
            ['red', 'orange', 'blue'],
            default='none'
        )

        max_count = max(max_count, int(timeline_df['Cumulative_Count'].max()))
        low = timeline_df['Date'].min()
        high = timeline_df['Date'].max()

        _add_cumulative_detection_overlay(
            fig,
            timeline_df,
            secondary_y=True,
            legend_seen=legend_seen,
            showlegend=True
        )

        if show_blue_band:
            fig.add_vrect(
                x0=low,
                x1=high,
                fillcolor='blue',
                opacity=0.18,
                line_width=0
            )

    y_max_limit = max_y * (1.6 if other_dates is not None else 1.32)
    fig.update_layout(
        xaxis=dict(
            title="Date",
            range=[x.min(), x.max()],
            rangeslider=dict(visible=True, thickness=0.15, bgcolor="#EAEAEA"),
            type="date"
        ),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        plot_bgcolor='white',
        margin=dict(l=40, r=40, t=95, b=40)
    )

    fig.update_yaxes(title_text=f"<b>{epi}</b>", range=[0, y_max_limit], secondary_y=False, showgrid=False)
    fig.update_yaxes(
        title_text="<b>Cumulative bootstrap detections</b>",
        range=[0, max(1, max_count) * 1.2],
        secondary_y=True,
        gridcolor='lightgray'
    )
    return fig

def _build_bootstrap_detection_timeline_shared_axis_experiment(
    data,
    epi,
    other_dates,
    hockey_dates,
    date_df,
    sample_window,
    show_blue_band=True,
):
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.82, 0.18],
        vertical_spacing=0.035,
        specs=[[{"secondary_y": True}], [{"secondary_y": True}]]
    )
    data = data.sort_values('Date').reset_index(drop=True)
    x = data['Date']
    y = data[epi]
    max_y = y.max()
    top_line_y = max_y * 1.28

    fig.add_trace(
        go.Bar(
            x=x,
            y=y,
            name=f"{epi} Patients",
            marker_color='gray',
            opacity=0.35
        ),
        row=1,
        col=1,
        secondary_y=False,
    )
    fig.add_trace(
        go.Bar(
            x=x,
            y=y,
            name=f"{epi} Overview",
            marker_color='gray',
            opacity=0.30,
            showlegend=False
        ),
        row=2,
        col=1,
        secondary_y=False,
    )

    if other_dates is not None:
        fig.add_hline(y=top_line_y, line_color='black', line_width=1, row=1, col=1)
        ref_colors = ['#ef4444', '#f97316', '#a855f7']
        ref_markers = ['star', 'triangle-up', 'diamond']
        ref_dashes = ['solid', 'dash', 'dot']

        for j, (key, dates) in enumerate(other_dates.items()):
            color = ref_colors[j % len(ref_colors)]
            marker = ref_markers[j % len(ref_markers)]
            dash = ref_dashes[j % len(ref_dashes)]

            for i, date in enumerate(pd.to_datetime(dates, errors='coerce')):
                if pd.isna(date):
                    continue
                if date < x.min() or date > x.max():
                    continue
                fig.add_trace(
                    go.Scatter(
                        x=[date],
                        y=[max_y * (1.38 + 0.08 * j)],
                        mode='markers',
                        marker=dict(color=color, symbol=marker, size=10),
                        name=key,
                        showlegend=(i == 0),
                        hoverinfo='name+x'
                    ),
                    row=1,
                    col=1,
                    secondary_y=False,
                )
                fig.add_shape(
                    type='line',
                    x0=date,
                    x1=date,
                    y0=top_line_y,
                    y1=max_y * (1.34 + 0.08 * j),
                    line=dict(color=color, dash=dash, width=2),
                    row=1,
                    col=1,
                )

    max_probability = 0
    legend_seen = set()
    thin_width = 3 * 24 * 60 * 60 * 1000
    half_week = pd.Timedelta(days=3.5)
    season_columns = sorted(date_df.columns) if len(date_df.columns) > 0 else []
    for col in season_columns:
        detect_dates = pd.to_datetime(date_df[col], errors='coerce').dt.normalize()
        valid_dates = detect_dates.dropna()
        if valid_dates.empty:
            continue

        counts = valid_dates.value_counts().sort_index()
        counts = counts[(counts.index >= x.min()) & (counts.index <= x.max())]
        if counts.empty:
            continue

        cumulative_counts = counts.cumsum()
        timeline_df = pd.DataFrame({
            'Date': cumulative_counts.index,
            'Cumulative_Count': cumulative_counts.values,
            'Cumulative_Ratio': cumulative_counts.values / len(detect_dates),
        })
        timeline_df['Cumulative_Probability'] = timeline_df['Cumulative_Ratio'] * 100
        timeline_df['Level'] = np.select(
            [
                timeline_df['Cumulative_Ratio'] >= 0.10,
                timeline_df['Cumulative_Ratio'] >= 0.05,
                timeline_df['Cumulative_Ratio'] > 0,
            ],
            ['red', 'orange', 'blue'],
            default='none'
        )
        max_probability = max(max_probability, float(timeline_df['Cumulative_Probability'].max()))

        for level, name, color in [
            ('blue', 'Caution', 'blue'),
            ('orange', 'Alert', 'orange'),
            ('red', 'Severe', 'red'),
        ]:
            level_df = timeline_df[timeline_df['Level'] == level]
            if level_df.empty:
                continue
            fig.add_trace(
                go.Bar(
                    x=level_df['Date'],
                    y=level_df['Cumulative_Probability'],
                    name=name,
                    marker_color=color,
                    opacity=0.6,
                    width=thin_width,
                    showlegend=(name not in legend_seen)
                ),
                row=1,
                col=1,
                secondary_y=True,
            )
            legend_seen.add(name)

        level_dates = _level_dates_from_timeline(timeline_df)
        band_end = counts.index.max() + pd.Timedelta(days=7)
        stage_ranges = [
            ('blue', level_dates['blue'], level_dates['orange'] if pd.notnull(level_dates['orange']) else level_dates['red'] if pd.notnull(level_dates['red']) else band_end),
            ('orange', level_dates['orange'], level_dates['red'] if pd.notnull(level_dates['red']) else band_end),
            ('red', level_dates['red'], band_end),
        ]
        for color, start, end in stage_ranges:
            if pd.isna(start) or pd.isna(end):
                continue
            if end <= start:
                continue
            x0 = start - half_week
            x1 = end - half_week
            if x1 <= x0:
                continue
            fig.add_shape(
                type='rect',
                x0=x0,
                x1=x1,
                y0=0,
                y1=max_y * 1.36,
                fillcolor=color,
                opacity=0.45,
                line_width=0,
                layer='below',
                row=2,
                col=1,
            )

    y_max_limit = max_y * (1.6 if other_dates is not None else 1.32)
    fig.update_layout(
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        plot_bgcolor='white',
        margin=dict(l=40, r=40, t=95, b=55),
        bargap=0.05
    )
    fig.update_xaxes(range=[x.min(), x.max()], type="date", showticklabels=False, row=1, col=1)
    fig.update_xaxes(title_text="Date", range=[x.min(), x.max()], type="date", showticklabels=True, row=2, col=1)
    fig.update_yaxes(title_text=f"<b>{epi}</b>", range=[0, y_max_limit], secondary_y=False, showgrid=False, row=1, col=1)
    fig.update_yaxes(
        title_text="<b>Probability (%)</b>",
        range=[0, max(100, max_probability) * 1.05],
        secondary_y=True,
        gridcolor='lightgray',
        fixedrange=True,
        row=1,
        col=1,
    )
    fig.update_yaxes(
        title_text=f"<b>{epi}</b>",
        range=[0, y_max_limit],
        secondary_y=False,
        showgrid=False,
        fixedrange=True,
        row=1,
        col=1,
    )
    fig.update_yaxes(
        range=[0, max_y * 1.36],
        showticklabels=False,
        title_text="",
        showgrid=False,
        fixedrange=True,
        row=2,
        col=1,
        secondary_y=False,
    )
    fig.update_yaxes(
        range=[0, 100],
        showticklabels=False,
        title_text="",
        showgrid=False,
        fixedrange=True,
        row=2,
        col=1,
        secondary_y=True,
    )
    return fig

# Render the retrospective bootstrap result across the full analysis period.
def early_warning_visualization_bootstrap(data, data_all, epi, other_dates, Hockey_date, date_df, sample_window):
    if epi != 'ILI':
        other_dates = None

    analysis_data = data.reset_index(drop=True)

    return _build_bootstrap_detection_timeline(
        data=analysis_data,
        epi=epi,
        other_dates=other_dates,
        hockey_dates=[],
        date_df=date_df,
        sample_window=sample_window,
        fitting_end_date=None,
        split_labels=False,
        show_blue_band=True,
        show_season_boundaries=False,
    )

def early_warning_visualization_bootstrap_shared_axis_experiment(data, data_all, epi, other_dates, Hockey_date, date_df, sample_window):
    if epi != 'ILI':
        other_dates = None

    analysis_data = data.reset_index(drop=True)

    return _build_bootstrap_detection_timeline_shared_axis_experiment(
        data=analysis_data,
        epi=epi,
        other_dates=other_dates,
        hockey_dates=[],
        date_df=date_df,
        sample_window=sample_window,
        show_blue_band=True,
    )

# Render the full retrospective timeline.
def overall_period_visualization_bootstrap(data, epi, other_dates, Hockey_date, date_df, sample_window, fitting_end_date=None):
    data = data.sort_values('Date').reset_index(drop=True)
    fig = go.Figure()
    max_y = data[epi].max()

    fig.add_trace(
        go.Bar(
            x=data['Date'],
            y=data[epi],
            name=f"{epi} Signal",
            marker_color='gray',
            opacity=0.35,
        )
    )

    if other_dates is not None:
        ref_colors = ['#ef4444', '#f97316', '#a855f7']
        ref_markers = ['star', 'triangle-up', 'diamond']
        for j, (key, dates) in enumerate(other_dates.items()):
            color = ref_colors[j % len(ref_colors)]
            marker = ref_markers[j % len(ref_markers)]
            for i, date in enumerate(pd.to_datetime(dates, errors='coerce')):
                if pd.isna(date):
                    continue
                if date < data['Date'].min() or date > data['Date'].max():
                    continue
                fig.add_trace(
                    go.Scatter(
                        x=[date],
                        y=[max_y * (1.16 + 0.08 * j)],
                        mode='markers',
                        marker=dict(color=color, symbol=marker, size=10),
                        name=key,
                        showlegend=(i == 0),
                        hoverinfo='name+x'
                    )
                )

    fig.update_layout(
        xaxis=dict(
            title="Date",
            range=[data['Date'].min(), data['Date'].max()],
            rangeslider=dict(visible=True, thickness=0.15, bgcolor="#EAEAEA"),
            type="date"
        ),
        yaxis=dict(range=[0, max_y * (1.32 if other_dates is not None else 1.15)]),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        plot_bgcolor='white',
        margin=dict(l=40, r=40, t=90, b=40)
    )
    fig.update_yaxes(title_text=f"<b>{epi}</b>", showgrid=False)
    return fig

def visualize_3d_incremental_detection_weekly(iteration_results):
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    columns = iteration_results.columns
    
    def to_num(val):
        try:
            if pd.isnull(val): return np.nan
            return mdates.date2num(pd.to_datetime(val))
        except:
            return np.nan

    numeric_df = iteration_results.applymap(to_num)
    all_values = numeric_df.values.flatten()
    all_values = all_values[~np.isnan(all_values)]
    
    if len(all_values) == 0:
        print("No data to display.")
        return

    x_min, x_max = np.min(all_values), np.max(all_values)
    x_bins = np.arange(int(x_min) - 7, int(x_max) + 14, 7) 

    for i, col in enumerate(columns):
        data_nums = numeric_df[col].dropna().values
        if len(data_nums) == 0: continue
            
        counts, bins = np.histogram(data_nums, bins=x_bins)
        percent = (counts / len(data_nums)) * 100
        x_centers = (bins[:-1] + bins[1:]) / 2
        y_pos = np.ones_like(x_centers) * i
        mask = percent > 0
        if not np.any(mask): continue

        ax.bar3d(x_centers[mask], y_pos[mask], 0, dx=6.0, dy=0.6, dz=percent[mask], 
                 color=plt.cm.coolwarm(i / len(columns)), alpha=0.7, edgecolor='gray', linewidth=0.2)

    ax.set_xlabel('\nDetection Date (Weekly Bin)', linespacing=3)
    ax.set_ylabel('\nData Accumulated Until', linespacing=6)
    ax.set_zlabel('Bootstrap Probability (%)', linespacing=3)
    
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    
    ax.set_ylabel('Data Accumulated Until', labelpad=30)
    y_indices = np.arange(len(columns))
    step_size = max(1, len(columns) // 10) 
    ax.set_yticks(y_indices[::step_size])
    ax.set_yticklabels([columns[i] for i in y_indices[::step_size]], rotation=-15, va='center', ha='left', fontsize=9)

    ax.view_init(elev=30, azim=-70)
    plt.subplots_adjust(right=0.9)
    plt.show()

def visualization_real_time_early_detection(data_all, date_table, prob_table, epi, other_dates, Hockey_date, bootstrap_dates, batch_size = 10):
    generated_figs = []
    fig_summary, axes = plt.subplots(2, 1, figsize=(10, 4), dpi=300)

    x = data_all['Date']
    y = data_all[epi]
    max_y = y.max()

    ax1 = axes[0]
    ax1.bar(x, y, color='gray', alpha=0.3, label=epi, width=5)
    ax1.set_ylabel(epi)
    ax1.grid(False)

    ax2 = ax1.twinx()
    ax2.plot(x, prob_table['Warning_Probability'], color='blue', alpha=0.8)
    ax2.grid(True, axis='y', color='gray', linestyle='--', linewidth=1, alpha=0.2)

    ax1 = axes[1]
    ax1.bar(x, y, color='gray', alpha=0.3, label=epi, width=5)
    ax1.set_ylabel(epi)
    ax1.grid(False)

    ax2 = ax1.twinx()
    data_bootstrap = pd.to_datetime(date_table['Detect_date']).dt.normalize().dropna()
    sns.histplot(data_bootstrap, ax=ax2, color='blue', kde=False,
                            stat="probability",
                            alpha=0.5,
                            label='Bootstrap', legend=False, discrete=True,
                            shrink=2.0)
    ax2.grid(True, axis='y', color='gray', linestyle='--', linewidth=1, alpha=0.2)

    colors = ['orange', 'red']
    markers = ['^', '*']
    line_style = ['--', '-']
    if other_dates is not None:
        for j, key in enumerate(other_dates.keys()):
            for i, date in enumerate(other_dates[key]):
                ax1.scatter(date, max_y*(1+0.1*j), color=colors[j], marker=markers[j], s=50, label=key if i==0 else "")
                ax1.vlines(date, ymin=0, ymax=max_y*1.2, color=colors[j], linestyle=line_style[j], linewidth=2)

    for i, date in enumerate(Hockey_date):
        label = 'Hockey' if i == 0 else ""
        ax1.scatter(date, max_y*1.1, color='green', marker='D', s=50, label=label)
        ax1.vlines(x=date, ymin=0, ymax=max_y*1.2, color='green', linestyle='--', linewidth=2, alpha=0.7)
    
    ax1.set_xlim(x.min(), x.max())
    plt.tight_layout()
    generated_figs.append(fig_summary)

    col_list = list(bootstrap_dates.columns)
    first_blue = None
    first_orange = None
    first_red = None

    for start in range(0, len(col_list), batch_size):

        batch_list = col_list[start:start + batch_size]
        n_rows = len(batch_list)

        fig_batch, axes = plt.subplots(n_rows, 1, figsize=(10, 2 * n_rows))

        if n_rows == 1:
            axes = [axes]

        for i, col in enumerate(batch_list):
            ax1 = axes[i]

            data = data_all.set_index('Date').loc[:col].reset_index()
            x = data['Date']
            y = data[epi]

            ax1.bar(x, y, color='gray', alpha=0.3, width=5)
            
            start_view = x.min() - pd.Timedelta(days=7)
            end_view = pd.to_datetime(col) + pd.Timedelta(days=7)
            ax1.set_xlim(start_view, end_view)
            ax1.set_ylim(0, max(y) * 1.2)
            ax1.set_title(f'End of test date: {col}')
            ax1.set_ylabel(epi)
            ax1.grid(False)

            colors = ['orange', 'red']
            markers = ['^', '*']
            line_style = ['--', '-']
            if other_dates is not None:
                for j, key in enumerate(other_dates.keys()):
                    for i, date in enumerate(other_dates[key]):
                        ax1.scatter(date, max_y*(1+0.1*j), color=colors[j], marker=markers[j], s=50, label=key if i==0 else "")
                        ax1.vlines(date, ymin=0, ymax=max_y*1.2, color=colors[j], linestyle=line_style[j], linewidth=2)

            ax2 = ax1.twinx()
            bootstrap_dates[col] = bootstrap_dates[col].apply(lambda x: x[0] if isinstance(x, list) else x)
            data_bootstrap = (pd.to_datetime(bootstrap_dates[col]).dt.normalize().dropna())
            
            if not data_bootstrap.empty:
                counts = data_bootstrap.value_counts().sort_index()
                total_n = len(bootstrap_dates[col])

                first_blue = None
                first_orange = None
                first_red = None

                for date, count in counts.items():
                    ratio = count / total_n
                    if ratio >= 0.10:
                        color = 'red'
                        if first_red is None:
                            ax2.axvline(date, color='red', linestyle='--', linewidth=2, alpha=0.5)
                            first_red = date
                    elif ratio >= 0.05:
                        color = 'orange'
                        if first_orange is None:
                            ax2.axvline(date, color='orange', linestyle='--', linewidth=2, alpha=0.5)
                            first_orange = date
                    else:
                        color = 'blue'
                        if first_blue is None:
                            ax2.axvline(date, color='blue', linestyle='--', linewidth=2, alpha=0.5)
                            first_blue = date
                    ax2.bar(date, count, color=color, alpha=0.6, width=2)

            ax2.set_ylabel('Detection dates', rotation=270, labelpad=15)
            ax2.set_ylim(0, len(bootstrap_dates))

            if len(y) <= 10:
                ax1.set_xticks(x)
                ax1.set_xticklabels(x.dt.strftime('%Y-%m-%d'))

        plt.tight_layout()
        generated_figs.append(fig_batch)
        
    print("First blue date:", first_blue)
    print("First orange date:", first_orange)
    print("First red date:", first_red)

    return generated_figs
# Build a single-season real-time chart with optional gray context shading.
def interactive_real_time_chart(data_all, detection_timeline, other_dates, epi, shaded_range=None):
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    min_date = data_all['Date'].min()
    max_date = data_all['Date'].max()

    fig.add_trace(
        go.Bar(
            x=data_all['Date'], y=data_all[epi], 
            name=f"{epi} Patients", marker_color='gray', opacity=0.3
        ),
        secondary_y=False,
    )

    if shaded_range is not None:
        shade_start, shade_end = shaded_range
        if (shade_start is not None) and (shade_end is not None):
            fig.add_vrect(
                x0=shade_start,
                x1=shade_end,
                fillcolor='gray',
                opacity=0.18,
                line_width=0
            )

    level_dates = _add_cumulative_detection_overlay(fig, detection_timeline, secondary_y=True)
    b_date_str = level_dates['blue'].strftime('%Y-%m-%d') if pd.notnull(level_dates['blue']) else "Not detected"
    o_date_str = level_dates['orange'].strftime('%Y-%m-%d') if pd.notnull(level_dates['orange']) else "Not detected"
    r_date_str = level_dates['red'].strftime('%Y-%m-%d') if pd.notnull(level_dates['red']) else "Not detected"
    fig.update_layout(
        xaxis=dict(
            title="Date",
            rangeslider=dict(visible=True, thickness=0.15, bgcolor="#EAEAEA"), 
            type="date",
            range=[min_date, max_date] 
        ),
        hovermode="x unified",
        legend=dict(
            orientation="h", 
            yanchor="bottom", y=1.02,
            xanchor="center", x=0.5
        ),
        
        plot_bgcolor='white', margin=dict(l=40, r=40, t=80, b=40)
    )
    
    max_count = detection_timeline['Cumulative_Count'].max() if detection_timeline is not None and not detection_timeline.empty else 1
    fig.update_yaxes(title_text=f"<b>{epi}</b>", secondary_y=False, showgrid=False)
    fig.update_yaxes(title_text="<b>Cumulative bootstrap detections</b>", secondary_y=True, gridcolor='lightgray', range=[0, max(1, max_count) * 1.2])

    return fig, b_date_str, o_date_str, r_date_str

# Merge season-specific real-time results into a single timeline.
def interactive_real_time_chart_combined(season_results, epi):
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    if not season_results:
        return fig

    combined_data = pd.concat([item['display_data'] for item in season_results], ignore_index=True)
    combined_data = combined_data.sort_values('Date').drop_duplicates(subset=['Date']).reset_index(drop=True)

    fig.add_trace(
        go.Bar(
            x=combined_data['Date'],
            y=combined_data[epi],
            name=f"{epi} Patients",
            marker_color='gray',
            opacity=0.3
        ),
        secondary_y=False,
    )

    legend_seen = set()
    max_count = 0

    for item in season_results:
        detection_timeline = item.get('detection_timeline')
        if detection_timeline is not None and not detection_timeline.empty:
            max_count = max(max_count, int(detection_timeline['Cumulative_Count'].max()))
            _add_cumulative_detection_overlay(
                fig,
                detection_timeline,
                secondary_y=True,
                legend_seen=legend_seen,
                showlegend=True
            )

        shaded_range = item.get('shaded_range')
        if shaded_range is not None:
            shade_start, shade_end = shaded_range
            if (shade_start is not None) and (shade_end is not None):
                fig.add_vrect(
                    x0=shade_start,
                    x1=shade_end,
                    fillcolor='gray',
                    opacity=0.18,
                    line_width=0
                )

    fig.update_layout(
        xaxis=dict(
            title="Date",
            rangeslider=dict(visible=True, thickness=0.15, bgcolor="#EAEAEA"),
            type="date",
            range=[combined_data['Date'].min(), combined_data['Date'].max()]
        ),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="center", x=0.5
        ),
        plot_bgcolor='white',
        margin=dict(l=40, r=40, t=90, b=40)
    )

    fig.update_yaxes(title_text=f"<b>{epi}</b>", secondary_y=False, showgrid=False)
    fig.update_yaxes(title_text="<b>Cumulative bootstrap detections</b>", secondary_y=True, gridcolor='lightgray', range=[0, max(1, max_count) * 1.2])

    return fig

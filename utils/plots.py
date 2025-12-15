# использовал Claude Haiku 4.5

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from phik.phik import phik_matrix
import numpy as np


def plot_target_distribution(data: pd.Series) -> go.Figure:

    fig = go.Figure()
    
    fig.add_histogram(
        x=data,
        nbinsx=50,
        opacity=0.7,
        showlegend=False
    )
    
    median = data.median()
    mean = data.mean()
    std = data.std()
    
    fig.add_vline(
        x=median,
        line=dict(color='#FF5722', width=2, dash='dash')
    )
    
    fig.add_vline(
        x=mean,
        line=dict(color='#4CAF50', width=2, dash='dot')
    )
    
    fig.update_layout(
        title=dict(
            text=f"<b>{"Распределение цены автомобиля"}</b>",
            x=0.5,
            xanchor='center',
            font=dict(size=18)
        ),
        xaxis_title="Цена автомобиля, $",
        yaxis_title="Частота",
        hovermode='x unified',
        template='plotly_white',
        showlegend=True,
        legend=dict(
            title="<b>Статистика</b>",
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="#999",
            borderwidth=1,
            font=dict(size=12)
        ),
        height=700,
        margin=dict(l=50, r=50, t=100, b=50)
    )
    
    stats_text = f"<b style='font-size:14px'>Статистика</b><br><br><span style='color:#FF5722; font-size:13px'><b>━ Медиана:</b> {median:,.0f}</span><br><span style='color:#4CAF50; font-size:13px'><b>┈ Среднее:</b> {mean:,.0f}</span><br><br><span style='font-size:12px'>Стд. отклонение: {std:,.2f}<br>Минимум: {data.min():,.0f}<br>Максимум: {data.max():,.0f}</span>"
    fig.add_annotation(
        text=stats_text,
        xref="paper", yref="paper",
        x=0.97, y=0.45,
        showarrow=False,
        bgcolor="rgba(200, 220, 255, 0.9)",
        bordercolor="#1E88E5",
        borderwidth=3,
        borderpad=18,
        font=dict(size=13),
        xanchor="right",
        yanchor="middle"
    )
    
    return fig


def plot_phik_matrix(df: pd.DataFrame) -> go.Figure:
    phik_corr = phik_matrix(df)
    fig = go.Figure(data=go.Heatmap(
        z=phik_corr.values,
        x=phik_corr.columns,
        y=phik_corr.index,
        text=np.round(phik_corr.values, 2),
        texttemplate='%{text:.2f}',
        textfont={"size": 15},
        colorscale='Reds',
        zmin=0,
        zmax=1,
        colorbar=dict(
            title="Phik<br>Корреляция",
            thickness=15,
            len=0.7
        ),
        hovertemplate='<b>%{y}</b> vs <b>%{x}</b><br>Корреляция: %{z:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text="<b>Phik Матрица Корреляций</b>",
            x=0.5,
            xanchor='center',
            font=dict(size=18)
        ),
        xaxis_title="Признаки",
        yaxis_title="Признаки",
        width=1000,
        height=1000,
        hovermode='closest'
    )
    fig.update_xaxes(tickangle=-45)
    
    return fig


def plot_feature_pairs(data: pd.DataFrame, target_column: str) -> go.Figure:

    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numeric_cols if col != target_column]
    
    n_features = len(feature_cols)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[f"<b>{col}</b> vs {target_column}" for col in feature_cols],
        vertical_spacing=0.2,
        horizontal_spacing=0.1
    )
    
    for idx, feature in enumerate(feature_cols):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        
        fig.add_trace(
            go.Scatter(
                x=data[feature],
                y=data[target_column],
                mode='markers',
                marker=dict(
                    size=5,
                    color='#1E88E5',
                    opacity=0.6,
                    line=dict(width=0)
                ),
                name=feature,
                hovertemplate=f'<b>{feature}</b>: %{{x:.2f}}<br><b>{target_column}</b>: %{{y:.0f}}<extra></extra>'
            ),
            row=row, col=col
        )

        fig.update_xaxes(title_text=feature, row=row, col=col)
        fig.update_yaxes(title_text=target_column, row=row, col=col)
    
    fig.update_layout(
        title=dict(
            text="<b>Зависимость признаков от целевой переменной</b>",
            x=0.5,
            xanchor='center',
            font=dict(size=18)
        ),
        height=300 * n_rows,
        width=1200,
        hovermode='closest',
        showlegend=False
    )
    
    return fig


def plot_log_transform(data: pd.DataFrame, feature_column: str, target_column: str) -> go.Figure:

    feature_log = np.log1p(data[feature_column])
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(f"<b>Распределение log({feature_column})</b>", 
                       f"<b>log({feature_column}) vs {target_column}</b>"),
        specs=[[{"type": "histogram"}, {"type": "scatter"}]]
    )

    fig.add_trace(
        go.Histogram(
            x=feature_log,
            nbinsx=40,
            name=f'log({feature_column})',
            marker=dict(color='#1E88E5'),
            opacity=0.7,
            showlegend=False,
            hovertemplate='Диапазон: %{x:.2f}<br>Частота: %{y}<extra></extra>'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=feature_log,
            y=data[target_column],
            mode='markers',
            name=target_column,
            marker=dict(
                size=5,
                color='#1E88E5',
                opacity=0.6,
                line=dict(width=0)
            ),
            hovertemplate=f'<b>log({feature_column})</b>: %{{x:.2f}}<br><b>{target_column}</b>: %{{y:.0f}}<extra></extra>',
            showlegend=False
        ),
        row=1, col=2
    )
    
    fig.update_xaxes(title_text=f"log({feature_column})", row=1, col=1)
    fig.update_yaxes(title_text="Частота", row=1, col=1)
    fig.update_xaxes(title_text=f"log({feature_column})", row=1, col=2)
    fig.update_yaxes(title_text=target_column, row=1, col=2)
    
    fig.update_layout(
        title=dict(
            text=f"<b>Log-трансформация признака: {feature_column}</b>",
            x=0.5,
            xanchor='center',
            font=dict(size=18)
        ),
        height=500,
        width=1200,
        hovermode='closest',
        showlegend=False
    )
    
    return fig


def plot_top_coefficients(coef_df: pd.DataFrame, top_n: int = 20) -> go.Figure:
    top_features = coef_df.head(top_n).iloc[::-1]
    colors = ['#d62728' if x < 0 else '#2ca02c' for x in top_features['Coefficient']]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=top_features['Coefficient'],
        y=top_features['Feature'],
        orientation='h',
        marker=dict(color=colors),
        text=top_features['Coefficient'].round(4),
        textposition='auto',
    ))
    
    fig.update_layout(
        title="Коэффициенты 20 наиболее важных признаков",
        xaxis_title="Коэффициент",
        yaxis_title="Признак",
        height=500,
        showlegend=False,
        hovermode='y unified'
    )
    
    return fig


def plot_coefficient_distribution(coefficients: np.ndarray) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=coefficients,
        nbinsx=50,
        marker_color='#1f77b4',
        name='Коэффициенты'
    ))
    
    fig.update_layout(
        title="Распределение всех коэффициентов модели",
        xaxis_title="Значение коэффициента",
        yaxis_title="Частота",
        height=400,
        showlegend=False
    )
    
    return fig


def plot_positive_negative_features(coef_df: pd.DataFrame, top_n: int = 10) -> tuple[go.Figure, go.Figure]:
    top_positive = coef_df[coef_df['Coefficient'] > 0].head(top_n).iloc[::-1]
    top_negative = coef_df[coef_df['Coefficient'] < 0].head(top_n).iloc[::-1]
    fig_pos = go.Figure()
    fig_pos.add_trace(go.Bar(
        x=top_positive['Coefficient'],
        y=top_positive['Feature'],
        orientation='h',
        marker_color='#2ca02c'
    ))
    fig_pos.update_layout(
        height=400,
        showlegend=False,
        xaxis_title="Коэффициент",
        margin=dict(l=150)
    )
    fig_neg = go.Figure()
    fig_neg.add_trace(go.Bar(
        x=top_negative['Coefficient'],
        y=top_negative['Feature'],
        orientation='h',
        marker_color='#d62728'
    ))
    fig_neg.update_layout(
        height=400,
        showlegend=False,
        xaxis_title="Коэффициент",
        margin=dict(l=150)
    )
    
    return fig_pos, fig_neg



import streamlit as st
import pandas as pd

from utils.sidebar import render_sidebar
from utils.constants import PAGES
from utils.general import set_style
from utils.data_processing import (
    load_df,
    remove_duplicates,
    convert_df_columns,
    NoneEncoder
)
from utils.plots import (
    plot_target_distribution,
    plot_phik_matrix,
    plot_feature_pairs,
    plot_log_transform
)


STYLE_CSS_PATH = "styles/main.css"
TRAIN_DATA_PATH = "resources/train.csv"


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = load_df(path)
    df = remove_duplicates(df)
    df = convert_df_columns(df)
    encoder = NoneEncoder()
    df = encoder.fit_transform(df)
    return df


@st.cache_data
def get_target_dist_plot(target_series):
    return plot_target_distribution(target_series)


@st.cache_data
def get_phik_plot(df):
    return plot_phik_matrix(df)


@st.cache_data
def get_feature_pairs_plot(df, target_column):
    return plot_feature_pairs(df, target_column=target_column)


@st.cache_data
def get_log_transform_plot(df, feature, target):
    return plot_log_transform(df, feature, target)


data = load_data(TRAIN_DATA_PATH)

st.set_page_config(
    page_title="ML Homework #1 Pro - EDA",
    layout="wide"
)

render_sidebar(PAGES, "EDA")
set_style(STYLE_CSS_PATH)


st.markdown('<h1 class="page-title">Разведочный Анализ Данных</h1>', unsafe_allow_html=True)
st.markdown("---")
st.subheader("Распределение целевой переменной: Цена продажи автомобиля")
fig = get_target_dist_plot(data['selling_price'])
st.plotly_chart(fig, width='stretch')
st.markdown("""
**Выводы**:
- Распределение смещено вправо и не является нормальным;
- Большинство машин имеют относительно низкую стоимость, в то время как несколько дорогих автомобилей увеличивают среднее значение.
""")
st.markdown("---")
st.subheader("Матрица корреляций признаков")
st.markdown("Посмотри на $\\phi_k$ корреляции между признаками в датасете.")
fig = get_phik_plot(data.drop(columns=['name']))
st.plotly_chart(fig, width='stretch')
st.markdown("""
**Выводы**:
- Наиболее сильная положительная зависимость наблюдается между признаками `selling_price` и `max_power`, `torque` и `max_power`;
- Корреляция позволила установить зависимость между целевой переменной и категориальными фичами `owner`, `transmission` и `seller_type`.
""")
st.markdown("---")
st.subheader("Визуализация зависимостей признаков")
fig = get_feature_pairs_plot(data[['selling_price', 'km_driven', 'max_power', 'torque', 'year', 'mileage', 'max_torque_rpm']], target_column='selling_price')
st.plotly_chart(fig, width='stretch')
st.markdown("""
**Выводы**:
- У признаков `max_power` и `torque` наблюдается сильная положительная линейная зависимость с целевой переменной `selling_price`;
- Признаки `km_driven` и `year` демонстрируют более сложные нелинейные зависимости с целевой переменной;
- Признаки `mileage` и `max_torque_rpm` имеют слабую зависимость от целевой переменной.
""")
st.markdown("---")
st.subheader("Обработка признака `km_driven`")
st.markdown("""
Признак `km_driven` имеет сильно ненормальное распределение. Для улучшения модели можно применить логарифмическое преобразование:
""")
fig = get_log_transform_plot(data, 'km_driven', 'selling_price')
st.plotly_chart(fig, width='stretch')
st.markdown("""
**Вывод**: Логарифмирование признака `km_driven` немного сгладило распределение, сделав его более похожим на нормальное, а также улучшило видимость зависимости с целевой переменной `selling_price`.
""")
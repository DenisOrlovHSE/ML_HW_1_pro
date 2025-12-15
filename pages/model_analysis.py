# частично использовал Claude Haiku 4.5

import streamlit as st
import pandas as pd
import numpy as np

from utils.sidebar import render_sidebar
from utils.constants import PAGES
from utils.general import set_style
from utils.model import load_model
from utils.data_processing import (
    load_df,
    remove_duplicates,
    convert_df_columns,
    add_company_feature,
    NoneEncoder,
    TargetEncoder
)
from utils.plots import (
    plot_top_coefficients,
    plot_coefficient_distribution,
    plot_positive_negative_features
)
from sklearn.preprocessing import OneHotEncoder


STYLE_CSS_PATH = "styles/main.css"
MODEL_PATH = "models/best_model.pickle"
TRAIN_DATA_PATH = "resources/train.csv"


@st.cache_resource
def load_trained_model():
    return load_model(MODEL_PATH)


@st.cache_data
def load_and_prepare_data() -> list[str]:
    df_train = load_df(TRAIN_DATA_PATH)
    df_train = remove_duplicates(df_train)
    df_train = convert_df_columns(df_train)
    none_encoder = NoneEncoder()
    df_train = none_encoder.fit_transform(df_train)
    df_train = add_company_feature(df_train)
    target_encoder = TargetEncoder(smoothing=1)
    target_encoder.fit(df_train[['company', 'name']], df_train['selling_price'])
    df_train['seats'] = df_train['seats'].astype(int)
    one_hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    one_hot_encoder.fit(df_train[['fuel', 'seller_type', 'transmission', 'owner', 'seats']])
    numeric_cols = ['name', 'year', 'km_driven', 'mileage', 'engine', 'max_power', 'torque', 'max_torque_rpm']
    ohe_cols = list(one_hot_encoder.get_feature_names_out())
    categorical_cols = ['company']
    return numeric_cols + ohe_cols + categorical_cols


st.set_page_config(
    page_title="ML Homework #1 Pro - Model Analysis",
    layout="wide"
)

render_sidebar(PAGES, "Model Analysis")
set_style(STYLE_CSS_PATH)

st.markdown('<h1 class="page-title">Анализ обученной модели</h1>', unsafe_allow_html=True)
st.markdown("---")

try:
    model = load_trained_model()
    feature_names = load_and_prepare_data()
except Exception as e:
    st.error(f"❌ Ошибка при загрузке модели: {str(e)}")
    st.stop()

st.subheader("Информация о модели")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Тип модели", "ElasticNet")
with col2:
    st.metric("Alpha (α)", "0.1")
with col3:
    st.metric("L1 Ratio", "0.5 (Ridge/Lasso)")

st.markdown("""
**Pipeline**:
1. PolynomialFeatures (степень 2, без смещения)
2. StandardScaler (нормализация)
3. ElasticNet (α=0.1, l1_ratio=0.5)
""")

st.markdown("---")

elastic_net = model.steps[-1][1]
coefficients = elastic_net.coef_
intercept = elastic_net.intercept_

poly = model.named_steps['poly']
poly_feature_names = poly.get_feature_names_out(feature_names)

coef_df = pd.DataFrame({
    'Feature': poly_feature_names,
    'Coefficient': coefficients,
    'Abs_Coefficient': np.abs(coefficients)
})

coef_df = coef_df.sort_values('Abs_Coefficient', ascending=False)

st.subheader("Топ-20 наиболее важных признаков")

fig = plot_top_coefficients(coef_df, top_n=20)
st.plotly_chart(fig, width='stretch')

st.markdown("""
**Интерпретация**:
- **Зеленые столбцы** (положительные коэффициенты): увеличение признака увеличивает цену
- **Красные столбцы** (отрицательные коэффициенты): увеличение признака уменьшает цену
""")

st.markdown("---")

st.subheader("Анализ разреженности модели")

non_zero_coefs = np.count_nonzero(coefficients)
total_coefs = len(coefficients)
sparsity = (1 - non_zero_coefs / total_coefs) * 100

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Всего коэффициентов", f"{total_coefs:,}")
with col2:
    st.metric("Ненулевых коэффициентов", f"{non_zero_coefs:,}")
with col3:
    st.metric("Разреженность (L1)", f"{sparsity:.1f}%")

st.markdown(f"""
L1 регуляризация обнулила **{total_coefs - non_zero_coefs:,}** коэффициентов из {total_coefs:,}.
Это помогает модели избежать переобучения и улучшает интерпретируемость.
""")

st.markdown("---")

st.subheader("Распределение коэффициентов")

fig = plot_coefficient_distribution(coefficients)
st.plotly_chart(fig, width='stretch')

st.markdown("---")

st.subheader("Самые положительные и отрицательные признаки")

fig_pos, fig_neg = plot_positive_negative_features(coef_df, top_n=10)

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Увеличивают цену")
    st.plotly_chart(fig_pos, width='stretch')

with col2:
    st.markdown("#### Уменьшают цену")
    st.plotly_chart(fig_neg, width='stretch')

st.markdown("---")

st.subheader("Полная таблица коэффициентов")

display_df = coef_df[['Feature', 'Coefficient']].copy()
display_df.columns = ['Признак', 'Коэффициент']
display_df['Коэффициент'] = display_df['Коэффициент'].round(6)

st.dataframe(display_df, width='stretch', hide_index=True)


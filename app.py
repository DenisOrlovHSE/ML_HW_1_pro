import streamlit as st

from utils.sidebar import render_sidebar
from utils.constants import PAGES
from utils.general import set_style


STYLE_CSS_PATH = "styles/main.css"


st.set_page_config(
    page_title="ML Homework #1 Pro",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

render_sidebar(PAGES, "Home")
set_style(STYLE_CSS_PATH)

st.markdown('<h1 class="page-title">Предсказание Цен на Автомобили</h1>', unsafe_allow_html=True)

st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 📊 EDA")
    st.markdown("**Разведочный Анализ Данных**")
    st.markdown(
    "- Распределение целевой переменной\n"
    "- Корреляционная матрица признаков\n"
    "- Анализ пар признаков\n"
    "- Преобразование признаков"
    )
    if st.button("Открыть", key="home_eda"):
        st.query_params["page"] = "EDA"
        st.switch_page(PAGES["EDA"])

with col2:
    st.markdown("### 🔮 Inference")
    st.markdown("**Сделать предсказания на новых данных**")
    st.markdown(
    "- Загрузка данных через CSV файл\n"
    "- Ручной ввод характеристик\n"
    "- Получение предсказания цены\n"
    "- Скачивание результатов"
    )    
    if st.button("Открыть", key="home_inf"):
        st.query_params["page"] = "Inference"
        st.switch_page(PAGES["Inference"])

with col3:
    st.markdown("### 🧠 Model Analysis")
    st.markdown("**Анализ весов модели**")
    st.markdown(
    "- Коэффициенты важных признаков\n"
    "- Анализ разреженности (L1)\n"
    "- Положительные & отрицательные признаки\n"
    "- Полная таблица коэффициентов"
    )
    if st.button("Открыть", key="home_analysis"):
        st.query_params["page"] = "Model Analysis"
        st.switch_page(PAGES["Model Analysis"])
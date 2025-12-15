# частично использовал Claude Haiku 4.5

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

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


STYLE_CSS_PATH = "styles/main.css"
MODEL_PATH = "models/best_model.pickle"
TRAIN_DATA_PATH = "resources/train.csv"


@st.cache_resource
def load_trained_model():
    return load_model(MODEL_PATH)


@st.cache_resource
def load_encoders() -> tuple[NoneEncoder, TargetEncoder, OneHotEncoder]:
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
    return none_encoder, target_encoder, one_hot_encoder


def preprocess_input_data(
    df: pd.DataFrame,
    none_encoder: NoneEncoder,
    target_encoder: TargetEncoder,
    one_hot_encoder: OneHotEncoder
) -> pd.DataFrame:
    df = convert_df_columns(df)
    df = none_encoder.transform(df)
    df = add_company_feature(df)
    target_df = target_encoder.transform(df[['company', 'name']])
    categorical_data = one_hot_encoder.transform(df[['fuel', 'seller_type', 'transmission', 'owner', 'seats']])
    df['km_driven'] = np.log1p(df['km_driven'])
    df_final = pd.concat(
        [
            df.drop(columns=[col for col in ['company', 'name', 'fuel', 'seller_type', 'transmission', 'owner', 'seats', 'selling_price'] if col in df.columns]),
            pd.DataFrame(categorical_data, columns=one_hot_encoder.get_feature_names_out()),
            target_df
        ],
        axis=1
    )
    numeric_cols = ['name', 'year', 'km_driven', 'mileage', 'engine', 'max_power', 'torque', 'max_torque_rpm']
    ohe_cols = [col for col in df_final.columns if col.startswith(('fuel_', 'seller_type_', 'transmission_', 'owner_', 'seats_'))]
    categorical_cols = ['company']
    column_order = numeric_cols + ohe_cols + categorical_cols
    df_final = df_final[column_order]
    return df_final


st.set_page_config(
    page_title="ML Homework #1 Pro - Inference",
    layout="wide"
)

render_sidebar(PAGES, "Inference")
set_style(STYLE_CSS_PATH)

st.markdown('<h1 class="page-title">Сделать предсказания на новых данных</h1>', unsafe_allow_html=True)
st.markdown("---")

try:
    model = load_trained_model()
    none_encoder, target_encoder, one_hot_encoder = load_encoders()
except Exception as e:
    st.error(f"❌ Ошибка при загрузке модели: {str(e)}")
    st.stop()


if 'input_data_list' not in st.session_state:
    st.session_state.input_data_list = []

if 'uploaded_file_id' not in st.session_state:
    st.session_state.uploaded_file_id = None

st.subheader("📤 Загрузить CSV файл")

uploaded_file = st.file_uploader(
    "Выберите CSV файл",
    type=['csv'],
    help="Файл должен содержать все необходимые признаки для модели",
    key="csv_uploader"
)

if uploaded_file is not None:
    file_id = f"{uploaded_file.name}_{uploaded_file.size}"
    
    if file_id != st.session_state.uploaded_file_id:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Файл загружен успешно ({len(df)} строк)")
            with st.expander("Просмотр данных"):
                st.dataframe(df.head(10), width='stretch')
            for idx, row in df.iterrows():
                st.session_state.input_data_list.append(pd.DataFrame([row]))
            st.session_state.uploaded_file_id = file_id
        except Exception as e:
            st.error(f"❌ Ошибка при загрузке файла: {str(e)}")
else:
    st.session_state.uploaded_file_id = None

st.markdown("---")
st.subheader("📝 Ввести данные об автомобиле вручную")

with st.form("car_input_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        name = st.text_input("Модель автомобиля", value="Toyota Fortuner")
        year = st.number_input("Год выпуска", min_value=1990, max_value=2025, value=2015)
        km_driven = st.number_input("Пройденные км", min_value=0, value=50000)
        mileage = st.number_input("Расход топлива (kmpl)", min_value=0.0, value=10.0)
        engine = st.number_input("Объем двигателя (CC)", min_value=0, value=2400)
    
    with col2:
        max_power = st.number_input("Максимальная мощность (bhp)", min_value=0.0, value=150.0)
        torque = st.text_input("Крутящий момент (Nm @ RPM)", value="343 Nm @ 1400 rpm")
        fuel = st.selectbox("Тип топлива", ["Petrol", "Diesel", "CNG", "LPG"])
        seller_type = st.selectbox("Тип продавца", ["Individual", "Dealer", "Trustmark Dealer"])
        transmission = st.selectbox("Коробка передач", ["Manual", "Automatic"])
    
    col3, col4 = st.columns(2)
    
    with col3:
        owner = st.selectbox("Количество владельцев", ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner"])
        seats = st.selectbox("Количество мест", [2, 5, 7, 8, 9, 14])
    
    submit_button = st.form_submit_button("➕ Добавить строку", width='stretch')

if submit_button:
    input_data = pd.DataFrame({
        'name': [name],
        'year': [year],
        'km_driven': [km_driven],
        'mileage': [f"{mileage} kmpl"],
        'engine': [f"{engine} CC"],
        'max_power': [f"{max_power} bhp"],
        'torque': [torque],
        'fuel': [fuel],
        'seller_type': [seller_type],
        'transmission': [transmission],
        'owner': [owner],
        'seats': [seats]
    })
    
    st.session_state.input_data_list.append(input_data)
    st.success("✅ Строка добавлена!")

if st.session_state.input_data_list:
    st.markdown("---")
    st.subheader("🎯 Данные для предсказания")
    
    combined_df = pd.concat(st.session_state.input_data_list, ignore_index=True)
    st.dataframe(combined_df, width='stretch')
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Очистить", width='stretch', key="clear_btn"):
            st.session_state.input_data_list = []
            st.rerun()
    
    with col2:
        if st.button("❌ Удалить строку", width='stretch', key="delete_btn"):
            if st.session_state.input_data_list:
                st.session_state.input_data_list.pop()
                st.rerun()
    
    if st.button("🚀 Сделать предсказания", width='stretch', key="predict_btn"):
        try:
            st.subheader("🔄 Обработка данных")
            with st.spinner("Обрабатываю данные..."):
                df_processed = preprocess_input_data(combined_df.copy(), none_encoder, target_encoder, one_hot_encoder)
                st.success("✅ Данные обработаны")
            
            st.subheader("🎯 Предсказания")
            with st.spinner("Делаю предсказания..."):
                predictions = model.predict(df_processed)
                df_results = combined_df.copy()
                df_results['predicted_price'] = predictions
                st.success("✅ Предсказания готовы!")
            
            st.dataframe(df_results[['predicted_price']], width='stretch')
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Средняя цена", f"${predictions.mean():,.0f}")
            with col2:
                st.metric("Минимальная цена", f"${predictions.min():,.0f}")
            with col3:
                st.metric("Максимальная цена", f"${predictions.max():,.0f}")
            with col4:
                st.metric("Стд. отклонение", f"${predictions.std():,.0f}")
            
            csv = df_results.to_csv(index=False)
            st.download_button(
                label="📥 Скачать результаты (CSV)",
                data=csv,
                file_name="predictions.csv",
                mime="text/csv"
            )
            
        except Exception as e:
            st.error(f"❌ Ошибка при обработке данных: {str(e)}")


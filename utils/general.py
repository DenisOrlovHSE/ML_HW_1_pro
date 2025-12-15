import streamlit as st


def set_style(css_path: str) -> None:
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
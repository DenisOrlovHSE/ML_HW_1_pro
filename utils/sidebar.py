# использовал Claude Haiku 4.5

import streamlit as st


STYLE_CSS_PATH = "styles/sidebar.css"


def render_sidebar(pages: dict[str, str], current_page_name: str = "Home") -> str:
    with open(STYLE_CSS_PATH) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    st.markdown('<style>ul[data-testid="stSidebarNav"] {display: none;}</style>', unsafe_allow_html=True)

    with st.sidebar:
        st.markdown(
            '<div class="sidebar-title">'
            '<h1>Car Price</h1>'
            '<h1>Prediction</h1>'
            '</div>',
            unsafe_allow_html=True
        )
        st.markdown("---")

        for page_name, page_file in pages.items():
            if st.button(
                page_name,
                key=f"nav_{page_name}",
                use_container_width=True,
                type="primary" if page_name == current_page_name else "secondary"
            ):
                st.query_params["page"] = page_name
                st.switch_page(page_file)
        
        st.markdown("---")
        st.markdown("### Информация:")
        st.markdown("Орлов Денис")
        st.markdown("Магистратура ФКН \"Искусственный Интелект\"")

    return current_page_name
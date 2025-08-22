import streamlit as st

def dashboard_page():
    st.sidebar.title("TradingAI")
    page = st.sidebar.selectbox("Page", ["Home", "Data Explorer", "Variable Tuner", "Prediction"])
    if page == "Home":
        st.header("Home — Overview")
        st.write("Welcome to TradingAI — choose a page from the sidebar.")
    elif page == "Data Explorer":
        from ui.pages.data_explorer import page as data_page

        data_page()
    elif page == "Variable Tuner":
        from ui.pages.variable_tuner import page as tuner_page

        tuner_page()
    else:
        st.header("Prediction / Model")
        st.write("Model predictions and quick optimize actions will appear here.")

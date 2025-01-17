import streamlit as st
from streamlit_option_menu import option_menu
from home import home
from duration import duration
from status import status
# from status_analysis import show_status_analysis
st.set_page_config(

    layout="wide",

)
def main():
    # Sidebar navigation
    st.sidebar.image("ip-label-logo-grey.png", width=200)
    with st.sidebar:
        page = option_menu("Navigation", ["Home", "Scenario & Step Data Analysis", "Status Data Analysis"],icons=["house", "bar-chart-fill", "activity"],default_index=0)

    if page == "Home":
        home()
    elif page == "Scenario & Step Data Analysis":
        duration()
    elif page == "Status Data Analysis":
        status()


if __name__ == "__main__":
    main()  
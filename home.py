import streamlit as st


def home():

    st.title(":blue[Welcome to the Performance Analysis Application]👋")
    st.sidebar.success("Select a page above.")

    st.markdown(
        """
    This application provides insights into the performance of different scenarios and steps. 
    You can analyze execution durations, aggregate performance metrics, and compare results across different steps and scenarios. 
    Here's a brief overview of the functionalities available:

    - **Scenario Information**: Analyze and visualize performance data for selected scenarios, including aggregations and feature distributions.
    - **Step Information**: Compare performance metrics for individual steps across different queues, with detailed visualizations and summary statistics.
    - **Status Data**: Explore status data with visualizations and insights into various scenarios.

    Use the sidebar to navigate to different sections of the application.
    """
    )

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from status_pred import predict


@st.cache_data(show_spinner=False)
def load_data():
    data = pd.read_csv("Data/streamlit_data_status.csv")
    que = pd.read_excel("Data/queue.xlsx")
    scn = pd.read_excel("Data/scenario.xlsx")
    return data, que, scn


def create_pie_chart(data, column):
    fig = px.pie(data, names=column, title=f"Distribution of {column}")
    fig.update_layout(width=350, height=350)
    return fig


def create_percentage_plot(filtered_data, x_column, x_title, status_filter=None):
    if status_filter:
        filtered_data = filtered_data[filtered_data["scs_status"].isin(status_filter)]

    grouped_data = (
        filtered_data.groupby([x_column, "scs_status"]).size().reset_index(name="count")
    )

    total_count = grouped_data.groupby(x_column)["count"].transform("sum")

    grouped_data["percentage"] = (grouped_data["count"] / total_count) * 100
    color_map = {
        "Available": "#1f77b4     ",
        "Not Available": "#ff4d4d      ",
    }  # Light green and light red

    fig = go.Figure()

    # Check the number of unique statuses
    unique_statuses = grouped_data["scs_status"].unique()
    show_percentage = len(unique_statuses) > 1

    # Add bars for each scs_status
    for status in unique_statuses:
        status_data = grouped_data[grouped_data["scs_status"] == status]
        fig.add_trace(
            go.Bar(
                x=status_data[x_column],
                y=status_data["count"],
                name=status,
                marker_color=color_map[status],
                text=(
                    status_data["percentage"].apply(lambda x: f"{x:.1f}%")
                    if show_percentage
                    else []
                ),
                textposition=(
                    "outside" if show_percentage else "inside"
                ),  # Adjust text position based on the number of statuses
                hovertemplate="<b>%{x}</b><br>Count: %{y}<br>Percentage: %{text}<extra></extra>",
            )
        )

    # Update layout for better readability
    fig.update_layout(
        barmode="group",  # Group bars instead of stacking
        xaxis_title=x_title,
        yaxis_title="Count",
        xaxis_tickangle=-20,  # Rotate x-axis labels
        title={
            "text": f"Count and Percentage Plot of {x_title} by Status",
            "x": 0.5,
            "y": 0.95,
            "xanchor": "center",
            "yanchor": "top",
        },
        margin=dict(l=0, r=0, t=80, b=0),
    )

    # Display the chart in Streamlit
    st.plotly_chart(fig, use_container_width=True)


def generate_daily_distribution_pie_chart(unavailable_df):

    daily_unavailability = (
        unavailable_df.groupby("day_of_week")["duration_minutes"].sum().reset_index()
    )

    fig = px.pie(
        daily_unavailability,
        names="day_of_week",
        values="duration_minutes",
        title="Daily Distribution of Unavailability",
        labels={
            "day_of_week": "Day of the Week",
            "duration_minutes": "Duration (minutes)",
        },
        color="duration_minutes",
        color_discrete_sequence=px.colors.sequential.Sunsetdark,
    )

    fig.update_traces(textinfo="percent+label", hole=0.3)

    return fig


def generate_visualization(unavailable_df):

    daily_unavailability = (
        unavailable_df.groupby("day_of_week")["duration_minutes"].sum().reset_index()
    )

    day_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    all_days = pd.DataFrame(day_order, columns=["day_of_week"])
    daily_unavailability = all_days.merge(
        daily_unavailability, on="day_of_week", how="left"
    )
    daily_unavailability["duration_minutes"].fillna(0, inplace=True)

    daily_unavailability["day_of_week"] = pd.Categorical(
        daily_unavailability["day_of_week"], categories=day_order, ordered=True
    )
    daily_unavailability = daily_unavailability.sort_values("day_of_week")
    # Create a horizontal bar chart using Plotly
    fig = px.bar(
        daily_unavailability,
        x="duration_minutes",
        y="day_of_week",
        orientation="h",
        title="Total Duration of Unavailability by Day of the Week",
        labels={
            "duration_minutes": "Duration (minutes)",
            "day_of_week": "Day of the Week",
        },
        color="duration_minutes",
        color_continuous_scale=px.colors.sequential.Sunsetdark,
    )

    st.plotly_chart(fig)


def generate_hourly_breakdown_bar_chart(unavailable_df):

    # Aggregate duration by hour of the day
    hourly_unavailability = (
        unavailable_df.groupby("hour")["duration_minutes"].sum().reset_index()
    )

    # Create a bar chart using Plotly
    fig = px.bar(
        hourly_unavailability,
        x="hour",
        y="duration_minutes",
        title="Hourly Breakdown of Unavailability",
        labels={"hour": "Hour of the Day", "duration_minutes": "Duration (minutes)"},
        color="duration_minutes",
        color_continuous_scale=px.colors.sequential.Sunsetdark,
    )

    fig.update_layout(xaxis_title="Hour of the Day", yaxis_title="Duration (minutes)")

    return fig


def generate_heatmap_by_day_hour(unavailable_df):

    day_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    hour_range = list(range(24))  # Hours from 0 to 23

    # Create a MultiIndex DataFrame
    all_combinations = pd.MultiIndex.from_product(
        [hour_range, day_order], names=["hour", "day_of_week"]
    )
    heatmap_data = pd.DataFrame(index=all_combinations).reset_index()

    # Aggregate data to fill the heatmap_data DataFrame
    heatmap_data_agg = (
        unavailable_df.groupby(["hour", "day_of_week"])["duration_minutes"]
        .sum()
        .reset_index()
    )

    # Merge with the full combinations DataFrame
    heatmap_data = heatmap_data.merge(
        heatmap_data_agg, on=["hour", "day_of_week"], how="left"
    )
    heatmap_data["duration_minutes"].fillna(
        0, inplace=True
    )  # Fill missing values with 0

    # Pivot the data to create a matrix for the heatmap
    heatmap_data_pivot = heatmap_data.pivot_table(
        index="hour", columns="day_of_week", values="duration_minutes", fill_value=0
    )

    # Reorder columns to ensure day order is maintained
    heatmap_data_pivot = heatmap_data_pivot[day_order]

    # Create a heatmap using Plotly
    fig = px.imshow(
        heatmap_data_pivot,
        color_continuous_scale="Sunsetdark",
        labels={
            "x": "Day of the Week",
            "y": "Hour of the Day",
            "color": "Duration (minutes)",
        },
        title="Heatmap of Unavailability by Day and Hour",
        origin="lower",
    )

    fig.update_layout(
        xaxis_title="Day of the Week",
        yaxis_title="Hour of the Day",
        yaxis=dict(
            tickmode="array",
            tickvals=list(range(24)),
            ticktext=[f"{i}" for i in range(24)],
        ),
    )

    return fig


def status():
    with st.spinner("Loading data..."):
        data, que, scn = load_data()
    que_id_to_name = que.set_index("que_id")["que_name"].to_dict()
    scn_id_to_name = scn.set_index("scn_id")["scn_name"].to_dict()
    scenarios = (
        pd.Series(data["scn_id"].unique())
        .map(scn_id_to_name)
        .fillna(pd.Series(data["scn_id"].unique()))
        .values
    )

    shared_scn = st.session_state.get("shared_scn", None)

    if shared_scn in scn["scn_id"].values:
        default_index = list(scenarios).index(
            scn.loc[scn["scn_id"] == shared_scn, "scn_name"].values[0]
        )
    else:
        default_index = 0

    selected_scenario_name = st.sidebar.selectbox(
        "Select Scenario", scenarios, index=default_index
    )

    if (
        st.session_state.get("shared_scn") is None
        or st.session_state["shared_scn"] != selected_scenario_name
    ):
        st.session_state["shared_scn"] = scn.loc[
            scn["scn_name"] == selected_scenario_name, "scn_id"
        ].values[0]

    selected_scenario = st.session_state["shared_scn"]

    filtered_data = data[data["scn_id"] == selected_scenario]
    # filtered_data['scs_planningtime'] = filtered_data['scs_planningtime'].astype(str)

    filtered_data["scs_planningtime"] = pd.to_datetime(
        filtered_data["scs_planningtime"], errors="coerce"
    )
    filtered_data = filtered_data.dropna(subset=["scs_planningtime"])

    month_mapping = {
        1: "January",
        2: "February",
        3: "March",
        4: "April",
        5: "May",
        6: "June",
        7: "July",
        8: "August",
        9: "September",
        10: "October",
        11: "November",
        12: "December",
    }
    day_of_week_mapping = {
        0: "Monday",
        1: "Tuesday",
        2: "Wednesday",
        3: "Thursday",
        4: "Friday",
        5: "Saturday",
        6: "Sunday",
    }

    status_mapping = {1: "Available", 2: "Not Available"}
    # Apply mappings
    filtered_data["month"] = filtered_data["month"].map(month_mapping)
    filtered_data["day_of_week"] = filtered_data["day_of_week"].map(day_of_week_mapping)
    filtered_data["scs_status"] = filtered_data["scs_status"].map(status_mapping)
    filtered_data["que name"] = (
        filtered_data["que_id"].map(que_id_to_name).fillna(filtered_data["que_id"])
    )

    min_date = filtered_data["scs_planningtime"].min()
    max_date = filtered_data["scs_planningtime"].max()

    start_date, end_date = st.sidebar.date_input(
        "Select period",
        [min_date.date(), max_date.date()],
        min_value=min_date.date(),
        max_value=max_date.date(),
    )

    filtered_data = filtered_data[
        (filtered_data["scs_planningtime"] >= pd.Timestamp(start_date))
        & (filtered_data["scs_planningtime"] <= pd.Timestamp(end_date))
    ]

    st.title(f":grey[Status Data Analysis for Scenario:] {selected_scenario_name}")

    status_counts = filtered_data["scs_status"].value_counts()
    available_count = status_counts.get("Available", 0)
    not_available_count = status_counts.get("Not Available", 0)

    with st.container(border=True):

        col1, col2 = st.columns(2)
        col1.metric(f"Number of Available Executions: ", available_count)
        col2.metric("Number of Not Available Executions: ", not_available_count)
        if st.button("Generate Predictions For the next 7 Days", type="primary"):
            with st.spinner("Calculating predictions, please wait..."):
                predictions = predict()
            with st.expander("Visualize Predictions"):
                df_predictions = pd.DataFrame(predictions)
                unique_statuses = df_predictions["status"].unique()
                if len(unique_statuses) == 1:
                    only_value = unique_statuses[0]
                    if only_value == 1:
                        st.warning(
                            "The prediction results in all **AVAILABLE** data across the 7 days",
                            icon="⚠️",
                        )

                    else:
                        st.warning(
                            "The prediction results in all **UNAVAILABLE** data across the 7 days",
                            icon="⚠️",
                        )

                else:
                    df_predictions["start_time"] = pd.to_datetime(
                        df_predictions["x"], unit="s"
                    )
                    df_predictions["end_time"] = pd.to_datetime(
                        df_predictions["x2"], unit="s"
                    )
                    df_predictions["status"] = df_predictions["status"].apply(
                        lambda x: "Unavailable" if x == 2 else "Available"
                    )

                    # Filter for status 2 (Unavailable)
                    unavailable_df = df_predictions[
                        df_predictions["status"] == "Unavailable"
                    ]

                    unavailable_df["day_of_week"] = unavailable_df[
                        "start_time"
                    ].dt.day_name()
                    unavailable_df["hour"] = unavailable_df["start_time"].dt.hour

                    unavailable_df["duration_minutes"] = (
                        unavailable_df["end_time"] - unavailable_df["start_time"]
                    ).dt.total_seconds() / 60
                    if predictions:
                        # with st.expander("View Predictions Details"):
                        #     st.write("Predictions:")
                        #     for prediction in predictions:
                        #         st.write(f"From: {datetime.fromtimestamp(prediction['x'] )} To: {datetime.fromtimestamp(prediction['x2'] )} - Status: {prediction['status']}")
                        col1, col2 = st.columns(2)
                        with col1:
                            with st.container(border=False):
                                generate_visualization(unavailable_df)
                        with col2:
                            with st.container(border=False):
                                pie_chart = generate_daily_distribution_pie_chart(
                                    unavailable_df
                                )
                                st.plotly_chart(pie_chart, use_container_width=True)

                        col11, col22 = st.columns(2)
                        with col11:
                            with st.container(border=False):
                                bar_chart = generate_hourly_breakdown_bar_chart(
                                    unavailable_df
                                )
                                st.plotly_chart(bar_chart, use_container_width=True)
                        with col22:
                            with st.container(border=False):
                                heatmap = generate_heatmap_by_day_hour(unavailable_df)
                                st.plotly_chart(heatmap, use_container_width=True)
                    else:
                        st.write("No predictions available.")

    with st.container(border=True):
        st.subheader("Que Name and Site")
        col1, col2 = st.columns(2)
        with col1:
            create_percentage_plot(filtered_data, "que name", "Queue Name")

        with col2:
            create_percentage_plot(filtered_data, "sit_id", "Site ID")
    with st.container(border=True):
        selected_statuses = st.multiselect(
            "Select Statuses to Display",
            options=["Available", "Not Available"],
            default=["Available", "Not Available"],
        )
        st.subheader("Temporal Analysis")

        if not selected_statuses:
            st.error("Please select at least one status.")
        else:
            with st.container(border=True):
                col1, col2 = st.columns(2)
                with col1:
                    create_percentage_plot(
                        filtered_data, "month", "Month", status_filter=selected_statuses
                    )
                with col2:
                    create_percentage_plot(
                        filtered_data,
                        "day_of_week",
                        "Day of Week",
                        status_filter=selected_statuses,
                    )

            with st.container(border=True):
                col1, col2 = st.columns(2)
                with col1:
                    create_percentage_plot(
                        filtered_data,
                        "part_of_day",
                        "Part of Day",
                        status_filter=selected_statuses,
                    )
                with col2:
                    create_percentage_plot(
                        filtered_data,
                        "is_business_hour",
                        "is_business_hour",
                        status_filter=selected_statuses,
                    )

            # with st.container():
            #     # Create columns for the fourth row (1 plot in full width)

            #     create_percentage_plot(filtered_data, 'is_weekend', 'is_weekend', status_filter=selected_statuses)

            with st.container(border=True):
                create_percentage_plot(
                    filtered_data, "day", "Day", status_filter=selected_statuses
                )
            with st.container(border=True):
                create_percentage_plot(
                    filtered_data, "hour", "Hour", status_filter=selected_statuses
                )

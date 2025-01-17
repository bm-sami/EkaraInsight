import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import altair as alt


@st.cache_data(show_spinner=False)
def load_data():
    data = pd.read_csv("Data/streamlit_data.csv")
    que = pd.read_excel("Data/queue.xlsx")
    scn = pd.read_excel("Data/scenario.xlsx")
    return data, que, scn


def compute_aggregation(data, group_by_column, aggregation_type):
    if aggregation_type == "Mean":
        return data.groupby(group_by_column)["rsl_duration"].mean().reset_index()
    elif aggregation_type == "Min":
        return data.groupby(group_by_column)["rsl_duration"].min().reset_index()
    elif aggregation_type == "Max":
        return data.groupby(group_by_column)["rsl_duration"].max().reset_index()
    elif aggregation_type == "Median":
        return data.groupby(group_by_column)["rsl_duration"].median().reset_index()


def create_figure(data, x_col, y_col, aggregation_type, title, x_label, y_label):

    fig = px.bar(
        data,
        x=x_col,
        y=y_col,
        title=title.format(aggregation_type=aggregation_type),
        labels={x_col: x_label, y_col: f"{aggregation_type} Duration (s)"},
        color=y_col,
        height=500,
        color_continuous_scale=px.colors.sequential.YlGnBu,
        width=600,
    )
    fig.update_layout(
        xaxis_title=x_label,
        yaxis_title=y_label.format(aggregation_type=aggregation_type),
        xaxis_tickfont=dict(size=10),
        xaxis_tickmode="linear",
    )
    return fig


def create_pie_chart(data, column):
    fig = px.pie(
        data,
        names=column,
        title=f"Distribution of {column}",
    )
    fig.update_layout(width=600, height=400)
    return fig


def plot_bar_chart(data, x, y, title, labels, height=600, width=600):
    fig = px.bar(
        data,
        x=x,
        y=y,
        title=title,
        labels=labels,
        color=y,
        height=height,
        width=width,
        color_continuous_scale=px.colors.sequential.YlGnBu,
    )
    fig.update_layout(
        xaxis_title=labels[x],
        yaxis_title=labels[y],
        xaxis_tickfont=dict(size=10),
        xaxis_tickmode="linear",
    )
    return fig


def duration():
    with st.spinner("Loading Data ..."):
        data, que, scn = load_data()

    que_id_to_name = que.set_index("que_id")["que_name"].to_dict()
    scn_id_to_name = scn.set_index("scn_id")["scn_name"].to_dict()
    scenarios = (
        pd.Series(data["scn_id"].unique())
        .map(scn_id_to_name)
        .fillna(pd.Series(data["scn_id"].unique()))
        .values
    )

    if "shared_scn" not in st.session_state:
        st.session_state["shared_scn"] = None
    shared_scn = st.session_state.get("shared_scn", None)
    selected_scenario_name = st.sidebar.selectbox(
        "Select Scenario",
        scenarios,
        index=(
            list(scenarios).index(
                scn.loc[
                    scn["scn_id"] == st.session_state["shared_scn"], "scn_name"
                ].values[0]
            )
            if st.session_state["shared_scn"] in scn["scn_id"].values
            else 0
        ),
    )
    if (
        st.session_state["shared_scn"] is None
        or st.session_state["shared_scn"] != selected_scenario_name
    ):
        st.session_state["shared_scn"] = scn.loc[
            scn["scn_name"] == selected_scenario_name, "scn_id"
        ].values[0]

    try:
        selected_scenario = st.session_state["shared_scn"]
    except IndexError:
        selected_scenario = selected_scenario_name
    # Filter data based on the selected scenario
    filtered_data = data[data["scn_id"] == selected_scenario]
    # filtered_data['rsl_planningtime'] = filtered_data['rsl_planningtime'].astype(str)

    filtered_data["rsl_planningtime"] = pd.to_datetime(
        filtered_data["rsl_planningtime"], errors="coerce", format="%Y-%m-%d %H:%M:%S"
    )
    filtered_data = filtered_data.dropna(subset=["rsl_planningtime"])

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

    # Apply mappings
    filtered_data["month"] = filtered_data["month"].map(month_mapping)
    filtered_data["day_of_week"] = filtered_data["day_of_week"].map(day_of_week_mapping)

    min_date = filtered_data["rsl_planningtime"].min()
    max_date = filtered_data["rsl_planningtime"].max()

    start_date, end_date = st.sidebar.date_input(
        "Select period",
        [min_date.date(), max_date.date()],
        min_value=min_date.date(),
        max_value=max_date.date(),
    )

    filtered_data = filtered_data[
        (filtered_data["rsl_planningtime"] >= pd.Timestamp(start_date))
        & (filtered_data["rsl_planningtime"] <= pd.Timestamp(end_date))
    ]

    new_min_date = filtered_data["rsl_planningtime"].min()
    new_max_date = filtered_data["rsl_planningtime"].max()
    num_days = (new_max_date - new_min_date).days
    tab1, tab2 = st.tabs(["Scenario Information", "Step Information"])
    with tab1:
        st.title(f":grey[General Informations for Scenario:] {selected_scenario_name}")

        nb_exec = filtered_data["rsl_planningtime"].nunique()
        sum_steps = filtered_data.groupby(filtered_data["rsl_planningtime"].dt.date)[
            "rsl_duration"
        ].max()

        max_sum = sum_steps.max()
        max_planningtime = sum_steps.idxmax()

        min_sum = sum_steps.min()
        min_planningtime = sum_steps.idxmin()
        with st.container(border=True):
            col1, col2, col3 = st.columns(3)
            col1.markdown(
                f"""
                <strong>Number of executions for {num_days} days:</strong> 
                <span style='color: #4CAF50; font-size: 18px;'>{nb_exec}</span>
                """,
                unsafe_allow_html=True,
            )

            col2.markdown(
                f"""
                <strong>Highest Recorded Total Duration:</strong> 
                <span style='color: #FF5722; font-size: 18px;'>{max_sum} (s)</span> 
                at {max_planningtime}
                """,
                unsafe_allow_html=True,
            )

            col3.markdown(
                f"""
                <strong>Lowest Recorded Total Duration:</strong> 
                <span style='color: #2196F3; font-size: 18px;'>{min_sum} (s)</span> 
                at {min_planningtime}
                """,
                unsafe_allow_html=True,
            )
        st.write("## ")

        with st.expander(":orange[Aggregation Analysis of Execution Duration]"):

            aggregation_type = st.selectbox(
                "Select Aggregation Type", ["Mean", "Median", "Min", "Max"]
            )

            data_day_sum = (
                filtered_data.groupby(["rsl_planningtime"])["rsl_duration"]
                .sum()
                .reset_index()
            )
            data_day_sum["year"] = data_day_sum["rsl_planningtime"].dt.year
            data_day_sum["month"] = data_day_sum["rsl_planningtime"].dt.month
            data_day_sum["day"] = data_day_sum["rsl_planningtime"].dt.day
            data_day_sum["hour"] = data_day_sum["rsl_planningtime"].dt.hour
            data_day_sum["day_of_week"] = data_day_sum["rsl_planningtime"].dt.dayofweek
            data_day_sum["date"] = pd.to_datetime(
                data_day_sum[["year", "month", "day"]]
            )

            data_day_sum["month"] = data_day_sum["month"].map(month_mapping)
            data_day_sum["day_of_week"] = data_day_sum["day_of_week"].map(
                day_of_week_mapping
            )

            daily_mean = compute_aggregation(data_day_sum, "date", aggregation_type)
            date_range = pd.date_range(
                start=daily_mean["date"].min(), end=daily_mean["date"].max(), freq="D"
            )

            daily_mean = daily_mean.set_index("date").reindex(date_range).reset_index()
            daily_mean.columns = ["date", "rsl_duration"]

            # st.write(filtered_data.describe())
            # Plot the results using Plotly
            chart = (
                alt.Chart(daily_mean)
                .mark_line(point=False)
                .encode(
                    x=alt.X("date:T", title="Date"),
                    y=alt.Y("rsl_duration:Q", title=f"{aggregation_type} Duration (s)"),
                    tooltip=["date:T", "rsl_duration:Q"],
                )
                .properties(
                    title=f"{aggregation_type} Duration Per Day", width=700, height=400
                )
                .configure_title(fontSize=24, anchor="start")
                .configure_axis(labelFontSize=12, titleFontSize=14)
                .configure_legend(labelFontSize=12, titleFontSize=14)
            )

            # Display in Streamlit
            st.altair_chart(chart, use_container_width=True)

            # st.write("#### Summary Statistics")

            grouped_data_part = (
                filtered_data.groupby(["rsl_planningtime", "part_of_day"])[
                    "rsl_duration"
                ]
                .sum()
                .reset_index()
            )
            grouped_data_buisness_hour = (
                filtered_data.groupby(["rsl_planningtime", "is_business_hour"])[
                    "rsl_duration"
                ]
                .sum()
                .reset_index()
            )

            aggregated_data_stepname = compute_aggregation(
                filtered_data, "rsl_stepname", aggregation_type
            )
            aggregated_data_stepname.columns = ["rsl_stepname", "rsl_duration"]
            aggregated_data_partofday = compute_aggregation(
                grouped_data_part, "part_of_day", aggregation_type
            )
            aggregated_data_partofday.columns = ["part_of_day", "rsl_duration"]
            aggregated_data_hour = compute_aggregation(
                data_day_sum, "hour", aggregation_type
            )
            aggregated_data_hour.columns = ["hour", "rsl_duration"]
            aggregated_data_day = compute_aggregation(
                data_day_sum, "day", aggregation_type
            )
            aggregated_data_day.columns = ["day", "rsl_duration"]
            aggregated_data_day_of_week = compute_aggregation(
                data_day_sum, "day_of_week", aggregation_type
            )
            aggregated_data_day_of_week.columns = ["day_of_week", "rsl_duration"]
            aggregated_data_month = compute_aggregation(
                data_day_sum, "month", aggregation_type
            )
            aggregated_data_month.columns = ["month", "rsl_duration"]
            aggregated_buisness_hour = compute_aggregation(
                grouped_data_buisness_hour, "is_business_hour", aggregation_type
            )
            aggregated_buisness_hour.columns = ["is_business_hour", "rsl_duration"]

            # Sort the groups based on the selected aggregation and select top 20
            top_categories_stepname = aggregated_data_stepname.sort_values(
                by="rsl_duration", ascending=False
            )
            top_categories_partofday = aggregated_data_partofday.sort_values(
                by="rsl_duration", ascending=False
            )

            # Plot the results using Plotly for 'rsl_stepname'
            # st.write(f"### Steps Ranked by {aggregation_type} rsl_duration")

            chart_data_mapping = {
                "Hour": {
                    "data": aggregated_data_hour,
                    "x_col": "hour",
                    "y_col": "rsl_duration",
                    "title": "{aggregation_type} duration per hour",
                    "x_label": "Hour",
                    "y_label": "{aggregation_type} Duration (s)",
                },
                "Part of Day": {
                    "data": aggregated_data_partofday,
                    "x_col": "part_of_day",
                    "y_col": "rsl_duration",
                    "title": "{aggregation_type} duration per part of the day",
                    "x_label": "Part of Day",
                    "y_label": "{aggregation_type} Duration (s)",
                },
                "Day": {
                    "data": aggregated_data_day,
                    "x_col": "day",
                    "y_col": "rsl_duration",
                    "title": "{aggregation_type} duration per day",
                    "x_label": "Day",
                    "y_label": "{aggregation_type} Duration (s)",
                },
                "Month": {
                    "data": aggregated_data_month,
                    "x_col": "month",
                    "y_col": "rsl_duration",
                    "title": "{aggregation_type} duration per month",
                    "x_label": "Month",
                    "y_label": "{aggregation_type} Duration (s)",
                },
                "Day of Week": {
                    "data": aggregated_data_day_of_week,
                    "x_col": "day_of_week",
                    "y_col": "rsl_duration",
                    "title": "{aggregation_type} duration per day of week",
                    "x_label": "Day of Week",
                    "y_label": "{aggregation_type} Duration (s)",
                },
                "Business Hour": {
                    "data": aggregated_buisness_hour,
                    "x_col": "is_business_hour",
                    "y_col": "rsl_duration",
                    "title": "{aggregation_type} duration for business hour",
                    "x_label": "Business Hour",
                    "y_label": "{aggregation_type} Duration (s)",
                },
            }
            selected_charts = st.multiselect(
                "Select charts to display:",
                options=list(chart_data_mapping.keys()),
                default=["Month", "Day"],
            )

            figs = []
            for chart_name in selected_charts:
                if chart_name in chart_data_mapping:
                    chart_config = chart_data_mapping[chart_name]
                    fig = create_figure(
                        data=chart_config["data"],
                        x_col=chart_config["x_col"],
                        y_col=chart_config["y_col"],
                        aggregation_type=aggregation_type,
                        title=chart_config["title"],
                        x_label=chart_config["x_label"],
                        y_label=chart_config["y_label"],
                    )
                    figs.append(fig)

            cols = st.columns(len(figs))
            for i in range(0, len(figs), 2):
                cols = st.columns(2)
                for col, fig in zip(cols, figs[i : i + 2]):
                    with col:
                        st.plotly_chart(fig)

            st.write("## ")
        with st.expander(
            f":orange[Scenario Execution Distribution for {num_days} days]"
        ):

            columns = [
                "month",
                "day_of_week",
                "is_weekend",
                "part_of_day",
                "is_business_hour",
            ]

            for i in range(0, len(columns), 2):
                col1, col2 = st.columns([2, 2])
                with col1:
                    if i < len(columns):
                        with st.container(border=True):
                            st.plotly_chart(create_pie_chart(filtered_data, columns[i]))
                with col2:
                    if i + 1 < len(columns):
                        with st.container(border=True):
                            st.plotly_chart(
                                create_pie_chart(filtered_data, columns[i + 1])
                            )
    with tab2:
        st.title(f":grey[Step Information for Scenario:] {selected_scenario_name}")
        st.write("## ")
        steps = filtered_data["rsl_stepname"].unique()
        selected_step = st.selectbox("Select Step:", steps)

        st.write("## ")

        step_data = filtered_data[filtered_data["rsl_stepname"] == selected_step]
        group_que = step_data.groupby("que_id")["rsl_duration"].mean().reset_index()
        group_que["que name"] = (
            group_que["que_id"].map(que_id_to_name).fillna(group_que["que_id"])
        )

        group_site = step_data.groupby("sit_id")["rsl_duration"].mean().reset_index()
        group_site["site id"] = (
            group_site["sit_id"].map(scn_id_to_name).fillna(group_site["sit_id"])
        )

        bb_sorted = group_que.sort_values(by="rsl_duration", ascending=False).head(10)
        chart = (
            alt.Chart(bb_sorted)
            .mark_bar()
            .encode(
                x=alt.X("que name:N", sort=None, title="Queue Name"),
                y=alt.Y("rsl_duration:Q", title="Mean of duration"),
                color=alt.Color(
                    "que name:N", scale=alt.Scale(scheme="tableau10")
                ),  # Optional: vibrant color scheme
            )
            .properties(
                width=700,  # Adjust width as needed
                height=450,
                title="ques ranked by average duration",
            )
        )
        group_site_sorted = group_site.sort_values(
            by="rsl_duration", ascending=False
        ).head(10)
        chart_site = (
            alt.Chart(group_site_sorted)
            .mark_bar()
            .encode(
                x=alt.X("site id:N", sort=None, title="Site ID"),
                y=alt.Y("rsl_duration:Q", title="Mean of duration"),
                color=alt.Color(
                    "site id:N", scale=alt.Scale(scheme="tableau10")
                ),  # Optional: vibrant color scheme
            )
            .properties(
                width=700,  # Adjust width as needed
                height=450,
                title="sites ranked by average duration",
            )
        )
        col1, col2 = st.columns(2)
        with col1:
            with st.container(border=True):
                st.altair_chart(chart_site, use_container_width=True)

        with col2:
            with st.container(border=True):
                st.altair_chart(chart, use_container_width=True)

        with st.expander(f":orange[General statistics for selected ques]"):
            ques = (
                pd.Series(step_data["que_id"].unique())
                .map(que_id_to_name)
                .fillna(pd.Series(step_data["que_id"].unique()))
                .values
            )
            # with st.container(border=True):
            selected_que = st.multiselect("Choose ques", ques, default=ques[:1])

            if not selected_que:
                st.error("Please select at least one que.")
            else:
                # step_data = filtered_data[filtered_data['rsl_stepname'].isin(selected_steps)]
                # step_data.rename(columns={'rsl_stepname': 'step name'}, inplace=True)
                # Calculate summary statistics

                selected_que_id = que.loc[
                    que["que_name"].isin(selected_que), "que_id"
                ].values
                data_que_step = step_data[step_data["que_id"].isin(selected_que_id)]
                summary_stats = (
                    data_que_step.groupby("que_id")["rsl_duration"]
                    .agg(["mean", "min", "max", "median"])
                    .reset_index()
                )
                summary_stats["Queue Name"] = (
                    summary_stats["que_id"]
                    .map(que_id_to_name)
                    .fillna(summary_stats["que_id"])
                )
                summary_stats = summary_stats[
                    ["Queue Name", "mean", "min", "max", "median"]
                ]
                # Display the summary statistics
                data_que_step["day_planningtime"] = pd.to_datetime(
                    data_que_step["rsl_planningtime"]
                ).dt.floor("D")

                st.write("Summary Statistics for Selected ques")
                st.dataframe(summary_stats)

                aa = (
                    data_que_step.groupby(["day_planningtime", "que_id"])[
                        "rsl_duration"
                    ]
                    .mean()
                    .reset_index()
                )
                aa["que name"] = aa["que_id"].map(que_id_to_name).fillna(aa["que_id"])

                chart = (
                    alt.Chart(aa)
                    .mark_line(opacity=0.5, interpolate="linear")
                    .encode(
                        x=alt.X("day_planningtime:T", title="Date"),
                        y=alt.Y("rsl_duration:Q", title="Duration"),
                        color=alt.Color(
                            "que name:N", scale=alt.Scale(scheme="tableau10")
                        ),
                    )
                    .properties(
                        width=800,  # Adjust width as needed
                        height=400,  # Adjust height as needed
                    )
                    .interactive()
                )

                st.altair_chart(chart, use_container_width=True)

        with st.expander(
            f":orange[General temporal informations for the selected step]"
        ):
            aggregation_type_step = st.selectbox(
                "Select Aggregation Type:", ["Mean", "Median", "Min", "Max"]
            )

            aggregated_data_day_step = compute_aggregation(
                step_data, "day", aggregation_type_step
            )
            aggregated_data_hour_step = compute_aggregation(
                step_data, "hour", aggregation_type_step
            )
            aggregated_data_partofday_step = compute_aggregation(
                step_data, "part_of_day", aggregation_type_step
            )
            aggregated_data_day_of_week_step = compute_aggregation(
                step_data, "day_of_week", aggregation_type_step
            )
            aggregated_data_month_step = compute_aggregation(
                step_data, "month", aggregation_type_step
            )
            aggregated_data_buisness_hour_step = compute_aggregation(
                step_data, "is_business_hour", aggregation_type_step
            )

            selected_charts = st.multiselect(
                "Select charts to display",
                ["Month", "Day", "Hour", "Part of Day", "Day of Week", "Business Hour"],
                default=["Month", "Day"],
            )

            # Display the selected charts in columns
            col1, col2 = st.columns(2)

            with col1:
                if "Month" in selected_charts:
                    fig_month_step = plot_bar_chart(
                        aggregated_data_month_step,
                        "month",
                        "rsl_duration",
                        title=f"{aggregation_type_step} duration per month",
                        labels={
                            "month": "Month",
                            "rsl_duration": f"{aggregation_type_step} Duration (s)",
                        },
                    )
                    with st.container(border=True):
                        st.plotly_chart(fig_month_step)

                if "Business Hour" in selected_charts:
                    fig_buisness_hour_step = plot_bar_chart(
                        aggregated_data_buisness_hour_step,
                        "is_business_hour",
                        "rsl_duration",
                        title=f"{aggregation_type_step} duration for business hour",
                        labels={
                            "is_business_hour": "Business Hour",
                            "rsl_duration": f"{aggregation_type_step} Duration (s)",
                        },
                    )
                    with st.container(border=True):
                        st.plotly_chart(fig_buisness_hour_step)

                if "Day of Week" in selected_charts:
                    fig_day_of_week_step = plot_bar_chart(
                        aggregated_data_day_of_week_step,
                        "day_of_week",
                        "rsl_duration",
                        title=f"{aggregation_type_step} duration per day of the week",
                        labels={
                            "day_of_week": "Day of Week",
                            "rsl_duration": f"{aggregation_type_step} Duration (s)",
                        },
                    )
                    with st.container(border=True):
                        st.plotly_chart(fig_day_of_week_step)

            with col2:
                if "Day" in selected_charts:
                    fig_day_step = plot_bar_chart(
                        aggregated_data_day_step,
                        "day",
                        "rsl_duration",
                        title=f"{aggregation_type_step} duration per day",
                        labels={
                            "day": "Day",
                            "rsl_duration": f"{aggregation_type_step} Duration (s)",
                        },
                    )
                    with st.container(border=True):
                        st.plotly_chart(fig_day_step)

                if "Hour" in selected_charts:
                    fig_hour_step = plot_bar_chart(
                        aggregated_data_hour_step,
                        "hour",
                        "rsl_duration",
                        title=f"{aggregation_type_step} duration per hour",
                        labels={
                            "hour": "Hour",
                            "rsl_duration": f"{aggregation_type_step} Duration (s)",
                        },
                    )
                    with st.container(border=True):
                        st.plotly_chart(fig_hour_step)

                if "Part of Day" in selected_charts:
                    fig_partofday_step = plot_bar_chart(
                        aggregated_data_partofday_step,
                        "part_of_day",
                        "rsl_duration",
                        title=f"{aggregation_type_step} duration per part of the day",
                        labels={
                            "part_of_day": "Part of Day",
                            "rsl_duration": f"{aggregation_type_step} Duration (s)",
                        },
                    )
                    with st.container(border=True):
                        st.plotly_chart(fig_partofday_step)

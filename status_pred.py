import joblib
import holidays
from pandas import Timestamp
import pandas as pd
import numpy as np
import warnings
from datetime import datetime, timedelta
import streamlit as st

# from sklearn.exceptions import InconsistentVersionWarning
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

model_path_status = "Model_Status/status_prediction_2.joblib"

scn_with_one_status = pd.read_csv("Data_Status/scn_with_one_status.csv")

scn_status_encoder = joblib.load("Model_Status/status_one_hot_scn_id_cli_2.pkl")
sit_status_encoder = joblib.load("Model_Status/status_one_hot_sit_id_cli_2.pkl")
que_status_encoder = joblib.load("Model_Status/status_le_rsl_que_id_cli_2.pkl")

model_status = joblib.load(model_path_status)
# booster = model_status.get_booster()

# # Get the feature names from the model
# feature_names = booster.feature_names

# # Display the feature names
# print("Features the model was trained on:")
# for feature in feature_names:
#     print(feature)


que_sit = pd.read_csv("Data_Status/last_que_sit_status.csv")


def prepare_data(
    datetime_obj,
    scn_id,
    sit_id,
):

    bins = [0, 6, 12, 18, 22, 24]
    labels = [3, 0, 1, 2, 3]

    fr_holidays = holidays.France()
    tn_holidays = holidays.Tunisia()

    is_fr_holiday = datetime_obj in fr_holidays
    is_tn_holiday = datetime_obj in tn_holidays

    month = datetime_obj.month
    day = datetime_obj.day
    day_of_week = datetime_obj.day_of_week
    quarter = datetime_obj.quarter
    is_month_end = datetime_obj.is_month_end
    is_month_start = datetime_obj.is_month_start
    hour = datetime_obj.hour
    minute = datetime_obj.minute

    # hour_normalized = hour / 24
    # hour_cos = np.cos(2 * np.pi * hour_normalized)
    # hour_sin = np.sin(2 * np.pi * hour_normalized)

    minute_normalized = minute / 60
    minute_cos = np.cos(2 * np.pi * minute_normalized)
    minute_sin = np.sin(2 * np.pi * minute_normalized)
    minutes_squared = minute**2
    log_minutes_que = np.log1p(minute)

    part_of_day = pd.cut(
        [hour],
        bins=bins,
        labels=labels,
        right=False,
        include_lowest=True,
        ordered=False,
    )[0]
    is_weekend = day_of_week in [5, 6]

    is_business_hour = hour >= 8 or hour <= 17
    # minute_is_weekend = minute * is_weekend
    # minute_is_business_hour = minute * is_business_hour

    # minute_day_of_week = minute * day_of_week

    # encoded features

    scn_status_encoded = scn_status_encoder.transform([[scn_id]]).toarray()
    scn_status_encoded_list = scn_status_encoded[0].tolist()
    # scn_status_encoded = scn_status_encoder.transform([scn_id])

    # que_status_encoded = que_status_encoder.transform([que_id])

    sit_status_encoded = sit_status_encoder.transform([[sit_id]]).toarray()
    sit_status_encoded_list = sit_status_encoded[0].tolist()
    result_status = [
        month,
        day,
        quarter,
        is_month_end,
        is_month_start,
        hour,
        minute,
        is_weekend,
        minute_sin,
        minute_cos,
        part_of_day,
        is_tn_holiday,
        is_fr_holiday,
        minutes_squared,
        log_minutes_que,
        is_business_hour,
    ]
    result_status.extend(scn_status_encoded_list)
    result_status.extend(sit_status_encoded_list)
    result_status.extend([0.0, 0.0])
    # result_status.append(que_status_encoded[0])
    return result_status


def predict():
    now = pd.Timestamp.now()

    start_of_today = now.normalize()

    date = start_of_today + pd.Timedelta(days=1)
    scneario_id = st.session_state.get("shared_scn", None)
    retrieved_que_sit = que_sit[que_sit["scn_id"] == scneario_id]

    sit_status = retrieved_que_sit["sit_id"].values[0]

    period = 15

    datetime_obj = Timestamp(date)
    prediction_intervall = 10080
    if date is None or scneario_id is None or period is None:
        return []
    else:

        try:
            current_datetime = datetime_obj

            if scneario_id in scn_with_one_status["scn_id"]:
                pred = []
                new_start_date = datetime_obj
                stat = int(
                    scn_with_one_status.scs_status[
                        scn_with_one_status["scn_id"] == scneario_id
                    ].iloc[0]
                )
                # end_date = datetime_obj + timedelta(minutes=prediction_intervall)
                for _ in range(7):
                    end_date = new_start_date + timedelta(days=1)
                    prediction_status_constant = [
                        {
                            "x": int(new_start_date.timestamp()),
                            "x2": int(end_date.timestamp()),
                            "status": stat + 1,
                        }
                    ]
                    new_start_date = end_date
                pred.append(prediction_status_constant)
                return pred

            else:

                all_predictions = []

                counter = period  # we initialised this instead of 0 because will make a prediction of the first element before executing the while loop

                data = prepare_data(
                    datetime_obj,
                    scneario_id,
                    sit_status,
                )

                data = np.array(data).reshape(1, -1)
                currecnt_status = model_status.predict(data)
                datetime_obj += pd.Timedelta(minutes=period)

                while datetime_obj <= current_datetime:
                    datetime_obj += pd.Timedelta(minutes=period)

                s = 1

                while counter <= prediction_intervall:
                    data = prepare_data(
                        datetime_obj,
                        scneario_id,
                        sit_status,
                    )
                    data = np.array(data).reshape(1, -1)
                    status = model_status.predict(data)
                    s += 1
                    if status[0] != currecnt_status[0]:
                        diff_days = datetime_obj - current_datetime
                        if diff_days.days > 1:
                            print("TEST")
                            d = datetime_obj - pd.Timedelta(days=diff_days.days)
                            for i in range(diff_days.days):
                                prediction_item = {
                                    "x": int(d.timestamp()),
                                    "x2": int((d + pd.Timedelta(days=1)).timestamp()),
                                    "status": int(currecnt_status[0]) + 1,
                                }
                                all_predictions.append(prediction_item)
                                d += pd.Timedelta(days=1)
                                currecnt_status = status
                        else:
                            prediction_item = {
                                "x": int(current_datetime.timestamp()),
                                "x2": int(datetime_obj.timestamp()),
                                "status": int(currecnt_status[0]) + 1,
                            }
                            all_predictions.append(prediction_item)

                            currecnt_status = status
                            current_datetime = datetime_obj

                    counter += period
                    datetime_obj += pd.Timedelta(minutes=period)
                diff_days = datetime_obj - current_datetime
                if diff_days.days > 1:
                    d = datetime_obj - pd.Timedelta(days=diff_days.days)
                    for i in range(diff_days.days):
                        prediction_item = {
                            "x": int(d.timestamp()),
                            "x2": int((d + pd.Timedelta(days=1)).timestamp()),
                            "status": int(currecnt_status[0]) + 1,
                        }
                        all_predictions.append(prediction_item)
                        d += pd.Timedelta(days=1)
                else:
                    prediction_item = {
                        "x": int(current_datetime.timestamp()),
                        "x2": int(datetime_obj.timestamp()),
                        "status": int(currecnt_status[0]) + 1,
                    }
                    all_predictions.append(prediction_item)
        except Exception as e:

            print(f"An error occurred bb: {str(e)}")
        return all_predictions

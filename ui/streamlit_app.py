from pathlib import Path
import sys
import os


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import logging
from typing import Any, Dict

import requests
import streamlit as st

from app.contracts.property_features import (
    NOMINAL_FEATURES,
    NUMERIC_FEATURES,
    ORDINAL_FEATURES,
    ORDINAL_USER_TO_MODEL,
    REQUIRED_FEATURES,
)
from app.core.display_mapping import (
    get_fallback_reply,
    get_feature_display_name,
    get_feature_display_value,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

BACKEND_BASE_URL = os.getenv("BACKEND_BASE_URL", "http://127.0.0.1:8000")
HEALTH_ENDPOINT = f"{BACKEND_BASE_URL}/health"
PREDICT_ENDPOINT = f"{BACKEND_BASE_URL}/predict"

SESSION_CHAT_MESSAGES = "chat_messages"
SESSION_ACCUMULATED_FEATURES = "accumulated_features"
SESSION_IS_COMPLETE = "is_complete"
SESSION_FINAL_RESULT = "final_result"
SESSION_BACKEND_HEALTH = "backend_health"
SESSION_EDITED_FEATURES = "edited_features"


def initialize_session_state() -> None:
    defaults = {
        SESSION_CHAT_MESSAGES: [],
        SESSION_ACCUMULATED_FEATURES: {},
        SESSION_IS_COMPLETE: False,
        SESSION_FINAL_RESULT: None,
        SESSION_BACKEND_HEALTH: False,
        SESSION_EDITED_FEATURES: None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def check_backend_health() -> bool:
    try:
        response = requests.get(HEALTH_ENDPOINT, timeout=2)
        return response.status_code == 200
    except requests.exceptions.RequestException as exc:
        logger.error("Backend health check failed: %s", exc)
        return False


def call_predict_api(query: str, accumulated_features: Dict[str, Any]) -> Dict[str, Any] | None:
    try:
        payload = {
            "query": query,
            "accumulated_features": accumulated_features,
        }
        response = requests.post(PREDICT_ENDPOINT, json=payload, timeout=15)

        if response.status_code != 200:
            logger.error("Predict API returned status %s", response.status_code)
            return None

        return response.json()
    except requests.exceptions.RequestException as exc:
        logger.error("Predict API call failed: %s", exc)
        return None
    except ValueError as exc:
        logger.error("Failed to parse Predict API response: %s", exc)
        return None


def reset_conversation() -> None:
    st.session_state[SESSION_CHAT_MESSAGES] = []
    st.session_state[SESSION_ACCUMULATED_FEATURES] = {}
    st.session_state[SESSION_IS_COMPLETE] = False
    st.session_state[SESSION_FINAL_RESULT] = None
    st.session_state[SESSION_EDITED_FEATURES] = None


def convert_user_value_to_backend(feature_name: str, user_value: Any) -> Any:
    if user_value is None or user_value == "":
        return None

    if feature_name in ORDINAL_FEATURES:
        return ORDINAL_USER_TO_MODEL.get(user_value, user_value)

    if feature_name in NUMERIC_FEATURES:
        try:
            if feature_name in {"OverallQual", "GarageCars", "FullBath", "TotRmsAbvGrd", "YearBuilt"}:
                return int(user_value)
            numeric_value = float(user_value)
            return int(numeric_value) if numeric_value.is_integer() else numeric_value
        except (ValueError, TypeError):
            return None

    if feature_name in NOMINAL_FEATURES:
        cleaned = str(user_value).strip()
        return cleaned if cleaned else None

    return user_value


def render_chat_history() -> None:
    for message in st.session_state[SESSION_CHAT_MESSAGES]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def render_result_section() -> None:
    if not st.session_state[SESSION_IS_COMPLETE]:
        return

    final_result = st.session_state[SESSION_FINAL_RESULT]
    if not final_result:
        return

    st.divider()
    st.subheader("🎯 Prediction Result")

    predicted_price = final_result.get("predicted_price")
    if predicted_price is not None:
        st.metric("Estimated Sale Price", f"${predicted_price:,.2f}")

    interpretation = final_result.get("interpretation")
    if interpretation:
        st.write("**Market Analysis:**")
        st.write(interpretation)

    global_stats = final_result.get("global_stats", {})
    if global_stats:
        with st.expander("📊 Dataset Statistics"):
            col1, col2 = st.columns(2)
            with col1:
                if "median_price" in global_stats:
                    st.metric("Median Price", f"${global_stats['median_price']:,.0f}")
                if "mean_price" in global_stats:
                    st.metric("Mean Price", f"${global_stats['mean_price']:,.0f}")
                if "min_price" in global_stats:
                    st.metric("Min Price", f"${global_stats['min_price']:,.0f}")
            with col2:
                if "max_price" in global_stats:
                    st.metric("Max Price", f"${global_stats['max_price']:,.0f}")
                if "q1_price" in global_stats:
                    st.metric("Q1 (25th %ile)", f"${global_stats['q1_price']:,.0f}")
                if "q3_price" in global_stats:
                    st.metric("Q3 (75th %ile)", f"${global_stats['q3_price']:,.0f}")

    accumulated = st.session_state[SESSION_ACCUMULATED_FEATURES]
    if accumulated:
        with st.expander("🏠 Extracted Property Features"):
            for feature_name, feature_value in sorted(accumulated.items()):
                display_name = get_feature_display_name(feature_name)
                display_value = get_feature_display_value(feature_name, feature_value)
                st.write(f"- **{display_name}:** {display_value}")


def render_review_edit_section() -> None:
    accumulated = st.session_state[SESSION_ACCUMULATED_FEATURES]
    if not accumulated:
        return

    with st.expander("✏️ Review and Edit Extracted Details", expanded=False):
        st.write(
            "Review the extracted property details below. Edit any values that need "
            "correction, then click **Update Estimate** to re-run the estimate."
        )
        st.divider()

        with st.form("review_edit_form"):
            col1, col2 = st.columns(2)
            edited_values: Dict[str, Any] = {}

            for index, feature_name in enumerate(REQUIRED_FEATURES):
                current_value = accumulated.get(feature_name)
                display_name = get_feature_display_name(feature_name)
                col = col1 if index % 2 == 0 else col2
                widget_key = f"edit_{feature_name}"

                with col:
                    if feature_name == "OverallQual":
                        edited_values[feature_name] = st.number_input(
                            f"{display_name} (1–10)",
                            min_value=1,
                            max_value=10,
                            value=int(current_value) if current_value is not None else 5,
                            step=1,
                            key=widget_key,
                        )
                    elif feature_name in {"GarageCars", "FullBath", "TotRmsAbvGrd"}:
                        edited_values[feature_name] = st.number_input(
                            display_name,
                            min_value=0,
                            value=int(current_value) if current_value is not None else 0,
                            step=1,
                            key=widget_key,
                        )
                    elif feature_name == "YearBuilt":
                        edited_values[feature_name] = st.number_input(
                            display_name,
                            min_value=1800,
                            max_value=2100,
                            value=int(current_value) if current_value is not None else 2000,
                            step=1,
                            key=widget_key,
                        )
                    elif feature_name in ORDINAL_FEATURES:
                        ordinal_options = [
                            "None",
                            "Poor",
                            "Fair",
                            "Typical/Average",
                            "Good",
                            "Excellent",
                        ]
                        current_display = next(
                            (
                                user_value
                                for user_value, model_value in ORDINAL_USER_TO_MODEL.items()
                                if model_value == current_value
                            ),
                            "None",
                        )
                        if current_display not in ordinal_options:
                            current_display = "None"

                        edited_values[feature_name] = st.selectbox(
                            display_name,
                            options=ordinal_options,
                            index=ordinal_options.index(current_display),
                            key=widget_key,
                        )
                    elif feature_name in NOMINAL_FEATURES:
                        edited_values[feature_name] = st.text_input(
                            display_name,
                            value=str(current_value) if current_value else "",
                            key=widget_key,
                        )
                    else:
                        edited_values[feature_name] = st.number_input(
                            f"{display_name} (sq ft)",
                            min_value=0.0,
                            value=float(current_value) if current_value is not None else 0.0,
                            step=50.0,
                            key=widget_key,
                        )

            submitted = st.form_submit_button("🔄 Update Estimate", use_container_width=True)

        if submitted:
            backend_values: Dict[str, Any] = {}
            for feature_name, user_value in edited_values.items():
                backend_value = convert_user_value_to_backend(feature_name, user_value)
                if backend_value is not None:
                    backend_values[feature_name] = backend_value

            st.session_state[SESSION_ACCUMULATED_FEATURES] = backend_values
            st.session_state[SESSION_EDITED_FEATURES] = backend_values

            with st.spinner("Re-running estimate with edited details..."):
                response = call_predict_api(
                    query="Use these exact property details and continue.",
                    accumulated_features=backend_values,
                )

            if response is None:
                st.error("❌ Failed to update estimate. Please try again.")
                return

            st.session_state[SESSION_ACCUMULATED_FEATURES] = response.get(
                "accumulated_features", backend_values
            )

            prediction_ready = response.get("is_complete", False)
            has_prediction = response.get("predicted_price") is not None

            if prediction_ready and has_prediction:
                st.session_state[SESSION_IS_COMPLETE] = True
                st.session_state[SESSION_FINAL_RESULT] = response
                st.success("✅ Estimate updated successfully!")
            else:
                st.session_state[SESSION_IS_COMPLETE] = False
                st.session_state[SESSION_FINAL_RESULT] = None
                st.warning(
                    "⚠️ The estimate could not be calculated yet. "
                    "Some values may still need adjustment."
                )

            st.rerun()


def main() -> None:
    st.set_page_config(
        page_title="AI Real Estate Agent",
        page_icon="🏡",
        layout="wide",
    )

    initialize_session_state()

    st.title("🏡 AI Real Estate Agent")
    st.write(
        "Describe your property in natural language, and I'll estimate its market price "
        "based on the Ames Housing dataset."
    )

    st.session_state[SESSION_BACKEND_HEALTH] = check_backend_health()
    if not st.session_state[SESSION_BACKEND_HEALTH]:
        st.error(
            "⚠️ Backend is not available. Please make sure the server is running at "
            f"{BACKEND_BASE_URL}"
        )
        return

    with st.sidebar:
        st.header("Controls")
        if st.button("🔄 New Conversation", use_container_width=True):
            reset_conversation()
            st.rerun()

        if st.session_state[SESSION_IS_COMPLETE] and st.session_state[SESSION_FINAL_RESULT]:
            st.success("✅ Prediction complete!")
        else:
            st.info("💬 Continue describing the property...")

    render_chat_history()

    if not st.session_state[SESSION_IS_COMPLETE]:
        user_input = st.chat_input("Describe the property...")

        if user_input:
            st.session_state[SESSION_CHAT_MESSAGES].append(
                {"role": "user", "content": user_input}
            )

            with st.spinner("Analyzing property..."):
                response = call_predict_api(
                    query=user_input,
                    accumulated_features=st.session_state[SESSION_ACCUMULATED_FEATURES],
                )

            if response is None:
                st.error("❌ Failed to get response from backend. Please check the server and try again.")
                st.session_state[SESSION_CHAT_MESSAGES].pop()
                st.rerun()

            st.session_state[SESSION_ACCUMULATED_FEATURES] = response.get(
                "accumulated_features", {}
            )

            reply = response.get("reply", "").strip()
            if not reply:
                reply = get_fallback_reply()

            prediction_ready = response.get("is_complete", False)
            has_prediction = response.get("predicted_price") is not None

            if prediction_ready and not has_prediction:
                st.session_state[SESSION_IS_COMPLETE] = False
                st.session_state[SESSION_FINAL_RESULT] = None
                reply = (
                    "I gathered most of the property details, but I still could not "
                    "generate the estimate. Please provide exact numeric values for "
                    "any vague details so I can continue."
                )
            elif prediction_ready and has_prediction:
                st.session_state[SESSION_IS_COMPLETE] = True
                st.session_state[SESSION_FINAL_RESULT] = response
            else:
                st.session_state[SESSION_IS_COMPLETE] = False
                st.session_state[SESSION_FINAL_RESULT] = None

            st.session_state[SESSION_CHAT_MESSAGES].append(
                {"role": "assistant", "content": reply}
            )
            st.rerun()
    else:
        st.info(
            "✅ Property analysis complete! Click 'New Conversation' in the sidebar "
            "to analyze another property."
        )

    render_result_section()
    render_review_edit_section()


if __name__ == "__main__":
    main()

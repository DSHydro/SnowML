""" Simple Dashboard to Display LSTM Results By HUC"""

# pylint: disable=C0103

import streamlit as st
from streamlit_folium import st_folium
import geopandas as gpd
from snowML.dashboard import dashboard_utils as dash

# Use full browser width
st.set_page_config(layout="wide")

# Handle navigation between main dashboard and model page
if "page" not in st.session_state:
    st.session_state["page"] = "main"


# --- Model It page ---
if st.session_state["page"] == "model_page":
    st.title("Pacific Northwest Snow Water Equivalent")
    st.subheader("Model It")
    st.write("Coming Soon")

    # Back button
    if st.button("⬅️ Back to Dashboard"):
        st.session_state["page"] = "main"

    st.stop()  # Prevent main dashboard from rendering


# --- Main Page ---

# Page Set Up
st.title("Pacific Northwest Snow Water Equivalent")
col1, col2 = st.columns([1, 1.3], gap="medium")
if "huc_input" not in st.session_state:
    st.session_state["huc_input"] = None
if "huc12" not in st.session_state:
    st.session_state["huc12"] = None


# Basin Map
with col1:

    # Map of all R17
    #st.subheader(f"Map Of All Huc8 Sub-Basin's in Region 17")
    #R17 = dash.cached_region_geos()
    #m1 = R17.explore()
    #region_map_data = st_folium(m1, use_container_width=True, height=300)

   # User Input for Huc8
    permitted_values = ["17020009", "17030001", "17030002", "17110005", "17110006", "17110008", "17110009"]
    display_names = [
        'Lake Chelan (17020009)',
        'Upper Yakima (17030001)',
        'Naches (17030002)',
        'Upper Skagit (17110005)',
        'Sauk (17110006)',
        'Stillaguamish (17110008)',
        'Skykomish (17110009)'
        ]

    # Mapping from huc_id to display name
    huc_mapping = dict(zip(permitted_values, display_names))

    # Dropdown options with placeholder first
    display_options = ["Select a HUC8..."] + display_names

    # Store previous selection (display_name)
    prev_selection = st.session_state.get("huc_input")

    # Selectbox using display names
    selected_display_name = st.selectbox(
        "Select the HUC8 Sub-Basin you wish to Visualize:",
        options=display_options,
        index=display_options.index(prev_selection) if prev_selection in display_options else 0
    )

    # If selection changed, reset huc12
    if selected_display_name != prev_selection:
        st.session_state["huc12"] = None

    # Store the selected display_name in session_state
    st.session_state["huc_input"] = selected_display_name

    # Convert selected display name to huc_id
    if selected_display_name == "Select a HUC8...":
        huc_input = None
    else:
        # Reverse lookup in huc_mapping
        huc_input = None
        for k, v in huc_mapping.items():
            if v == selected_display_name:
                huc_input = k
                break

    # Only create map if user has made a selection
    if huc_input is not None:
        st.subheader(f"Basin Map For Huc {huc_input}")
        geos = dash.cached_get_geos(huc_input)
        m = geos.explore()

        # Display map in Streamlit
        map_data = st_folium(m, use_container_width=True, height=600)

    # Check if a feature was clicked
        if map_data and map_data.get("last_active_drawing"):
            feature = map_data["last_active_drawing"]
            # Extract huc_id from clicked feature’s properties
            huc12 = feature.get("properties", {}).get("huc_id")
            st.session_state["huc12"] = huc12
        else:
            huc12 = st.session_state.get("huc12")


# SWE History Chart
with col2:
    # Only proceed if the user has selected a HUC8 and there is a selected HUC12
    huc_input = st.session_state.get("huc_input")
    huc12 = st.session_state.get("huc12")

    if huc_input is not None:
        if huc12:
            st.subheader(f"SWE History for HUC {huc12}")

            # Load data from S3
            df_UA, df_UCLA = dash.cached_get_data(huc12)

            if df_UA is None:
                st.info(f"Sorry. Model-ready data for HUC {huc12} is not currently available")
            else:
                # Display plot and data
                fig = dash.cached_plot_swe(df_UA, df_UCLA, huc12)
                st.pyplot(fig)

            # Show "Model It" button
            st.markdown("---")
            if st.button("Model It"):
                st.session_state["page"] = "model_page"
                st.rerun()
        else:
            st.info("Select a HUC8 sub-basin from the map on the left to view SWE history.")


# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 22:17:04 2025

@author: vince (simplified)
"""

import streamlit as st
import geopandas as gpd
import pandas as pd
import folium
import time
import os
import requests
import zipfile
import io
import tempfile
import ephem
from datetime import datetime, timedelta
from streamlit_folium import folium_static, st_folium
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
import numpy as np
from folium.plugins import Draw, MarkerCluster
import math
from scipy.interpolate import interp1d
from scipy import integrate
import psycopg2 as sql
import statistics
import plotly.express as px
import json

import sys



sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import others.solar_algo_new as solar


# Set page config
<<<<<<< HEAD
st.set_page_config(layout="wide", page_title="Philippine Solar Potential Assessment")
st.markdown("""
    <h1 style='text-align: center;'>
        Solar Potential Assessment
    </h1>
""", unsafe_allow_html=True)
=======
st.set_page_config(layout="wide", page_title="Philippine Solar Energy Potential Assessment")

st.title("Solar Potential Assessment")
>>>>>>> 041e7b1e4e0add571bf32715b874e22c3b73cbd8
st.markdown("Select a municipality, or draw a custom area (minimum 3km x 3km) to assess solar potential.")

# --- Static file paths --- 
municipality_github_url = "https://media.githubusercontent.com/media/aikasaurus/philippines-geodata/main/Municipalities.zip"
points_github_url = "https://media.githubusercontent.com/media/aikasaurus/philippines-geodata/main/NoConstraintsID.zip"

# Geometry simplification tolerance (adjust as needed for performance vs. detail)
simplify_tolerance = 0.001  # Roughly 100m at equator

# --- Fix for JSON serialization of timestamps ---
def convert_to_jsonable(obj):
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif hasattr(obj, 'to_dict'):
        return obj.to_dict()
    elif isinstance(obj, datetime):
        return obj.isoformat()
    else:
        return obj

def clean_gdf_for_json(gdf):
    if gdf is None or gdf.empty:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        
    gdf_clean = gdf.copy()
    for col in gdf_clean.columns:
        if col != 'geometry':
            col_dtype = gdf_clean[col].dtype
            if np.issubdtype(col_dtype, np.datetime64):
                gdf_clean[col] = gdf_clean[col].astype(str)
            elif gdf_clean[col].apply(lambda x: isinstance(x, pd.Timestamp)).any():
                gdf_clean[col] = gdf_clean[col].astype(str)
            elif np.issubdtype(col_dtype, np.floating):
                gdf_clean[col] = gdf_clean[col].astype(float)
            elif np.issubdtype(col_dtype, np.integer):
                gdf_clean[col] = gdf_clean[col].astype(int)
    return gdf_clean


# --- Data Loading ---
@st.cache_data
def load_geodata(simplify_tolerance):
    try:
        response = requests.get(municipality_github_url, headers={"Accept": "application/octet-stream"})
        response.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
            with tempfile.TemporaryDirectory() as tmpdirname:
                zip_ref.extractall(tmpdirname)
                shp_files = []
                for root, dirs, files in os.walk(tmpdirname):
                    for file in files:
                        if file.endswith('.shp'):
                            shp_files.append(os.path.join(root, file))

                if not shp_files:
                    st.error("No shapefile found in municipalities data.")
                    return None, None
                gdf_municipalities = gpd.read_file(shp_files[0])
    except Exception as e:
        st.error(f"Error downloading or processing municipalities data: {str(e)}")
        return None, None
    
    try:
        response_points = requests.get(points_github_url, headers={"Accept": "application/octet-stream"})
        response_points.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(response_points.content)) as zip_ref:
            with tempfile.TemporaryDirectory() as tmpdirname:
                zip_ref.extractall(tmpdirname)
                shp_files = [os.path.join(tmpdirname, f) for f in os.listdir(tmpdirname) if f.endswith('.shp')]
                if not shp_files:
                    st.error("No shapefile found in points data.")
                    return gdf_municipalities, None
                gdf_points = gpd.read_file(shp_files[0])
    except Exception as e:
        st.error(f"Error downloading or processing points data: {str(e)}")
        return gdf_municipalities, None

    # Check for empty datasets
    if gdf_municipalities is None or gdf_municipalities.empty:
        st.error("Municipalities dataset is empty.")
        return None, gdf_points
    
    if gdf_points is None or gdf_points.empty:
        st.error("Points dataset is empty.")
        return gdf_municipalities, None

    # CRS checking
    for gdf in [gdf_municipalities, gdf_points]:
        if gdf.crs is None:
            gdf.set_crs(epsg=4326, inplace=True)
        elif gdf.crs.to_epsg() != 4326:
            gdf.to_crs(epsg=4326, inplace=True)

    # Simplify geometries with error handling
    try:
        gdf_municipalities['geometry'] = gdf_municipalities.geometry.simplify(tolerance=simplify_tolerance)
    except Exception as e:
        st.warning(f"Could not simplify municipality geometries: {e}")

    return gdf_municipalities, gdf_points

# --- Load Data ---
with st.spinner("Loading data from GitHub..."):
    gdf_municipalities, gdf_points = load_geodata(simplify_tolerance)

# Check if data loaded successfully
if gdf_municipalities is None or gdf_points is None:
    st.error("Failed to load required datasets. Please check your internet connection and try again.")
    st.stop()

# --- Clean GeoDataFrames ---
gdf_municipalities_clean = clean_gdf_for_json(gdf_municipalities)
gdf_points_clean = clean_gdf_for_json(gdf_points)

# Determine name column
potential_name_cols = [col for col in gdf_municipalities.columns if 'adm3_en' in col.lower()]
if potential_name_cols:
    municipality_name_col = potential_name_cols[0]
else:
    municipality_name_col = gdf_municipalities.columns[0]
    st.warning(f"No column with 'ADM3_EN' found. Using '{municipality_name_col}' as municipality name.")

# Calculate Philippines center for initial map view
try:
    gdf_municipalities_projected = gdf_municipalities.to_crs(epsg=3857)  # Web Mercator projection
    ph_center = gdf_municipalities_projected.geometry.centroid.to_crs(epsg=4326)
    ph_center_lat = ph_center.y.mean()
    ph_center_lon = ph_center.x.mean()
except Exception as e:
    st.warning(f"Error calculating center point: {e}. Using default center.")
    ph_center_lat = 12.8797  # Default center for Philippines
    ph_center_lon = 121.7740  # Default center for Philippines

# Function to calculate distance in km between two points
def haversine_distance(lat1, lon1, lat2, lon2):
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371  # Radius of Earth in kilometers
    return c * r

# Function to create map with drawing tools
def create_drawing_map():
    m = folium.Map(location=[ph_center_lat, ph_center_lon], zoom_start=6)
    
    # Add simplified municipalities layer as reference
    simplified_gdf = gdf_municipalities_clean.copy()
    try:
        simplified_gdf['geometry'] = simplified_gdf['geometry'].simplify(tolerance=0.005)
    except Exception as e:
        st.warning(f"Could not simplify geometry for drawing map: {e}")
    
    folium.Choropleth(
        geo_data=simplified_gdf,
        name="Municipalities",
        fill_color='YlOrRd',
        fill_opacity=0.2,
        line_opacity=0.3,
        highlight=False
    ).add_to(m)
    
    # Add drawing tools plugin
    draw = Draw(
        draw_options={
            'polyline': False,
            'polygon': False,
            'circle': False,
            'marker': False,
            'circlemarker': False,
            'rectangle': True,
        },
        edit_options={
            'featureGroup': None,
            'remove': True
        }
    )
    draw.add_to(m)
    
    # Add scale
    folium.plugins.MeasureControl(position='bottomleft', primary_length_unit='kilometers').add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    return m

# Function to create municipality selection map
def create_selection_map():
    m = folium.Map(location=[ph_center_lat, ph_center_lon], zoom_start=6)

    # Add simplified GeoJSON with fewer features to improve performance
    simplified_gdf = gdf_municipalities_clean.copy()
    
    # Try to simplify the geometry if there are too many points
    try:
        simplified_gdf['geometry'] = simplified_gdf['geometry'].simplify(tolerance=0.005)
    except Exception as e:
        st.warning(f"Could not simplify geometry for selection map: {e}")
    
    # Add municipalities as choropleth layer with tooltips
    folium.Choropleth(
        geo_data=simplified_gdf,
        name="Municipalities",
        fill_color='YlOrRd',
        fill_opacity=0.3,
        line_opacity=0.5,
        highlight=True
    ).add_to(m)
    
    # Add tooltip layer
    tooltip_layer = folium.features.GeoJson(
        simplified_gdf,
        name="Municipalities",
        style_function=lambda x: {"color": "transparent", "fillColor": "transparent"},
        highlight_function=lambda x: {"fillColor": "#ff0000", "fillOpacity": 0.5},
        tooltip=folium.features.GeoJsonTooltip(
            fields=[municipality_name_col],
            aliases=["Municipality:"],
            sticky=True
        )
    )
    m.add_child(tooltip_layer)
    
    return m

# Function to analyze drawn area
def analyze_drawn_area(drawn_shapes):
    if not drawn_shapes or len(drawn_shapes) == 0:
        st.info("Please draw a rectangle on the map to analyze the area.")
        return None
    
    # Get the latest drawn shape (should be a rectangle)
    latest_shape = drawn_shapes[-1]
    
    if latest_shape["geometry"]["type"] == "Polygon":
        # Extract coordinates from GeoJSON
        coordinates = latest_shape["geometry"]["coordinates"][0]  # First element is exterior ring
        
        # Check if it's a valid polygon (at least 3 points)
        if len(coordinates) < 3:
            st.error("Invalid polygon: not enough vertices")
            return None
        
        # Convert coordinates to a Shapely polygon
        try:
            polygon = Polygon(coordinates)
            if not polygon.is_valid:
                st.error("Invalid polygon geometry")
                return None
        except Exception as e:
            st.error(f"Error creating polygon: {e}")
            return None
        
        # Create a GeoDataFrame with the drawn polygon
        drawn_gdf = gpd.GeoDataFrame([{"geometry": polygon}], crs="EPSG:4326")
        
        # Calculate rough dimensions of the rectangle
        min_x = min(coord[0] for coord in coordinates)
        max_x = max(coord[0] for coord in coordinates)
        min_y = min(coord[1] for coord in coordinates)
        max_y = max(coord[1] for coord in coordinates)
        
        # Calculate width and height in kilometers
        width_km = haversine_distance(min_y, min_x, min_y, max_x)
        height_km = haversine_distance(min_y, min_x, max_y, min_x)
        area_km2 = width_km * height_km
        
        if area_km2 < 9:  # Minimum 3km x 3km = 9km²
            st.error(f"The drawn area is too small: {area_km2:.2f} km² (minimum required: 9 km²)")
            st.markdown(f"Width: {width_km:.2f} km, Height: {height_km:.2f} km")
            st.markdown("Please draw a larger rectangle (at least 3km x 3km)")
            return None
        
        # Create result object
        result = {
            "geometry": polygon,
            "width_km": width_km,
            "height_km": height_km,
            "area_km2": area_km2,
            "center": [min_y + (max_y - min_y)/2, min_x + (max_x - min_x)/2]
        }
        
        return result
    else:
        st.warning("Please draw a rectangle using the rectangle tool.")
        return None

# Initialize session state
if 'selected_muni' not in st.session_state:
    st.session_state.selected_muni = None

if 'map_click_source' not in st.session_state:
    st.session_state.map_click_source = False

# Initialize coordinate_tuples in session state
if 'coordinate_tuples' not in st.session_state:
    st.session_state.coordinate_tuples = []

# Initialize drawn_area session state
if 'drawn_area' not in st.session_state:
    st.session_state.drawn_area = None

# Initialize previous_drawings to detect new drawings
if 'previous_drawings' not in st.session_state:
    st.session_state.previous_drawings = None

# Initialize current_drawings session state to store drawings between refreshes
if 'current_drawings' not in st.session_state:
    st.session_state.current_drawings = None

if 'selection_mode' not in st.session_state:
    st.session_state.selection_mode = "municipality"

# Function to handle municipality selection from any source
def select_municipality(municipality_name, source="dropdown"):
    """
    Central function to handle municipality selection.
    Source parameter helps track where the selection came from.
    """
    if municipality_name != st.session_state.selected_muni:
        st.session_state.selected_muni = municipality_name
        st.session_state.map_click_source = (source == "map")
        return True  # Selection changed
    return False  # No change

# Selection mode toggle
selection_mode = st.radio(
    "Selection Mode:",
    ["Municipality Selection", "Draw Custom Area (min 3km x 3km)"],
    horizontal=True,
    key="mode_selector"
)



# Update selection mode in session state and reset relevant states when switching modes
if (selection_mode == "Municipality Selection" and st.session_state.selection_mode != "municipality") or \
   (selection_mode == "Draw Custom Area (min 3km x 3km)" and st.session_state.selection_mode != "draw"):
    # Reset relevant state when changing modes
    if selection_mode == "Municipality Selection":
        st.session_state.selection_mode = "municipality"
        st.session_state.current_drawings = None
        st.session_state.drawn_area = None
    else:
        st.session_state.selection_mode = "draw"
        st.session_state.selected_muni = None

# Create columns for the selection UI 
col1, col2 = st.columns([1, 2])

if st.session_state.selection_mode == "municipality":
    with col1:
        # st.subheader("Municipality Selection")
        # st.markdown("You can select a municipality either by:")
        # st.markdown("1. Clicking on the map")
        # st.markdown("2. Using the dropdown selector")
        
        # Create the dropdown first
        municipality_list = [""] + sorted(gdf_municipalities[municipality_name_col].tolist())

        # Create dropdown without on_change handler to avoid circular dependencies
        selected_from_dropdown = st.selectbox(
            "Select from dropdown", 
            municipality_list,
            index=0 if st.session_state.selected_muni is None else 
                municipality_list.index(st.session_state.selected_muni) if st.session_state.selected_muni in municipality_list else 0,
            key="dropdown_muni"  # Changed key name to avoid conflict
        )
        st.write("Choose which constraints to apply.")
        constraints_table = {0:"BuiltUp Constraints Removed", 1: "CADTs Constraints Removed", 2: "Forest Constraints Removed", 3: "Protected Areas Removed"}
        choose_from = []

        with st.container():
            col1sub, col2sub = st.columns(2)

            with col1sub:
                ancestral = st.checkbox("Ancestral Domains")
                if ancestral:
                    choose_from.append(constraints_table[1])

                tree_cover = st.checkbox("Tree Covers")
                if tree_cover:
                    choose_from.append(constraints_table[2])

            with col2sub:
                land_use = st.checkbox("Land Use")
                if land_use:
                    choose_from.append(constraints_table[0])
                protected_areas =  st.checkbox("Protected Areas")
                if protected_areas:
                    choose_from.append(constraints_table[3])
                
                    

        # let user choose the slope restrictions
        # let user choose the slope restrictions
        slope_checker = st.checkbox("Do you wish to apply slope limits?")
        if slope_checker:
            slope = st.number_input("Enter a value (in %) to filter out slope of the land", min_value=None, max_value=None, value=5.00, step= 0.01, format ="%.4f")

        #af and eta slider
        af = st.slider(
            "Area Factor (af)", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.7, 
            step=0.01,
            key="nsrdb_af"  # Add unique key
        )
        eta = st.slider(
            "Panel Efficiency (eta)", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.2, 
            step=0.01,
            key="nsrdb_eta"  # Add unique key
        )


        # Handle dropdown selection - separate from widget creation
        if selected_from_dropdown and not st.session_state.map_click_source:
            if select_municipality(selected_from_dropdown):
                st.rerun()
        # Reset map click source flag after dropdown handling
        st.session_state.map_click_source = False
        
        # MOVED METRICS BELOW SELECTION INSTRUCTIONS
        # st.markdown("### 📊 Dataset Information")
        # col_metric1, col_metric2 = st.columns(2)
        
        # # Center-align the metrics with custom markdown and HTML
        # with col_metric1:
        #     st.markdown(f"""
        #     <div style="text-align: center;">
        #         <p style="font-size: 14px; color: gray;">Municipalities</p>
        #         <p style="font-size: 28px; font-weight: bold;">{len(gdf_municipalities)}</p>
        #     </div>
        #     """, unsafe_allow_html=True)
        
        # with col_metric2:
        #     st.markdown(f"""
        #     <div style="text-align: center;">
        #         <p style="font-size: 14px; color: gray;">Points</p>
        #         <p style="font-size: 28px; font-weight: bold;">{len(gdf_points)}</p>
        #     </div>
        #     """, unsafe_allow_html=True)

    with col2:
        # Create and display selection map in main content area with DOUBLED HEIGHT
        try:
            selection_map = create_selection_map()
            map_output = st_folium(selection_map, height=600, width=None, returned_objects=["last_active_drawing", "last_clicked"])
        except Exception as e:
            st.error(f"Error creating selection map: {e}")
            map_output = None

        # MAP CLICK HANDLING WITH SPATIAL JOIN
        if map_output and map_output.get("last_clicked"):
            clicked_location = map_output["last_clicked"]
            point = Point(clicked_location["lng"], clicked_location["lat"])
            click_gdf = gpd.GeoDataFrame([{"geometry": point}], crs="EPSG:4326")

            current_zoom = 6  # Default zoom level
            if hasattr(selection_map, 'get_zoom'):
                try:
                    current_zoom = selection_map.get_zoom()
                except:
                    pass
    
    # Adjust search radius based on zoom level
    # Lower zoom = larger search radius
            search_radius = max(0.05, 0.5 / (current_zoom or 1))

            # Use spatial join to find municipality containing the clicked point
            try:
                # First try: Check if point is directly within any municipality
                matched = gpd.sjoin(click_gdf, gdf_municipalities, predicate="within", how="left")
                
                if not matched.empty and not pd.isna(matched.iloc[0][municipality_name_col]):
                    selected_municipality = matched.iloc[0][municipality_name_col]
                    st.session_state.click_confidence = "exact"
                else:
                    # Second try: Create a buffer around the clicked point based on zoom level
                    buffered_point = point.buffer(search_radius)
                    buffer_gdf = gpd.GeoDataFrame([{"geometry": buffered_point}], crs="EPSG:4326")
                    
                    # Find municipalities that intersect with the buffer
                    intersecting = gpd.sjoin(gdf_municipalities, buffer_gdf, predicate="intersects", how="inner")
                    
                    if not intersecting.empty:
                        # Find municipality with centroid closest to click point
                        intersecting["distance"] = intersecting.geometry.centroid.distance(point)
                        nearest_row = intersecting.loc[intersecting["distance"].idxmin()]
                        selected_municipality = nearest_row[municipality_name_col]
                        st.session_state.click_confidence = "buffer"
                    else:
                        # If still no match, find overall nearest municipality by centroid
                        gdf_municipalities["distance"] = gdf_municipalities.geometry.centroid.distance(point)
                        nearest_row = gdf_municipalities.loc[gdf_municipalities["distance"].idxmin()]
                        selected_municipality = nearest_row[municipality_name_col]
                        st.session_state.click_confidence = "nearest"
                
                # Add confirmation when using approximate methods at low zoom levels
                if current_zoom < 8 and st.session_state.click_confidence != "exact":
                    if "last_click_time" not in st.session_state:
                        st.session_state.last_click_time = 0
                        
                    # Prevent rapid double-clicks from causing issues
                    current_time = time.time()
                    if current_time - st.session_state.last_click_time > 1.0:  # 1 second debounce
                        st.session_state.last_click_time = current_time
                        
                        # Show confirmation with option to cancel
                        st.session_state.pending_municipality = selected_municipality
                        confirm = st.toast(f"Selected {selected_municipality}. Click again to confirm.")
                        
                        # Only update if this is a confirmation click
                        if (st.session_state.get("pending_municipality") == selected_municipality and 
                            st.session_state.get("pending_confirmed", False)):
                            st.session_state.pending_confirmed = False
                            if select_municipality(selected_municipality, source="map"):
                                time.sleep(0.3)  # Small delay to ensure UI updates properly
                                st.rerun()
                        else:
                            st.session_state.pending_confirmed = True
                else:
                    # Direct selection for higher zoom levels or exact matches
                    if select_municipality(selected_municipality, source="map"):
                        time.sleep(0.2)  # Small delay to prevent accidental clicks
                        st.rerun()
                        
            except Exception as e:
                st.error(f"Error selecting municipality: {e}")

    # Check if a municipality is selected
    if not st.session_state.selected_muni:
        st.info("👆 Please select a municipality by clicking on the map or using the dropdown menu.")
        st.stop()

    # Display the selected municipality
    # st.subheader(f"Selected Municipality: {st.session_state.selected_muni}")

    # --- Geo selection ---
    selected_gdf = gdf_municipalities[gdf_municipalities[municipality_name_col] == st.session_state.selected_muni]
    if selected_gdf.empty:
        st.error(f"Could not find data for municipality: {st.session_state.selected_muni}")
        st.stop()

    selected_geom = selected_gdf.geometry.unary_union
    try:
        intersecting = gdf_points[gdf_points.geometry.within(selected_geom)]
    except Exception as e:
        st.error(f"Error finding points within municipality: {e}")
        intersecting = gpd.GeoDataFrame(geometry=[], crs=gdf_points.crs)

    # Create coordinate_tuples list from intersecting points
    coordinate_tuples = [(point.y, point.x) for point in intersecting.geometry]
    st.session_state.coordinate_tuples = coordinate_tuples

    # Get centroid coordinates for the selected municipality
    try:
        selected_geom_projected = selected_gdf.to_crs(epsg=3857).geometry.unary_union
        center_point = selected_geom_projected.centroid
        center_point_wgs84 = gpd.GeoSeries([center_point], crs=3857).to_crs(4326)[0]
        center_lat = center_point_wgs84.y
        center_lon = center_point_wgs84.x
        
        # Compute area (approx)
        area_km2 = selected_gdf.to_crs(epsg=3857).geometry.area.iloc[0] / 1e6  # in km²
    except Exception as e:
        st.warning(f"Error calculating municipality center or area: {e}. Using approximate values.")
        # Fallback calculation using bounds
        bounds = selected_gdf.bounds.iloc[0]
        center_lat = (bounds.miny + bounds.maxy) / 2
        center_lon = (bounds.minx + bounds.maxx) / 2
        # Rough area estimation using bounds
        width_km = haversine_distance(bounds.miny, bounds.minx, bounds.miny, bounds.maxx)
        height_km = haversine_distance(bounds.miny, bounds.minx, bounds.maxy, bounds.minx)
        area_km2 = width_km * height_km


    metric, graphs  = st.columns(2)
    # Display municipality metrics
    st.markdown("""
    <div style="
        background-color: #F75A5A;
        padding: 12px 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        font-size: 24px;
        font-weight: 600;
        color: #FFFDF6;
        margin-bottom: 15px;
    ">
        Municipality Information
    </div>
""", unsafe_allow_html=True)
<<<<<<< HEAD
    c1, c2 = st.columns(2)
    disp1, disp2, disp3 = st.columns([1,1,1])
=======

    col_metric1, col_metric2 = st.columns([1,2])
    #c1, c2 = st.columns(2)
    #c1.metric("Area", f"{area_km2:.2f} km²")
    #c2.metric("Points within Municipality", f"{len(intersecting)}")

    disp1, disp2, disp3 = st.columns([1,1,1])

>>>>>>> 041e7b1e4e0add571bf32715b874e22c3b73cbd8
    interactive, summary = st.columns([2,1])
    # --- Map ---
    with interactive:
        m = folium.Map(location=[center_lat, center_lon], zoom_start=9)

        # Add municipality boundary
        folium.GeoJson(
            selected_gdf.geometry,
            name="Municipality",
            style_function=lambda x: {"color": "#0000ff", "fillColor": "#3388ff", "weight": 2, "fillOpacity": 0.2}
        ).add_to(m)

        # Use marker cluster for large datasets to improve performance
        use_clustering = len(intersecting) > 100
        
        if use_clustering:
            # Create marker cluster
            marker_cluster = MarkerCluster(name="Solar Points (Clustered)")
            
            # Add points within the municipality to cluster
            for _, row in intersecting.iterrows():
                folium.CircleMarker(
                    location=[row.geometry.y, row.geometry.x], 
                    radius=5, 
                    color="red",
                    fill=True,
                    fill_color="red",
                    fill_opacity=0.7,
                    popup=f"Point ID: {row.name}"
                ).add_to(marker_cluster)
                
            marker_cluster.add_to(m)
        else:
            # For smaller datasets, use regular feature group
            point_group = folium.FeatureGroup(name="Solar Points")
            
            # Add points within the municipality
            for _, row in intersecting.iterrows():
                folium.CircleMarker(
                    location=[row.geometry.y, row.geometry.x], 
                    radius=5, 
                    color="red",
                    fill=True,
                    fill_color="red",
                    fill_opacity=0.7,
                    popup=f"Point ID: {row.name}"
                ).add_to(point_group)
            
            point_group.add_to(m)

        # Add a limited number of background points for context
        if len(gdf_points) > len(intersecting):
            # Limit number of background points for performance
            background_limit = min(500, len(gdf_points) - len(intersecting))
            if background_limit > 0:
                background_points = gdf_points[~gdf_points.index.isin(intersecting.index)].sample(background_limit)
                bg_group = folium.FeatureGroup(name="Background Points")
                
                for _, row in background_points.iterrows():
                    folium.CircleMarker(
                        location=[row.geometry.y, row.geometry.x], 
                        radius=2, 
                        color='gray', 
                        fill_opacity=0.4
                    ).add_to(bg_group)
                    
                bg_group.add_to(m)

        # Add layer control
        folium.LayerControl().add_to(m)

        # Display the map
        folium_static(m)

    # --- Table of points ---
<<<<<<< HEAD
    # st.subheader(f"📍 {len(intersecting)} Points Inside {st.session_state.selected_muni}")
    # if not intersecting.empty:
    #     display_cols = [col for col in intersecting.columns if col != 'geometry']
    #     st.dataframe(intersecting[display_cols])
    # else:
    #     st.info("No points found within this municipality.")
=======
    #st.subheader(f"📍 {len(intersecting)} Points Inside {st.session_state.selected_muni}")
    #if not intersecting.empty:
    #    display_cols = [col for col in intersecting.columns if col != 'geometry']
    #    st.dataframe(intersecting[display_cols])
    #else:
    #    st.info("No points found within this municipality.")
>>>>>>> 041e7b1e4e0add571bf32715b874e22c3b73cbd8

else:  # Draw Custom Area mode
    with col1:

            # # Create the dropdown first
            # municipality_list = [""] + sorted(gdf_municipalities[municipality_name_col].tolist())

            # # Create dropdown without on_change handler to avoid circular dependencies
            # selected_from_dropdown = st.selectbox(
            #     "Select a municipality:", 
            #     municipality_list,
            #     index=0 if st.session_state.selected_muni is None else 
            #         municipality_list.index(st.session_state.selected_muni) if st.session_state.selected_muni in municipality_list else 0,
            #     key="dropdown_muni"  # Changed key name to avoid conflict
            # )

            # # Handle dropdown selection - separate from widget creation
            # if selected_from_dropdown and not st.session_state.map_click_source:
            #     if select_municipality(selected_from_dropdown):
            #         st.rerun()
            # # Reset map click source flag after dropdown handling
            # st.session_state.map_click_source = False
        
<<<<<<< HEAD
            # col_metric1, col_metric2 = st.columns(2)


            # let user choose constraints first
            st.write("Choose which constraints to apply.")

            constraints_table = {0:"BuiltUp Constraints Removed", 1: "CADTs Constraints Removed", 2: "Forest Constraints Removed", 3: "Protected Areas Removed"}
            choose_from = []

            with st.container():
                col1sub, col2sub = st.columns(2)

                with col1sub:
                    ancestral = st.checkbox("Ancestral Domains")
                    if ancestral:
                        choose_from.append(constraints_table[1])

                    tree_cover = st.checkbox("Tree Covers")
                    if tree_cover:
                        choose_from.append(constraints_table[2])

                with col2sub:
                    land_use = st.checkbox("Land Use")
                    if land_use:
                        choose_from.append(constraints_table[0])
                    protected_areas =  st.checkbox("Protected Areas")
                    if protected_areas:
                        choose_from.append(constraints_table[3])

            # let user choose the slope restrictions
            # let user choose the slope restrictions
            slope_checker = st.checkbox("Do you wish to apply slope limits?")
            if slope_checker:
                slope = st.number_input("Enter a value (in %) to filter out slope of the land", min_value=None, max_value=None, value=5.00, step= 0.01, format ="%.4f")

                    #af and eta slider
            af = st.slider(
                "Area Factor (af)", 
                min_value=0.0, 
                max_value=1.0, 
                value=0.7, 
                step=0.01,
                key="nsrdb_af"  # Add unique key
            )
            eta = st.slider(
                "Panel Efficiency (eta)", 
                min_value=0.0, 
                max_value=1.0, 
                value=0.2, 
                step=0.01,
                key="nsrdb_eta"  # Add unique key
            )


=======
        # METRICS BELOW DRAWING INSTRUCTIONS
        #st.markdown("### 📊 Dataset Information")
        #col_metric1, col_metric2 = st.columns(2)
        
        # let user choose constraints first
        st.write("Choose which constraints to apply.")

        constraints_table = {0:"BuiltUp Constraints Removed", 1: "CADTs Constraints Removed", 2: "Forest Constraints Removed", 3: "Protected Areas Removed"}
        choose_from = []

        with st.container():
            col1sub, col2sub = st.columns(2)

            with col1sub:
                ancestral = st.checkbox("Ancestral Domains")
                if ancestral:
                    choose_from.append(constraints_table[1])

                tree_cover = st.checkbox("Tree Covers")
                if tree_cover:
                    choose_from.append(constraints_table[2])

            with col2sub:
                land_use = st.checkbox("Land Use")
                if land_use:
                    choose_from.append(constraints_table[0])
                protected_areas =  st.checkbox("Protected Areas")
                if protected_areas:
                    choose_from.append(constraints_table[3])

        

        # let user choose the slope restrictions
        slope = st.slider("Filter out slope of the land (%):", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

        st.markdown(f"""
        <div style="
            background-color: #ffffff;
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            font-size: 25px;
            font-weight: bold;
            color: #333;
            ">
            Total Area: {9 * len(valid_points)} km²
        </div>
        <br>
    """, unsafe_allow_html=True)
        
>>>>>>> 041e7b1e4e0add571bf32715b874e22c3b73cbd8
    with col2:
            # Create map with drawing tools
        try:
            drawing_map = create_drawing_map()
                # Store previous drawings to detect changes
            previous_drawings = st.session_state.current_drawings
                
                # Use key parameter for the st_folium component to help Streamlit manage its state
            map_data = st_folium(drawing_map, height=600, width=None, key="draw_map",
                            returned_objects=["all_drawings"])
                
            # Store drawings in session state 
            # Store drawings in session state 
            if map_data and map_data.get("all_drawings"):
                current_drawings = map_data.get("all_drawings")
                st.session_state.current_drawings = current_drawings
                
                # Check if we have new drawings compared to the previous state
                if (previous_drawings is None and current_drawings) or \
                    (previous_drawings and len(current_drawings) != len(previous_drawings)):
                    # New drawing detected - analyze it automatically
                    drawn_area = analyze_drawn_area(current_drawings)
                    if drawn_area:
                        st.session_state.drawn_area = drawn_area
                        # Instead of rerun, we directly modify session state to trigger UI updates
                        st.session_state.refresh = True  # Just set a flag to trigger UI refresh

        except Exception as e:
            st.error(f"Error creating drawing map: {e}")
            map_data = None
    
    # Process the drawn area if it exists in session state
    if st.session_state.drawn_area:
        drawn_area = st.session_state.drawn_area
        
        # Display success message
        #st.success(f"Area selected: {drawn_area['area_km2']:.2f} km² (Width: {drawn_area['width_km']:.2f} km, Height: {drawn_area['height_km']:.2f} km)")
        
        # Find points within the drawn area
        try:
            intersecting = gdf_points[gdf_points.geometry.within(drawn_area['geometry'])]
        except Exception as e:
            st.error(f"Error finding points within custom area: {e}")
            intersecting = gpd.GeoDataFrame(geometry=[], crs=gdf_points.crs)
        
        # Display area metrics
        
        st.markdown("""
    <div style="
        background-color: #F75A5A;
        padding: 12px 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        font-size: 24px;
        font-weight: 600;
        color: #FFFDF6;
        margin-bottom: 15px;
    ">
        Custom Area Information
    </div>
""", unsafe_allow_html=True)
        
        disp1, disp2, disp3 = st.columns([1,1,1])
        interactive, summary = st.columns([2,1]) 
        dis1, dis2, dis3 = st.columns([1,1,1])
        
        # c1, c2, c3 = st.columns(3)
        # c1.metric("Area", f"{drawn_area['area_km2']:.2f} km²")
        # c2.metric("Width x Height", f"{drawn_area['width_km']:.2f} km x {drawn_area['height_km']:.2f} km")
        # c3.metric("Points within Area", f"{len(intersecting)}")

        #handle column sizes here ----
        
        # Create and display map
        with interactive:
            area_map = folium.Map(location=drawn_area["center"], zoom_start=9)
            
            # Add the drawn polygon
            drawn_gdf = gpd.GeoDataFrame([{"geometry": drawn_area["geometry"]}], crs="EPSG:4326")
            folium.GeoJson(
                drawn_gdf,
                name="Custom Area",
                style_function=lambda x: {"color": "#ff7800", "fillColor": "#ffff00", "weight": 2, "fillOpacity": 0.2}
            ).add_to(area_map)
            
            # Use marker cluster for large datasets to improve performance
            use_clustering = len(intersecting) > 100
            
            if use_clustering:
                # Create marker cluster
                marker_cluster = MarkerCluster(name="Solar Points (Clustered)")
                
                # Add points within the drawn area to cluster
                for _, row in intersecting.iterrows():
                    folium.CircleMarker(
                        location=[row.geometry.y, row.geometry.x], 
                        radius=5, 
                        color="red",
                        fill=True,
                        fill_color="red",
                        fill_opacity=0.7,
                        popup=f"Point ID: {row.name}"
                    ).add_to(marker_cluster)
                    
                marker_cluster.add_to(area_map)
            else:
                # For smaller datasets, use regular feature group
                point_group = folium.FeatureGroup(name="Solar Points")
                
                # Add points within the drawn area
                for _, row in intersecting.iterrows():
                    folium.CircleMarker(
                        location=[row.geometry.y, row.geometry.x], 
                        radius=5, 
                        color="red",
                        fill=True,
                        fill_color="red",
                        fill_opacity=0.7,
                        popup=f"Point ID: {row.name}"
                    ).add_to(point_group)
                
                point_group.add_to(area_map)
            
            # Add layer control
            folium.LayerControl().add_to(area_map)
            
            # Display the map
            folium_static(area_map)
        
        # # Display table of points
        # st.subheader(f"📍 {len(intersecting)} Points Inside Custom Area")
        # if not intersecting.empty:
        #     display_cols = [col for col in intersecting.columns if col != 'geometry']
        #     st.dataframe(intersecting[display_cols])
        # else:
        #     st.info("No points found within the drawn area.")
    else:
        st.info(" Draw a rectangle on the map using the rectangle tool. The area will be analyzed automatically.")

    #list of cooridnate tuples

    custom_coords = list(zip(intersecting['lat'], intersecting['lon']))
# First define once at the top
@st.cache_data
def convert_df_to_csv(_df):
    export_df = _df.drop(columns=['geometry'])
    return export_df.to_csv(index=False).encode('utf-8')

# Then in your logic:
if st.session_state.selection_mode == "municipality" and st.session_state.selected_muni:
    selected_gdf = gdf_municipalities[gdf_municipalities[municipality_name_col] == st.session_state.selected_muni]
    if not selected_gdf.empty:
        selected_geom = selected_gdf.geometry.unary_union
        intersecting = gdf_points[gdf_points.geometry.within(selected_geom)]
        # if not intersecting.empty:
        #     csv = convert_df_to_csv(intersecting)
        #     st.download_button(
        #         label="Download points as CSV",
        #         data=csv,
        #         file_name=f"{st.session_state.selected_muni}_points.csv",
        #         mime='text/csv',
        #     )

elif st.session_state.selection_mode == "draw" and st.session_state.drawn_area:
    try:
        intersecting = gdf_points[gdf_points.geometry.within(st.session_state.drawn_area['geometry'])]
        if not intersecting.empty:
            csv = convert_df_to_csv(intersecting)
            st.download_button(
                label="Download points as CSV",
                data=csv,
                file_name="custom_area_points.csv",
                mime='text/csv',
            )
    except Exception as e:
        st.error(f"Error preparing download: {e}")

###
<<<<<<< HEAD
# Database connection function
def get_db_connection():
    return psycopg2.connect(
        dbname="webGIS",  
        user="postgres",       
        password="ashley",    
        host="localhost",       
        port="5432"
    )

def connect_to_db():  # utilized
  try:
    connection = sql.connect(
            dbname= "webGIS",
            user="postgres",
            password="ashley",
            host="localhost",
            port= "5432"
        )
    
    #st.text("Database connection successful!")
    return connection
  except Exception as e:

    st.text("Database connection failed:")

    return None 
  
def filter_slope(valid_points, slope_constraint = 0):
    ''' call this function to filter out '''

    working_db = connect_to_db()
    while working_db == None:
      working_db = connect_to_db()

  #this will allow the sql queries
    pointer = working_db.cursor()

    prep_points = ', '.join(['(%s, %s)'] * len(valid_points))
    coords = [coord for point in valid_points for coord in point]

    query = f"""SELECT lon, lat
     FROM "Slope_Percentage"
     WHERE "slope_rounded" <= {slope_constraint}  AND (lon_rounded, lat_rounded) IN ({prep_points});"""
    
    pointer.execute(query, coords)

    slope_data = pointer.fetchall()

    #clean data
    temp_holder = []
    for point in slope_data:
      new = (round(float(point[0]), 6), round(float(point[1]), 6))
      temp_holder.append(new)
    
    slope_data = temp_holder


    pointer.close()
    working_db.close()

    return slope_data 



# Function to fetch data from PostgreSQL
def fetch_data_from_db(polygon_geojson):
    polygon_wkt = shape(polygon_geojson).wkt  # Convert GeoJSON to WKT format
    
    query = f"""
    SELECT name, ghi, wind_speed
    FROM municipalities
    WHERE ST_Within(geom, ST_GeomFromText('{polygon_wkt}', 4326));
    """

    # Connect to DB, execute query, and return GeoDataFrame
    conn = get_db_connection()
    gdf = gpd.read_postgis(query, conn, geom_col='geom')
    conn.close()
    
    return gdf

def db_fetch_hourly_solar(valid_points:list, municipality = None):
  """ returns in this format:           Lat | Long | Month | Day | Hour | GHI | MuniCipality 
                              point a                               1                        
                              point b                               1                        """

  working_db = connect_to_db()
  while working_db == None:
    working_db = connect_to_db()

  pointer = working_db.cursor()

  prep_points = ', '.join(['(%s, %s)'] * len(valid_points))
  coords = [coord for point in valid_points for coord in point]

  if municipality == None:
    query = f"""SELECT latitude, longitude, "Year", "Month", "Day", "Hour", "GHI", "municipality"
    FROM "NSRDB_SOLAR"
    WHERE (lon_rounded, lat_rounded) IN ({prep_points})
    ORDER BY "Month" ASC, "Day" ASC, "Hour" ASC;"""

    pointer.execute(query, coords)

  else:
    query = f"""SELECT latitude, longitude, "Year", "Month", "Day", "Hour", "GHI", "municipality"
    FROM "NSRDB_SOLAR"
    WHERE municipality = {municipality}
    AND (lon_rounded, lat_rounded) IN ({prep_points})
    ORDER BY "Month" ASC, "Day" ASC, "Hour" ASC;"""

  pointer.execute(query, coords)

  solar_data = pointer.fetchall()

  pointer.close()
  working_db.close()

  return solar_data
  
def db_fetch_IRENA_solar(valid_points:list, municipality = None):
  """ returns in this format:           longitude (xcoord) | latitude (ycoord) | jan ghi | ... | dec ghi | municipality
                              point a                                                       
                              point b                                                       """

  working_db = connect_to_db()
  while working_db == None:
    working_db = connect_to_db()

  pointer = working_db.cursor()
  prep_points = ', '.join(['(%s, %s)'] * len(valid_points))
  coords = [coord for point in valid_points for coord in point]

  if municipality == None:
    query = f"""
    SELECT xcoord, ycoord,"jan ghi1",
    "feb ghi1",
    "mar ghi1",
    "apr ghi1",
    "may ghi1",
    "jun ghi1",
    "jul ghi1",
    "aug ghi1",
    "sep ghi1",
    "oct ghi1",
    "nov ghi1",
    "dec ghi1"
    FROM "IRENA_GHI_WS20_WS60 "
    WHERE (ROUND(xcoord::NUMERIC, 6), ROUND(ycoord::NUMERIC, 6)) IN ({prep_points});"""

    pointer.execute(query, coords)
    solar_data = pointer.fetchall()
    pointer.close()
    working_db.close()

    final_solar_data = []
    temp = []

    for row in solar_data:
      for element in row:
        temp.append(float(element))
      final_solar_data.append(tuple(temp))
      temp = []
      
          
       
    return final_solar_data

munip_hourly_list = []  # List to store average GHI per hour
sum_months = [] # NSRDB monthly ghi data
monthly_ghi_data = [] # IRENA monthly ghi data


=======
>>>>>>> 041e7b1e4e0add571bf32715b874e22c3b73cbd8

with col1:
    #rearrange and then round off to match with SQL database
    temp_points =[]
    if st.session_state.selection_mode != "municipality":
        coordinate_tuples = custom_coords

    for tup in coordinate_tuples:
        new_tup = (round(tup[1],6), round(tup[0],6))
        temp_points.append(new_tup)


    if choose_from:

        valid_points = solar.look_up_points(temp_points, choose_from)

            # st.write(f"all of the points: {temp_points}")
            # st.write(f"filtered points: {valid_points}")

    else:
        # st.write(f"filtered points: No Constraint Selected. ")
        valid_points = temp_points

    # st.write(f"unfiltered: {valid_points}")

    if slope_checker:
        valid_points = filter_slope(valid_points, slope)
        # st.write(f"filtered slope {valid_points}")
        if len(valid_points) == 0:
            st.info("No results match your current selection criteria. ")

    st.markdown(f"""
        <div style="
            background-color: #ffffff;
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            font-size: 25px;
            font-weight: bold;
            color: #333;
            ">
            Total Area: {9 * len(valid_points)} km²
        </div>
        <br>
    """, unsafe_allow_html=True)

<<<<<<< HEAD
def ave_ghi_nsrdb(solar_data): #NSRDB
    """
    Compute the average GHI per hour for the given solar_data
    and populate monthly GHI sums.
    """
    # Debug the first entry to understand the data structure
    
    sum_months = [0] * 12
    
    # Group and sum GHI values by month
    for entry in solar_data:
        month = entry[3]
        ghi = entry[6]    
        
        sum_months[month - 1] += ghi
    
    #st.write(f"Monthly NSRDB GHI sums: {sum_months}")
    return sum_months

def IRENA_monthly_ghi(solar_data): #IRENA
    # Initialize the list here
    monthly_ghi_data = []
    
    for i in range(2, 14):
        # Create a tuple of values for this month from all entries
        month_data = tuple(entry[i] for entry in solar_data)
        monthly_sum = sum(month_data)*1000
        monthly_ghi_data.append(monthly_sum)
    
    # Display the data in Streamlit (if using Streamlit)
    #st.write(f"Monthly IRENA GHI sums: {monthly_ghi_data}")
    return monthly_ghi_data


# NSRDB Monthly Energy Yield function
def NSRDB_monthly_energy_yield(af, eta, sum_months, area=9, pixel_num=0): #NSRDB
    """
    Compute Monthly Energy Yield (MEY) for each month using total GHI.
    """
    mey_list_nsrdb = []
    for ghi_sum in sum_months:
        MEY = (ghi_sum * area * pixel_num * af * eta) # Converted to MWh
        mey_list_nsrdb.append(MEY)
        
    annual_energy_yield_nsrdb = sum(mey_list_nsrdb)

    return mey_list_nsrdb, annual_energy_yield_nsrdb



# IRENA Monthly Energy Yield function
def IRENA_monthly_energy_yield(af, eta, monthly_ghi_data, valid_points, area=9): #IRENA
    mey_list_irena = []
    for month_ghi in monthly_ghi_data:
        # month_ghi is already a sum, so don't try to sum it again
        MEY = (month_ghi * area * len(valid_points) * af * eta) # Convert to MWh
        mey_list_irena.append(MEY)
    
    annual_energy_yield_irena = sum(mey_list_irena)

    return mey_list_irena, annual_energy_yield_irena

#-------------------------------------------------------------------------------------------------------

    """
    Compute the solar noon hour (0-24) for each day of the year at a given coordinate, rounding to the nearest whole hour.

    Parameters:
    - latitude (float): Latitude of the location
    - longitude (float): Longitude of the location
    - year (int): Year for which to compute solar noon

    Returns:
    - solar_noon_hours (dict): Dictionary with dates as keys and rounded solar noon hour (local time) as values.
    """
#-------------------------------------------------------------------------------------------------------
def monthlyGHI(latitude, longitude):
    """
    Calculate solar noon hours for a given latitude and longitude for the year 2017.
    
    Parameters:
    - latitude: float - latitude coordinate
    - longitude: float - longitude coordinate
    
    Returns:
    - Dictionary mapping dates to solar noon hours
    """
    observer = ephem.Observer()
    observer.lat, observer.lon = str(latitude), str(longitude)
    observer.elev = 0  # Assume sea level

    solar_noon_hours = {}  # Store results
    # 2017 is not a leap year, so it has 365 days
    days_in_year = 365

    for day in range(1, days_in_year + 1):
        date = datetime(2017, 1, 1) + timedelta(days=day - 1)
        observer.date = date.strftime("%Y/%m/%d")
        solar_noon = observer.next_transit(ephem.Sun(), start=observer.date)
        solar_noon_local = ephem.localtime(solar_noon)
        rounded_hour = round(solar_noon_local.hour + solar_noon_local.minute / 60)
        rounded_hour = min(24, max(0, rounded_hour))
        solar_noon_hours[date.strftime("%Y-%m-%d")] = rounded_hour
    
    return solar_noon_hours

def highest_GHI_at_solar_noon(solar_data):
    """
    Find the highest GHI value at solar noon for the entire year 2017 from the provided solar data.
    
    Parameters:
    - solar_data: list of tuples in format (latitude, longitude, year, month, day, hour, ghi, location_name)
    
    Returns:
    - max_ghi: float - the highest GHI value at solar noon for the entire year
    """
    max_ghi = 0
    solar_noon_cache = {}  # Cache to avoid recalculating for same coordinates
    
    for entry in solar_data:
        # Extract the values from your tuple format
        latitude, longitude, year, month, day, hour, ghi, location_name = entry
        
        # Convert values to correct types
        latitude = float(latitude)
        longitude = float(longitude)
        month = int(month)
        day = int(day)
        hour = int(hour)
        ghi = float(ghi)
        
        # Since we know it's always 2017, we can simplify the cache key
        location_key = (latitude, longitude)
        
        # Calculate solar noon hours if not already in cache
        if location_key not in solar_noon_cache:
            solar_noon_cache[location_key] = monthlyGHI(latitude, longitude)
        
        solar_noon_hours = solar_noon_cache[location_key]
        date_key = f"2017-{month:02d}-{day:02d}"
        
        if date_key in solar_noon_hours and hour == solar_noon_hours[date_key]:
            if ghi > max_ghi:
                max_ghi = ghi
    
    return max_ghi
#-------------------------------------------------------------------------------------------------------

#SOLAR CAPACITY
def NSRDB_capacity(power_density, area=9):
    cap_nsrdb = (area * power_density * len(valid_points))
    #st.write(f"power density nsrdb: {power_density}")
    return cap_nsrdb


def IRENA_capacity (power_density=1000, area=9):
    cap_irena = (area * power_density * len(valid_points))
    return cap_irena
#-------------------------------------------------------------------------------------------------------
#SOLAR_CAPACITY_FACTOR
days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]  
hours_in_month = [days * 24 for days in days_in_month]

def NSRDB_capacity_factor(cap_nsrdb, mey_list_nsrdb): #NSRDB
    print(f"cap_nsrdb: {cap_nsrdb}")
    print(f"mey_list_nsrdb: {mey_list_nsrdb}")
    print(f"hours_in_month: {hours_in_month}")
    
    cf_list_nsrdb = []
    cf_percentage_list_nsrdb = []
    
    for i, (month, hours) in enumerate(zip(mey_list_nsrdb, hours_in_month)):
        print(f"Month {i+1}: month={month}, hours={hours}")
        
        # Check for zero division
        if cap_nsrdb == 0 or hours == 0:
            print(f"Warning: Division by zero detected in month {i+1}. cap_nsrdb={cap_nsrdb}, hours={hours}")
            capacity_factor_nsrdb = 0  # Set to zero or another appropriate default
            cf_percentage_nsrdb = 0
        else:
            capacity_factor_nsrdb = month / (cap_nsrdb * hours)
            cf_percentage_nsrdb = capacity_factor_nsrdb*100
        
        cf_list_nsrdb.append(capacity_factor_nsrdb)
        cf_percentage_list_nsrdb.append(cf_percentage_nsrdb)
    
    return cf_list_nsrdb, cf_percentage_list_nsrdb

def IRENA_capacity_factor(cap_irena, mey_list_irena):
    cf_list_irena = []
    cf_percentage_list_irena = []
    
    for month, hours in zip(mey_list_irena, hours_in_month):
        capacity_factor_irena = month / (cap_irena * hours)
        cf_percentage_irena = capacity_factor_irena*100
        cf_list_irena.append(capacity_factor_irena)
        cf_percentage_list_irena.append(cf_percentage_irena)
    
    return cf_list_irena, cf_percentage_list_irena

#-------------------------------------------------------------------------------------------------------

#SOLAR_LCOE
def NSRDB_lcoe(cf_list_nsrdb, fixed_charge_rate=0.092, capital_cost=75911092, fixed_om_cost=759111, variable_om_cost=0, fuel_cost=0):
    lcoe_list_nsrdb = []
    
    for cf, hours in zip(cf_list_nsrdb, hours_in_month):
        if cf == 0:
            lcoe_list_nsrdb.append(float('inf'))  # Avoid division by zero
        else:
            denominator = cf * hours
            lcoe_value_nsrdb = (((fixed_charge_rate * capital_cost + fixed_om_cost) / denominator) + variable_om_cost + fuel_cost) 
            lcoe_list_nsrdb.append(lcoe_value_nsrdb)
    
    return lcoe_list_nsrdb

def IRENA_lcoe(cf_list_irena, fixed_charge_rate=0.092, capital_cost=75911092, fixed_om_cost=759111, variable_om_cost=0, fuel_cost=0):
    lcoe_list_irena = []
    
    for cf, hours in zip(cf_list_irena, hours_in_month):
        if cf == 0:
            lcoe_list_irena.append(float('inf'))  # Avoid division by zero
        else:
            denominator = cf * hours
            lcoe_value_irena = (((fixed_charge_rate * capital_cost + fixed_om_cost) / denominator) + variable_om_cost + fuel_cost) 
            lcoe_list_irena.append(lcoe_value_irena)
    
    return lcoe_list_irena
=======
with col_metric1:
    # #fetch irena and nsrdb again for the valid points
    solar_data = solar.db_fetch_hourly_solar(valid_points:list, municipality = None)
    final_solar_data = solar.db_fetch_IRENA(valid_points, municipality = None)

    #fetch NSRDB spatial average
    sum_months = solar.ave_ghi_nsrdb(solar_data)

    #fetch IRENA monthly
    monthly_ghi_data = solar.IRENA_monthly_ghi(final_solar_data)

    # # initialize

    NSRDB_monthly_energy_yield = []
    IRENA_monthly_energy_yield = []
    cf_list_nsrdb = []
    cf_percentage_list_nsrdb = []
    cf_list_irena = []
    cf_percentage_list_irena = []
    lcoe_list_nsrdb = []
    lcoe_list_irena = []
    
    #MEY
    mey_list_nsrdb, annual_energy_yield_nsrdb = solar.NSRDB_monthly_energy_yield(sum_months, len(valid_points), area=9, af=0.7, eta=0.2)
    mey_list_irena, annual_energy_yield_irena = solar.IRENA_monthly_energy_yield(monthly_ghi_data, len(valid_points), area=9, af=0.7, eta=0.2)
>>>>>>> 041e7b1e4e0add571bf32715b874e22c3b73cbd8

    #CAP
    solar_noon_hours = solar.monthlyGHI(latitude, longitude, 2017)
    max_ghi = solar.highest_GHI_at_solar_noon(solar_data)
    cap_nsrdb = solar.NSRDB_capacity(max_ghi, area=9)
    cap_irena = IRENA_capacity (valid_points, len(valid_points), power_density=1000, area=9)

    #CF

    #LCOE

    # process per month
    for month in range(1, 13):
    # compute energy yield using NSRDB dataset first
        month_40ws = nsrdb_monthly40WS[month - 1]
        month_60ws = nsrdb_monthly60WS[month - 1]
        hours = len(month_40ws)

        e_yield1, std = solar.calc_energy_yield(hours, len(valid_points), 40, month_40ws, 60, month_60ws, turbine_model )
        NSRDB_monthly_energy_yield.append(e_yield1)

    # st.write("Using IRENA:")
    # st.write(f"month: {month} energy yield: {e_yield1} stdev: {std}")

    #then compute using IRENA dataset
        for point in range(len(irena_ds20)):

            e_yield2 = solar.calc_energy_yield(hours, len(valid_points), 20, [irena_ds20[point][month -1]], 60, [irena_ds60[point][month -1]], turbine_model, nsrdb_stdev= std, source= "IRENA" )

            month_point_yield.append(e_yield2)

        total = 0
        for point_yield in month_point_yield:
            total += point_yield
            
        IRENA_monthly_energy_yield.append(total)

    # st.write(f"e yield comparison ---- month: {month} nsrdb: {e_yield1} irena: {total}")
    # diff = abs(e_yield1 - total)

    # st.write(f"this is how much they differ: {str(round(diff, 2))}")

    # handle municipality-based assessment

    capacity = solar.compute_capacity(turbine_model, len(irena_ds20))
    # st.write(f"capacity: {capacity}")

    month_cntr = 0
    irena_lcoe = []
    nsrdb_lcoe = []

    for e_yield in NSRDB_monthly_energy_yield:
        month_60ws = nsrdb_monthly60WS[month_cntr]
        hours = len(month_40ws)
        nsrdb_cf = solar.compute_monthly_capacity_factor(e_yield, hours, capacity)
        NSRDB_monthly_cf.append(nsrdb_cf)
        nsrdb_lcoe.append(solar.compute_lcoe(nsrdb_cf))
        month_cntr += 1

    month_cntr = 0

    for e_yield in IRENA_monthly_energy_yield:
        month_60ws = nsrdb_monthly60WS[month_cntr]
        hours = len(month_40ws)
        irena_cf = solar.compute_monthly_capacity_factor(e_yield, hours, capacity)
        irena_lcoe.append(solar.compute_lcoe(irena_cf))
        IRENA_monthly_cf.append(irena_cf)
        month_cntr += 1

<<<<<<< HEAD
  valid_points = look_up_points(temp_points, choose_from)

else:
  #st.write(f"filtered points: No Constraint Selected. ")
  valid_points = temp_points

holder = []
for point in valid_points:
    new = (point[1], point[0])
    holder.append(new)

# valid_points = holder

#Data Visualization
def plot_monthly_value(IRENA: list, NSRDB: list, IRENA_cf: list, NSRDB_cf, IRENA_lcoe, NSRDB_lcoe):
    months = [
        "Jan", "Feb", "Mar", "Apr", "May", "June",
        "July", "Aug", "Sep", "Oct", "Nov", "Dec"
    ]

    energy_df = pd.DataFrame({
        "Month": months,
        "Energy Yield (IRENA)" : IRENA,
        "Energy Yield (NSRDB)" : NSRDB
    })

    capacity_df = pd.DataFrame({
        "Month": months,
        "Capacity Factor (IRENA)": IRENA_cf,
        "Capacity Factor (NSRDB)": NSRDB_cf
    })

    lcoe_df = pd.DataFrame({
        "Month": months,
        "LCOE (IRENA)": IRENA_lcoe,
        "LCOE (NSRDB)": NSRDB_lcoe
    })

    
    
    fig1 = px.bar(
        energy_df,
        x='Month',
        y=['Energy Yield (IRENA)', 'Energy Yield (NSRDB)'],
        barmode='group',
        title="Monthly Energy Yield (WH)",
        color_discrete_sequence=['#FFD63A', '#FFA955']  # Change color palette here
    )
  
    
    fig2 = px.bar(
        capacity_df, x='Month', 
        y=['Capacity Factor (IRENA)', 'Capacity Factor (NSRDB)'], 
        barmode='group', title="Monthly Capacity Factor (%)",
        color_discrete_sequence=['#67AE6E', '#90C67C']
    )

    
    fig3 = px.bar(
        lcoe_df, x='Month', 
        y=['LCOE (IRENA)', 'LCOE (NSRDB)'], 
        barmode='group', title="Monthly LCOE (₱)",
        color_discrete_sequence=['#3D90D7', '#7AC6D2']
    )

    return fig1, fig2, fig3
    


#st.write(f"all of the points: {temp_points}")
#st.write(f"filtered points: {valid_points}")
#call functions here

irena_solar_data = db_fetch_IRENA_solar(valid_points)
#st.write(f"this is irena: {irena_solar_data}")

nsrdb_solar_data = db_fetch_hourly_solar(valid_points)
#st.write(f"this is nsrdb: {nsrdb_solar_data[0]} and {nsrdb_solar_data[1]}")

monthly_ghi_data = IRENA_monthly_ghi(irena_solar_data)

mey_list_irena, annual_energy_yield_irena = IRENA_monthly_energy_yield(af, eta, monthly_ghi_data, valid_points, area=9)
cap_irena = IRENA_capacity (power_density=1000, area=9)
cf_list_irena, cf_percentage_list_irena = IRENA_capacity_factor(cap_irena, mey_list_irena)
lcoe_list_irena = IRENA_lcoe(cf_list_irena)

solar_data = [()]
sum_months = ave_ghi_nsrdb(nsrdb_solar_data)
mey_list_nsrdb, annual_energy_yield_nsrdb = NSRDB_monthly_energy_yield(af, eta,sum_months, area=9, pixel_num = len(valid_points))
power_density = highest_GHI_at_solar_noon(nsrdb_solar_data)
cap_nsrdb = NSRDB_capacity (power_density, area=9)
cf_list_nsrdb, cf_percentage_list_nsrdb = NSRDB_capacity_factor(cap_nsrdb, mey_list_nsrdb)
lcoe_list_nsrdb = NSRDB_lcoe(cf_list_nsrdb)

# st.write(f"MEY list irena: {mey_list_irena}, annual energy yield irena: {annual_energy_yield_irena}")
# st.write(f"cap irena: {cap_irena}")
# st.write(f"cf list: {cf_list_irena}, cf percent: {cf_percentage_list_irena}")
# st.write(f"lcoe list irena: {lcoe_list_irena}")

# st.write(f"MEY list nsrdb: {mey_list_nsrdb}, annual energy yield nsrdb: {annual_energy_yield_nsrdb}")
# st.write(f"cap nsrdb: {cap_nsrdb}")
# st.write(f"cf list nsrdb: {cf_list_nsrdb}, cf percent nsrdb: {cf_percentage_list_nsrdb}")
# st.write(f"lcoe list nsrdb: {lcoe_list_nsrdb}")

eyield, cfactor, lcoee = plot_monthly_value(mey_list_irena, mey_list_nsrdb, cf_percentage_list_irena, cf_percentage_list_nsrdb,lcoe_list_irena, lcoe_list_nsrdb)
st.write(f'mey irena: {mey_list_irena}')
st.write(f'mey nsrdb: {mey_list_nsrdb}')
st.write(f'cf_list_irena: {cf_percentage_list_irena}')
st.write(f'cf_list_nsrdb: {cf_percentage_list_nsrdb}')
st.write(f'lcoe_nsrdb: {lcoe_list_nsrdb}')
st.write(f'lcoe_irena: {lcoe_list_irena}')
with disp1:
    st.plotly_chart(eyield)

with disp2:
    st.plotly_chart(cfactor)

with disp3:
    st.plotly_chart(lcoee)

ave_yield_irena = annual_energy_yield_irena/12
ave_yield_nsrdb = annual_energy_yield_nsrdb/12
ave_lcoe_irena = sum(lcoe_list_irena)/12
ave_lcoe_nsrdb = sum(lcoe_list_nsrdb)/12


with summary:
    st.markdown(f"""
        <style>
            .metric-container {{
                background-color: #ffffff;
                border-radius: 5px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                margin-bottom: 8px;
                overflow: hidden;
            }}
            .metric-header {{
                background-color: #F75A5A;
                padding: 8px;
                text-align: center;
            }}
            .metric-title {{
                font-size: 20px;
                font-weight: bold;
                color: white !important;
                margin: 0;
            }}
            .metric-content {{
                display: flex;
                padding: 8px;
            }}
            .icon-container {{
                flex: 0 0 80px;
                display: flex;
                justify-content: center;
                align-items: center;
            }}
            .icon {{
                width: 80px;
                height: 80px;
            }}
            .data-container {{
                flex: 1;
                padding-left: 10px;
            }}
            .data-row {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 5px;
            }}
            .source-label {{
                font-size: 17px;
                font-weight: bold;
                color: #000;
                flex: 0 0 80px;
            }}
            .value {{
                font-size: 18px;
                font-weight: normal;
                color: #000;
                text-align: right;
                flex: 1;
            }}
        </style>
        
        <!-- Capacity -->
        <div class="metric-container">
            <div class="metric-header">
                <h2 class="metric-title">CAPACITY</h2>
            </div>
            <div class="metric-content">
                <div class="icon-container">
                    <img src="https://raw.githubusercontent.com/yelsha07/icons/refs/heads/main/1.png" class="icon" />
                </div>
                <div class="data-container">
                    <div class="data-row">
                        <div class="source-label">IRENA</div>
                        <div class="value">{round((cap_irena), 3)} MW</div>
                    </div>
                    <div class="data-row">
                        <div class="source-label">NSRDB</div>
                        <div class="value">{round((cap_nsrdb), 3)} MW</div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Energy Yield -->
        <div class="metric-container">
            <div class="metric-header">
                <h2 class="metric-title">AVE. ENERGY YIELD</h2>
            </div>
            <div class="metric-content">
                <div class="icon-container">
                    <img src="https://raw.githubusercontent.com/yelsha07/icons/refs/heads/main/2.png" class="icon" />
                </div>
                <div class="data-container">
                    <div class="data-row">
                        <div class="source-label">IRENA</div>
                        <div class="value">{round((ave_yield_irena), 3)} MWh</div>
                    </div>
                    <div class="data-row">
                        <div class="source-label">NSRDB</div>
                        <div class="value">{round((ave_yield_nsrdb), 3)} MWh</div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- LCOE -->
        <div class="metric-container">
            <div class="metric-header">
                <h2 class="metric-title">AVE. LCOE</h2>
            </div>
            <div class="metric-content">
                <div class="icon-container">
                    <img src="https://raw.githubusercontent.com/yelsha07/icons/refs/heads/main/3.png" class="icon" />
                </div>
                <div class="data-container">
                    <div class="data-row">
                        <div class="source-label">IRENA</div>
                        <div class="value">₱{round((ave_lcoe_irena), 3)}/MWh</div>
                    </div>
                    <div class="data-row">
                        <div class="source-label">NSRDB</div>
                        <div class="value">₱{round((ave_lcoe_nsrdb), 3)}/MWh</div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    #updated database
=======
ave_eyield = sum(IRENA_monthly_energy_yield)/12
ave_lcoe = sum(irena_lcoe)/12
with summary:
    st.markdown("""
    <div style="
        background-color: #F75A5A;
        padding: 12px 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        font-size: 24px;
        font-weight: 600;
        color: #FFFDF6;
        margin-bottom: 15px;
    ">
        Summary
    </div>
""", unsafe_allow_html=True)
>>>>>>> 041e7b1e4e0add571bf32715b874e22c3b73cbd8

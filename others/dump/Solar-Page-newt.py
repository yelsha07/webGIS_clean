# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 22:17:04 2025

@author: vince (simplified)
"""

import streamlit as st
import geopandas as gpd
import pandas as pd
import folium
import os
import requests
import zipfile
import io
import tempfile
from datetime import datetime
from streamlit_folium import st_folium
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

import others.solar_algo_newt as solar
import sys


# Add the parent directory to the system path so 'logic' can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# Set page config
st.set_page_config(layout="wide", page_title="Philippine Solar Energy Potential Assessment")

st.title("Solar Potential Assessment")
#st.markdown("Select a municipality, or draw a custom area (minimum 3km x 3km) to assess solar potential.")

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

st.success(f"Successfully loaded {len(gdf_municipalities)} municipalities and {len(gdf_points)} points!")

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
# municipality selection

        municipality_list = [""] + sorted(gdf_municipalities[municipality_name_col].tolist())

        # Create dropdown without on_change handler to avoid circular dependencies
        selected_from_dropdown = st.selectbox(
            "Select from dropdown", 
            municipality_list,
            index=0 if st.session_state.selected_muni is None else 
                municipality_list.index(st.session_state.selected_muni) if st.session_state.selected_muni in municipality_list else 0,
            key="dropdown_muni"  # Changed key name to avoid conflict
        )

        # Handle dropdown selection - separate from widget creation
        if selected_from_dropdown and not st.session_state.map_click_source:
            if select_municipality(selected_from_dropdown):
                st.rerun()
        # Reset map click source flag after dropdown handling
        st.session_state.map_click_source = False


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
            
            # Use spatial join to find municipality containing the clicked point
            try:
                matched = gpd.sjoin(click_gdf, gdf_municipalities, predicate="within", how="left")
                
                if not matched.empty and not pd.isna(matched.iloc[0][municipality_name_col]):
                    selected_municipality = matched.iloc[0][municipality_name_col]
                else:
                    # If no exact match found, find nearest municipality
                    gdf_municipalities["distance"] = gdf_municipalities.geometry.centroid.distance(point)
                    nearest_row = gdf_municipalities.loc[gdf_municipalities["distance"].idxmin()]
                    selected_municipality = nearest_row[municipality_name_col]
                    st.info(f"Selected nearest municipality: {selected_municipality}")
                
                # Update selection through the central function
                if select_municipality(selected_municipality, source="map"):
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

    selected_geom = selected_gdf.geometry.union_all()
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
        selected_geom_projected = selected_gdf.to_crs(epsg=3857).geometry.union_all()
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

    col_metric1, col_metric2 = st.columns([1,2])
    #c1, c2 = st.columns(2)
    #c1.metric("Area", f"{area_km2:.2f} km²")
    #c2.metric("Points within Municipality", f"{len(intersecting)}")

    disp1, disp2, disp3 = st.columns([1,1,1])

    interactive, summary = st.columns([2,1])

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
        st_folium(m)

else:  # Draw Custom Area mode
    with col1:
        
        # Create the dropdown first
        municipality_list = [""] + sorted(gdf_municipalities[municipality_name_col].tolist())

        # Create dropdown without on_change handler to avoid circular dependencies
        selected_from_dropdown = st.selectbox(
            "Select a municipality:", 
            municipality_list,
            index=0 if st.session_state.selected_muni is None else 
                municipality_list.index(st.session_state.selected_muni) if st.session_state.selected_muni in municipality_list else 0,
            key="dropdown_muni"  # Changed key name to avoid conflict
        )

        # Handle dropdown selection - separate from widget creation
        if selected_from_dropdown and not st.session_state.map_click_source:
            if select_municipality(selected_from_dropdown):
                st.rerun()
        # Reset map click source flag after dropdown handling
        st.session_state.map_click_source = False
    
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
        st.success(f"Area selected: {drawn_area['area_km2']:.2f} km² (Width: {drawn_area['width_km']:.2f} km, Height: {drawn_area['height_km']:.2f} km)")
        
        # Find points within the drawn area
        try:
            intersecting = gdf_points[gdf_points.geometry.within(drawn_area['geometry'])]
        except Exception as e:
            st.error(f"Error finding points within custom area: {e}")
            intersecting = gpd.GeoDataFrame(geometry=[], crs=gdf_points.crs)
        
        # Display area metrics
        st.subheader("📊 Custom Area Information")
        c1, c2, c3 = st.columns(3)
        c1.metric("Area", f"{drawn_area['area_km2']:.2f} km²")
        c2.metric("Width x Height", f"{drawn_area['width_km']:.2f} km x {drawn_area['height_km']:.2f} km")
        c3.metric("Points within Area", f"{len(intersecting)}")
        
        # Create and display map
        st.subheader("🗺️ Interactive Map of Custom Area")
        area_map = folium.Map(location=drawn_area["center"], zoom_start=10)
        
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
        st_folium(area_map)
        
        # # Display table of points
        # st.subheader(f"📍 {len(intersecting)} Points Inside Custom Area")
        # if not intersecting.empty:
        #     display_cols = [col for col in intersecting.columns if col != 'geometry']
        #     st.dataframe(intersecting[display_cols])
        # else:
        #     st.info("No points found within the drawn area.")
    else:
        st.info(" Draw a rectangle on the map using the rectangle tool. The area will be analyzed automatically.")
# First define once at the top
@st.cache_data
def convert_df_to_csv(_df):
    export_df = _df.drop(columns=['geometry'])
    return export_df.to_csv(index=False).encode('utf-8')

# Then in your logic:
if st.session_state.selection_mode == "municipality" and st.session_state.selected_muni:
    selected_gdf = gdf_municipalities[gdf_municipalities[municipality_name_col] == st.session_state.selected_muni]
    if not selected_gdf.empty:
        selected_geom = selected_gdf.geometry.union_all()
        intersecting = gdf_points[gdf_points.geometry.within(selected_geom)]
        if not intersecting.empty:
            csv = convert_df_to_csv(intersecting)
            st.download_button(
                label="Download points as CSV",
                data=csv,
                file_name=f"{st.session_state.selected_muni}_points.csv",
                mime='text/csv',
            )

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

with col1:
#rearrange and then round off to match with SQL database
    temp_points =[]
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

with col_metric1:
    # #fetch irena and nsrdb again for the valid points
    solar_data = solar.db_fetch_hourly_solar(valid_points, municipality = None)
    final_solar_data = solar.db_fetch_IRENA_solar(valid_points, municipality = None)

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

    #CAP
    solar_noon_hours = solar.monthlyGHI((entry[0] for entry in solar_data), (entry[1] for entry in solar_data), 2017)
    max_ghi = solar.highest_GHI_at_solar_noon(solar_data)
    cap_nsrdb = solar.NSRDB_capacity(max_ghi, area=9)
    cap_irena = solar.IRENA_capacity (valid_points, len(valid_points), power_density=1000, area=9)

    #CF
    cf_list_nsrdb, cf_percentage_list_nsrdb = solar.NSRDB_capacity_factor(cap_nsrdb, mey_list_nsrdb)
    cf_list_irena, cf_percentage_list_irena = solar.IRENA_capacity_factor(cap_irena, mey_list_irena)

    #LCOE
    lcoe_list_nsrdb = solar.NSRDB_lcoe(cf_list_nsrdb, fixed_charge_rate=0.092, capital_cost=75911092, fixed_om_cost=759111, variable_om_cost=0, fuel_cost=0)
    lcoe_list_irena = solar.IRENA_lcoe(cf_list_irena, fixed_charge_rate=0.092, capital_cost=75911092, fixed_om_cost=759111, variable_om_cost=0, fuel_cost=0)
ave_cap = sum(cap_irena)/12
ave_eyield = sum(IRENA_monthly_energy_yield)/12
ave_lcoe = sum(lcoe_list_irena)/12    
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
        
        # st.markdown(f"""
        #     <div style="text-align: center; margin-bottom: 20px;">
        #         <img src="https://raw.githubusercontent.com/yelsha07/icons/refs/heads/main/1.png" width="50" height="50" />
        #         Capacity: {capacity} W
        #     </div>
        # """, unsafe_allow_html=True)

        # st.markdown(f"""
        #     <div style="text-align: center; margin-bottom: 20px;">
        #         <img src="https://raw.githubusercontent.com/yelsha07/icons/refs/heads/main/2.png" width="50" height="50" />
        #         Energy Yield: {76} Wh
        #     </div>
        # """, unsafe_allow_html=True)

        # st.markdown(f"""
        #     <div style="text-align: center; margin-bottom: 20px;">
        #         <img src="https://raw.githubusercontent.com/yelsha07/icons/refs/heads/main/3.png" width="50" height="50" />
        #         LCOE: {76} USD/kWh
        #     </div>
        # """, unsafe_allow_html=True)

        
    st.markdown(f"""
    <style>
        .tooltip {{
            position: relative;
            display: inline-block;
            cursor: pointer;
        }}
        .tooltip .tooltiptext {{
            visibility: hidden;
            width: 220px;
            background-color: #555;
            color: #fff;
            text-align: left;
            border-radius: 6px;
            padding: 5px;
            position: absolute;
            z-index: 1;
            bottom: 125%;
            left: 50%;
            margin-left: -110px;
            opacity: 0;
            transition: opacity 0.3s;
            font-size: 14px;
        }}
        .tooltip:hover .tooltiptext {{
            visibility: visible;
            opacity: 1;
        }}
        .container {{
            text-align: center;
            padding: 10px;
            margin-bottom: 15px;
            display: block;
        }}
        .metric {{
            font-size: 20px;
            font-weight: normal;
            margin-top: 10px;
            line-height: 1.1;
        }}
        .number {{
            font-size: 25px;
            font-weight: bold;
            margin-top: 10px;
        }}
        .icon {{
            display: block;
            margin: 0 auto;
        }}
    </style>

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
        <div class="container">
            <img src="https://raw.githubusercontent.com/yelsha07/icons/refs/heads/main/1.png" width="40" height="40" class="icon" />
            <div class="tooltip">
                <p class="metric">Capacity</p>
                <span class="tooltiptext">The total capacity of the system in watts (W). It indicates the maximum output that can be generated by the wind system.</span>
            </div>
            <div class="number">{round((ave_cap/1000000), 3)} MW</div>
        </div>
    </div>
    <br>
    """, unsafe_allow_html=True)

    ey, lc = st.columns([1,1])

    with ey:
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
            width: 220px;
            height: 210px;
            ">
            <div class="container">
                <img src="https://raw.githubusercontent.com/yelsha07/icons/refs/heads/main/2.png" width="40" height="40" class="icon" />
                <div class="tooltip">
                    <p class="metric">Ave. Energy Yield</p>
                    <span class="tooltiptext">The energy generated by the wind system, typically measured in watt-hours (Wh) or kilowatt-hours (kWh). It depends on the wind speed and capacity factor.</span>
                </div>
                <div class="number">{round((ave_eyield), 3)} MWh</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with lc:
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
            width: 220px;
            height: 210px;
            ">
            <div class="container">
                <img src="https://raw.githubusercontent.com/yelsha07/icons/refs/heads/main/3.png" width="40" height="40" class="icon" />
                <div class="tooltip">
                    <p class="metric">Ave. LCOE</p>
                    <span class="tooltiptext">Levelized Cost of Energy: A measure of the cost per unit of energy produced over the system's lifetime, considering both capital and operational costs.</span>
                </div>
                <div class="number">₱{round((ave_lcoe), 3)}M</div>
            </div>
        </div>
        """, unsafe_allow_html=True)



        #ave monthly energy yield
        #ave cf
        #ave LCOE



eyield, cfactor, lcoe = solar.plot_monthly_value(IRENA_monthly_energy_yield, NSRDB_monthly_energy_yield, cf_list_irena, cf_list_nsrdb, lcoe_list_irena, lcoe_list_nsrdb)

with disp1:
    st.plotly_chart(eyield)
    st.write("Test")

with disp2:
    st.plotly_chart(cfactor)
    st.write("Test2")
with disp3:
    st.plotly_chart(lcoe)
    st.write("Test3")
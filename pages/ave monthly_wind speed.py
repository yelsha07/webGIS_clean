import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
from scipy.optimize import fsolve

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

# def connect_to_db():  # utilized
#   try:
#     connection = sql.connect(
#             dbname= "webGIS",
#             user="postgres",
#             password="ashley",
#             host="localhost",
#             port= "5432"
#         )
    
#     #st.text("Database connection successful!")
#     return connection
#   except Exception as e:

#     st.text("Database connection failed:")

#     return None 

# def db_fetch_hourly_solar(lat_rounded, lon_rounded, municipality = None):
#   """ returns in this format:           Lat | Long | Month | Day | Hour | GHI | MuniCipality 
#                               point a                               1                        
#                               point b                               1                        """

#   working_db = connect_to_db()
#   while working_db == None:
#     working_db = connect_to_db()

#   pointer = working_db.cursor()

# #   prep_points = ', '.join(['(%s, %s)'] * len(valid_points))
# #   coords = [coord for point in valid_points for coord in point]

#   if municipality == None:
#     query = f"""SELECT "Month", "GHI"
#     FROM "NSRDB_SOLAR"
#     WHERE lat_rounded = {lat_rounded} AND lon_rounded = {lon_rounded}
#     ORDER BY "Month" ASC, "Day" ASC, "Hour" ASC;"""

#     pointer.execute(query)

#   else:
#     query = f"""SELECT "Month", "GHI"
#     FROM "NSRDB_SOLAR"
#     WHERE lat_rounded = {lat_rounded} AND lon_rounded = {lon_rounded}
#     ORDER BY "Month" ASC, "Day" ASC, "Hour" ASC;"""

#     pointer.execute(query)


#   solar_data = pointer.fetchall()

#   pointer.close()
#   working_db.close()

#   return solar_data


# def db_fetch_IRENA_solar(lat_rounded, lon_rounded, municipality = None):
#   """ returns in this format:           longitude (xcoord) | latitude (ycoord) | jan ghi | ... | dec ghi | municipality
#                               point a                                                       
#                               point b                                                       """

#   working_db = connect_to_db()
#   while working_db == None:
#     working_db = connect_to_db()

#   pointer = working_db.cursor()
# #   prep_points = ', '.join(['(%s, %s)'] * len(valid_points))
# #   coords = [coord for point in valid_points for coord in point]

#   if municipality == None:
#     query = f"""
#     SELECT "jan ghi1",
#     "feb ghi1",
#     "mar ghi1",
#     "apr ghi1",
#     "may ghi1",
#     "jun ghi1",
#     "jul ghi1",
#     "aug ghi1",
#     "sep ghi1",
#     "oct ghi1",
#     "nov ghi1",
#     "dec ghi1"
#     FROM "IRENA_GHI_WS20_WS60 "
#     WHERE lat_rounded = {lat_rounded} AND lon_rounded = {lon_rounded}"""

#     pointer.execute(query)
#     solar_data = pointer.fetchall()
#     pointer.close()
#     working_db.close()

#     final_solar_data = []
#     temp = []

#     for row in solar_data:
#       for element in row:
#         temp.append(float(element))
#       final_solar_data.append(tuple(temp))
#       temp = []
      
          
       
#     return final_solar_data

# def ave_ghi_nsrdb(solar_data): #NSRDB
    
#     """
#     Compute the average GHI per hour for the given solar_data
#     and populate monthly GHI sums.
#     """
#     # Debug the first entry to understand the data structure
    
#     sum_months = [0] * 12
#     print("hatdog")
#     # Group and sum GHI values by month
#     for entry in solar_data:
#         month = entry[0]
#         ghi = entry[1]    
        
#         sum_months[month - 1] += ghi
    
#     #st.write(f"Monthly NSRDB GHI sums: {sum_months}")
#     return sum_months

# def IRENA_monthly_ghi(solar_data): #IRENA
#     # Initialize the list here
#     monthly_ghi_data = []
    
#     for i in range(0, 12):
#         # Create a tuple of values for this month from all entries
#         month_data = tuple(entry[i] for entry in solar_data)
#         monthly_sum = sum(month_data)*1000
#         monthly_ghi_data.append(monthly_sum)
#     # st.write(f"len valid points irena mey: {len(valid_points_final)}")
#     # Display the data in Streamlit (if using Streamlit)
#     #st.write(f"Monthly IRENA GHI sums: {monthly_ghi_data}")
#     return monthly_ghi_data
# lat_rounded = 16.262501
# lon_rounded = 119.887502
# solar_data = db_fetch_hourly_solar(lat_rounded, lon_rounded, municipality = None)
# final_solar_data = db_fetch_IRENA_solar(lat_rounded, lon_rounded, municipality = None)
# sum_months = ave_ghi_nsrdb(solar_data)
# monthly_ghi_data = IRENA_monthly_ghi(final_solar_data)
# print(f"NSRDB Monthly GHI:{sum_months}")
# print(f"IRENA Monthly GHI:{monthly_ghi_data}")

import psycopg2 as sql
import pandas as pd
import streamlit as st

def connect_to_db():
    try:
        connection = sql.connect(
            dbname="webGIS",
            user="postgres",
            password="ashley",
            host="localhost",
            port="5432"
        )
        return connection
    except Exception as e:
        print(f"Database connection failed: {e}")
        return None 

def db_fetch_hourly_solar(lat_rounded, lon_rounded, municipality=None):
    """Fetch NSRDB solar data"""
    working_db = connect_to_db()
    if working_db is None:
        return []

    pointer = working_db.cursor()

    query = f"""SELECT "Month", "GHI"
    FROM "NSRDB_SOLAR"
    WHERE lat_rounded = {lat_rounded} AND lon_rounded = {lon_rounded}
    ORDER BY "Month" ASC, "Day" ASC, "Hour" ASC;"""

    try:
        pointer.execute(query)
        solar_data = pointer.fetchall()
    except Exception as e:
        print(f"Error fetching NSRDB data for {lat_rounded}, {lon_rounded}: {e}")
        solar_data = []
    finally:
        pointer.close()
        working_db.close()

    return solar_data

def db_fetch_IRENA_solar(lat_rounded, lon_rounded, municipality=None):
    """Fetch IRENA solar data"""
    working_db = connect_to_db()
    if working_db is None:
        return []

    pointer = working_db.cursor()

    query = f"""
    SELECT "jan ghi1",
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
    WHERE lat_rounded = {lat_rounded} AND lon_rounded = {lon_rounded}"""

    try:
        pointer.execute(query)
        solar_data = pointer.fetchall()
        
        final_solar_data = []
        for row in solar_data:
            temp = []
            for element in row:
                temp.append(float(element))
            final_solar_data.append(tuple(temp))
        
        return final_solar_data
    except Exception as e:
        print(f"Error fetching IRENA data for {lat_rounded}, {lon_rounded}: {e}")
        return []
    finally:
        pointer.close()
        working_db.close()

def ave_ghi_nsrdb(solar_data):
    """Compute monthly GHI sums for NSRDB data"""
    sum_months = [0] * 12
    
    for entry in solar_data:
        if len(entry) >= 2:
            month = entry[0]
            ghi = entry[1]    
            if month is not None and ghi is not None:
                sum_months[month - 1] += ghi
    
    return sum_months

def IRENA_monthly_ghi(solar_data):
    """Process IRENA solar data to get monthly GHI"""
    if not solar_data:
        return [0] * 12
        
    monthly_ghi_data = []
    
    for i in range(0, 12):
        month_data = tuple(entry[i] for entry in solar_data)
        monthly_sum = sum(month_data) * 1000
        monthly_ghi_data.append(monthly_sum)
    
    return monthly_ghi_data

def process_coordinates_to_csv(coordinates_list, output_filename="solar_data_output.csv"):
    """
    Process multiple coordinates and save to CSV with separate NSRDB and IRENA columns
    
    Parameters:
    coordinates_list: List of tuples [(lat1, lon1), (lat2, lon2), ...]
    output_filename: Name of the output CSV file
    """
    
    # List to store all results
    all_results = []
    
    print(f"Processing {len(coordinates_list)} coordinate pairs...")
    
    for i, (lat, lon) in enumerate(coordinates_list):
        print(f"Processing coordinate {i+1}/{len(coordinates_list)}: ({lat}, {lon})")
        
        # Fetch data for current coordinates
        nsrdb_data = db_fetch_hourly_solar(lat, lon)
        irena_data = db_fetch_IRENA_solar(lat, lon)
        
        # Process both datasets
        nsrdb_monthly = ave_ghi_nsrdb(nsrdb_data) if nsrdb_data else [0] * 12
        irena_monthly = IRENA_monthly_ghi(irena_data) if irena_data else [0] * 12
        
        if not nsrdb_data:
            print(f"  No NSRDB data found for ({lat}, {lon})")
        if not irena_data:
            print(f"  No IRENA data found for ({lat}, {lon})")
        
        # Create one row per month with both NSRDB and IRENA values
        for month_idx in range(12):
            all_results.append({
                'lat': lat,
                'lon': lon,
                'month': month_idx + 1,  # 1-12 instead of 0-11
                'nsrdb_ghi': nsrdb_monthly[month_idx],
                'irena_ghi': irena_monthly[month_idx]
            })
        
        print(f"  Completed processing for ({lat}, {lon})")
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(all_results)
    df = df.sort_values(['lat', 'lon', 'month'])
    df.to_csv(output_filename, index=False)
    
    print(f"\nResults saved to {output_filename}")
    print(f"Total records: {len(df)}")
    print(f"Unique coordinates: {len(coordinates_list)}")
    print(f"Records per coordinate: {len(df) // len(coordinates_list)} months")
    
    # Display summary
    print("\nSample of the data:")
    print(df.head(10))
    
    return df

# Example usage with 10 coordinate pairs
if __name__ == "__main__":
    # Define your 10 lat/lon pairs here
    coordinates = [
        (16.262501, 119.887502),
        (17.587501,	121.587502),
        (13.762500,	122.337502),
        (12.937500,	123.862502),
        (12.962500,	120.837502),
        (11.087500,	122.887502),
        (12.337500,	125.112502),
        (8.012500,	123.537502),
        (7.612500,	124.812502),
        (9.912500,	124.287502)
    ]
    
    # Process all coordinates and save to CSV
    result_df = process_coordinates_to_csv(coordinates, "solar_data_batch_output.csv")
    
    print("\nProcessing complete!")


# def fetch_monthly_per_point(month, lat_rounded  , lon_rounded): #utilized
#   """ returns the hourly windspeed"""

#   working_db = wind.connect_to_db()
#   while working_db == None:
#     working_db =wind.connect_to_db()
    
#   pointer = working_db.cursor()

#   query1 = f"""SELECT "wind_speed_at_60m_(m/s)" FROM "NSRDB_WIND1"
#     WHERE lon_rounded = {lon_rounded} AND lat_rounded = {lat_rounded} and month = {month};"""
  
#   pointer.execute(query1)
#   data = pointer.fetchall()

#   pointer.close()
#   working_db.close()

#   if data:
#     data = [float(row[0]) for row in data]

#   return data

# # def read_wind_speeds_from_file(filename):
# #     """Read wind speeds from a text file (one value per line)"""
# #     try:
# #         with open(filename, 'r') as file:
# #             wind_speeds = []
# #             for line in file:
# #                 line = line.strip()
# #                 if line:  # Skip empty lines
# #                     wind_speeds.append(float(line))
# #         return np.array(wind_speeds)
# #     except FileNotFoundError:
# #         print(f"Error: File '{filename}' not found.")
# #         return None
# #     except ValueError as e:
# #         print(f"Error reading file: {e}")
# #         print("Make sure the file contains only numeric values, one per line.")
# #         return None
    
# # def weibull_from_mean_std(mean_wind, std_wind):
# #     """Calculate Weibull parameters from mean and standard deviation"""
# #     cv = std_wind / mean_wind
    
# #     # Solve for shape parameter k
# #     def shape_equation(k):
# #         return np.sqrt(gamma(1 + 2/k) - gamma(1 + 1/k)**2) / gamma(1 + 1/k) - cv
    
# #     k = fsolve(shape_equation, 2.0)[0]
# #     c = mean_wind / gamma(1 + 1/k)
    
# #     return k, c

# # def plot_histogram_with_two_weibulls(wind_speeds, given_mean, month):
# #     """Plot histogram with two Weibull PDF overlays"""
    
    
# #     actual_mean = np.mean(wind_speeds)
# #     actual_std = np.std(wind_speeds, ddof=1)
    
    
# #     k1, c1 = weibull_from_mean_std(actual_mean, actual_std)
    
    
# #     k2, c2 = weibull_from_mean_std(given_mean, actual_std)
    
# #     # Create the plot
# #     plt.figure(figsize=(12, 7))
    
# #     # Plot histogram with probability density
# #     plt.hist(wind_speeds, bins=25, density=True, alpha=0.6, 
# #              color='lightblue', edgecolor='black', label='Actual Wind Speed Data')
    
    
# #     x = np.linspace(0, max(wind_speeds), 1000)
    
# #     # Calculate first Weibull PDF (from actual data)
# #     weibull_pdf1 = (k1/c1) * (x/c1)**(k1-1) * np.exp(-(x/c1)**k1)
    
# #     # Calculate second Weibull PDF (from given mean)
# #     weibull_pdf2 = (k2/c2) * (x/c2)**(k2-1) * np.exp(-(x/c2)**k2)
    
# #     # Plot both Weibull curves
# #     plt.plot(x, weibull_pdf1, 'r-', linewidth=2, 
# #              label=f'NSRDB: Actual Data (k={k1:.2f}, c={c1:.2f})')
# #     plt.plot(x, weibull_pdf2, 'g--', linewidth=2, 
# #              label=f'IRENA: Given Mean (k={k2:.2f}, c={c2:.2f})')
    
# #     months = ['January', 'February', 'March', 'April', 'May', 'June',
# #           'July', 'August', 'September', 'October', 'November', 'December']
    
# #     # Labels and formatting
# #     plt.xlabel('Wind Speed (m/s)')
# #     plt.ylabel('Probability Density')
# #     plt.title(months[month-1])
# #     plt.legend()
# #     plt.grid(True, alpha=0.3)
    
# #     # Print parameters
# #     print("ACTUAL DATA:")
# #     print(f"  Mean: {actual_mean:.3f} m/s")
# #     print(f"  Std Dev: {actual_std:.3f} m/s")
# #     print(f"  Weibull Shape (k): {k1:.3f}")
# #     print(f"  Weibull Scale (c): {c1:.3f}")
    
# #     print("\nGIVEN MEAN WITH SAME STD DEV:")
# #     print(f"  Mean: {given_mean:.3f} m/s")
# #     print(f"  Std Dev: {actual_std:.3f} m/s (same as actual)")
# #     print(f"  Weibull Shape (k): {k2:.3f}")
# #     print(f"  Weibull Scale (c): {c2:.3f}")
    
# #     # Save the plot
# #     plt.savefig(f'wind_speed_two_weibulls{month}.png', dpi=300, bbox_inches='tight')
# #     print("\nPlot saved as 'wind_speed_two_weibulls.png'")
    
# #     return (k1, c1), (k2, c2)

# # # filename = "C:\\Users\\student\\dummywebGIS\\pages\\wind_speeds.txt"  # Change this to your file name
# # # wind_speeds = read_wind_speeds_from_file(filename)

# # IRENA_windspeeds = [6.539347649,
# # 5.821983814,
# # 5.36273098,
# # 4.361451149,
# # 3.406647205,
# # 4.108593464,
# # 4.371004581,
# # 5.546800613,
# # 4.118494987,
# # 4.76050663,
# # 5.988661289,
# # 6.489982605,]

# #     # Plot with IRENA mean
# # for month in range(1, 13):
# #     wind_speeds = fetch_monthly_per_point(month, 15.862501, 121.437502)
# #     print(wind_speeds)
# #     plot_histogram_with_two_weibulls(wind_speeds, IRENA_windspeeds[month - 1], month)




# #PLOT MONTHLY AVERAGES
# # Month labels
# months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 
#           'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']

# # Sample data - replace these with your actual IRENA and NSRDB averages
# # IRENA data (example: solar irradiance averages in kWh/m²/day)
# irena_averages = [6.539347649,
# 5.821983814,
# 5.36273098,
# 4.361451149,
# 3.406647205,
# 4.108593464,
# 4.371004581,
# 5.546800613,
# 4.118494987,
# 4.76050663,
# 5.988661289,
# 6.489982605,]

# # NSRDB data (example: solar irradiance averages in kWh/m²/day)

# nsrdb_averages = []

# for month in range(1, 13):
#     month_ws = fetch_monthly_per_point(month, 15.862501, 121.437502)
#     ave_ws = np.mean(month_ws)
#     nsrdb_averages.append(ave_ws)

# # nsrdb_averages = [4.717123656,
# # 4.01078869,
# # 2.701895161,
# # 2.486361111,
# # 2.270134409,
# # 2.925680556,
# # 3.323481183,
# # 3.493010753,
# # 2.107097222,
# # 3.611424731,
# # 3.750222222,
# # 6.709032258,]

# # Create the plot
# plt.figure(figsize=(12, 8))

# # Plot both datasets
# plt.plot(months, irena_averages, marker='o', linewidth=2, 
#          label='IRENA', color='blue', markersize=8)
# plt.plot(months, nsrdb_averages, marker='s', linewidth=2, 
#          label='NSRDB', color='red', markersize=8)

# # Customize the plot
# plt.title('Monthly Wind Speed Averages: IRENA vs NSRDB', 
#           fontsize=16, fontweight='bold')
# plt.xlabel('Month', fontsize=14)
# plt.ylabel('Wind Speed Average (m/s)', fontsize=14)
# plt.legend(fontsize=12)
# plt.grid(True, alpha=0.3)

# # Rotate x-axis labels for better readability
# plt.xticks(rotation=45)

# # Adjust layout to prevent label cutoff
# plt.tight_layout()

# # Optional: Add value labels on data points
# for i, (irena, nsrdb) in enumerate(zip(irena_averages, nsrdb_averages)):
#     plt.annotate(f'{irena:.1f}', (i, irena), textcoords="offset points", 
#                 xytext=(0,10), ha='center', fontsize=9, color='blue')
#     plt.annotate(f'{nsrdb:.1f}', (i, nsrdb), textcoords="offset points", 
#                 xytext=(0,-15), ha='center', fontsize=9, color='red')

# # Optional: Save the plot
# plt.savefig('irena_vs_nsrdb_comparison.png', dpi=300, bbox_inches='tight')

# # Print some basic statistics
# print("IRENA Statistics:")
# print(f"  Mean: {np.mean(irena_averages):.2f}")
# print(f"  Max: {np.max(irena_averages):.2f} (Month: {months[np.argmax(irena_averages)]})")
# print(f"  Min: {np.min(irena_averages):.2f} (Month: {months[np.argmin(irena_averages)]})")

# print("\nNSRDB Statistics:")
# print(f"  Mean: {np.mean(nsrdb_averages):.2f}")
# print(f"  Max: {np.max(nsrdb_averages):.2f} (Month: {months[np.argmax(nsrdb_averages)]})")
# print(f"  Min: {np.min(nsrdb_averages):.2f} (Month: {months[np.argmin(nsrdb_averages)]})")

# # print(f"\nMean Absolute Difference: {np.mean(np.abs(np.array(irena_averages) - np.array(nsrdb_averages))):.2f}")

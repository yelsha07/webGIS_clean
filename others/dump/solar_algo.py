import calendar
from datetime import datetime, timedelta
import ephem

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy import integrate
import pandas as pd
import streamlit as st
import psycopg2 as sql
import math
import statistics
import plotly.express as px
import json

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
    WHERE (ROUND(lon_rounded::NUMERIC, 6), ROUND(lat_rounded::NUMERIC, 6)) IN ({prep_points})
    ORDER BY "Month" ASC, "Day" ASC, "Hour" ASC;"""

    pointer.execute(query, coords)

    solar_data = pointer.fetchall()

    pointer.close()
    working_db.close()

    return solar_data
  
  else:
    query = f"""SELECT latitude, longitude, "Year", "Month", "Day", "Hour", "GHI", "municipality"
    FROM "NSRDB_SOLAR"
    WHERE municipality = {municipality}
    AND (ROUND(lon_rounded::NUMERIC, 6), ROUND(lat_rounded::NUMERIC, 6)) IN ({prep_points})
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

# munip_hourly_list = []  # List to store average GHI per hour
# sum_months = [] # NSRDB monthly ghi data
# monthly_ghi_data = [] # IRENA monthly ghi data

# def ave_ghi_nsrdb(solar_data): #NSRDB
#     """
#     Compute the average GHI per hour and store in munip_hourly_list.
#     """
#     ghi_hourly = [entry[6] for entry in solar_data]  # Extract hourly GHI column (index 8)
#     average_ghi = sum(ghi_hourly) / len(ghi_hourly)  # Compute average GHI
#     munip_hourly_list.append(average_ghi)  # Store in the list

#     sum_months = [0] * 12  # List to hold GHI sums for each month

#     # Define the hourly index range for each month in 2017
#     month_ranges = [
#         (0, 744),   # January
#         (744, 1416),  # February
#         (1416, 2160), # March
#         (2160, 2880), # April
#         (2880, 3624), # May
#         (3624, 4344), # June
#         (4344, 5088), # July
#         (5088, 5832), # August
#         (5832, 6552), # September
#         (6552, 7296), # October
#         (7296, 8016), # November
#         (8016, 8760)  # December
#     ]

#     # Iterate over hour-based GHI values and sum them into months
#     for month_index, (start, end) in enumerate(month_ranges):
#         sum_months[month_index] = sum(munip_hourly_list[start:end])
    
#     return sum_months

# def IRENA_monthly_ghi(solar_data): #IRENA
#     monthly_ghi_data.extend([tuple(entry[i] for entry in solar_data) for i in range(2, 14)])
#     return monthly_ghi_data

# #MONTHLY ENERGY YIELD
# def NSRDB_monthly_energy_yield(sum_months, area=9, af=0.7, eta=0.2): #NSRDB
#     """
#     Compute Monthly Energy Yield (MEY) for each month using total GHI.
#     """
#     mey_list_nsrdb = []
#     for ghi_sum in sum_months:
#         MEY = (ghi_sum * area * pixel_num * af * eta) # Converted to MWh
#         mey_list_nsrdb.append(MEY)
        
#         annual_energy_yield_nsrdb = sum(mey_list_nsrdb)

#     return mey_list_nsrdb, annual_energy_yield_nsrdb

# def IRENA_monthly_energy_yield(monthly_ghi_data, valid_points, area=9, af=0.7, eta=0.2): #IRENA
#     """
#     Compute Monthly Energy Yield (MEY) for each month using total GHI.
#     """
#     mey_list_irena = []
#     for month_ghi in monthly_ghi_data:
#         ghi_sum = sum(month_ghi)  # Compute total GHI for the month
#         MEY = (ghi_sum * area * len(valid_points) * af * eta)/1000 # Convert to MWh
#         mey_list_irena.append(MEY)
    

#     annual_energy_yield_irena = sum(mey_list_irena)

#     return mey_list_irena, annual_energy_yield_irena

# #-------------------------------------------------------------------------------------------------------

#     """
#     Compute the solar noon hour (0-24) for each day of the year at a given coordinate, rounding to the nearest whole hour.

#     Parameters:
#     - latitude (float): Latitude of the location
#     - longitude (float): Longitude of the location
#     - year (int): Year for which to compute solar noon

#     Returns:
#     - solar_noon_hours (dict): Dictionary with dates as keys and rounded solar noon hour (local time) as values.
#     """
# #-------------------------------------------------------------------------------------------------------
# #COMPUTING FOR NSRDB POWER DENSITY
# def monthlyGHI(latitude, longitude, year):
#     observer = ephem.Observer()
#     observer.lat, observer.lon = str(latitude), str(longitude)
#     observer.elev = 0  # Assume sea level

#     solar_noon_hours = {}  # Store results
#     days_in_year = 366 if calendar.isleap(year) else 365

#     for day in range(1, days_in_year + 1):
#         date = datetime(year, 1, 1) + timedelta(days=day - 1)
#         observer.date = date.strftime("%Y/%m/%d")
#         solar_noon = observer.next_transit(ephem.Sun(), start=observer.date)
#         solar_noon_local = ephem.localtime(solar_noon)
#         rounded_hour = round(solar_noon_local.hour + solar_noon_local.minute / 60)
#         rounded_hour = min(24, max(0, rounded_hour))
#         solar_noon_hours[date.strftime("%Y-%m-%d")] = rounded_hour
    
#     return solar_noon_hours

# def highest_GHI_at_solar_noon(solar_data):
#     max_ghi = 0
    
#     for entry in solar_data:
#         unique_id, lat, lon, year, month, day, hour, _, ghi, *_ = entry
#         solar_noon_hours = monthlyGHI(lat, lon, year)
#         date_key = f"{year}-{month:02d}-{day:02d}"
        
#         if date_key in solar_noon_hours and hour == solar_noon_hours[date_key]:
#             max_ghi = max(max_ghi, ghi)
    
#     return max_ghi

# #SOLAR CAPACITY
# def NSRDB_capacity(power_density, area=9):
#     cap_nsrdb = (area * power_density * 3000) / 1000000
#     return cap_nsrdb


# def IRENA_capacity (valid_points, power_density=1000, area=9):
#     cap_irena = (area * power_density * len(valid_points))/1000000
#     return cap_irena
# #-------------------------------------------------------------------------------------------------------
# #SOLAR_CAPACITY_FACTOR
# days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]  
# hours_in_month = [days * 24 for days in days_in_month]

# def NSRDB_capacity_factor(cap_nsrdb, mey_list_nsrdb): #NSRDB
#     cf_list_nsrdb = []
#     cf_percentage_list_nsrdb = []
    
#     for month, hours in zip(mey_list_nsrdb, hours_in_month):
#         capacity_factor_nsrdb = month / (cap_nsrdb * hours)
#         cf_percentage_nsrdb = capacity_factor_nsrdb 
#         cf_list_nsrdb.append(capacity_factor_nsrdb)
#         cf_percentage_list_nsrdb.append(cf_percentage_nsrdb)
    
#     return cf_list_nsrdb, cf_percentage_list_nsrdb

# def IRENA_capacity_factor(cap_irena, mey_list_irena):
#     cf_list_irena = []
#     cf_percentage_list_irena = []
    
#     for month, hours in zip(mey_list_irena, hours_in_month):
#         capacity_factor_irena = month / (cap_irena * hours)
#         cf_percentage_irena = capacity_factor_irena 
#         cf_list_irena.append(capacity_factor_irena)
#         cf_percentage_list_irena.append(cf_percentage_irena)
    
#     return cf_list_irena, cf_percentage_list_irena

# #-------------------------------------------------------------------------------------------------------

# #SOLAR_LCOE
# def NSRDB_lcoe(cf_list_nsrdb, fixed_charge_rate=0.092, capital_cost=75911092, fixed_om_cost=759111, variable_om_cost=0, fuel_cost=0):
#     lcoe_list_nsrdb = []
    
#     for cf, hours in zip(cf_list_nsrdb, hours_in_month):
#         if cf == 0:
#             lcoe_list_nsrdb.append(float('inf'))  # Avoid division by zero
#         else:
#             denominator = cf * hours
#             lcoe_value_nsrdb = (((fixed_charge_rate * capital_cost + fixed_om_cost) / denominator) + variable_om_cost + fuel_cost) / 1000
#             lcoe_list_nsrdb.append(lcoe_value_nsrdb)
    
#     return lcoe_list_nsrdb

# def IRENA_lcoe(cf_list_irena, fixed_charge_rate=0.092, capital_cost=75911092, fixed_om_cost=759111, variable_om_cost=0, fuel_cost=0):
#     lcoe_list_irena = []
    
#     for cf, hours in zip(cf_list_irena, hours_in_month):
#         if cf == 0:
#             lcoe_list_irena.append(float('inf'))  # Avoid division by zero
#         else:
#             denominator = cf * hours
#             lcoe_value_irena = (((fixed_charge_rate * capital_cost + fixed_om_cost) / denominator) + variable_om_cost + fuel_cost) / 1000
#             lcoe_list_irena.append(lcoe_value_irena)
    
#     return lcoe_list_irena

# def obtain_municip(): # utilized
#   #establishes connection to the database first
#   #st.text("Trying to connect to database...")
#   working_db = connect_to_db()
#   while working_db == None:
#     working_db = connect_to_db()

#   #this will allow the sql queries
#   #st.text("Successful!")
#   pointer = working_db.cursor()
  
  
#   query = f'''SELECT DISTINCT adm3_en
#   FROM "IRENA_GHI_WS20_WS60 ";'''

#   pointer.execute(query)
#   municip_data = pointer.fetchall() 

#   pointer.close()
#   working_db.close()


#   return municip_data

def look_up_points(points:list, tables:list):
  ''' checks whether the points are not part of exclusion areas '''

  if len(tables) == 0:
    return "default"
  
  exists_conditions = [
      f"EXISTS (SELECT 1 FROM \"{table}\" WHERE ROUND(\"{table}\".xcoord, 6) = input_points.xcoord "
      f"AND ROUND(\"{table}\".ycoord, 6) = input_points.ycoord)"
      for table in tables
  ]
  sql_query = f"""
        SELECT xcoord, ycoord
        FROM (VALUES {', '.join(['(%s, %s)'] * len(points))}) AS input_points(xcoord, ycoord)
        WHERE {' AND '.join(exists_conditions)};
    """
  
  query_params = [coord for point in points for coord in point]

  connection = connect_to_db()
  with connection.cursor() as cursor:
        cursor.execute(sql_query, query_params)
        matching_points = cursor.fetchall()

  matching_points = [(float(row[0]), float(row[1])) for row in matching_points]

  return matching_points

# def db_fetch_sample_points(valid_points = None, municipality = None): # utilized
#   """ """
#   #establishes connection to the database first
#   #st.text("Trying to connect to database...")
#   working_db = connect_to_db()
#   while working_db == None:
#     working_db = connect_to_db()

#   #this will allow the sql queries
#   #st.text("Successful!")
#   pointer = working_db.cursor()
  
#   if municipality != None:
#     query = f'''SELECT xcoord, ycoord, adm3_en
#       FROM "IRENA_GHI_WS20_WS60 "
#       WHERE adm3_en = '{municipality}';'''
    
#     pointer.execute(query)
#     data = pointer.fetchall()
#     return data
# #----------------------------------------------------------------------------------------
# # the following lines are used to display on the website
# sample_munip = obtain_municip()
# sample_munip = sorted([row[0] for row in sample_munip])


# # let user choose constraints first
# st.write("Choose which constraints to apply.")

# constraints_table = {0:"BuiltUp Constraints Removed", 1: "CADTs Constraints Removed", 2: "Forest Constraints Removed", 3: "Protected Areas Removed"}
# choose_from = []

# with st.container():
#     col1, col2 = st.columns(2)

#     with col1:
#         ancestral = st.checkbox("Ancestral Domains")
#         if ancestral:
#           choose_from.append(constraints_table[1])

#         tree_cover = st.checkbox("Tree Covers")
#         if tree_cover:
#           choose_from.append(constraints_table[2])

#     with col2:
#         land_use = st.checkbox("Land Use")
#         if land_use:
#           choose_from.append(constraints_table[0])
#         protected_areas =  st.checkbox("Protected Areas")
#         if protected_areas:
#           choose_from.append(constraints_table[3])


# # Display dropdown
# selected_option = st.selectbox("Choose a municipality:", sample_munip)

# # filter out invalid points (those that are exclusion areas based on user's constraint selection)
# temp_points = db_fetch_sample_points(municipality = selected_option)
# temp_points = [(round(float(row[0]), 6), round(float(row[1]), 6)) for row in temp_points]


# if choose_from:

#   valid_points = look_up_points(temp_points, choose_from)

#   st.write(f"all of the points: {temp_points}")
#   st.write(f"filtered points: {valid_points}")

# else:
#   st.write(f"filtered points: No Constraint Selected. ")
#   valid_points = temp_points

# #call functions here

# irena_solar_data = db_fetch_IRENA_solar(valid_points)
# # nsrdb_solar_data = db_fetch_hourly_solar(valid_points)

# monthly_ghi_data = IRENA_monthly_ghi(irena_solar_data)

# mey_list_irena, annual_energy_yield_irena = IRENA_monthly_energy_yield(monthly_ghi_data, valid_points, area=9, af=0.7, eta=0.2)
# cap_irena = IRENA_capacity (valid_points,power_density=1000, area=9)
# cf_list_irena, cf_percentage_list_irena = IRENA_capacity_factor(cap_irena, mey_list_irena)
# lcoe_list_irena = IRENA_lcoe(cf_list_irena)

# # solar_data = [()]
# # sum_months = ave_ghi_nsrdb(solar_data)
# # mey_list_irena, annual_energy_yield_irena = IRENA_monthly_energy_yield(area=9, af=0.7, eta=0.2)
# # cap_irena = IRENA_capacity (len(valid_points),power_density=1000, area=9)
# # cf_list_irena, cf_percentage_list_irena = IRENA_capacity_factor(cap_irena, mey_list_irena)
# # lcoe_list_irena = IRENA_lcoe(cf_list_irena)

# st.write(f"MEY list irena: {mey_list_irena}, annual energy yield irena: {annual_energy_yield_irena}")
# st.write(f"cap irena: {cap_irena}")
# st.write(f"cf list: {cf_list_irena}, cf percent: {cf_percentage_list_irena}")
# st.write(f"lcoe list irena: {lcoe_list_irena}")

# # st.write(valid_points)
# # st.write(str(irena_solar_data)

# # st.write(mey_list_irena, annual_energy_yield_irena)
# # st.write(cap_irena)
# # st.write(cf_list_irena, cf_percentage_list_irena)
# # st.write(lcoe_list_irena)
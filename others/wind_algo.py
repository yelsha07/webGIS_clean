# main code for computing wind technical potential
# author: @ashley_celis

# import all packages in this section ----------------------------------------

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
from scipy.stats import weibull_min
from scipy.special import gamma
from scipy.optimize import fsolve
import warnings


# -----------------------------------------------------------------------------
def testing_printer(str):
  st.write(str)
  
class Turbine:
  '''   model name
        p_rated (KW)
        rotor_diameter (m)
        hub_height (m)
        w_speed (m/s)
        p_output (KW) '''
 
  def __init__(self, model, max_ws, min_ws, p_rated, rotor_diameter, hub_height, w_speed, p_output):


    self.model_name = model
    self.p_rated = p_rated
    self.rotor_diameter = rotor_diameter
    self.hub_height = hub_height
    self.max = max_ws
    self.min = min_ws
    self.w_speed = np.array(w_speed)
    self.p_output = np.array(p_output)
    


    self.region2 = interp1d(self.w_speed, self.p_output, 'linear', fill_value= 'extrapolate')


  def output_power(self, wind_speeds):

    ''' returns an array of the output power for the respective wind speed data '''


    output = self.region2(wind_speeds)


    #replace negative values with zero
    output = np.clip(output, 0, None)


    return output

#initializations should go here--------------------------------------------------------------------------------------------------------------------------------------------------

combined_table_cols = [
    "main_id", "xcoord", "ycoord", "jan ws201", "feb ws201", "mar ws201", 
    "apr ws201", "may ws201", "jun ws201", "jul ws201", "aug ws201", "sep ws201", 
    "sample_1", "nov ws201", "dec ws201", "jan ws601", "feb ws601", "mar ws601", 
    "apr ws601", "may ws601", "jun ws601", "jul ws601", "aug ws601", "sep ws601", 
    "oct ws601", "nov ws601", "dec ws601", "year", "month", "day", "hour", 
    "minute", "hourly_ghi", "hourly_windspeed", "municipality", "region", "province"
]

# #datasheet link:  
V165 = Turbine("V66-1.65", 25, 4, 1650, 66, 78, np.arange(3, 28.5, 0.5), [0, 0, 20, 47.5, 78.2, 123.3, 170.67, 222, 278, 365.2, 465.85, 568.5, 673.3, 808.9, 913.6, 984, 1054.4, 1143.35, 1242.9, 1347, 1439.35, 1509.2, 1567, 1595, 1620.67, 1633.5, 1639, 1644.5, 1650, 1650, 1650, 1650, 1650, 1650, 1650, 1650, 1650, 1650, 1650, 1650, 1650, 1650, 1650, 1650, 1650, 0, 0, 0, 0, 0, 0])


# #datasheet link: https://en.wind-turbine-models.com/turbines/1249-vestas-v126-3.45
V126 = Turbine("V126-3.45MW", 22.5, 3, 3450, 126, 149, np.arange(3, 26, 0.5), [35, 101, 184, 283, 404, 550, 725, 932, 1172, 1446, 1760, 2104, 2482, 2865, 3187, 3366, 3433, 3448, 3450, 3450, 3450, 3450, 3450, 3450, 3450, 3450, 3450, 3450, 3450, 3450, 3450, 3450, 3450, 3450, 3450, 3450, 3450, 3450, 3450, 3450, 0, 0, 0, 0, 0 ,0])


# #datasheet link: https://www.thewindpower.net/turbine_en_1490_vestas_v150-4000-4200.php
V150 = Turbine("V150/4000-4200",3,22.5, 4200, 150, 166, np.arange(3, 26, 0.5), [78, 172, 287, 426, 601, 814, 1069, 1367, 1717, 2110, 2546, 3002, 3428, 3773, 4012, 4131, 4186, 4198, 4200, 4200, 4200, 4200, 4200, 4200, 4200, 4200, 4200, 4200, 4200, 4200, 4200, 4200, 4200, 4200, 4200, 4200, 4200, 4200, 4200, 4200, 0, 0, 0, 0, 0, 0])

turbines = {1:V126, 2:V150, 3:V165}


# BACK TRACKING THE CODE -----
# Create smooth wind speed array for plotting
# wind_speeds = np.linspace(0, 30, 1000)

# # Calculate power output for each turbine
# power_V165 = V165.output_power(wind_speeds)
# power_V126 = V126.output_power(wind_speeds)
# power_V150 = V150.output_power(wind_speeds)

# # Create figure for Streamlit
# fig, ax = plt.subplots(figsize=(10, 6))

# # Plot power curves
# ax.plot(wind_speeds, power_V165, 'b-', linewidth=2.5, label=f'{V165.model_name} ({V165.p_rated} kW)')
# ax.plot(wind_speeds, power_V126, 'r-', linewidth=2.5, label=f'{V126.model_name} ({V126.p_rated} kW)')
# ax.plot(wind_speeds, power_V150, 'g-', linewidth=2.5, label=f'{V150.model_name} ({V150.p_rated} kW)')

# # Add cut-in and cut-out wind speeds as vertical lines for each turbine
# ax.axvline(x=V165.min, color='b', linestyle='--', alpha=0.5, label=f'{V165.model_name} cut-in')
# ax.axvline(x=V165.max, color='b', linestyle=':', alpha=0.5, label=f'{V165.model_name} cut-out')

# ax.axvline(x=V126.min, color='r', linestyle='--', alpha=0.5, label=f'{V126.model_name} cut-in')
# ax.axvline(x=V126.max, color='r', linestyle=':', alpha=0.5, label=f'{V126.model_name} cut-out')

# ax.axvline(x=V150.min, color='g', linestyle='--', alpha=0.5, label=f'{V150.model_name} cut-in')
# ax.axvline(x=V150.max, color='g', linestyle=':', alpha=0.5, label=f'{V150.model_name} cut-out')

# # Add rated power for each turbine as horizontal lines
# ax.axhline(y=V165.p_rated, color='b', linestyle='-.', alpha=0.5)
# ax.axhline(y=V126.p_rated, color='r', linestyle='-.', alpha=0.5)
# ax.axhline(y=V150.p_rated, color='g', linestyle='-.', alpha=0.5)

# # Add labels and title
# ax.set_xlabel('Wind Speed (m/s)', fontsize=12)
# ax.set_ylabel('Power Output (kW)', fontsize=12)
# ax.set_title('Wind Turbine Power Curves', fontsize=14)
# ax.grid(True, alpha=0.3)

# # Set axis limits
# ax.set_xlim(0, 30)
# ax.set_ylim(0, 4500)

# # Create a more compact legend
# ax.legend(fontsize=10, loc='upper right')

# # Adjust layout
# plt.tight_layout()

# # Display the plot in Streamlit
# st.pyplot(fig)
# #--------------------------------------------------------------------------------------------------------------------------------------------------
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
    st.text(e)

    return None  
  
def obtain_municip(): # utilized
  #establishes connection to the database first
  #st.text("Trying to connect to database...")
  working_db = connect_to_db()
  while working_db == None:
    working_db = connect_to_db()

  #this will allow the sql queries
  #st.text("Successful!")
  pointer = working_db.cursor()
  
  
  query = f'''SELECT DISTINCT adm3_en
  FROM "IRENA_GHI_WS20_WS60 ";'''

  pointer.execute(query)
  municip_data = pointer.fetchall() 

  pointer.close()
  working_db.close()


  return municip_data

def fetch_spatial_average(month, valid_points, ws_height = 40): #utilized
  """ returns the spatial average for the specified month of the given valid points. """

  working_db = connect_to_db()
  while working_db == None:
    working_db = connect_to_db()
    
  pointer = working_db.cursor()
  prep_points = ', '.join(['(%s, %s)'] * len(valid_points))

    #fetch the 40m wind speed
  if ws_height == 40:
    query_1 = f"""SELECT AVG("wind_speed_at_40m_(m/s)")
        FROM "NSRDB_WIND1"
        WHERE (lon_rounded, lat_rounded) 
        IN ({prep_points})
        AND month = {month}
        GROUP BY month, day, hour
        ORDER BY month, day, hour;"""
  
    params = [coord for point in valid_points for coord in point]

    pointer.execute(query_1, params)
    data = pointer.fetchall()

  else:
    #fetch the 60m wind speed
    query_1 = f"""SELECT AVG("wind_speed_at_60m_(m/s)")
        FROM "NSRDB_WIND1"
        WHERE (lon_rounded, lat_rounded) 
        IN ({prep_points})
        AND month = {month}
        GROUP BY month, day, hour
        ORDER BY month, day, hour;"""
  
    params = [coord for point in valid_points for coord in point]

    pointer.execute(query_1, params)
    data = pointer.fetchall()



  pointer.close()
  working_db.close()
  if data:
    data = [row[0] for row in data]

  return data

def fetch_point_NSRDB(month, valid_point, ws_height = 40): #utilized
  """ returns the spatial average for the specified month of the given valid points. """

  working_db = connect_to_db()
  while working_db == None:
    working_db = connect_to_db()
    
  pointer = working_db.cursor()

    #fetch the 40m wind speed
  if ws_height == 40:

    query_1 = f"""SELECT "wind_speed_at_40m_(m/s)" FROM "NSRDB_WIND1"
    WHERE lon_rounded = {valid_point[0]} AND lat_rounded = {valid_point[1]}
    AND month = {month}
    ORDER BY month, day, hour;"""

    pointer.execute(query_1)
    data = pointer.fetchall()

  else:
    #fetch the 60m wind speed
    query_1 = f"""SELECT "wind_speed_at_60m_(m/s)" FROM "NSRDB_WIND1"
    WHERE lon_rounded = {valid_point[0]} AND lat_rounded = {valid_point[1]}
    AND month = {month}
    ORDER BY month, day, hour;"""
  
    pointer.execute(query_1)
    data = pointer.fetchall()

  pointer.close()
  working_db.close()

  if data:
    data = [float(row[0]) for row in data]

  return data

def db_fetch_IRENA(valid_points = None, municipality = None): # utilized
  """ """
  #establishes connection to the database first
  #st.text("Trying to connect to database...")
  working_db = connect_to_db()
  while working_db == None:
    working_db = connect_to_db()

  #this will allow the sql queries
  #st.text("Successful!")
  pointer = working_db.cursor()

  # st.write(f"checking valid points: {valid_points}")
  
  if municipality != None:
    query = f'''SELECT xcoord, ycoord, adm3_en, "jan ws201", "feb ws201", "mar ws201", 
      "apr ws201", "may ws201", "jun ws201", "jul ws201", "aug ws201", "sep ws201", 
      "sample_1", "nov ws201", "dec ws201", "jan ws601", "feb ws601", "mar ws601", 
      "apr ws601", "may ws601", "jun ws601", "jul ws601", "aug ws601", "sep ws601", 
      "oct ws601", "nov ws601", "dec ws601" 
      FROM "IRENA_GHI_WS20_WS60 "
      WHERE adm3_en = '{municipality}';'''
    
    pointer.execute(query)
    data = pointer.fetchall()
    return data

  else:
      prep_points = ', '.join(['(%s, %s)'] * len(valid_points))
      query = f'''SELECT xcoord, ycoord, adm3_en, "jan ws201", "feb ws201", "mar ws201", 
      "apr ws201", "may ws201", "jun ws201", "jul ws201", "aug ws201", "sep ws201", 
      "sample_1", "nov ws201", "dec ws201", "jan ws601", "feb ws601", "mar ws601", 
      "apr ws601", "may ws601", "jun ws601", "jul ws601", "aug ws601", "sep ws601", 
      "oct ws601", "nov ws601", "dec ws601" 
      FROM "IRENA_GHI_WS20_WS60 "
      WHERE (lon_rounded, lat_rounded) 
      IN ({prep_points});'''
    
      params = [coord for point in valid_points for coord in point]

      pointer.execute(query, params)

      data = pointer.fetchall()

      # parse the IRENA_DATA
      if data:  
          ws_20m_all = []  
          ws_60m_all = []
          points = []  

          for row in data:
              xcoord = float(row[0])
              ycoord = float(row[1]) 
              ws_20m = list(map(float, row[3:15]))  
              ws_60m = list(map(float, row[15:27]))  
              
              ws_20m_all.append(ws_20m)  
              ws_60m_all.append(ws_60m)
              points.append((xcoord,ycoord))

          pointer.close()
          working_db.close()


          return ws_20m_all, ws_60m_all, points
      
def fetch_monthly_per_point(month, lat_rounded, lon_rounded, height = 60 ): #utilized
  """ returns the hourly windspeed"""

  working_db = connect_to_db()
  while working_db == None:
    working_db = connect_to_db()
    
  pointer = working_db.cursor()

  if height == 60:
    query1 = f"""SELECT "wind_speed_at_60m_(m/s)" FROM "NSRDB_WIND1"
    WHERE lon_rounded = {lon_rounded} AND lat_rounded = {lat_rounded} and month = {month};"""

  else:
    query1 = f"""SELECT "wind_speed_at_40m_(m/s)" FROM "NSRDB_WIND1"
    WHERE lon_rounded = {lon_rounded} AND lat_rounded = {lat_rounded} and month = {month};"""
  
  pointer.execute(query1)
  data = pointer.fetchall()

  pointer.close()
  working_db.close()

  if data:
    data = [float(row[0]) for row in data]

  return data

# def get_alpha(height1:int, windspeed1: list, height2:int, windspeed2:list):
#   """ computes the friction coefficient for each wind speed pair measured from different heights. """
#   fric_coeffs = []

#   for index in range(0, len(windspeed1)):

#     # x > 0 for log(x)

#     #WARNING: CHANGE THIS WHEN THE NEW DATA IS IN !!

#     # wind1 = max(windspeed1[index], 2.0)  
#     # wind2 = max(windspeed2[index], 2.0)

#     alpha = (math.log(windspeed2[index]) - math.log(windspeed1[index])) / (math.log(height2) - math.log(height1))
#     fric_coeffs.append(alpha)

#   return fric_coeffs


# #FOR DEBUGGING
# def get_alpha(height1: int, windspeed1: list, height2: int, windspeed2: list):
#     """Debug version to identify NaN sources"""
#     fric_coeffs = []
#     nan_count = 0
    
#     print(f"Heights: {height1}m to {height2}m")
#     print(f"Processing {len(windspeed1)} wind speed pairs...")
    
#     for index in range(len(windspeed1)):
#         ws1, ws2 = windspeed1[index], windspeed2[index]
        
#         # Check each step
#         print(f"\nIndex {index}: ws1={ws1}, ws2={ws2}")
        
#         if ws1 <= 0 or ws2 <= 0:
#             print(f"  -> PROBLEM: Zero/negative wind speed!")
        
#         if np.isnan(ws1) or np.isnan(ws2):
#             print(f"  -> PROBLEM: NaN in input data!")
        
#         try:
#             log_ws1 = math.log(ws1)
#             log_ws2 = math.log(ws2)
#             print(f"  -> Logs: log({ws1})={log_ws1:.3f}, log({ws2})={log_ws2:.3f}")
            
#             alpha = (log_ws2 - log_ws1) / (math.log(height2) - math.log(height1))
#             print(f"  -> Alpha: {alpha:.3f}")
            
#             if np.isnan(alpha):
#                 nan_count += 1
#                 print(f"  -> PROBLEM: Alpha is NaN!")
            
#             fric_coeffs.append(alpha)
            
#         except Exception as e:
#             print(f"  -> ERROR: {e}")
#             fric_coeffs.append(np.nan)
#             nan_count += 1
    
#     print(f"\nSummary: {nan_count} NaN values out of {len(windspeed1)} calculations")
#     return fric_coeffs

def get_alpha(height1: int, windspeed1: list, height2: int, windspeed2: list):
    """solves friction coefficient"""
    
    ws1 = np.array(windspeed1)
    ws2 = np.array(windspeed2)
    
    valid_mask = (
        (ws1 > 0) & (ws2 > 0) & 
        np.isfinite(ws1) & np.isfinite(ws2) &
        (np.abs(ws2/ws1 - 1.0) > 0.01)  #sometimes values are too equal
    )
    
    alpha = np.full(len(ws1), 0.14)
    
    if np.any(valid_mask):
        
        ws1_safe = np.maximum(ws1[valid_mask], 0.1)
        ws2_safe = np.maximum(ws2[valid_mask], 0.1)
        
        
        height_log_diff = math.log(height2) - math.log(height1)
        wind_log_diff = np.log(ws2_safe) - np.log(ws1_safe)
        alpha_calc = wind_log_diff / height_log_diff
        
        
        alpha_calc = np.clip(alpha_calc, 0.05, 0.5)
        
        
        alpha[valid_mask] = alpha_calc
    
    
    invalid_count = np.sum(~valid_mask)
    if invalid_count > 0:
        print(f"WARNING: {invalid_count}/{len(ws1)} wind speed pairs were invalid, using default alpha=0.14")

    alpha_list = alpha.tolist()
    ave = sum(alpha_list)/len(alpha_list)
    
    return alpha_list, ave


# def extrapolate_windspeed(height1:int, wind_speed1:list, turbine_height:int, alpha:list):
#   """ extrapolates the wind speed data to the height of the turbine using the power law. """
#   new_windspeeds = []
#   for index in range(0, len(wind_speed1)):
#     new_wind = wind_speed1[index] * (turbine_height/height1) ** alpha[index] 
#     new_windspeeds.append(new_wind)

#   # st.write(f"this is extrapolated: {new_windspeeds}")

#   return new_windspeeds

#FOR DEBUGGING
def extrapolate_windspeed(height1: int, wind_speed1: list, turbine_height: int, alpha: list):
    """Production-ready version with realistic bounds"""
    
    new_windspeeds = []
    height_ratio = turbine_height / height1
    
    for index in range(len(wind_speed1)):
        ws, a = wind_speed1[index], alpha[index]
        
        if np.isnan(ws) or np.isnan(a) or ws < 0:
            new_windspeeds.append(np.nan)
            continue
        
        if ws == 0:
            new_windspeeds.append(0.0)
            continue
        
        
        if height_ratio != 1.0:  # Only if extrapolating
            max_safe_alpha = np.log(10) / np.log(height_ratio) if height_ratio > 1 else np.log(0.1) / np.log(height_ratio)
            
            if abs(a) > abs(max_safe_alpha):
                original_a = a
                a = np.sign(a) * min(abs(a), abs(max_safe_alpha))
                if index < 5:
                    print(f"Capping alpha[{index}] from {original_a:.3f} to {a:.3f} to prevent extreme extrapolation")
        
        # Extrapolate
        try:
            new_wind = ws * (height_ratio ** a)
            new_wind = min(new_wind, 100)  
            new_windspeeds.append(new_wind)
        except:
            new_windspeeds.append(ws)  # Fallback: no extrapolation
        
        ave = (sum(new_windspeeds)/len(new_windspeeds))
    
    return new_windspeeds, ave

# def get_weibull_params(extrapolated_WS:list, stdev = None, source = "NSRDB"):
#   """ computes shape and scale paramaters. by default, the computation is suited for the NSRDB database. """

#   ave_ws =  statistics.mean(extrapolated_WS)

#   if source == "NSRDB":
    
#     std_ws = statistics.stdev(extrapolated_WS)
#     cv = std_ws / ave_ws
#     # Function to solve for k
#     def equation(k):
#         return np.sqrt(gamma(1 + 2/k) - gamma(1 + 1/k)**2) / gamma(1 + 1/k) - cv
    
#     shape = fsolve(equation, 2.0)[0]
    
#     # Calculate c
#     scale = ave_ws / gamma(1 + 1/shape)

#     return shape, scale, std_ws

#   else:

#     cv = stdev / ave_ws
#     # Function to solve for k
#     def equation(k):
#         return np.sqrt(gamma(1 + 2/k) - gamma(1 + 1/k)**2) / gamma(1 + 1/k) - cv
    
#     shape = fsolve(equation, 2.0)[0]
    
#     # Calculate c
#     scale = ave_ws / gamma(1 + 1/shape)

#     return shape, scale

def get_weibull_params(extrapolated_WS: list, stdev=None, source="NSRDB"):
    """Robust version with proper error handling"""
    
    # Clean and validate input
    ws_clean = [ws for ws in extrapolated_WS if not (np.isnan(ws) or ws <= 0)]
    
    if len(ws_clean) == 0:
        raise ValueError("No valid wind speed data")
    
    ave_ws = statistics.mean(ws_clean)
    
    # Get standard deviation
    if source == "NSRDB":
        if len(ws_clean) < 2:
            raise ValueError("NSRDB mode requires at least 2 wind speed values")
        std_ws = statistics.stdev(ws_clean)
    else:
        if stdev is None or stdev <= 0:
            raise ValueError("IRENA mode requires valid stdev parameter")
        std_ws = stdev
    
    # Calculate CV with bounds
    cv = std_ws / ave_ws
    cv = np.clip(cv, 0.05, 3.0)  # Reasonable bounds
    
    # Robust equation solving
    def safe_equation(k):
        try:
            if k <= 0.1 or k > 20:
                return 1e6
                
            g1 = gamma(1 + 1/k)
            g2 = gamma(1 + 2/k)
            
            # Check for numerical issues
            if g1 <= 0 or g2 <= 0 or np.isnan(g1) or np.isnan(g2):
                return 1e6
            
            under_sqrt = g2 - g1**2
            if under_sqrt <= 0:  # Prevent sqrt of negative
                return 1e6
            
            theoretical_cv = np.sqrt(under_sqrt) / g1
            return abs(theoretical_cv - cv)  # Use absolute difference
            
        except:
            return 1e6
    
    # Try multiple approaches
    shape = None
    
    # Method 1: Multiple initial guesses for fsolve
    for initial_k in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                result = fsolve(safe_equation, initial_k)
                candidate_k = result[0]
                
                if (0.1 < candidate_k < 20 and 
                    safe_equation(candidate_k) < 0.001):  # Good convergence
                    shape = candidate_k
                    break
        except:
            continue
    
    # Method 2: Fallback to method of moments
    if shape is None:
        shape = cv ** (-1.086)
        shape = np.clip(shape, 0.5, 10.0)
    
    # Calculate scale parameter
    try:
        gamma_val = gamma(1 + 1/shape)
        scale = ave_ws / gamma_val
    except:
        scale = ave_ws  # Fallback
    
    # Final validation
    if np.isnan(shape) or np.isnan(scale) or shape <= 0 or scale <= 0:
        raise ValueError(f"Invalid Weibull parameters: k={shape}, c={scale}")
    
    if source == "NSRDB":
        return shape, scale, std_ws
    else:
        return shape, scale
  

# def get_weibull_params(extrapolated_WS: list, stdev=None, source="NSRDB"):
#     """Calculate Weibull parameters from mean + stdev, NO defaults"""
    
#     # Clean data first
#     ws_clean = [ws for ws in extrapolated_WS if not np.isnan(ws) and ws > 0]
    
#     if len(ws_clean) == 0:
#         raise ValueError("No valid wind speed data provided")
    
#     # Always calculate mean from available data
#     mean_ws = statistics.mean(ws_clean)
#     mean_ws = max(mean_ws, 0.1)  # Safety check
    
#     print(f"DEBUG: mean_ws = {mean_ws:.6f}")
    
#     # Get standard deviation
#     if source == "NSRDB":
#         if len(ws_clean) >= 2:
#             # Multiple values: calculate stdev from data
#             std_ws = statistics.stdev(ws_clean)
#             print(f"DEBUG: Calculated std_ws from {len(ws_clean)} values = {std_ws:.6f}")
#         else:
#             # Single value: must provide external stdev or error
#             if stdev is None or stdev <= 0:
#                 raise ValueError("NSRDB mode with single value requires valid stdev parameter")
#             std_ws = stdev
#             print(f"DEBUG: Using provided stdev for NSRDB = {std_ws:.6f}")
#     else:
#         # IRENA mode: always use provided stdev
#         if stdev is None or stdev <= 0:
#             raise ValueError("IRENA mode requires valid stdev parameter")
#         std_ws = stdev
#         print(f"DEBUG: Using provided stdev for IRENA = {std_ws:.6f}")
    
#     # Calculate coefficient of variation
#     cv = std_ws / mean_ws
#     cv = np.clip(cv, 0.05, 5.0)  # Wider bounds, no artificial defaults
    
#     print(f"DEBUG: cv = {cv:.6f}")
    
#     # Method 1: Try equation solving (your original approach)
#     def weibull_cv_equation(k):
#         try:
#             if k <= 0.1 or k > 50:  # Wider bounds
#                 return 1e6
#             g1 = gamma(1 + 1/k)
#             g2 = gamma(1 + 2/k)
#             theoretical_cv = np.sqrt(g2 - g1**2) / g1
#             return abs(theoretical_cv - cv)
#         except:
#             return 1e6
    
#     try:
#         with warnings.catch_warnings():
#             warnings.simplefilter("ignore")
            
#             # Try multiple initial guesses for robustness
#             best_k = None
#             best_error = float('inf')
            
#             for initial_guess in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
#                 try:
#                     result = fsolve(weibull_cv_equation, initial_guess)[0]
#                     error = weibull_cv_equation(result)
                    
#                     if error < best_error and 0.1 < result < 50:
#                         best_k = result
#                         best_error = error
#                 except:
#                     continue
            
#             if best_k is not None and best_error < 0.01:
#                 shape = best_k
#                 print(f"DEBUG: fsolve successful with k = {shape:.6f}, error = {best_error:.6f}")
#             else:
#                 raise ValueError("fsolve failed to converge")
                
#     except:
#         # Method 2: Fallback to method of moments approximation
#         print("DEBUG: fsolve failed, using method of moments")
#         shape = cv ** (-1.086)  # Empirical relationship
#         shape = np.clip(shape, 0.1, 20.0)
#         print(f"DEBUG: Method of moments k = {shape:.6f}")
    
#     # Calculate scale parameter
#     try:
#         gamma_val = gamma(1 + 1/shape)
#         scale = mean_ws / gamma_val
        
#         if scale <= 0 or np.isnan(scale) or np.isinf(scale):
#             raise ValueError("Invalid scale calculation")
            
#         print(f"DEBUG: scale = {scale:.6f}")
        
#     except Exception as e:
#         print(f"ERROR: Scale calculation failed: {e}")
#         raise ValueError(f"Cannot calculate valid Weibull scale parameter: {e}")
    
#     # Final validation - NO defaults allowed
#     if (np.isnan(shape) or np.isnan(scale) or 
#         shape <= 0 or scale <= 0 or
#         np.isinf(shape) or np.isinf(scale)):
#         raise ValueError(f"Invalid Weibull parameters calculated: k={shape}, c={scale}")
    
#     print(f"SUCCESS: Final Weibull parameters - k={shape:.6f}, c={scale:.6f}")
    
#     if source == "NSRDB":
#         return shape, scale, std_ws
#     else:
#         return shape, scale
  
# def calc_energy_yield_discrete(hours: int, points_num: int, height1: int, ws_height1: list, 
#                               height2: int, ws_height2, turbine_model, nsrdb_stdev=None, source="NSRDB"):
    
#     # Your existing code for getting k, c parameters...
#     frix = get_alpha(height1, ws_height1, height2, ws_height2)
#     conv_ws = extrapolate_windspeed(height2, ws_height2, turbine_model.hub_height, frix)
    
#     if source == "NSRDB":
#         k, c, stdev = get_weibull_params(conv_ws)
#     else:
#         k, c = get_weibull_params(conv_ws, nsrdb_stdev, "IRENA")
    
#     # Create Weibull distribution
#     weibull_dist = weibull_min(k, scale=c)
    
#     # Define wind speed bins
#     bin_width = 0.5
#     max_wind_speed = min(25, turbine_model.max + 2)  # Cap at reasonable value
#     bin_edges = np.arange(0, max_wind_speed + bin_width, bin_width)
#     wind_speeds = bin_edges[:-1] + bin_width/2  # Use bin centers
    
#     # Get probabilities for each bin
#     probabilities = np.diff(weibull_dist.cdf(bin_edges))
    
#     # Get power output for each wind speed
#     power_outputs = np.array([turbine_model.output_power(v) for v in wind_speeds])
    
#     # Calculate expected power (this replaces your integral!)
#     expected_power = np.sum(power_outputs * probabilities)
    
#     # Rest of your calculations...
#     turbines_per_pixel = (9*1000000) / ((7*turbine_model.rotor_diameter) * (5*turbine_model.rotor_diameter))
#     energy_per_pixel = (int(turbines_per_pixel) * expected_power * hours) / 1000
    
#     if source == "NSRDB": 
#         monthly_yield = energy_per_pixel * points_num
#         return monthly_yield, stdev
#     else:
#         return energy_per_pixel

#FOR DEBUGGING
# def calc_energy_yield_discrete(hours: int, points_num: int, height1: int, ws_height1: list, 
#                               height2: int, ws_height2, turbine_model, nsrdb_stdev=None, source="NSRDB"):
    
#     print("=" * 80)
#     print("DEBUG: calc_energy_yield_discrete STARTING")
#     print("=" * 80)
    
#     # Print all inputs
#     print(f"INPUT PARAMETERS:")
#     print(f"  hours: {hours}")
#     print(f"  points_num: {points_num}")
#     print(f"  height1: {height1}m")
#     print(f"  height2: {height2}m")
#     print(f"  turbine_model.hub_height: {turbine_model.hub_height}m")
#     print(f"  turbine_model.rotor_diameter: {turbine_model.rotor_diameter}m")
#     print(f"  turbine_model.max: {getattr(turbine_model, 'max', 'N/A')}")
#     print(f"  nsrdb_stdev: {nsrdb_stdev}")
#     print(f"  source: {source}")
#     print()
    
#     # Print wind speed data statistics
#     print(f"WIND SPEED DATA:")
#     print(f"  ws_height1 length: {len(ws_height1)}")
#     print(f"  ws_height1 first 10: {ws_height1[:10]}")
#     print(f"  ws_height1 stats: min={min(ws_height1):.3f}, max={max(ws_height1):.3f}, mean={np.mean(ws_height1):.3f}")
#     print(f"  ws_height1 NaN count: {sum(1 for x in ws_height1 if np.isnan(x))}")
#     print(f"  ws_height1 zero/negative count: {sum(1 for x in ws_height1 if x <= 0)}")
#     print()
    
#     print(f"  ws_height2 length: {len(ws_height2)}")
#     print(f"  ws_height2 first 10: {ws_height2[:10]}")
#     print(f"  ws_height2 stats: min={min(ws_height2):.3f}, max={max(ws_height2):.3f}, mean={np.mean(ws_height2):.3f}")
#     print(f"  ws_height2 NaN count: {sum(1 for x in ws_height2 if np.isnan(x))}")
#     print(f"  ws_height2 zero/negative count: {sum(1 for x in ws_height2 if x <= 0)}")
#     print()
    
#     # Step 1: Calculate friction coefficients
#     print("STEP 1: Calculating friction coefficients...")
#     try:
#         frix = get_alpha(height1, ws_height1, height2, ws_height2)
#         print(f"  frix length: {len(frix)}")
#         print(f"  frix first 10: {frix[:10]}")
#         print(f"  frix stats: min={min(frix):.3f}, max={max(frix):.3f}, mean={np.mean(frix):.3f}")
#         print(f"  frix NaN count: {sum(1 for x in frix if np.isnan(x))}")
#         print(f"  frix extreme values (>1.0): {sum(1 for x in frix if abs(x) > 1.0)}")
#         print(f"  ✓ get_alpha SUCCESS")
#     except Exception as e:
#         print(f"  ✗ get_alpha FAILED: {e}")
#         return None
#     print()
    
#     # Step 2: Extrapolate wind speeds
#     print("STEP 2: Extrapolating wind speeds...")
#     try:
#         conv_ws = extrapolate_windspeed(height2, ws_height2, turbine_model.hub_height, frix)
#         print(f"  conv_ws length: {len(conv_ws)}")
#         print(f"  conv_ws first 10: {conv_ws[:10]}")
#         print(f"  conv_ws stats: min={min(conv_ws):.3f}, max={max(conv_ws):.3f}, mean={np.mean(conv_ws):.3f}")
#         print(f"  conv_ws NaN count: {sum(1 for x in conv_ws if np.isnan(x))}")
#         print(f"  conv_ws extreme values (>50): {sum(1 for x in conv_ws if x > 50)}")
#         print(f"  ✓ extrapolate_windspeed SUCCESS")
#     except Exception as e:
#         print(f"  ✗ extrapolate_windspeed FAILED: {e}")
#         return None
#     print()
    
#     # Step 3: Get Weibull parameters
#     print("STEP 3: Calculating Weibull parameters...")
#     try:
#         if source == "NSRDB":
#             k, c, stdev = get_weibull_params(conv_ws)
#             print(f"  Weibull shape (k): {k:.6f}")
#             print(f"  Weibull scale (c): {c:.6f}")
#             print(f"  Standard deviation: {stdev:.6f}")
#         else:
#             k, c = get_weibull_params(conv_ws, nsrdb_stdev, "IRENA")
#             print(f"  Weibull shape (k): {k:.6f}")
#             print(f"  Weibull scale (c): {c:.6f}")
#             stdev = nsrdb_stdev
        
#         # Validate Weibull parameters
#         if np.isnan(k) or np.isnan(c) or k <= 0 or c <= 0:
#             print(f"  ✗ INVALID Weibull parameters!")
#             return None
#         else:
#             print(f"  ✓ get_weibull_params SUCCESS")
#     except Exception as e:
#         print(f"  ✗ get_weibull_params FAILED: {e}")
#         return None
#     print()
    
#     # Step 4: Create Weibull distribution
#     print("STEP 4: Creating Weibull distribution...")
#     try:
#         from scipy.stats import weibull_min
#         weibull_dist = weibull_min(k, scale=c)
        
#         # Test the distribution
#         test_vals = [1, 5, 10, 15]
#         for val in test_vals:
#             pdf_val = weibull_dist.pdf(val)
#             cdf_val = weibull_dist.cdf(val)
#             print(f"  Test: PDF({val}) = {pdf_val:.6f}, CDF({val}) = {cdf_val:.6f}")
        
#         print(f"  ✓ weibull_dist creation SUCCESS")
#     except Exception as e:
#         print(f"  ✗ weibull_dist creation FAILED: {e}")
#         return None
#     print()
    
#     # Step 5: Define wind speed bins
#     print("STEP 5: Defining wind speed bins...")
#     try:
#         bin_width = 0.5
#         max_wind_speed = min(25, getattr(turbine_model, 'max', 25) + 2)
#         bin_edges = np.arange(0, max_wind_speed + bin_width, bin_width)
#         wind_speeds = bin_edges[:-1] + bin_width/2
        
#         print(f"  bin_width: {bin_width}")
#         print(f"  max_wind_speed: {max_wind_speed}")
#         print(f"  bin_edges length: {len(bin_edges)} (from {bin_edges[0]} to {bin_edges[-1]})")
#         print(f"  wind_speeds length: {len(wind_speeds)} (from {wind_speeds[0]} to {wind_speeds[-1]})")
#         print(f"  wind_speeds sample: {wind_speeds[:10]}...")
#         print(f"  ✓ wind speed bins SUCCESS")
#     except Exception as e:
#         print(f"  ✗ wind speed bins FAILED: {e}")
#         return None
#     print()
    
#     # Step 6: Get probabilities for each bin
#     print("STEP 6: Calculating probabilities...")
#     try:
#         cdf_values = weibull_dist.cdf(bin_edges)
#         probabilities = np.diff(cdf_values)
        
#         print(f"  cdf_values length: {len(cdf_values)}")
#         print(f"  cdf_values range: {cdf_values[0]:.6f} to {cdf_values[-1]:.6f}")
#         print(f"  probabilities length: {len(probabilities)}")
#         print(f"  probabilities sum: {np.sum(probabilities):.6f} (should be ~1.0)")
#         print(f"  probabilities max: {np.max(probabilities):.6f}")
#         print(f"  probabilities first 10: {probabilities[:10]}")
#         print(f"  probabilities NaN count: {np.sum(np.isnan(probabilities))}")
        
#         if np.sum(probabilities) < 0.95 or np.sum(probabilities) > 1.05:
#             print(f"  ⚠ WARNING: Probabilities don't sum to 1!")
        
#         print(f"  ✓ probabilities calculation SUCCESS")
#     except Exception as e:
#         print(f"  ✗ probabilities calculation FAILED: {e}")
#         return None
#     print()
    
#     # Step 7: Get power outputs
#     print("STEP 7: Calculating power outputs...")
#     try:
#         power_outputs = []
#         for i, v in enumerate(wind_speeds):
#             try:
#                 power = turbine_model.output_power(v)
#                 power_outputs.append(power)
#                 if i < 5:  # Show first 5
#                     print(f"  Power at {v:.1f} m/s: {power:.2f} kW")
#             except Exception as e:
#                 print(f"  Error getting power at {v:.1f} m/s: {e}")
#                 power_outputs.append(0)
        
#         power_outputs = np.array(power_outputs)
        
#         print(f"  power_outputs length: {len(power_outputs)}")
#         print(f"  power_outputs stats: min={np.min(power_outputs):.2f}, max={np.max(power_outputs):.2f}, mean={np.mean(power_outputs):.2f}")
#         print(f"  power_outputs NaN count: {np.sum(np.isnan(power_outputs))}")
#         print(f"  power_outputs non-zero count: {np.sum(power_outputs > 0)}")
#         print(f"  ✓ power outputs calculation SUCCESS")
#     except Exception as e:
#         print(f"  ✗ power outputs calculation FAILED: {e}")
#         return None
#     print()
    
#     # Step 8: Calculate expected power
#     print("STEP 8: Calculating expected power...")
#     try:
#         if len(power_outputs) != len(probabilities):
#             print(f"  ✗ LENGTH MISMATCH: power_outputs={len(power_outputs)}, probabilities={len(probabilities)}")
#             return None
        
#         power_prob_products = power_outputs * probabilities
#         expected_power = np.sum(power_prob_products)
        
#         print(f"  power × probability products stats:")
#         print(f"    min: {np.min(power_prob_products):.6f}")
#         print(f"    max: {np.max(power_prob_products):.6f}")
#         print(f"    sum (expected_power): {expected_power:.6f} kW")
        
#         if np.isnan(expected_power) or np.isinf(expected_power):
#             print(f"  ✗ INVALID expected_power: {expected_power}")
#             return None
        
#         print(f"  ✓ expected power calculation SUCCESS: {expected_power:.2f} kW")
#     except Exception as e:
#         print(f"  ✗ expected power calculation FAILED: {e}")
#         return None
#     print()
    
#     # Step 9: Calculate final results
#     print("STEP 9: Final calculations...")
#     try:
#         turbines_per_pixel = (9*1000000) / ((7*turbine_model.rotor_diameter) * (5*turbine_model.rotor_diameter))
#         energy_per_pixel = (int(turbines_per_pixel) * expected_power * hours)
        
#         print(f"  Turbine spacing calculation:")
#         print(f"    Rotor diameter: {turbine_model.rotor_diameter}m")
#         print(f"    Spacing: 7D × 5D = {7*turbine_model.rotor_diameter}m × {5*turbine_model.rotor_diameter}m")
#         print(f"    Area per turbine: {(7*turbine_model.rotor_diameter) * (5*turbine_model.rotor_diameter):,.0f} m²")
#         print(f"    Pixel area: 9,000,000 m²")
#         print(f"    Turbines per pixel (float): {turbines_per_pixel:.2f}")
#         print(f"    Turbines per pixel (int): {int(turbines_per_pixel)}")
#         print()
#         print(f"  Energy calculation:")
#         print(f"    Expected power per turbine: {expected_power:.2f} kW")
#         print(f"    Number of turbines: {int(turbines_per_pixel)}")
#         print(f"    Hours: {hours}")
#         print(f"    Total energy per pixel: {energy_per_pixel:.2f} MWh")
        
#         if source == "NSRDB":
#             monthly_yield = energy_per_pixel * points_num
#             print(f"    Points in municipality: {points_num}")
#             print(f"    Total municipal yield: {monthly_yield:.2f} MWh")
#             result = (monthly_yield, stdev)
#             print(f"  FINAL RESULT (NSRDB): {result}")
#         else:
#             result = energy_per_pixel
#             print(f"  FINAL RESULT (IRENA): {result:.2f} MWh")
        
#         print(f"  ✓ Final calculations SUCCESS")
        
#     except Exception as e:
#         print(f"  ✗ Final calculations FAILED: {e}")
#         return None
    
#     print("=" * 80)
#     print("DEBUG: calc_energy_yield_discrete COMPLETED SUCCESSFULLY")
#     print("=" * 80)
#     print()
    
#     return result

def calc_energy_yield_discrete(hours: int, points_num:int, height1:int, ws_height1:list, height2:int, ws_height2, turbine_model, nsrdb_stdev = None, source = "NSRDB"):
   """ input: list of wind speeds from two diff heights"""

   #debugging
  #  if source == 'IRENA':
  #     print("debugging")
  #     print(f'source: {source}')
  #     print("inputs:")
  #     print(f"hours: {hours}, points_num: {points_num}")
  #     print(f"height1: {height1}\n ws_height1:{ws_height1}") 
  #     print(f"height2: {height2}\n ws_height2: {ws_height2} ")
  #     print(f"turbine hubheight: {turbine_model.hub_height}")
  #     print(f"nsrdb_stdev: {nsrdb_stdev}")
   
   # obtain first the friction coefficients
   frix, frix_monthly_ave = get_alpha(height1, ws_height1, height2, ws_height2)
  #  print(f"fiction coefficient: {frix}")

   # extrapolate to turbine height
   conv_ws, ave_conv_ws = extrapolate_windspeed(height2, ws_height2, turbine_model.hub_height, frix)
  #  print(f"extrapolated windspeeds: \n {conv_ws}")

   # obtain shape and scale parameters for the weibull
   if source == "NSRDB":
     k, c, stdev = get_weibull_params(conv_ws)
   else:
     k, c = get_weibull_params(conv_ws, nsrdb_stdev, "IRENA" )
    #  print(f"shape: {k}")
    #  print(f"scale: {c}")


   # ========== OLD METHOD (DELETE THIS) ==========
  #  def weibull_pdf(v):
  #    return (k/c) * ((v/c) ** (k-1) ) * np.exp(-(v / c) ** k)
   
  #  n_bins = 100
  #  percentiles = np.linspace(0.5, 99.5, n_bins)  # Avoid extreme tails
  #  wind_speeds = c * (-np.log(1 - percentiles/100)) ** (1/k)
  #  if source == "IRENA":
  #   print(f'wind_speeds: {wind_speeds}')
   
  #  prob_weight = 1/n_bins
   
  #  power_outputs = np.array([turbine_model.output_power(v) for v in wind_speeds])
  #  expected_power_per_turbine = np.sum(power_outputs * prob_weight)
   # ========== END OLD METHOD ==========

   # NEW METHOD 1: Traditional Numerical Integration using your PDF
   def weibull_pdf(v):
     return (k/c) * ((v/c) ** (k-1) ) * np.exp(-(v / c) ** k)
   
   v_min = 0.001
   v_max = 30  
   n_points = 1000 
   wind_speeds = np.linspace(v_min, v_max, n_points)
   
   pdf_values = weibull_pdf(wind_speeds)
  
   power_outputs = np.array([turbine_model.output_power(v) for v in wind_speeds])
   
   # Integrate: P(v) * f(v) * dv
   dv = wind_speeds[1] - wind_speeds[0]
   expected_power_per_turbine = np.sum(power_outputs * pdf_values * dv)
   
  #  if source == "IRENA":
  #     # print(f"wind_speeds range: {v_min} to {v_max} m/s")
  #     print(f"power_output per turbine: {expected_power_per_turbine}")

   # ========== DEBUG LINES HERE ==========
   
   # ========== END DEBUG SECTION ==========
   
   # Rest of your code stays exactly the same...
   turbines_per_pixel = ((1/3) * 9*1000000)/ ((7*turbine_model.rotor_diameter) * (5*turbine_model.rotor_diameter))
   energy_per_pixel = (int(turbines_per_pixel) * expected_power_per_turbine * hours) #in kWH

  #  if source != "IRENA":
  #     print(f"energy per pixel in nsrdb: {energy_per_pixel}")
  #  if source == 'IRENA':
  #     print(f"turbines_per pixel = {int(turbines_per_pixel)}")
  #     print(f"energy per pixel: {energy_per_pixel}")

   if source == "NSRDB": 
    monthly_yield = energy_per_pixel 
    # print(f"nsrdb monthly_yield: {monthly_yield}")
    return monthly_yield, stdev, frix_monthly_ave, ave_conv_ws, k, c, conv_ws
   else:
     return energy_per_pixel, frix_monthly_ave, ave_conv_ws, k, c, conv_ws



def compute_capacity(turbine_model, pixel_num):
  turbines_per_pixel = (9*1000000)/ ((7*turbine_model.rotor_diameter) * (5*turbine_model.rotor_diameter))

  capacity = turbine_model.p_rated * turbines_per_pixel * pixel_num

  return capacity #in KW since the power rated is in kw

def compute_monthly_capacity_factor(monthly_yield, num_hours, capacity):
  print(f"inputs in computing cf: \n e_yield: {monthly_yield} \n hours: {num_hours} capacity: {capacity}")
  # st.write(f"{monthly_yield} hours: {num_hours}, capacity: {capacity}")
  # st.write('debugging')
  cf = (monthly_yield / (capacity * num_hours)) * 100
  return cf

def plot_monthly_value(IRENA: list, NSRDB: list, IRENA_cf: list, NSRDB_cf, capacity, IRENA_lcoe, NSRDB_lcoe):
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

def compute_lcoe(cf:float):
   
   #st.write(f"this is cf: {cf}")

   '''takes capacity factor as an input and outputs the LCOE in ___ (unit)'''


   fixed_charge_rate = 0.092 #unitless
   capital_cost = 126518487 #php/mw why 
   fixed_OM_cost = 1265185 #php/mw/year 
   variable_OM_cost = 0 
   fuel_cost =  0 


   lcoe = (((fixed_charge_rate * capital_cost) + fixed_OM_cost) / ((cf/100) * 8760) ) + (variable_OM_cost) + fuel_cost


   return lcoe

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
    



# def db_fetch_hourly_solar(valid_points:list, municipality = None):
#   """ returns in this format:           Lat | Long | Month | Day | Hour | GHI | MuniCipality 
#                               point a                               1                        
#                               point b                               1                        """

#   working_db = connect_to_db()
#   while working_db == None:
#     working_db = connect_to_db()

#   pointer = working_db.cursor()

#   prep_points = ', '.join(['(%s, %s)'] * len(valid_points))
#   coords = [coord for point in valid_points for coord in point]

#   if municipality == None:
#     query = f"""SELECT latitude, longitude, "Month", "Day", "Hour", "GHI", "municipality"
#     FROM "NSRDB_SOLAR"
#     WHERE (ROUND(longitude::NUMERIC, 6), ROUND(latitude::NUMERIC, 6)) IN ({prep_points})
#     ORDER BY "Month" ASC, "Day" ASC, "Hour" ASC;"""

#     pointer.execute(query, coords)

#     solar_data = pointer.fetchall()

#     pointer.close()
#     working_db.close()

#     return solar_data
  
#   else:
#     query = f"""SELECT latitude, longitude, "Month", "Day", "Hour", "GHI", "municipality"
#     FROM "NSRDB_SOLAR"
#     WHERE municipality = {municipality}
#     AND (ROUND(longitude::NUMERIC, 6), ROUND(latitude::NUMERIC, 6)) IN ({prep_points})
#     ORDER BY "Month" ASC, "Day" ASC, "Hour" ASC;"""

#     pointer.execute(query, coords)

#     solar_data = pointer.fetchall()

#     pointer.close()
#     working_db.close()

#     return solar_data
  

# def db_fetch_IRENA_solar(valid_points:list, municipality = None):
#   """ returns in this format:           longitude (xcoord) | latitude (ycoord) | jan ghi | ... | dec ghi | municipality
#                               point a                                                       
#                               point b                                                       """

#   working_db = connect_to_db()
#   while working_db == None:
#     working_db = connect_to_db()

#   pointer = working_db.cursor()
#   prep_points = ', '.join(['(%s, %s)'] * len(valid_points))
#   coords = [coord for point in valid_points for coord in point]

#   if municipality == None:
#     query = f"""
#     SELECT xcoord, ycoord,"jan ghi1",
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
#     WHERE (ROUND(xcoord::NUMERIC, 6), ROUND(ycoord::NUMERIC, 6)) IN ({prep_points});"""

#     pointer.execute(query, coords)
#     solar_data = pointer.fetchall()
#     pointer.close()
#     working_db.close()
#     return solar_data


# #----------------------------------------------------------------------------------------
# # the following lines are used to display on the website
# sample_munip = obtain_municip()
# sample_munip = sorted([row[0] for row in sample_munip])


# # let user choose turbine model first
# wind_turbine_options = [f"1. model: {V126.model_name}, hub height: {V126.hub_height}, rotor diameter: {V126.rotor_diameter}", f"2. model: {V150.model_name}, hub height: {V150.hub_height}, rotor diameter: {V150.rotor_diameter}", f"3. model: {V165.model_name}, hub height: {V165.hub_height}, rotor diameter: {V165.rotor_diameter}" ]
# selected_turbine = st.selectbox("Choose a wind turbine model:", wind_turbine_options)
# turbine_model = turbines[int(selected_turbine[0])]

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


# # Display 
# selected_option = st.selectbox("Choose a municipality:", sample_munip)

# # filter out invalid points (those that are exclusion areas based on user's constraint selection)
# temp_points = db_fetch_IRENA(municipality = selected_option)
# temp_points = [(round(float(row[0]), 6), round(float(row[1]), 6)) for row in temp_points]

# if choose_from:

#   valid_points = look_up_points(temp_points, choose_from)

#   st.write(f"all of the points: {temp_points}")
#   st.write(f"filtered points: {valid_points}")

# else:
#   st.write(f"filtered points: No Constraint Selected. ")
#   valid_points = temp_points

# # st.write(f"this are the valid points: {valid_points}")

# #fetch irena again for the valid points
# irena_ds20, irena_ds60, irena_points = db_fetch_IRENA(valid_points, municipality = None)
# # st.write(irena_ds20)
# # st.write(irena_ds60)

# #fetch NSRDB data
# nsrdb_monthly40WS = []
# nsrdb_monthly60WS = []
# for month in range(1, 13):
#   monthly_40ws_data = fetch_spatial_average(month, valid_points, 40)
#   monthly_60ws_data = fetch_spatial_average(month, valid_points, 60)

#   nsrdb_monthly40WS.append(monthly_40ws_data)
#   nsrdb_monthly60WS.append(monthly_60ws_data)
  

# # initialize

# NSRDB_monthly_energy_yield = []
# IRENA_monthly_energy_yield = []
# month_point_yield = [] #for IRENA
# NSRDB_monthly_cf = []
# IRENA_monthly_cf = []
# capacity = 0
# lcoe = 0

# # process per month
# for month in range(1, 13):
#   # compute energy yield using NSRDB dataset first
#   month_40ws = nsrdb_monthly40WS[month - 1]
#   month_60ws = nsrdb_monthly60WS[month - 1]
#   hours = len(month_40ws)

#   e_yield1, std = calc_energy_yield(hours, len(valid_points), 40, month_40ws, 60, month_60ws, turbine_model )
#   NSRDB_monthly_energy_yield.append(e_yield1)

#   # st.write("Using IRENA:")
#   # st.write(f"month: {month} energy yield: {e_yield1} stdev: {std}")

#   #then compute using IRENA dataset

#   for point in range(len(irena_ds20)):

#     e_yield2 = calc_energy_yield(hours, len(valid_points), 20, [irena_ds20[point][month -1]], 60, [irena_ds60[point][month -1]], turbine_model, nsrdb_stdev= std, source= "IRENA" )

#     month_point_yield.append(e_yield2)

#   total = 0
#   for point_yield in month_point_yield:
#     total += point_yield

#   IRENA_monthly_energy_yield.append(total)

#   st.write(f"e yield comparison ---- month: {month} nsrdb: {e_yield1} irena: {total}")
#   diff = abs(e_yield1 - total)

#   st.write(f"this is how much they differ: {str(round(diff, 2))}")

# # handle municipality-based assessment

# capacity = compute_capacity(turbine_model, len(irena_ds20))
# st.write(f"capacity: {capacity}")

# month_cntr = 0
# irena_lcoe = []
# nsrdb_lcoe = []

# for e_yield in NSRDB_monthly_energy_yield:
#   month_60ws = nsrdb_monthly60WS[month_cntr]
#   hours = len(month_40ws)
#   nsrdb_cf = compute_monthly_capacity_factor(e_yield, hours, capacity)
#   NSRDB_monthly_cf.append(nsrdb_cf)
#   nsrdb_lcoe.append(compute_lcoe(nsrdb_cf))
#   month_cntr += 1

# month_cntr = 0

# for e_yield in IRENA_monthly_energy_yield:
#   month_60ws = nsrdb_monthly60WS[month_cntr]
#   hours = len(month_40ws)
#   irena_cf = compute_monthly_capacity_factor(e_yield, hours, capacity)
#   irena_lcoe.append(compute_lcoe(irena_cf))
#   IRENA_monthly_cf.append(irena_cf)
#   month_cntr += 1


# for ind in range(0,12):
#   st.write(f"month: {ind+1} IRENA E-yeild: {IRENA_monthly_energy_yield[ind]} NSRDB yield: {NSRDB_monthly_energy_yield[ind]}")
#   diff_cf = abs(IRENA_monthly_energy_yield[ind] - NSRDB_monthly_energy_yield[ind] )
#   st.write(f"this is how much they differ: {str(round(diff_cf, 2))}")


# for ind in range(0,12):
#   st.write(f"month: {ind+1} IRENA CF: {IRENA_monthly_cf[ind]} NSRDB CF: {NSRDB_monthly_cf[ind]}")
#   diff_cf = abs(IRENA_monthly_cf[ind] - NSRDB_monthly_cf[ind] )
#   st.write(f"this is how much they differ: {str(round(diff_cf, 2))}")

#   st.write(f"month: {ind+1} IRENA lcoe: {irena_lcoe[ind]} NSRDB lcoe: {nsrdb_lcoe[ind]}")
#   diff_lcoe = abs(irena_lcoe[ind] - nsrdb_lcoe[ind] )
#   st.write(f"this is how much they differ: {str(round(diff_lcoe, 2))}")


# plot_monthly_value(IRENA_monthly_energy_yield, NSRDB_monthly_energy_yield, IRENA_monthly_cf, NSRDB_monthly_cf, capacity)

# fetch data from supabase cloud through API -------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# from supabase import create_client, Client

# url = "https://qlozqioxlituanstspdi.supabase.co"
# key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InFsb3pxaW94bGl0dWFuc3RzcGRpIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDUzNzgzNjUsImV4cCI6MjA2MDk1NDM2NX0.FK6SfWmrRICaqUdFTV7aHBvbUm2EnAQXTmBLIWcQJeQ"
# supabase: Client = create_client(url, key)

# response = supabase.table("NSRDB_WIND").select("*").execute()
# st.write(response.data)

#plotting using plotly for interactive plots -------------------------------------------------------------------------------


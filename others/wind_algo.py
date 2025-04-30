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

# #datasheet link: https://en.wind-turbine-models.com/turbines/15-vestas-v66-1.65
V165 = Turbine("V66-1.65", 25, 4, 1650, 66, 78, np.arange(3, 28.5, 0.5), [0, 0, 20, 47.5, 78.2, 123.3, 170.67, 222, 278, 365.2, 465.85, 568.5, 673.3, 808.9, 913.6, 984, 1054.4, 1143.35, 1242.9, 1347, 1439.35, 1509.2, 1567, 1595, 1620.67, 1633.5, 1639, 1644.5, 1650, 1650, 1650, 1650, 1650, 1650, 1650, 1650, 1650, 1650, 1650, 1650, 1650, 1650, 1650, 1650, 1650, 0, 0, 0, 0, 0, 0])


# #datasheet link: https://en.wind-turbine-models.com/turbines/1249-vestas-v126-3.45
V126 = Turbine("V126-3.45MW", 22.5, 3, 3450, 126, 149, np.arange(3, 26, 0.5), [35, 101, 184, 283, 404, 550, 725, 932, 1172, 1446, 1760, 2104, 2482, 2865, 3187, 3366, 3433, 3448, 3450, 3450, 3450, 3450, 3450, 3450, 3450, 3450, 3450, 3450, 3450, 3450, 3450, 3450, 3450, 3450, 3450, 3450, 3450, 3450, 3450, 3450, 0, 0, 0, 0, 0 ,0])


# #datasheet link: https://www.thewindpower.net/turbine_en_1490_vestas_v150-4000-4200.php
V150 = Turbine("V150/4000-4200",3,22.5, 4200, 150, 166, np.arange(3, 26, 0.5), [78, 172, 287, 426, 601, 814, 1069, 1367, 1717, 2110, 2546, 3002, 3428, 3773, 4012, 4131, 4186, 4198, 4200, 4200, 4200, 4200, 4200, 4200, 4200, 4200, 4200, 4200, 4200, 4200, 4200, 4200, 4200, 4200, 4200, 4200, 4200, 4200, 4200, 4200, 0, 0, 0, 0, 0, 0])

turbines = {1:V126, 2:V150, 3:V165}
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

def get_alpha(height1:int, windspeed1: list, height2:int, windspeed2:list):
  """ computes the friction coefficient for each wind speed pair measured from different heights. """
  fric_coeffs = []

  for index in range(0, len(windspeed1)):

    # x > 0 for log(x)

    #WARNING: CHANGE THIS WHEN THE NEW DATA IS IN !!

    # wind1 = max(windspeed1[index], 2.0)  
    # wind2 = max(windspeed2[index], 2.0)

    alpha = (math.log(windspeed2[index]) - math.log(windspeed1[index])) / (math.log(height2) - math.log(height1))
    fric_coeffs.append(alpha)

  return fric_coeffs


def extrapolate_windspeed(height1:int, wind_speed1:list, turbine_height:int, alpha:list):
  """ extrapolates the wind speed data to the height of the turbine using the power law. """
  new_windspeeds = []
  for index in range(0, len(wind_speed1)):
    new_wind = wind_speed1[index] * (turbine_height/height1) ** alpha[index] 
    new_windspeeds.append(new_wind)

  return new_windspeeds


def get_weibull_params(extrapolated_WS:list, stdev = None, source = "NSRDB"):
  """ computes shape and scale paramaters. by default, the computation is suited for the NSRDB database. """

  ave_ws =  statistics.mean(extrapolated_WS)

  if source == "NSRDB":
    
    std_ws = statistics.stdev(extrapolated_WS)
    shape = (std_ws/ave_ws) ** -1.086
    scale = ave_ws / math.gamma(1/(1+shape))
    return shape, scale, std_ws

  else:
    shape = (stdev/ave_ws) ** -1.086
    scale = ave_ws / math.gamma(1/(1+shape))

    return (shape, scale)


def calc_energy_yield(hours: int, points_num:int, height1:int, ws_height1:list, height2:int, ws_height2, turbine_model, nsrdb_stdev = None, source = "NSRDB"):
   """ input: list of wind speeds from two diff heights"""
   
   # obtain first the friction coefficients
   frix = get_alpha(height1, ws_height1, height2, ws_height2)
   #st.write(f"40m: {ws_40m}, 60m: {ws_60m}, alphas: {frix}")

   # extrapolate to turbine height
   conv_ws = extrapolate_windspeed(height1, ws_height1, turbine_model.hub_height, frix)
   #st.write(f"converted in munip func: {conv_ws}")

   # obtain shape and scale parameters for the weibull
   if source == "NSRDB":
     k, c, stdev = get_weibull_params(conv_ws)
    #  st.write(f"NSRDB: this is stdev used: {stdev}")

   else:
     k, c = get_weibull_params(conv_ws, nsrdb_stdev, "IRENA" )
    #  st.write(f"IRENA: this is stdev used: {nsrdb_stdev}")

  #  st.write(f"this is shape used: {k}, this is scale used: {c}")

   def weibull_pdf(v):
     return (k/c) * ((v/c) ** (k-1) ) * np.exp(-(v / c) ** k)

   # expected energy per turbine

   # assuming this is correct first !!
   turbine_yield = integrate.quad(lambda v: turbine_model.output_power(v) * weibull_pdf(v), turbine_model.min, turbine_model.max, limit = 200)


   # multiply by the number of turbines that can fit in a 3km x 3km pixel. (This is energy generated per pixel)
   # follow IRENA's recommended turbine layout usable_area/7Dx5D
   turbines_per_pixel = (9*1000000)/ ((7*turbine_model.rotor_diameter) * (5*turbine_model.rotor_diameter))
   energy_per_pixel = turbines_per_pixel * turbine_yield[0] * hours

   if source == "NSRDB": 
    # multiply this number by the number of UNIQUE centroids of the municipality. (This is the energy generated by the MUNICIPALITY)
    monthly_yield = energy_per_pixel * points_num
    return monthly_yield, stdev
   
   else:
     return energy_per_pixel



def compute_capacity(turbine_model, pixel_num):
  turbines_per_pixel = (9*1000000)/ ((7*turbine_model.rotor_diameter) * (5*turbine_model.rotor_diameter))
  capacity = turbine_model.p_rated * turbines_per_pixel * pixel_num
  return capacity

def compute_monthly_capacity_factor(monthly_yield, num_hours, capacity):
  cf = (monthly_yield / (capacity * num_hours))
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

   '''takes capacity factor as an input and outputs the LCOE in ___ (unit)'''


   fixed_charge_rate = 0.092 #unitless
   capital_cost = 126518487 #php/mw why is this the same with the solar counterpart
   fixed_OM_cost = 1265185 #php/mw/year same ques in above
   variable_OM_cost = 00 #MISSING VALUE NOT IN MANUSCRIPT!!!
   fuel_cost =  0 #MISSING VALUE NOT IN MANUSCRIPT !!!!


   lcoe = (((fixed_charge_rate * capital_cost) + fixed_OM_cost) / (cf * 8760) ) + (variable_OM_cost) + fuel_cost


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


# # import pandas as pd
# # import numpy as np
# # import matplotlib.pyplot as plt
# # import seaborn as sns
# # from scipy import stats
# # import os
# # from pathlib import Path

# # # Set style for better plots
# # plt.style.use('seaborn-v0_8')
# # sns.set_palette("husl")

# # class WindDataAnalyzer:
# #     def __init__(self, coordinates_list):
# #         """
# #         Initialize the wind data analyzer
        
# #         Parameters:
# #         coordinates_list: List of coordinate tuples [(lat1, lon1), (lat2, lon2), ...]
# #         """
# #         self.coordinates = coordinates_list
# #         self.months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
# #                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
# #     def inspect_csv_structure(self, lat, lon):
# #         """
# #         Inspect the CSV file structure to help with debugging
# #         """
# #         filename = f"ninja_wind_{lat}_{lon}_uncorrected.csv"
# #         try:
# #             df_inspect = pd.read_csv(filename, nrows=10)
# #             print(f"\nInspecting {filename}:")
# #             print("First 10 rows:")
# #             print(df_inspect)
# #             print(f"\nColumn names: {list(df_inspect.columns)}")
            
# #             df_data = pd.read_csv(filename, skiprows=4, nrows=5)
# #             print(f"\nData starting from row 5:")
# #             print(df_data)
            
# #         except Exception as e:
# #             print(f"Error inspecting {filename}: {str(e)}")

# #     def load_ninja_data(self, lat, lon):
# #         """
# #         Load ninja wind data from CSV file
# #         CSV structure: Data starts at row 5 (D5 for wind speed, B5 for timestamp)
# #         """
# #         filename = f"ninja_wind_{lat}_{lon}_uncorrected.csv"
# #         try:
# #             df = pd.read_csv(filename, skiprows=4)
            
# #             if df.shape[1] < 4:
# #                 print(f"Warning: {filename} has only {df.shape[1]} columns, expected at least 4")
# #                 return None
            
# #             # Extract timestamp (column B) and wind speed (column D)
# #             timestamp_col = df.iloc[:, 1]  # Column B 
# #             windspeed_col = df.iloc[:, 3]   # Column D
            
# #             clean_df = pd.DataFrame({
# #                 'timestamp': timestamp_col,
# #                 'wind_speed': pd.to_numeric(windspeed_col, errors='coerce')
# #             })
            
# #             clean_df['timestamp'] = pd.to_datetime(clean_df['timestamp'], errors='coerce')
            
# #             initial_rows = len(clean_df)
# #             clean_df = clean_df.dropna()
# #             final_rows = len(clean_df)
            
# #             if initial_rows != final_rows:
# #                 print(f"Removed {initial_rows - final_rows} rows with NaN values")
            
# #             if len(clean_df) == 0:
# #                 print(f"Warning: No valid data found in {filename}")
# #                 return None
            
# #             print(f"Loaded {len(clean_df)} records from {filename}")
# #             print(f"Date range: {clean_df['timestamp'].min()} to {clean_df['timestamp'].max()}")
# #             print(f"Wind speed range: {clean_df['wind_speed'].min():.2f} to {clean_df['wind_speed'].max():.2f} m/s")
            
# #             return clean_df
            
# #         except FileNotFoundError:
# #             print(f"Warning: File {filename} not found")
# #             return None
# #         except Exception as e:
# #             print(f"Error loading {filename}: {str(e)}")
# #             return None

# #     def calculate_monthly_averages_from_ninja(self, ninja_data):
# #         """
# #         Calculate monthly averages from ninja CSV data using timestamps
# #         """
# #         if ninja_data is None or len(ninja_data) == 0:
# #             return [0] * 12
        
# #         ninja_data['month'] = ninja_data['timestamp'].dt.month
# #         monthly_avgs = []
        
# #         for month in range(1, 13):
# #             month_data = ninja_data[ninja_data['month'] == month]['wind_speed']
# #             if len(month_data) > 0:
# #                 monthly_avgs.append(month_data.mean())
# #             else:
# #                 monthly_avgs.append(0)
        
# #         return monthly_avgs

# #     def calculate_monthly_averages_from_hourly(self, monthly_hourly_data):
# #         """
# #         Calculate monthly averages from hourly data organized by month
        
# #         Parameters:
# #         monthly_hourly_data: Can be either:
# #             - List of 12 arrays/lists (one per month) containing hourly data
# #             - Single array of 8760 hours (will be split by month)
        
# #         Returns:
# #         List of 12 monthly averages
# #         """
# #         # Check if data is already organized by month (list of 12 monthly arrays)
# #         if isinstance(monthly_hourly_data, list) and len(monthly_hourly_data) == 12:
# #             # Data is already organized by month
# #             monthly_avgs = []
# #             for month_idx, month_data in enumerate(monthly_hourly_data):
# #                 if isinstance(month_data, (list, np.ndarray)) and len(month_data) > 0:
# #                     avg = np.mean(month_data)
# #                     monthly_avgs.append(avg)
# #                     print(f"Month {month_idx+1} ({self.months[month_idx]}): {len(month_data)} hours, avg = {avg:.2f} m/s")
# #                 else:
# #                     monthly_avgs.append(0)
# #                     print(f"Month {month_idx+1} ({self.months[month_idx]}): No data")
# #             return monthly_avgs
        
# #         # Otherwise, treat as single array and split by month
# #         else:
# #             if isinstance(monthly_hourly_data, (list, np.ndarray)):
# #                 wind_speeds = np.array(monthly_hourly_data)
# #             else:
# #                 wind_speeds = monthly_hourly_data.values if hasattr(monthly_hourly_data, 'values') else monthly_hourly_data
            
# #             # Hours per month for 2019 (non-leap year)
# #             hours_per_month = [
# #                 31 * 24,  # January: 744 hours
# #                 28 * 24,  # February: 672 hours
# #                 31 * 24,  # March: 744 hours
# #                 30 * 24,  # April: 720 hours
# #                 31 * 24,  # May: 744 hours
# #                 30 * 24,  # June: 720 hours
# #                 31 * 24,  # July: 744 hours
# #                 31 * 24,  # August: 744 hours
# #                 30 * 24,  # September: 720 hours
# #                 31 * 24,  # October: 744 hours
# #                 30 * 24,  # November: 720 hours
# #                 31 * 24   # December: 744 hours
# #             ]
            
# #             monthly_avgs = []
# #             start_hour = 0
            
# #             for month_idx, hours in enumerate(hours_per_month):
# #                 end_hour = start_hour + hours
# #                 if end_hour <= len(wind_speeds):
# #                     month_data = wind_speeds[start_hour:end_hour]
# #                     avg = np.mean(month_data)
# #                     monthly_avgs.append(avg)
# #                     print(f"Month {month_idx+1} ({self.months[month_idx]}): {hours} hours, avg = {avg:.2f} m/s")
# #                 else:
# #                     month_data = wind_speeds[start_hour:]
# #                     if len(month_data) > 0:
# #                         avg = np.mean(month_data)
# #                         monthly_avgs.append(avg)
# #                         print(f"Month {month_idx+1} ({self.months[month_idx]}): {len(month_data)} hours (partial), avg = {avg:.2f} m/s")
# #                     else:
# #                         monthly_avgs.append(0)
# #                         print(f"Month {month_idx+1} ({self.months[month_idx]}): No data")
# #                     break
# #                 start_hour = end_hour
            
# #             return monthly_avgs

# #     def plot_monthly_wind_speeds(self, coord_idx, monthly_avg_dataset1, hourly_dataset2, ninja_data):
# #         """
# #         Plot overlayed monthly average wind speeds for three datasets
# #         """
# #         lat, lon = self.coordinates[coord_idx]
        
# #         # Calculate monthly averages
# #         avg_dataset1 = monthly_avg_dataset1[:12]  # Ensure 12 months
# #         avg_dataset2 = self.calculate_monthly_averages_from_hourly(hourly_dataset2)
# #         avg_ninja = self.calculate_monthly_averages_from_ninja(ninja_data)
        
# #         # Create the plot
# #         fig, ax = plt.subplots(figsize=(14, 8))
        
# #         x = np.arange(len(self.months))
# #         width = 0.25
        
# #         # Plot bars
# #         bars1 = ax.bar(x - width, avg_dataset1, width, 
# #                       label='Monthly Average Dataset 1', alpha=0.8, color='skyblue')
# #         bars2 = ax.bar(x, avg_dataset2, width, 
# #                       label='Hourly Dataset 2', alpha=0.8, color='lightcoral')
# #         bars3 = ax.bar(x + width, avg_ninja, width, 
# #                       label='Ninja CSV Dataset', alpha=0.8, color='lightgreen')
        
# #         # Customize plot
# #         ax.set_xlabel('Month', fontsize=12)
# #         ax.set_ylabel('Average Wind Speed (m/s)', fontsize=12)
# #         ax.set_title(f'Monthly Average Wind Speeds - Coordinate {coord_idx+1} ({lat}, {lon})', fontsize=14)
# #         ax.set_xticks(x)
# #         ax.set_xticklabels(self.months)
# #         ax.legend(fontsize=11)
# #         ax.grid(True, alpha=0.3)
        
# #         # Add value labels on bars
# #         for bars in [bars1, bars2, bars3]:
# #             for bar in bars:
# #                 height = bar.get_height()
# #                 ax.annotate(f'{height:.1f}',
# #                            xy=(bar.get_x() + bar.get_width() / 2, height),
# #                            xytext=(0, 3),
# #                            textcoords="offset points",
# #                            ha='center', va='bottom', fontsize=9)
        
# #         plt.tight_layout()
# #         plt.savefig(f'monthly_wind_speeds_coord_{coord_idx+1}.png', dpi=300, bbox_inches='tight')
# #         plt.show()

# #     def plot_monthly_weibull_parameters(self, coord_idx, shape_params, scale_params):
# #         """
# #         Plot monthly Weibull shape and scale parameters
# #         """
# #         lat, lon = self.coordinates[coord_idx]
        
# #         fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
# #         x = np.arange(len(self.months))
        
# #         # Plot shape parameters
# #         bars1 = ax1.bar(x, shape_params[:12], alpha=0.8, color='orange', width=0.6)
# #         ax1.set_xlabel('Month')
# #         ax1.set_ylabel('Shape Parameter (k)')
# #         ax1.set_title(f'Monthly Weibull Shape Parameters - Coordinate {coord_idx+1} ({lat}, {lon})')
# #         ax1.set_xticks(x)
# #         ax1.set_xticklabels(self.months)
# #         ax1.grid(True, alpha=0.3)
        
# #         # Add value labels for shape
# #         for bar in bars1:
# #             height = bar.get_height()
# #             ax1.annotate(f'{height:.2f}',
# #                         xy=(bar.get_x() + bar.get_width() / 2, height),
# #                         xytext=(0, 3),
# #                         textcoords="offset points",
# #                         ha='center', va='bottom', fontsize=9)
        
# #         # Plot scale parameters
# #         bars2 = ax2.bar(x, scale_params[:12], alpha=0.8, color='purple', width=0.6)
# #         ax2.set_xlabel('Month')
# #         ax2.set_ylabel('Scale Parameter (λ)')
# #         ax2.set_title(f'Monthly Weibull Scale Parameters - Coordinate {coord_idx+1} ({lat}, {lon})')
# #         ax2.set_xticks(x)
# #         ax2.set_xticklabels(self.months)
# #         ax2.grid(True, alpha=0.3)
        
# #         # Add value labels for scale
# #         for bar in bars2:
# #             height = bar.get_height()
# #             ax2.annotate(f'{height:.2f}',
# #                         xy=(bar.get_x() + bar.get_width() / 2, height),
# #                         xytext=(0, 3),
# #                         textcoords="offset points",
# #                         ha='center', va='bottom', fontsize=9)
        
# #         plt.tight_layout()
# #         plt.savefig(f'weibull_parameters_coord_{coord_idx+1}.png', dpi=300, bbox_inches='tight')
# #         plt.show()

# #     def plot_weibull_distributions(self, coord_idx, shape_params, scale_params):
# #         """
# #         Plot monthly Weibull distributions using provided shape and scale parameters
# #         """
# #         lat, lon = self.coordinates[coord_idx]
        
# #         # Create subplots for each month
# #         fig, axes = plt.subplots(3, 4, figsize=(16, 12))
# #         fig.suptitle(f'Monthly Weibull Distributions - Coordinate {coord_idx+1} ({lat}, {lon})', 
# #                     fontsize=16)
        
# #         # Wind speed range for plotting
# #         x_wind = np.linspace(0, 25, 1000)
        
# #         for i, (ax, month) in enumerate(zip(axes.flat, self.months)):
# #             if i < 12:  # Ensure we don't exceed 12 months
# #                 shape = shape_params[i]
# #                 scale = scale_params[i]
                
# #                 # Calculate Weibull PDF using your parameters
# #                 pdf = stats.weibull_min.pdf(x_wind, shape, loc=0, scale=scale)
                
# #                 # Plot the distribution
# #                 ax.plot(x_wind, pdf, linewidth=2, color='red', 
# #                        label=f'k={shape:.2f}, λ={scale:.2f}')
# #                 ax.fill_between(x_wind, pdf, alpha=0.3, color='red')
                
# #                 ax.set_title(f'{month}')
# #                 ax.set_xlabel('Wind Speed (m/s)')
# #                 ax.set_ylabel('Probability Density')
# #                 ax.grid(True, alpha=0.3)
# #                 ax.legend(fontsize=9)
# #                 ax.set_xlim(0, 20)
                
# #                 # Add mean wind speed annotation
# #                 mean_wind = scale * stats.gamma(1 + 1/shape)
# #                 ax.axvline(mean_wind, color='blue', linestyle='--', alpha=0.7)
# #                 ax.text(0.7, 0.9, f'Mean: {mean_wind:.1f} m/s', 
# #                        transform=ax.transAxes, fontsize=8,
# #                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
# #         plt.tight_layout()
# #         plt.savefig(f'weibull_distributions_coord_{coord_idx+1}.png', dpi=300, bbox_inches='tight')
# #         plt.show()

# #     def analyze_coordinate(self, coord_idx, monthly_avg_dataset1, hourly_dataset2, 
# #                           shape_params, scale_params):
# #         """
# #         Analyze a single coordinate and create all plots
# #         """
# #         lat, lon = self.coordinates[coord_idx]
# #         print(f"\nProcessing coordinate {coord_idx+1}: ({lat}, {lon})")
        
# #         # Load ninja data
# #         ninja_data = self.load_ninja_data(lat, lon)
        
# #         # Create all plots
# #         print("Creating monthly wind speed plot...")
# #         self.plot_monthly_wind_speeds(coord_idx, monthly_avg_dataset1, hourly_dataset2, ninja_data)
        
# #         print("Creating Weibull parameters plot...")
# #         self.plot_monthly_weibull_parameters(coord_idx, shape_params, scale_params)
        
# #         print("Creating Weibull distributions plot...")
# #         self.plot_weibull_distributions(coord_idx, shape_params, scale_params)

# #     def analyze_all_coordinates(self, monthly_avg_data_list, monthly_hourly_data_list, 
# #                                shape_params_list, scale_params_list):
# #         """
# #         Analyze all coordinates and create plots
        
# #         Parameters:
# #         monthly_avg_data_list: List of monthly average data for each coordinate
# #         monthly_hourly_data_list: List of monthly hourly data for each coordinate
# #                                  Each item can be either:
# #                                  - List of 12 monthly arrays (preferred for memory efficiency)
# #                                  - Single array of 8760 hours
# #         shape_params_list: List of monthly shape parameters for each coordinate
# #         scale_params_list: List of monthly scale parameters for each coordinate
# #         """
# #         for i, (lat, lon) in enumerate(self.coordinates):
# #             if i < len(monthly_avg_data_list) and i < len(monthly_hourly_data_list) and \
# #                i < len(shape_params_list) and i < len(scale_params_list):
                
# #                 print(f"\n{'='*60}")
# #                 print(f"Processing coordinate {i+1}: ({lat}, {lon})")
# #                 print(f"{'='*60}")
                
# #                 self.analyze_coordinate(
# #                     i, 
# #                     monthly_avg_data_list[i], 
# #                     monthly_hourly_data_list[i],
# #                     shape_params_list[i], 
# #                     scale_params_list[i]
# #                 )
# #             else:
# #                 print(f"Warning: Missing data for coordinate {i+1}")

# #     def load_hourly_data_by_month(self, coordinate_idx):
# #         """
# #         Example function showing how to load hourly data month by month
# #         This is a template - replace with your actual data loading logic
        
# #         Returns:
# #         List of 12 monthly arrays containing hourly wind speed data
# #         """
# #         lat, lon = self.coordinates[coordinate_idx]
# #         monthly_data = []
        
# #         # Hours per month for 2019
# #         hours_per_month = [744, 672, 744, 720, 744, 720, 744, 744, 720, 744, 720, 744]
        
# #         print(f"Loading hourly data for coordinate {coordinate_idx+1}: ({lat}, {lon})")
        
# #         for month in range(1, 13):
# #             # Replace this with your actual data loading logic
# #             # Example: loading from files, databases, APIs, etc.
            
# #             print(f"  Loading {self.months[month-1]} data...")
            
# #             # Simulate loading monthly data (replace with your actual code)
# #             # Example options:
# #             # 1. Load from monthly files
# #             # month_data = pd.read_csv(f'data_coord_{coordinate_idx}_month_{month:02d}.csv')['wind_speed'].values
            
# #             # 2. Load from database query
# #             # month_data = query_database(lat, lon, year=2019, month=month)
            
# #             # 3. Generate example data (REPLACE THIS with your actual loading)
# #             hours_in_month = hours_per_month[month-1]
# #             month_data = np.random.weibull(2, hours_in_month) * 5 + 2  # Example data
            
# #             monthly_data.append(month_data)
# #             print(f"    Loaded {len(month_data)} hours for {self.months[month-1]}")
        
# #         return monthly_data

# # # Example usage
# # if __name__ == "__main__":
# #     # Define your 10 coordinates
# #     coordinates = [
# #         (12.9375, 123.8625),  # Coordinate 1
# #         (13.0000, 124.0000),  # Coordinate 2
# #         # Add your other 8 coordinates here
# #         # (lat3, lon3),
# #         # (lat4, lon4),
# #         # ... up to 10 coordinates
# #     ]
    
# #     # Initialize analyzer
# #     analyzer = WindDataAnalyzer(coordinates)
    
# #     # 1. List of monthly averages for each coordinate
# #     monthly_averages_dataset1 = [
# #         [5.2, 5.8, 6.1, 5.9, 5.3, 4.8, 4.2, 4.5, 4.9, 5.1, 5.4, 5.0],  # Coord 1
# #         [6.1, 6.5, 6.8, 6.2, 5.7, 5.1, 4.6, 4.9, 5.3, 5.6, 5.9, 5.5],  # Coord 2
# #         # Add for all 10 coordinates
# #     ]
    
# #     # 2. OPTION A: Load hourly data month by month (RECOMMENDED for memory efficiency)
# #     monthly_hourly_data_dataset2 = []
    
# #     for coord_idx in range(len(coordinates)):
# #         print(f"\nPreparing hourly data for coordinate {coord_idx+1}...")
        
# #         # Method 1: Load month by month (better memory management)
# #         coord_monthly_data = []
# #         for month in range(1, 13):
# #             # Replace this with your actual data loading logic
# #             # Examples:
# #             # - Load from monthly files: pd.read_csv(f'hourly_data_coord_{coord_idx}_month_{month:02d}.csv')
# #             # - Load from database: query_hourly_data(coordinates[coord_idx], month)
# #             # - Load from API: fetch_hourly_data(lat, lon, year=2019, month=month)
            
# #             # Hours per month in 2019
# #             hours_in_month = [744, 672, 744, 720, 744, 720, 744, 744, 720, 744, 720, 744][month-1]
            
# #             # Example data generation (REPLACE with your actual loading)
# #             month_hourly_data = np.random.weibull(2, hours_in_month) * 5 + 2
# #             coord_monthly_data.append(month_hourly_data)
            
# #             print(f"  Loaded {len(month_hourly_data)} hours for {analyzer.months[month-1]}")
        
# #         monthly_hourly_data_dataset2.append(coord_monthly_data)
    
# #     # 2. OPTION B: If you prefer to load all 8760 hours at once per coordinate
# #     # hourly_data_dataset2_alternative = [
# #     #     np.random.weibull(2, 8760) * 5 + 2,  # All hours for coord 1
# #     #     np.random.weibull(2, 8760) * 6 + 2,  # All hours for coord 2
# #     #     # Add for all 10 coordinates
# #     # ]
    
# #     # 4. List of monthly shape parameters for each coordinate
# #     shape_parameters = [
# #         [2.1, 2.3, 2.5, 2.2, 1.9, 1.7, 1.6, 1.8, 2.0, 2.2, 2.4, 2.0],  # Coord 1
# #         [2.2, 2.4, 2.6, 2.3, 2.0, 1.8, 1.7, 1.9, 2.1, 2.3, 2.5, 2.1],  # Coord 2
# #         # Add for all 10 coordinates
# #     ]
    
# #     # 5. List of monthly scale parameters for each coordinate
# #     scale_parameters = [
# #         [5.8, 6.5, 6.9, 6.6, 6.0, 5.4, 4.7, 5.1, 5.5, 5.7, 6.1, 5.6],  # Coord 1
# #         [6.9, 7.3, 7.7, 7.0, 6.4, 5.7, 5.2, 5.5, 6.0, 6.3, 6.6, 6.2],  # Coord 2
# #         # Add for all 10 coordinates
# #     ]
    
# #     # Optional: Inspect CSV structure for debugging
# #     # analyzer.inspect_csv_structure(12.9375, 123.8625)
    
# #     # Run analysis for all coordinates
# #     analyzer.analyze_all_coordinates(
# #         monthly_averages_dataset1,
# #         monthly_hourly_data_dataset2,  # Using month-by-month data
# #         shape_parameters,
# #         scale_parameters
# #     )
    
# #     # Alternative: Use the template function to load data systematically
# #     # monthly_hourly_alternative = []
# #     # for i in range(len(coordinates)):
# #     #     coord_data = analyzer.load_hourly_data_by_month(i)
# #     #     monthly_hourly_alternative.append(coord_data)
    
# #     # Or analyze a single coordinate
# #     # analyzer.analyze_coordinate(0, monthly_averages_dataset1[0], monthly_hourly_data_dataset2[0], 
# #     #                           shape_parameters[0], scale_parameters[0])

# import os
# from PIL import Image

# def compress_images_recursive(root_folder, quality=40, max_width=2000):
#     """
#     Compress all images in nested folders (replaces originals)
#     """
#     image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
#     processed_count = 0
    
#     for root, dirs, files in os.walk(root_folder):
#         for file in files:
#             if any(file.lower().endswith(ext) for ext in image_extensions):
#                 file_path = os.path.join(root, file)
                
#                 try:
#                     # Get original file size for comparison
#                     original_size = os.path.getsize(file_path)
                    
#                     # Open and compress image
#                     with Image.open(file_path) as img:
#                         # Resize if too large
#                         if img.width > max_width:
#                             ratio = max_width / img.width
#                             new_height = int(img.height * ratio)
#                             img = img.resize((max_width, new_height), Image.Resampling.LANCZOS)
                        
#                         # Save compressed version (replaces original)
#                         if file.lower().endswith('.png'):
#                             img.save(file_path, 'PNG', optimize=True)
#                         else:
#                             img.save(file_path, 'JPEG', quality=quality, optimize=True)
                        
#                         # Get new file size
#                         new_size = os.path.getsize(file_path)
#                         reduction = ((original_size - new_size) / original_size) * 100
                        
#                         processed_count += 1
#                         print(f"Compressed: {file_path}")
#                         print(f"  Size: {original_size/1024:.1f}KB → {new_size/1024:.1f}KB ({reduction:.1f}% reduction)")
                        
#                 except Exception as e:
#                     print(f"Error processing {file_path}: {e}")
    
#     print(f"\nProcessed {processed_count} images!")

# # Usage - put this script in your main folder and run
# compress_images_recursive("C:\\Users\\student\\dummywebGIS\\wind data plots per point", quality=40, max_width=2000)
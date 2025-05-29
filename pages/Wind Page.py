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
import time
import requests
import zipfile
import io
import tempfile
from datetime import datetime
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
from scipy.stats import weibull_min
import numpy as np
from scipy.special import gamma
from scipy.optimize import fsolve
# import seaborn as sns
from scipy import stats



import others.wind_algo as wind



# Add the parent directory to the system path so 'logic' can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# Set page config
st.set_page_config(layout="wide", page_title="Philippine Renewable Energy Potential Assessment")


st.markdown("""
    <h1 style='text-align: center;'>
        Wind Potential Assessment
    </h1>
""", unsafe_allow_html=True)

#st.markdown("Select a municipality, or draw a custom area (minimum 3km x 3km) to evaluate wind potential.")

# --- Static file paths --- 
municipality_github_url = "https://media.githubusercontent.com/media/aikasaurus/philippines-geodata/main/Municipalities.zip"
points_github_url = "https://media.githubusercontent.com/media/aikasaurus/philippines-geodata/main/NoConstraintsID.zip"

# Geometry simplification tolerance (adjust as needed for performance vs. detail)
simplify_tolerance = 0.001  # Roughly 100m at equator

# --- Fix for JSON serialization of timestamps ---
# Set style for better plots


class WindEnergyAnalyzer:
    def __init__(self):
        """
        Initialize the wind energy analyzer
        """
        self.months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        # Create output directory if it doesn't exist
        os.makedirs('output_plots', exist_ok=True)
        os.makedirs('output_text', exist_ok=True)
    
    def plot_monthly_windspeed_overlays(self, lat, lon, windspeed_list1, windspeed_list2, windspeed_list3,
                                       dataset_names=['RE Ninja', 'NSRDB', 'IRENA']):
        """
        Plot overlayed monthly wind speeds at 78m height
        """
        fig, ax = plt.subplots(figsize=(14, 8))
        
        x = np.arange(len(self.months))
        width = 0.25
        
        # Plot bars for each dataset
        bars1 = ax.bar(x - width, windspeed_list1[:12], width, 
                      label=f'{dataset_names[0]} (78m)', alpha=0.8, color='skyblue')
        bars2 = ax.bar(x, windspeed_list2[:12], width, 
                      label=f'{dataset_names[1]} (78m)', alpha=0.8, color='lightcoral')
        bars3 = ax.bar(x + width, windspeed_list3[:12], width, 
                      label=f'{dataset_names[2]} (78m)', alpha=0.8, color='lightgreen')
        
        # Customize plot
        ax.set_xlabel('Month', fontsize=12)
        ax.set_ylabel('Average Wind Speed (m/s)', fontsize=12)
        ax.set_title(f'Monthly Average Wind Speed at 78m Height for Point ({lat}, {lon})', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(self.months)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.1f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('output_plots/monthly_windspeed_78m_overlay.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✓ Saved: monthly_windspeed_78m_overlay.png")

    
    def plot_wind_speed_boxplots(self, wind_data_list, dataset_names=None, 
                                title="Wind Speed Distribution Box Plots"):
        """
        Plot box and whisker plots for wind speed comparison
        
        Parameters:
        wind_data_list: List of wind speed datasets (hourly data)
        dataset_names: List of names for each dataset
        title: Plot title
        """
        if dataset_names is None:
            dataset_names = [f'Dataset {i+1}' for i in range(len(wind_data_list))]
        
        # Prepare data for box plotting
        clean_data_list = []
        valid_names = []
        
        for i, (data, name) in enumerate(zip(wind_data_list, dataset_names)):
            # Handle different data structures
            wind_speeds = []
            
            if isinstance(data, list):
                # Check if it's a list of monthly arrays
                if len(data) == 12 and all(isinstance(month_data, (list, np.ndarray)) for month_data in data):
                    # It's monthly organized data - flatten all months
                    for month_data in data:
                        if isinstance(month_data, (list, np.ndarray)) and len(month_data) > 0:
                            wind_speeds.extend(np.array(month_data).flatten())
                else:
                    # It's a regular list of numbers
                    wind_speeds = np.array(data).flatten()
            elif isinstance(data, np.ndarray):
                wind_speeds = data.flatten()
            else:
                print(f"Warning: Unrecognized data format for {name}")
                continue
            
            # Convert to numpy array and clean
            wind_speeds = np.array(wind_speeds)
            wind_speeds = wind_speeds[~np.isnan(wind_speeds)]
            wind_speeds = wind_speeds[wind_speeds >= 0]  # Remove negative values
            
            if len(wind_speeds) > 0:
                clean_data_list.append(wind_speeds)
                valid_names.append(name)
                print(f"Dataset {name}: {len(wind_speeds)} total hours of data")
                
                # Print statistics
                print(f"  {name} stats:")
                print(f"    Mean: {np.mean(wind_speeds):.2f} m/s")
                print(f"    Median: {np.median(wind_speeds):.2f} m/s")
                print(f"    Q25: {np.percentile(wind_speeds, 25):.2f} m/s")
                print(f"    Q75: {np.percentile(wind_speeds, 75):.2f} m/s")
                print(f"    Min: {np.min(wind_speeds):.2f} m/s")
                print(f"    Max: {np.max(wind_speeds):.2f} m/s")
                print(f"    Zero values: {np.sum(wind_speeds == 0)}")
                print(f"    Values < 1 m/s: {np.sum(wind_speeds < 1)}")
                print(f"    Values > 10 m/s: {np.sum(wind_speeds > 10)}")
        
        if len(clean_data_list) == 0:
            print("No valid data found for box plotting!")
            return
        
        # Create the box plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create box plot
        box_plot = ax.boxplot(clean_data_list, 
                             labels=valid_names,
                             patch_artist=True,  # Enable fill colors
                             showmeans=True,     # Show mean markers
                             meanline=False,     # Show mean as point, not line
                             showfliers=True,    # Show outliers
                             whis=1.5)          # Whisker length (1.5 * IQR)
        
        # Customize box colors
        colors = ['lightblue', 'lightcoral', 'lightgreen', 'gold', 'mediumpurple', 'orange']
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Customize other elements
        for element in ['whiskers', 'fliers', 'medians', 'caps']:
            plt.setp(box_plot[element], color='black')
        
        # Customize means
        plt.setp(box_plot['means'], marker='D', markerfacecolor='red', 
                markeredgecolor='darkred', markersize=6)
        
        # Set labels and title
        ax.set_xlabel('Dataset', fontsize=12)
        ax.set_ylabel('Wind Speed (m/s)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add legend explaining box plot elements
        legend_elements = [
            plt.Line2D([0], [0], color='black', linewidth=2, label='Median'),
            plt.Line2D([0], [0], marker='D', color='red', linewidth=0, 
                      markersize=6, label='Mean'),
            plt.Rectangle((0, 0), 1, 1, facecolor='lightblue', alpha=0.7, 
                         edgecolor='black', label='IQR (25th-75th percentile)'),
            plt.Line2D([0], [0], color='black', linewidth=1, label='Whiskers (1.5×IQR)'),
            plt.Line2D([0], [0], marker='o', color='black', linewidth=0, 
                      markersize=4, label='Outliers')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        # Rotate x-axis labels if needed
        if len(valid_names) > 2:
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('output_plots/wind_speed_boxplots.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✓ Saved: wind_speed_boxplots.png")
    
    def save_friction_coefficients_to_text(self, lat, lon, friction_list1, friction_list2,
                                          dataset_names=['NSRDB', 'IRENA']):
        """
        Save friction coefficients to organized text file
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open('output_text/friction_coefficients.txt', 'w') as f:
            f.write("="*60 + "\n")
            f.write(f"MONTHLY FRICTION COEFFICIENTS for Point ({lat}, {lon})\n")
            f.write("="*60 + "\n")
            f.write(f"Generated on: {timestamp}\n\n")
            
            # Header
            f.write(f"{'Month':<10} {dataset_names[0]:<15} {dataset_names[1]:<15}\n")
            f.write("-" * 50 + "\n")
            
            # Data rows
            for i, month in enumerate(self.months):
                val1 = friction_list1[i] if i < len(friction_list1) else 'N/A'
                val2 = friction_list2[i] if i < len(friction_list2) else 'N/A'
                f.write(f"{month:<10} {val1:<15.4f} {val2:<15.4f}\n")
            
            # Statistics
            f.write("\n" + "="*50 + "\n")
            f.write("STATISTICS\n")
            f.write("="*50 + "\n")
            f.write(f"{dataset_names[0]} - Mean: {np.mean(friction_list1[:12]):.4f}, "
                   f"Std: {np.std(friction_list1[:12]):.4f}\n")
            f.write(f"{dataset_names[1]} - Mean: {np.mean(friction_list2[:12]):.4f}, "
                   f"Std: {np.std(friction_list2[:12]):.4f}\n")
        
        print("✓ Saved: friction_coefficients.txt")
    
    def save_weibull_parameters_to_text(self, lat, lon, shape_list1, shape_list2, shape_list3,
                                       scale_list1, scale_list2, scale_list3,
                                       dataset_names=['NSRDB', 'IRENA', 'RE Ninja']):
        """
        Save Weibull shape and scale parameters to organized text file
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open('output_text/weibull_parameters.txt', 'w') as f:
            f.write("="*80 + "\n")
            f.write(f"MONTHLY WEIBULL DISTRIBUTION PARAMETERS for Point ({lat}, {lon})\n")
            f.write("="*80 + "\n")
            f.write(f"Generated on: {timestamp}\n\n")
            
            # Shape Parameters Section
            f.write("SHAPE PARAMETERS (k)\n")
            f.write("-" * 70 + "\n")
            f.write(f"{'Month':<10} {dataset_names[0]:<15} {dataset_names[1]:<15} {dataset_names[2]:<15}\n")
            f.write("-" * 70 + "\n")
            
            for i, month in enumerate(self.months):
                k1 = shape_list1[i] if i < len(shape_list1) else 'N/A'
                k2 = shape_list2[i] if i < len(shape_list2) else 'N/A'
                k3 = shape_list3[i] if i < len(shape_list3) else 'N/A'
                f.write(f"{month:<10} {k1:<15.4f} {k2:<15.4f} {k3:<15.4f}\n")
            
            f.write("\n" + "="*70 + "\n")
            
            # Scale Parameters Section
            f.write("SCALE PARAMETERS (c)\n")
            f.write("-" * 70 + "\n")
            f.write(f"{'Month':<10} {dataset_names[0]:<15} {dataset_names[1]:<15} {dataset_names[2]:<15}\n")
            f.write("-" * 70 + "\n")
            
            for i, month in enumerate(self.months):
                l1 = scale_list1[i] if i < len(scale_list1) else 'N/A'
                l2 = scale_list2[i] if i < len(scale_list2) else 'N/A'
                l3 = scale_list3[i] if i < len(scale_list3) else 'N/A'
                f.write(f"{month:<10} {l1:<15.4f} {l2:<15.4f} {l3:<15.4f}\n")
            
            # Statistics
            f.write("\n" + "="*70 + "\n")
            f.write("STATISTICS\n")
            f.write("="*70 + "\n")
            f.write("Shape Parameters:\n")
            for j, name in enumerate(dataset_names):
                shape_data = [shape_list1, shape_list2, shape_list3][j]
                f.write(f"  {name} - Mean: {np.mean(shape_data[:12]):.4f}, "
                       f"Std: {np.std(shape_data[:12]):.4f}\n")
            
            f.write("\nScale Parameters:\n")
            for j, name in enumerate(dataset_names):
                scale_data = [scale_list1, scale_list2, scale_list3][j]
                f.write(f"  {name} - Mean: {np.mean(scale_data[:12]):.4f}, "
                       f"Std: {np.std(scale_data[:12]):.4f}\n")
        
        print("✓ Saved: weibull_parameters.txt")
    
    def plot_monthly_energy_yield_overlays(self, lat, lon, energy_list1, energy_list2, energy_list3,
                                          dataset_names=['NSRDB', 'IRENA', 'RE Ninja']):
        """
        Plot overlayed monthly energy yields
        """
        fig, ax = plt.subplots(figsize=(14, 8))
        
        x = np.arange(len(self.months))
        width = 0.25
        
        # Plot bars for each dataset
        bars1 = ax.bar(x - width, energy_list1[:12], width, 
                      label=f'{dataset_names[0]}', alpha=0.8, color='gold')
        bars2 = ax.bar(x, energy_list2[:12], width, 
                      label=f'{dataset_names[1]}', alpha=0.8, color='orange')
        bars3 = ax.bar(x + width, energy_list3[:12], width, 
                      label=f'{dataset_names[2]}', alpha=0.8, color='darkorange')
        
        # Customize plot
        ax.set_xlabel('Month', fontsize=12)
        ax.set_ylabel('Energy Yield (kWh)', fontsize=12)
        ax.set_title(f'Monthly Energy Yield Comparison for Point ({lat}, {lon})', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(self.months)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.0f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=9, rotation=90)
        
        plt.tight_layout()
        plt.savefig('output_plots/monthly_energy_yield_overlay.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✓ Saved: monthly_energy_yield_overlay.png")

    def plot_monthly_wind_speed_boxplots(self, wind_data_list, dataset_names=None, 
                                        title="Monthly Wind Speed Distribution Box Plots"):
        """
        Plot monthly box and whisker plots showing seasonal patterns
        
        Parameters:
        wind_data_list: List of monthly organized wind speed datasets 
                       Each should be [jan_data, feb_data, ..., dec_data]
        dataset_names: List of names for each dataset
        title: Plot title
        """
        if dataset_names is None:
            dataset_names = [f'Dataset {i+1}' for i in range(len(wind_data_list))]
        
        # Number of datasets
        n_datasets = len(wind_data_list)
        
        # Create subplots - one for each dataset
        if n_datasets == 1:
            fig, axes = plt.subplots(1, 1, figsize=(14, 8))
            axes = [axes]
        elif n_datasets == 2:
            fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        elif n_datasets == 3:
            fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        else:
            # For more than 3, use 2 rows
            cols = (n_datasets + 1) // 2
            fig, axes = plt.subplots(2, cols, figsize=(6*cols, 12))
            axes = axes.flatten()
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Colors for box plots
        box_colors = ['lightblue', 'lightcoral', 'lightgreen', 'gold', 'mediumpurple', 'orange']
        
        for dataset_idx, (data, name) in enumerate(zip(wind_data_list, dataset_names)):
            ax = axes[dataset_idx]
            
            # Prepare monthly data
            monthly_data = []
            valid_months = []
            
            # Check if data is monthly organized
            if isinstance(data, list) and len(data) == 12:
                for month_idx, month_data in enumerate(data):
                    if isinstance(month_data, (list, np.ndarray)) and len(month_data) > 0:
                        # Clean the data
                        wind_speeds = np.array(month_data).flatten()
                        wind_speeds = wind_speeds[~np.isnan(wind_speeds)]
                        wind_speeds = wind_speeds[wind_speeds >= 0]
                        
                        if len(wind_speeds) > 0:
                            monthly_data.append(wind_speeds)
                            valid_months.append(self.months[month_idx])
                        else:
                            print(f"Warning: No valid data for {name} - {self.months[month_idx]}")
            else:
                print(f"Warning: {name} data is not in monthly format. Skipping monthly box plot.")
                ax.text(0.5, 0.5, f'{name}\nData not in monthly format', 
                       transform=ax.transAxes, ha='center', va='center', fontsize=12)
                ax.set_title(name)
                continue
            
            if len(monthly_data) == 0:
                ax.text(0.5, 0.5, f'{name}\nNo valid monthly data', 
                       transform=ax.transAxes, ha='center', va='center', fontsize=12)
                ax.set_title(name)
                continue
            
            # Create box plot for this dataset
            box_plot = ax.boxplot(monthly_data,
                                 labels=valid_months,
                                 patch_artist=True,
                                 showmeans=True,
                                 meanline=False,
                                 showfliers=True,
                                 whis=1.5)
            
            # Customize box colors (same color for all months of one dataset)
            color = box_colors[dataset_idx % len(box_colors)]
            for patch in box_plot['boxes']:
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            # Customize other elements
            for element in ['whiskers', 'fliers', 'medians', 'caps']:
                plt.setp(box_plot[element], color='black')
            
            # Customize means
            plt.setp(box_plot['means'], marker='D', markerfacecolor='red',
                    markeredgecolor='darkred', markersize=5)
            
            # Set labels and title
            ax.set_xlabel('Month', fontsize=11)
            ax.set_ylabel('Wind Speed (m/s)', fontsize=11)
            ax.set_title(f'{name} - Monthly Distribution', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Rotate month labels
            ax.tick_params(axis='x', rotation=45)
            
            # Print monthly statistics
            print(f"\n{name} Monthly Statistics:")
            for i, (month, month_data_array) in enumerate(zip(valid_months, monthly_data)):
                mean_val = np.mean(month_data_array)
                median_val = np.median(month_data_array)
                print(f"  {month}: Mean={mean_val:.2f}, Median={median_val:.2f}, "
                      f"Count={len(month_data_array)}, Zeros={np.sum(month_data_array == 0)}")
        
        # Hide unused subplots
        for j in range(n_datasets, len(axes)):
            axes[j].set_visible(False)
        
        # Add overall legend
        legend_elements = [
            plt.Line2D([0], [0], color='black', linewidth=2, label='Median'),
            plt.Line2D([0], [0], marker='D', color='red', linewidth=0, 
                      markersize=5, label='Mean'),
            plt.Rectangle((0, 0), 1, 1, facecolor='lightblue', alpha=0.7, 
                         edgecolor='black', label='IQR (25th-75th percentile)'),
            plt.Line2D([0], [0], color='black', linewidth=1, label='Whiskers (1.5×IQR)'),
            plt.Line2D([0], [0], marker='o', color='black', linewidth=0, 
                      markersize=3, label='Outliers')
        ]
        
        # Place legend outside the plot area
        fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.02), 
                  ncol=5, fontsize=10)
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1)  # Make room for legend
        plt.savefig('output_plots/monthly_wind_speed_boxplots.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✓ Saved: monthly_wind_speed_boxplots.png")
    
    def plot_monthly_capacity_factors_overlays(self, lat, lon, cf_list1, cf_list2,
                                              dataset_names=['NSRDB', 'IRENA']):
        """
        Plot overlayed monthly capacity factors
        """
        fig, ax = plt.subplots(figsize=(14, 8))
        
        x = np.arange(len(self.months))
        width = 0.35
        
        # Plot bars for each dataset
        bars1 = ax.bar(x - width/2, cf_list1[:12], width, 
                      label=f'{dataset_names[0]}', alpha=0.8, color='mediumpurple')
        bars2 = ax.bar(x + width/2, cf_list2[:12], width, 
                      label=f'{dataset_names[1]}', alpha=0.8, color='mediumorchid')
        
        # Customize plot
        ax.set_xlabel('Month', fontsize=12)
        ax.set_ylabel('Capacity Factor (%)', fontsize=12)
        ax.set_title(f'Monthly Capacity Factor Comparison for Point ({lat}, {lon})', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(self.months)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.1f}%',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('output_plots/monthly_capacity_factor_overlay.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✓ Saved: monthly_capacity_factor_overlay.png")
    
    def plot_all_weibull_distributions(self, nsrdb_shape, nsrdb_scale, irena_shape, irena_scale, 
                                      ninja_shape, ninja_scale, lat, lon):
        """
        Plot monthly Weibull distributions for all three datasets (NSRDB, IRENA, Renewables.ninja)
        """
        # Add this debug print in the plot_all_weibull_distributions function
        print("DEBUG - First month values:")
        print(f"NSRDB: shape={nsrdb_shape[0]}, scale={nsrdb_scale[0]}")
        print(f"IRENA: shape={irena_shape[0]}, scale={irena_scale[0]}")  
        print(f"Ninja: shape={ninja_shape[0]}, scale={ninja_scale[0]}")

        # Create subplots for each month
        fig, axes = plt.subplots(3, 4, figsize=(18, 14))
        fig.suptitle(f'Monthly Weibull Distributions for Point ({lat}, {lon})', 
                    fontsize=16, fontweight='bold')
        
        # Wind speed range for plotting
        x_wind = np.linspace(0, 25, 1000)
        
        # Colors for each dataset
        colors = ['blue', 'red', 'green']
        labels = ['NSRDB', 'IRENA', 'Renewables.ninja']
        
        for i, (ax, month) in enumerate(zip(axes.flat, self.months)):
            if i < 12:
                ax.set_title(f'{month}')
                ax.set_xlabel('Wind Speed (m/s)')
                ax.set_ylabel('Probability Density')
                ax.grid(True, alpha=0.3)
                ax.set_xlim(0, 20)
                
                # Plot all three distributions
                datasets = [
                    (nsrdb_shape, nsrdb_scale, colors[0], labels[0]),
                    (irena_shape, irena_scale, colors[1], labels[1]),
                    (ninja_shape, ninja_scale, colors[2], labels[2])
                ]
                
                for shape_list, scale_list, color, label in datasets:
                    if i < len(shape_list) and i < len(scale_list):
                        try:
                            shape = float(shape_list[i])
                            scale = float(scale_list[i])
                            
                            if shape > 0 and scale > 0:
                                # Calculate Weibull PDF
                                pdf = stats.weibull_min.pdf(x_wind, shape, loc=0, scale=scale)
                                
                                # Plot the distribution
                                ax.plot(x_wind, pdf, linewidth=2, color=color, alpha=0.8,
                                       label=f'{label} (k={shape:.2f}, λ={scale:.2f})')
                                
                                # Calculate and show mean
                                try:
                                    mean_wind = scale * gamma(1 + 1/shape)
                                    ax.axvline(mean_wind, color=color, linestyle='--', alpha=0.5)
                                except:
                                    pass
                        except:
                            # Skip invalid data
                            continue
                
                # Add legend
                ax.legend(fontsize=7, loc='upper right')
            else:
                ax.set_visible(False)
        
        plt.tight_layout()
        plt.savefig('output_plots/weibull_distributions_all_datasets.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✓ Saved: weibull_distributions_all_datasets.png")

def get_monthly_energy_yield_from_csv(filename):
    """
    Extract monthly energy yield data from ninja CSV file
    Column C contains electricity production data starting from C5
    
    Returns:
    - monthly_energy_yield: List of 12 monthly total energy yields
    """
    # Load CSV starting from row 5 (skiprows=4)
    df = pd.read_csv(filename, skiprows=4)
    
    # Extract timestamp (column B = index 1) and electricity production (column C = index 2)
    timestamps = pd.to_datetime(df.iloc[:, 1], errors='coerce')
    electricity_production = pd.to_numeric(df.iloc[:, 2], errors='coerce')  # Column C
    
    # Create clean dataframe
    data = pd.DataFrame({
        'timestamp': timestamps,
        'electricity': electricity_production
    }).dropna()
    
    # Add month column
    data['month'] = data['timestamp'].dt.month
    
    # Calculate monthly energy yield (sum of electricity production per month)
    monthly_energy_yield = []
    
    for month in range(1, 13):
        month_data = data[data['month'] == month]['electricity'].values
        if month == 2:
            print("GALING CSV")
            print(month_data)
        
        if len(month_data) > 0:
            monthly_total = month_data.sum()
            monthly_energy_yield.append(monthly_total)
            print(f"Month {month}: {len(month_data)} hours, total energy = {monthly_total:.2f}")
        else:
            monthly_energy_yield.append(0)
            print(f"Month {month}: No data")
    print("eto sum")
    print(monthly_energy_yield)
    
    return monthly_energy_yield

def get_monthly_wind_speeds_from_csv(filename):
    """
    Extract monthly wind speed data from ninja CSV file
    
    Returns:
    - monthly_averages: List of 12 monthly average wind speeds
    - monthly_hourly_data: List of 12 arrays containing hourly data per month
    """
    # Load CSV starting from row 5 (skiprows=4)
    df = pd.read_csv(filename, skiprows=4)
    
    # Extract timestamp (column B = index 1) and wind speed (column D = index 3)
    timestamps = pd.to_datetime(df.iloc[:, 1], errors='coerce')
    wind_speeds = pd.to_numeric(df.iloc[:, 3], errors='coerce')
    
    # Create clean dataframe
    data = pd.DataFrame({
        'timestamp': timestamps,
        'wind_speed': wind_speeds
    }).dropna()
    
    # Add month column
    data['month'] = data['timestamp'].dt.month
    
    # Calculate monthly averages
    monthly_averages = []
    monthly_hourly_data = []
    
    for month in range(1, 13):
        month_data = data[data['month'] == month]['wind_speed'].values
        
        if len(month_data) > 0:
            monthly_averages.append(month_data.mean())
            monthly_hourly_data.append(month_data)
            print(f"Month {month}: {len(month_data)} hours, avg = {month_data.mean():.2f} m/s")
        else:
            monthly_averages.append(0)
            monthly_hourly_data.append(np.array([]))
            print(f"Month {month}: No data")
    
    return monthly_averages, monthly_hourly_data

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

# st.success(f"Successfully loaded {len(gdf_municipalities)} municipalities and {len(gdf_points)} points!")

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

        # let user choose turbine model first
        wind_turbine_options = [f"1. model: {wind.V126.model_name}, hub height: {wind.V126.hub_height}, rotor diameter: {wind.V126.rotor_diameter}", f"2. model: {wind.V150.model_name}, hub height: {wind.V150.hub_height}, rotor diameter: {wind.V150.rotor_diameter}", f"3. model: {wind.V165.model_name}, hub height: {wind.V165.hub_height}, rotor diameter: {wind.V165.rotor_diameter}" ]
        selected_turbine = st.selectbox("Choose a wind turbine model:", wind_turbine_options)
        turbine_model = wind.turbines[int(selected_turbine[0])]

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
        slope_checker = st.checkbox("Do you wish to apply slope limits?")
        if slope_checker:
            slope = st.number_input("Enter a value (in %) to filter out slope of the land", min_value=None, max_value=None, value=20.00, step= 0.01, format ="%.4f")


        


        # Center-align the metrics with custom markdown and HTML
        # with col_metric1:
        #     st.markdown(f"""
        #     <div style="text-align: center;">
        #         <p style="font-size: 14px; color: gray;">IRENA</p>
        #         <p style="font-size: 28px; font-weight: bold;">{len(gdf_municipalities)}</p>
        #     </div>
        #     """, unsafe_allow_html=True)
        
        # with col_metric2:
        #     st.markdown(f"""
        #     <div style="text-align: center;">
        #         <p style="font-size: 14px; color: gray;">NSRDB</p>
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
    
    # Get current zoom level from map if available
            current_zoom = 6  # Default zoom level
            if hasattr(selection_map, 'get_zoom'):
                try:
                    current_zoom = selection_map.get_zoom()
                except:
                    pass
    
            # Adjust search radius based on zoom level
            # Lower zoom = larger search radius
            search_radius = max(0.05, 0.5 / (current_zoom or 1))  # Prevents division by zero
            
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

    col_metric1, col_metric2 = st.columns([1,2])

    # with col_metric1:
    #     st.markdown(f"""
    #         <style>
    #         .tooltip {{
    #             position: relative;
    #             display: inline-block;
    #             cursor: pointer;
    #         }}
    #         .tooltip .tooltiptext {{
    #             visibility: hidden;
    #             width: 220px;
    #             background-color: #555;
    #             color: #fff;
    #             text-align: left;
    #             border-radius: 6px;
    #             padding: 5px;
    #             position: absolute;
    #             z-index: 1;
    #             bottom: 125%;
    #             left: 50%;
    #             margin-left: -90px;
    #             opacity: 0;
    #             transition: opacity 0.3s;
    #             font-size: 14px;
    #         }}
    #         .tooltip:hover .tooltiptext {{
    #             visibility: visible;
    #             opacity: 1;
    #         }}
    #         .container {{
    #             display: flex;
    #             justify-content: space-around;
    #             align-items: center;
    #         }}
    #         .item {{
    #             font-size: 20px;
    #             color: black;
    #         }}
    #         </style>

    #         <div class="container">
    #             <div class="item">
    #                 <p>
    #                     IRENA
    #                     <span class="tooltip">ⓘ 
    #                         <span class="tooltiptext"> Long-term monthly averaged wind speed value from 2000 to 2020 from International Renewable Energy Agency (IRENA) . </span>
    #                     </span>
    #                 </p>
    #             </div>

    #             <div class="item">
    #                 <p>
    #                     NSRDB
    #                     <span class="tooltip">ⓘ 
    #                         <span class="tooltiptext"> Hourly wind speed values for 2017 from National Solar Radiation Database.</span>
    #                     </span>
    #                 </p>
    #             </div>
    #         </div>
    #         """, unsafe_allow_html=True)

    disp1, disp2 = st.columns([1,1])
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
        folium_static(m)

    # # --- Table of points ---
    # st.subheader(f"📍 {len(intersecting)} Points Inside {st.session_state.selected_muni}")
    # if not intersecting.empty:
    #     display_cols = [col for col in intersecting.columns if col != 'geometry']
    #     st.dataframe(intersecting[display_cols])
    # else:
    #     st.info("No points found within this municipality.")

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
    
        # col_metric1, col_metric2 = st.columns(2)

        # let user choose turbine model first
        wind_turbine_options = [f"1. model: {wind.V126.model_name}, hub height: {wind.V126.hub_height}, rotor diameter: {wind.V126.rotor_diameter}", f"2. model: {wind.V150.model_name}, hub height: {wind.V150.hub_height}, rotor diameter: {wind.V150.rotor_diameter}", f"3. model: {wind.V165.model_name}, hub height: {wind.V165.hub_height}, rotor diameter: {wind.V165.rotor_diameter}" ]
        selected_turbine = st.selectbox("Choose a wind turbine model:", wind_turbine_options)
        turbine_model = wind.turbines[int(selected_turbine[0])]

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
        slope_checker = st.checkbox("Do you wish to apply slope limits?")
        if slope_checker:
            slope = st.number_input("Enter a value (in %) to filter out slope of the land", min_value=None, max_value=None, value=20.00, step= 0.01, format ="%.4f")



    
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
        
        disp1, disp2 = st.columns([1,1])
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


# handle algo here

# filter out invalid points (those that are exclusion areas based on user's constraint selection)

with col1:
    #rearrange and then round off to match with SQL database
    temp_points =[]
    if st.session_state.selection_mode != "municipality":
        coordinate_tuples = custom_coords

    for tup in coordinate_tuples:
        new_tup = (round(tup[1],6), round(tup[0],6))
        temp_points.append(new_tup)


    if choose_from:

        valid_points = wind.look_up_points(temp_points, choose_from)

        # st.write(f"all of the points: {temp_points}")
        # st.write(f"filtered points: {valid_points}")

    else:
        # st.write(f"filtered points: No Constraint Selected. ")
        valid_points = temp_points

    
    if slope_checker:
        valid_points = wind.filter_slope(valid_points, slope)


        if len(valid_points) == 0:
            st.info("No results match your current selection criteria. ")

            st.markdown("""
        You might want to:
        - Try adjusting your slope parameters
        - Select a different geographic area
        - Expand your search criteria
        """)

    #DATA VALIDATION PURPOSES


    #testing

    #debug slope
    # st.write(f"all of the points: {temp_points}")
    
    # st.write(f"testing remaining points: {valid_points}")

    

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

    # st.write(f"this is valid_points in block {valid_po


col_metric1, col_metric2 = st.columns(2)


with col_metric1:
    
    start_time = time.perf_counter()

    #DATA VALIDATION PURPOSES SINGLE POINTTT
    # valid_points = [(122.887502, 11.087500)]

    # # Usage example:
    # filename = "C:\\Users\\student\\Downloads\\ninja_wind_11.0757_122.8906_uncorrected.csv"
    # re_ninja_monthly_ws_avgs, re_ninja_monthly_hourly = get_monthly_wind_speeds_from_csv(filename)
    # re_ninja_monthly_energy = get_monthly_energy_yield_from_csv(filename)
    # print("eto monthly eergy yield ng csv")
    # print(re_ninja_monthly_energy)

    # nsrdb_monthly_ws_ave = []
    # irena_monthly_ws_ave = []

    # nsrdb_monthly_alpha_ave = []
    # irena_monthly_alpha_ave = []

    # nsrdb_monthly_shape = []
    # irena_monthly_shape = []
    # re_ninja_monthly_shape = []

    # nsrdb_monthly_scale = []
    # irena_monthly_scale = []
    # re_ninja_monthly_scale = []

# monthly_avgs will be a list of 12 monthly averages
# monthly_hourly will be a list of 12 arrays with hourly data per month

    NSRDB_monthly_energy_yield = [0,0,0,0,0,0,0,0,0,0,0,0]
    IRENA_monthly_energy_yield = []
    NSRDB_monthly_cf = []
    IRENA_monthly_cf = []
    capacity = 0
    lcoe = 0
    nsrdb_stdev = []
    nsrdb_conv_ws = []
    irena_conv_ws = []

    #sample point from Maria Aurora for Data Validation

    # #fetch irena again for the valid points
    irena_ds20, irena_ds60, irena_points = wind.db_fetch_IRENA(valid_points, municipality = None)


    # print(f"irena_ds20: {irena_

    # #fetch NSRDB data
    nsrdb_monthly40WS = []
    nsrdb_monthly60WS = []
    for point in valid_points:
        for month in range(1, 13):
            monthly_40ws_data = wind.fetch_monthly_per_point(month, point[1], point[0], height = 40 )
            monthly_60ws_data = wind.fetch_monthly_per_point(month, point[1], point[0], height = 60 )

            nsrdb_monthly40WS.append(monthly_40ws_data)
            nsrdb_monthly60WS.append(monthly_60ws_data)

        # process per month
        for month in range(1, 13):

            month_40ws = nsrdb_monthly40WS[month - 1]
            month_60ws = nsrdb_monthly60WS[month - 1]
            hours = len(month_40ws)

            e_yield1, std, nsrdb_alpha, nsrdb_extrapolated, nsrdb_shape, nsrdb_scale, nsrdb_extrap_list = wind.calc_energy_yield_discrete(hours, (1/3)*len(valid_points), 40, month_40ws, 60, month_60ws, turbine_model )
            NSRDB_monthly_energy_yield[month-1] += e_yield1
            # nsrdb_monthly_ws_ave.append(nsrdb_extrapolated)
            # nsrdb_monthly_shape.append(nsrdb_shape)
            # nsrdb_monthly_scale.append(nsrdb_scale)
            # nsrdb_monthly_alpha_ave.append(nsrdb_alpha)
            nsrdb_stdev.append(std)
            nsrdb_conv_ws.append(nsrdb_extrap_list)

    for month in range(1, 13):

        month_point_yield = [] #for IRENA

        for point in range(len(irena_ds20)):

            e_yield2, irena_alpha, irena_extrapolated, irena_shape, irena_scale, irena_extrap_list = wind.calc_energy_yield_discrete(hours, (1/3)* len(valid_points), 20, [irena_ds20[point][month -1]], 60, [irena_ds60[point][month -1]], turbine_model, nsrdb_stdev= nsrdb_stdev[month-1], source= "IRENA" )
            # print(f"this is e_yield2: {e_yield2}")
            # irena_monthly_ws_ave.append(irena_extrapolated)
            # irena_monthly_shape.append(irena_shape)
            # irena_monthly_scale.append(irena_scale)
            # irena_monthly_alpha_ave.append(irena_alpha)

            month_point_yield.append(abs(e_yield2))
        
            irena_conv_ws.append(irena_extrap_list)

        total = sum(month_point_yield)
        IRENA_monthly_energy_yield.append(total)

    capacity = wind.compute_capacity(turbine_model, (1/3)*len(irena_ds20))

    month_cntr = 0
    irena_lcoe = []
    nsrdb_lcoe = []


    for e_yield in NSRDB_monthly_energy_yield:
        month_40ws = nsrdb_monthly40WS[month_cntr]  # ← Add this line
        # month_60ws = nsrdb_monthly60WS[month_cntr]
        hours = len(month_40ws)                     # ← Now this is correct
        nsrdb_cf = wind.compute_monthly_capacity_factor(e_yield, hours, capacity)
        NSRDB_monthly_cf.append(nsrdb_cf)
        nsrdb_lcoe.append(wind.compute_lcoe(nsrdb_cf))
        month_cntr += 1

    month_cntr = 0

    for e_yield in IRENA_monthly_energy_yield:
        month_40ws = nsrdb_monthly60WS[month_cntr]
        hours = len(month_40ws)
        irena_cf = wind.compute_monthly_capacity_factor(e_yield, hours, capacity)
        irena_lcoe.append(wind.compute_lcoe(irena_cf))
        IRENA_monthly_cf.append(irena_cf)
        month_cntr += 1

ave_yield_irena = (sum(IRENA_monthly_energy_yield)/12)/1000
ave_lcoe_irena = wind.compute_lcoe(sum(IRENA_monthly_cf)/12)
ave_yield_nsrdb = (sum(NSRDB_monthly_energy_yield)/12)/1000
ave_lcoe_nsrdb = wind.compute_lcoe(sum(NSRDB_monthly_cf)/12)

#re ninja

# ave_yield_ninja = (sum(re_ninja_monthly_energy)/12)

# re_ninja_monthly_shape =[]
# re_ninja_monthly_scale = []
# re_ninja_monthly_std = []

# for month in range(1,13):
#     k, c, stdev = wind.get_weibull_params(re_ninja_monthly_hourly[month - 1])
#     re_ninja_monthly_shape.append(k)
#     re_ninja_monthly_scale.append(c)
#     re_ninja_monthly_std.append(stdev)


ave_cf_nsrdb = sum(NSRDB_monthly_cf)/12
ave_cf_irena = sum(IRENA_monthly_cf)/12

st.write(f"LCOE_IRENA: {ave_lcoe_irena}")
st.write(f"LCOE_NSRDB: {ave_lcoe_nsrdb}")



# #PLOT everything here
# # plt.style.use('seaborn-v0_8')
# # sns.set_palette("husl")

# analyzer = WindEnergyAnalyzer()

# # =================================================================
# # INPUT YOUR DATA HERE - Replace these example values with your actual data
# # =================================================================

# # 1. Three lists of monthly average wind speeds at 78m height
# windspeed_78m_1 = re_ninja_monthly_ws_avgs  # Dataset 1
# windspeed_78m_2 = nsrdb_monthly_ws_ave # Dataset 2
# windspeed_78m_3 = irena_monthly_ws_ave # Dataset 3

# # 2. Two lists of monthly friction coefficients
# friction_coeff_1 = nsrdb_monthly_alpha_ave
# friction_coeff_2 = irena_monthly_alpha_ave

# # 3. Three lists each of monthly shape and scale parameter
# nsrdb_shape_0 = nsrdb_monthly_shape  # Shape list 0
# nsrdb_shape_1 = irena_monthly_shape  # Shape list 1
# nsrdb_shape_2 = re_ninja_monthly_shape  # Shape list 2

# nsrdb_scale_0 = nsrdb_monthly_scale  # Scale list 0
# nsrdb_scale_1 = irena_monthly_scale  # Scale list 1
# nsrdb_scale_2 = re_ninja_monthly_scale  # Scale list 2

# # 4. Three lists of monthly energy yield
# energy_yield_1 = NSRDB_monthly_energy_yield
# energy_yield_2 = IRENA_monthly_energy_yield
# energy_yield_3 = re_ninja_monthly_energy

# # 4. Three lists of monthly energy yield
# print(NSRDB_monthly_energy_yield)
# print(IRENA_monthly_energy_yield)
# print(re_ninja_monthly_energy)

# # 5. Two lists of monthly capacity factors
# capacity_factor_1 =  NSRDB_monthly_cf
# capacity_factor_2 = IRENA_monthly_cf

# # =================================================================
#     # PRINT ALL VARIABLES FOR VERIFICATION
#     # =================================================================
    
# print("🔍 DATA VERIFICATION - Checking all input variables:")
# print("="*80)

# # 1. Wind speeds at 78m
# print("\n1️⃣ WIND SPEEDS AT 78M HEIGHT:")
# print(f"   Renewables.ninja: {windspeed_78m_1}")
# print(f"   NSRDB:           {windspeed_78m_2}")
# print(f"   IRENA:           {windspeed_78m_3}")

# # 2. Friction coefficients
# print("\n2️⃣ FRICTION COEFFICIENTS:")
# print(f"   NSRDB: {friction_coeff_1}")
# print(f"   IRENA: {friction_coeff_2}")

# # 3. Weibull shape parameters
# print("\n3️⃣ WEIBULL SHAPE PARAMETERS:")
# print(f"   NSRDB:           {nsrdb_shape_0}")
# print(f"   IRENA:           {nsrdb_shape_1}")
# print(f"   Renewables.ninja: {nsrdb_shape_2}")

# # 4. Weibull scale parameters
# print("\n4️⃣ WEIBULL SCALE PARAMETERS:")
# print(f"   NSRDB:           {nsrdb_scale_0}")
# print(f"   IRENA:           {nsrdb_scale_1}")
# print(f"   Renewables.ninja: {nsrdb_scale_2}")

# # 5. Energy yields
# print("\n5️⃣ MONTHLY ENERGY YIELDS:")
# print(f"   NSRDB:           {energy_yield_1}")
# print(f"   IRENA:           {energy_yield_2}")
# print(f"   Renewables.ninja: {energy_yield_3}")

# # 6. Capacity factors
# print("\n6️⃣ CAPACITY FACTORS:")
# print(f"   NSRDB: {capacity_factor_1}")
# print(f"   IRENA: {capacity_factor_2}")

# # Data validation summary
# print("\n" + "="*80)
# print("📊 DATA SUMMARY:")
# print(f"   Wind speeds (78m)    - NSRDB: {len(windspeed_78m_2)} values, IRENA: {len(windspeed_78m_3)} values, RE.ninja: {len(windspeed_78m_1)} values")
# print(f"   Friction coefficients - NSRDB: {len(friction_coeff_1)} values, IRENA: {len(friction_coeff_2)} values")
# print(f"   Shape parameters     - NSRDB: {len(nsrdb_shape_0)} values, IRENA: {len(nsrdb_shape_1)} values, RE.ninja: {len(nsrdb_shape_2)} values")
# print(f"   Scale parameters     - NSRDB: {len(nsrdb_scale_0)} values, IRENA: {len(nsrdb_scale_1)} values, RE.ninja: {len(nsrdb_scale_2)} values")
# print(f"   Energy yields        - NSRDB: {len(energy_yield_1)} values, IRENA: {len(energy_yield_2)} values, RE.ninja: {len(energy_yield_3)} values")
# print(f"   Capacity factors     - NSRDB: {len(capacity_factor_1)} values, IRENA: {len(capacity_factor_2)} values")
# print("="*80)

# # =================================================================
# # GENERATE ALL PLOTS AND TEXT FILES
# # =================================================================

# print("Starting Wind Energy Analysis...")
# print("="*50)


# # 1. Monthly wind speed overlays at 78m height
# print("\n1. Generating monthly wind speed overlay plot...")
# analyzer.plot_monthly_windspeed_overlays(valid_points[0][1], valid_points[0][0] ,
#     windspeed_78m_1, windspeed_78m_2, windspeed_78m_3,
#     dataset_names=['RE Ninja', 'NSRDB', 'IRENA']
# )

# # 2. Friction coefficients text file
# print("\n2. Generating friction coefficients text file...")
# analyzer.save_friction_coefficients_to_text(valid_points[0][1], valid_points[0][0] ,
#     friction_coeff_1, friction_coeff_2,
#     dataset_names=['NSRDB', 'IRENA']
# )

# # 3. Weibull parameters text file
# print("\n3. Generating Weibull parameters text file...")

# analyzer.save_weibull_parameters_to_text(valid_points[0][1], valid_points[0][0] ,
#     nsrdb_shape_0, nsrdb_shape_1, nsrdb_shape_2,
#     nsrdb_scale_0, nsrdb_scale_1, nsrdb_scale_2,
#     dataset_names=['NSRDB', 'IRENA', 'RE Ninja']
# )

# # 4. Monthly energy yield overlays
# print("\n4. Generating monthly energy yield overlay plot...")
# analyzer.plot_monthly_energy_yield_overlays(valid_points[0][1], valid_points[0][0] ,
#     energy_yield_1, energy_yield_2, energy_yield_3,
#     dataset_names=['NSRDB', 'IRENA', 'RE Ninja']
# )

# # 5. Monthly capacity factor overlays
# print("\n5. Generating monthly capacity factor overlay plot...")
# analyzer.plot_monthly_capacity_factors_overlays(valid_points[0][1], valid_points[0][0] ,
#     capacity_factor_1, capacity_factor_2,
#     dataset_names=['NSRDB', 'IRENA']
# )

# # 6. Weibull distributions using mixed parameters
# analyzer.plot_all_weibull_distributions(
#     nsrdb_shape_0, nsrdb_scale_0,    # NSRDB: shape_0 + scale_0
#     nsrdb_shape_1, nsrdb_scale_1,    # IRENA: shape_1 + scale_1  
#     nsrdb_shape_2, nsrdb_scale_2, valid_points[0][1], valid_points[0][0]     # Renewables.ninja: shape_2 + scale_2
# )

# print("\n7b. Generating monthly wind speed box plots...")
# analyzer.plot_monthly_wind_speed_boxplots(
#     [re_ninja_monthly_hourly, nsrdb_conv_ws, irena_conv_ws],
#     dataset_names=['RE ninja', 'NSRDB', 'IRENA'],
#     title= f"Monthly Wind Speed Distribution for Point ({valid_points[0][1]}, {valid_points[0][0]}) "
# )

# print("\n" + "="*50)
# print("✅ Analysis Complete!")
# print("📁 Check 'output_plots/' folder for image files")
# print("📄 Check 'output_text/' folder for text files")
# print("="*50)
# print(re_ninja_monthly_hourly)
# print("nsrd")
# print(nsrdb_extrapolated)

#plot first monthly wind speed averages (extrapolated to a height of 78m)

#table for monthly shape and scale parameters

#plot overlayed monthly weibull distribution using shape and scale pramaters

#plot monthly average energy yield


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
                <h2 class="metric-title">Capacity</h2>
            </div>
            <div class="metric-content">
                <div class="icon-container">
                    <img src="https://raw.githubusercontent.com/yelsha07/icons/refs/heads/main/1.png" class="icon" />
                </div>
                <div class="data-container" style="text-align: center; width: 100%;">
                    <div class="data-row" style="justify-content: center;">
                        <div class="source-label">IRENA & NSRDB</div>
                    </div>
                    <div class="value" style="font-size: 22px; text-align: center; width: 100%;">{round((capacity/1000), 3)} MW</div>
                </div>
            </div>
        </div>
        
        <!-- Energy Yield -->
        <div class="metric-container">
            <div class="metric-header">
                <h2 class="metric-title">Annual Ave. Capacity Factor</h2>
            </div>
            <div class="metric-content">
                <div class="icon-container">
                    <img src="https://raw.githubusercontent.com/yelsha07/icons/refs/heads/main/2.png" class="icon" />
                </div>
                <div class="data-container">
                    <div class="data-row">
                        <div class="source-label">IRENA</div>
                        <div class="value">{f"{ave_cf_irena:.2f}"} %</div>
                    </div>
                    <div class="data-row">
                        <div class="source-label">NSRDB</div>
                        <div class="value">{f"{ave_cf_nsrdb:.2f}"} %</div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- LCOE -->
        <div class="metric-container">
            <div class="metric-header">
                <h2 class="metric-title">LCOE</h2>
            </div>
            <div class="metric-content">
                <div class="icon-container">
                    <img src="https://raw.githubusercontent.com/yelsha07/icons/refs/heads/main/3.png" class="icon" />
                </div>
                <div class="data-container">
                    <div class="data-row">
                        <div class="source-label">IRENA</div>
                        <div class="value">₱{round((ave_lcoe_irena), 2)}/MWh</div>
                    </div>
                    <div class="data-row">
                        <div class="source-label">NSRDB</div>
                        <div class="value">₱{round((ave_lcoe_nsrdb), 2)}/MWh</div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    #updated database

#     #ave monthly energy yield
#     #ave cf
#     #ave LCOE

# st.write(f'NSRDB: {NSRDB_monthly_energy_yield}')
# st.write(f'IRENA: {IRENA_monthly_energy_yield}')
st.write(f'ave irena cf: {sum(IRENA_monthly_cf)/12}')
st.write(f'ave nsrdb cf: {sum(NSRDB_monthly_cf)/12}')

#refer to sheets for re ninja

eyield, cfactor, lcoee = wind.plot_monthly_value(IRENA_monthly_energy_yield, NSRDB_monthly_energy_yield, IRENA_monthly_cf, NSRDB_monthly_cf, capacity, irena_lcoe, nsrdb_lcoe)

with disp1:
    st.plotly_chart(eyield)

with disp2:
    st.plotly_chart(cfactor)

end_time = time.perf_counter()
execution_time = end_time - start_time
st.write(f"runtime: {execution_time}")

# # FOR DATA VALIDATION REMOVER EVYRHTING WHEN DONE (Plotting Weibull)


# def fetch_monthly_per_point(month, lat_rounded, lon_rounded): #utilized
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

# print(f"\nMean Absolute Difference: {np.mean(np.abs(np.array(irena_averages) - np.array(nsrdb_averages))):.2f}")

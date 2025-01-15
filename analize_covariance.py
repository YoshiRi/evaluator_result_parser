import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_error_vs_covariance(df: pd.DataFrame, label: str = ''):
    """
    Analyze the relationship between error and covariance for a specified axis.
    """
    if label:
        df = df[df['label'] == label]

    # Filter rows with pose_covariance and EST data
    df = df[df['object_type'] == 'EST']
    df = df[df['pose_covariance'].apply(len) > 0]
    
    # Extract covariance and error for the specified axis
    cov_indices = {'x': 0, 'y': 7, 'yaw': 35}  # Diagonal indices in covariance matrix
    cov_names = ["covariance_" + axis for axis in cov_indices]   
    errors = []
    covariances = {cov_name: [] for cov_name in cov_names}
    
    for _, row in df.iterrows():
        # Extract covariance value for the specified axis
        covariance_list = row['pose_covariance']
        if len(covariance_list) == 0:
            continue
        # str to list
        if isinstance(covariance_list, str):
            covariance_list = covariance_list.replace("[", "").replace("]", "").split(",")
            covariance_list = [float(cov) for cov in covariance_list]
        
        for cov_name, cov_index in cov_indices.items():
            covariance = covariance_list[cov_index]
            covariances["covariance_" + cov_name].append(covariance)
    
    # Create a DataFrame for plotting
    analysis_df = pd.DataFrame(covariances)
    copy_columns = ['pose_error_x', 'pose_error_y', 'heading_error_z', 'bev_error', 'distance_from_ego', 'label']
    for column in copy_columns:
        analysis_df[column] = df[column]
    
    
    # Plot error vs covariance
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=analysis_df, x='covariance_x', y='bev_error', alpha=0.7)
    plt.title(f'{label} Error vs Covariance (x-axis)')
    plt.xlabel('Covariance')
    plt.ylabel('Error')
    plt.grid(True)

    # Plot error vs covariance
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=analysis_df, x='covariance_y', y='bev_error', alpha=0.7)
    plt.title(f'{label} Error vs Covariance (y-axis)')
    plt.xlabel('Covariance')
    plt.ylabel('Error')
    plt.grid(True)

    # Plot error vs covariance
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=analysis_df, x='covariance_yaw', y='heading_error_z', alpha=0.7)
    plt.title(f'{label} Error vs Covariance (yaw-axis)')
    plt.xlabel('Covariance')
    plt.ylabel('Error')
    plt.grid(True)




# Load the 
import os
this_folder = os.path.dirname(os.path.abspath(__file__))
file_path = this_folder + "/extracted_objects.csv"
df = pd.read_csv(file_path)

# Analyze for x-axis
analyze_error_vs_covariance(df, 'car')

analyze_error_vs_covariance(df, 'pedestrian')
plt.show()

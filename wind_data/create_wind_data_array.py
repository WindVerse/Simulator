import numpy as np
import pandas as pd
import os
from pathlib import Path
import re

def extract_wind_data(base_dir, x_range=(-50, 50), y_range=(-50, 50), z_levels=[1, 2, 3, 4]):
    """
    Extract wind data into 5D numpy array with dimensions:
    [component, z, y, x, time]
    """
    
    # Get all time directories
    time_dirs = sorted([d for d in os.listdir(base_dir) 
                       if os.path.isdir(os.path.join(base_dir, d)) and re.match(r'^\d+\.?\d*$', d)],
                      key=float)
    
    # Create coordinate grids
    x_coords = np.arange(x_range[0], x_range[1] + 1)
    y_coords = np.arange(y_range[0], y_range[1] + 1)
    z_coords = np.array(z_levels)
    time_coords = np.array([float(t) for t in time_dirs])
    
    # Initialize 5D array: [component, z, y, x, time]
    wind_data = np.full((3, len(z_coords), len(y_coords), len(x_coords), len(time_coords)), np.nan)
    
    for time_idx, time_dir in enumerate(time_dirs):
        time_val = float(time_dir)
        print(f"Processing time {time_val}s ({time_idx+1}/{len(time_dirs)})")
        
        for z_idx, z_level in enumerate(z_levels):
            # Construct filename
            file_name = f"U_zNormal_{z_idx+1}.raw"
            file_path = os.path.join(base_dir, time_dir, file_name).replace("\\", "/")
            
            if not os.path.exists(file_path):
                print(f"Warning: File not found {file_path}")
                continue
            
            try:
                # Read the raw file
                df = pd.read_csv(file_path, sep=r'\s+', skiprows=2, 
                               names=['x', 'y', 'z', 'U_x', 'U_y', 'U_z'],
                               dtype={'x': float, 'y': float, 'z': float, 
                                      'U_x': float, 'U_y': float, 'U_z': float})
                
                # Round coordinates to nearest meter
                df['x_round'] = np.round(df['x']).astype(int)
                df['y_round'] = np.round(df['y']).astype(int)
                
                # Filter points within our desired grid
                mask = (df['x_round'] >= x_range[0]) & (df['x_round'] <= x_range[1]) & \
                       (df['y_round'] >= y_range[0]) & (df['y_round'] <= y_range[1])
                df_filtered = df[mask].copy()
                
                # Group by rounded coordinates and take first value
                grouped = df_filtered.groupby(['x_round', 'y_round']).first().reset_index()
                
                # Place data into the 5D array
                for _, row in grouped.iterrows():
                    x = int(row['x_round'])
                    y = int(row['y_round'])
                    
                    # Find array indices
                    x_arr_idx = np.where(x_coords == x)[0][0]
                    y_arr_idx = np.where(y_coords == y)[0][0]
                    
                    # Store velocity components
                    wind_data[0, z_idx, y_arr_idx, x_arr_idx, time_idx] = row['U_x']  # u-component
                    wind_data[1, z_idx, y_arr_idx, x_arr_idx, time_idx] = row['U_y']  # v-component  
                    wind_data[2, z_idx, y_arr_idx, x_arr_idx, time_idx] = row['U_z']  # w-component
                    
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
    
    return wind_data, x_coords, y_coords, z_coords, time_coords

def save_wind_data(wind_data, x_coords, y_coords, z_coords, time_coords, output_file):
    """Save the extracted wind data to a compressed numpy file"""
    
    # Create metadata dictionary
    metadata = {
        'component_names': ['u', 'v', 'w'],
        'x_coordinates': x_coords,
        'y_coordinates': y_coords, 
        'z_coordinates': z_coords,
        'time_coordinates': time_coords,
        'dimensions': ['component', 'z', 'y', 'x', 'time'],
        'units': {
            'velocity': 'm/s',
            'coordinates': 'meters',
            'time': 'seconds'
        }
    }
    
    # Save data and metadata
    np.savez_compressed(output_file, 
                       wind_data=wind_data,
                       metadata=metadata)
    
    print(f"Data saved to {output_file}")
    print(f"Array shape: {wind_data.shape}")

# Usage example
if __name__ == "__main__":
    base_directory = "D:/Academic/FYP/buildingSimulation/buildingSimulation/postProcessing/surfaces"
    
    # Extract data
    wind_data, x_coords, y_coords, z_coords, time_coords = extract_wind_data(
        base_directory, 
        x_range=(-50, 50), 
        y_range=(-50, 50),
        z_levels=[1, 2, 3, 4]
    )
    
    # Save results
    save_wind_data(wind_data, x_coords, y_coords, z_coords, time_coords, 
                  "wind_data_5d_array.npz")
    
    # Print summary
    print(f"\nExtraction complete!")
    print(f"Array shape: {wind_data.shape}")
    print(f"Components: u, v, w")
    print(f"Z levels: {z_coords}m")
    print(f"X range: {x_coords[0]} to {x_coords[-1]}m")
    print(f"Y range: {y_coords[0]} to {y_coords[-1]}m") 
    print(f"Time steps: {time_coords[0]} to {time_coords[-1]}s")
    print(f"Missing data: {np.isnan(wind_data).sum()} points ({np.isnan(wind_data).mean()*100:.2f}%)")
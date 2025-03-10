import pandas as pd
import numpy as np

def parse_ps_csv(fname):
    # Read the CSV file
    df = pd.read_csv(fname,skiprows=1,sep=", ",engine='python') # Skip the first row and use a regular expression to split on commas
    
    # Filter out rows with type='C'
    df = df[df['type'] == 'M']
    
    # Get unique timestamps
    timestamps = df['time'].unique()
    
    # Initialize a dictionary to hold the results
    result_data = {
        'timestamp': timestamps,
        'time_s': timestamps / 1e6  # Converting to seconds assuming linux convention
    }
    
    # Find all unique marker IDs
    marker_ids = sorted(df['id'].unique())
    
    # For each timestamp and marker, extract position data
    for marker_id in marker_ids:
        if 0 <= marker_id <= 4:  # Only considering markers with IDs 0-4
            # Initialize arrays for each position dimension
            result_data[f'marker{marker_id}_pos_x'] = np.full(len(timestamps), np.nan)
            result_data[f'marker{marker_id}_pos_y'] = np.full(len(timestamps), np.nan)
            result_data[f'marker{marker_id}_pos_z'] = np.full(len(timestamps), np.nan)
            
            # Extract marker data
            marker_data = df[df['id'] == marker_id]
            
            # For each timestamp in this marker's data, add to the appropriate position
            for _, row in marker_data.iterrows():
                timestamp_idx = np.where(timestamps == row['time'])[0][0]
                result_data[f'marker{marker_id}_pos_x'][timestamp_idx] = row['pos_x']
                result_data[f'marker{marker_id}_pos_y'][timestamp_idx] = row['pos_y']
                result_data[f'marker{marker_id}_pos_z'][timestamp_idx] = row['pos_z']
    
    # Convert to DataFrame
    result_df = pd.DataFrame(result_data)
  
    # Create a more convenient data structure for accessing marker data
    # Each marker will be an Nx3 array
    data = {}
    data["timestamp"] = result_df["timestamp"].values
    data["time_s"] = result_df["time_s"].values
    
    for marker_id in marker_ids:
        if 0 <= marker_id <= 4:
            cols = [f'marker{marker_id}_pos_x', f'marker{marker_id}_pos_y', f'marker{marker_id}_pos_z']
            if all(col in result_df.columns for col in cols):
                data[f"marker{marker_id}"] = result_df[cols].values
    
    data["time_s"] = data["time_s"] - data["time_s"][0]
    return data

def swap_cols(arr, frmcols, tocols):
    temp = np.copy(arr)
    for (i,frmcol) in enumerate(frmcols):
      temp[:,tocols[i]] = arr[:,frmcol]
    return temp
import os
import json
import pandas as pd
import tqdm

# Root directory containing the station folders (e.g., station1, station2, etc.)
root_dir = '/Users/adamtsou/clothingquality/clothing_v3_data'
output_dir = '/Users/adamtsou/clothingquality/'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# List to store all labels
all_labels = []

# Traverse the directory structure (station -> month -> images)
for station_folder in os.listdir(root_dir):
    station_path = os.path.join(root_dir, station_folder)
    if os.path.isdir(station_path):
        for month_folder in os.listdir(station_path):
            month_path = os.path.join(station_path, month_folder)
            if os.path.isdir(month_path):
                # Process each JSON label file in the month folder
                for label_file in os.listdir(month_path):
                    if label_file.endswith('.json'):
                        with open(os.path.join(month_path, label_file), 'r') as f:
                            label_data = json.load(f)
                            label_data['timestamp'] = label_file.split('.')[0]
                            label_data['station'] = station_folder
                            label_data['month'] = month_folder
                            all_labels.append(label_data)
# Convert the list of labels to a DataFrame
df = pd.DataFrame(all_labels)

# Save the training and testing sets to CSV files
df.to_csv(os.path.join(output_dir, 'train_labels.csv'), index=False)

print("Training and testing datasets created successfully.")
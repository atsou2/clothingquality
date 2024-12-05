import os
from PIL import Image
import re
import json
from tqdm import tqdm

# Root directory containing the station folders (e.g., station1, station2, etc.)
root_dir = '/Users/adamtsou/clothingquality/clothing_v3_data'

# Regex to extract the type (front/back/brand) and timestamp
pattern = re.compile(r'(\w+)_(\d+)\.jpg')

# List to hold timestamps
timestamps = []

# Save the list of timestamps to a JSON or text file
timestamp_file = '/Users/adamtsou/clothingquality/timestamps.json'

# Traverse the directory structure (station -> month -> images)
for station_folder in os.listdir(root_dir):
    station_path = os.path.join(root_dir, station_folder)
    if os.path.isdir(station_path):
        for month_folder in os.listdir(station_path):
            month_path = os.path.join(station_path, month_folder)
            if os.path.isdir(month_path):
                # Process each image file in the month folder
                for filename in tqdm(os.listdir(month_path)):
                    match = pattern.match(filename)
                    if match:
                        img_type = match.group(1)  # front, back, or brand
                        timestamp = match.group(2)  # extract timestamp
                        img_path = os.path.join(month_path, filename)
                        
                        # Add the timestamp to the list if it's new
                        if timestamp not in timestamps:
                            timestamps.append(timestamp)

                        # Load and preprocess the image
                        with Image.open(img_path) as img:
                            # Example: resizing images to 256x256
                            img = img.resize((320,180))
                            
                            # Construct new file path (same folder structure but different path)
                            new_img_path = f'/Users/adamtsou/clothingquality/new_data/{station_folder}/{month_folder}/{img_type}_{timestamp}.jpg'
                            os.makedirs(os.path.dirname(new_img_path), exist_ok=True)
                            
                            # Save the preprocessed image to the new path
                            img.save(new_img_path)

# Save the list of timestamps
with open(timestamp_file, 'w') as f:
    json.dump(timestamps, f)
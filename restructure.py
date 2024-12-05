import os
import shutil
import pandas as pd
from tqdm import tqdm

# Paths
csv_file = '/Users/adamtsou/clothingquality/train_labels.csv'
image_dir = '/Users/adamtsou/clothingquality/fused'
output_dir = '/Users/adamtsou/clothingquality/dataset'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Read the CSV file
df = pd.read_csv(csv_file)

# Create directories for each condition label
conditions = df['condition'].unique()
for condition in conditions:
    os.makedirs(os.path.join(output_dir, str(condition)), exist_ok=True)

# Move files to corresponding condition folders
for index, row in tqdm(df.iterrows()):
    condition = row['condition']
    timestamp = row['timestamp']
    timestamp = timestamp.replace('labels_', '')  # Adjust this to match the correct column name for the timestamp
    image_filename = f"{timestamp}.jpg"  # Assuming the image filenames are based on the timestamp
    src_path = os.path.join(image_dir, image_filename)
    dest_path = os.path.join(output_dir, str(condition), image_filename)
    
    if os.path.exists(src_path):
        shutil.move(src_path, dest_path)
    else:
        print(f"File {src_path} does not exist.")

print("Dataset restructuring complete.")
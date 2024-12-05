import os
from PIL import Image

# Directory containing the images
image_dir = '/Users/adamtsou/clothingquality/new_data'
output_dir = '/Users/adamtsou/clothingquality/fused'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

for station_folder in os.listdir(image_dir):
    station_path = os.path.join(image_dir, station_folder)
    if os.path.isdir(station_path):
        for month_folder in os.listdir(station_path):
            month_path = os.path.join(station_path, month_folder)
            if os.path.isdir(month_path):
                # Get the list of image files in the directory
                image_files = sorted(os.listdir(month_path))

                # Function to extract timestamp from filename
                def get_timestamp(filename):
                    return '_'.join(filename.split('_')[1:])

                # Dictionary to store images by timestamp
                images_by_timestamp = {}

                # Organize images by timestamp
                for image_file in image_files:
                    if image_file.startswith('front_') or image_file.startswith('back_'):
                        timestamp = get_timestamp(image_file)
                        if timestamp not in images_by_timestamp:
                            images_by_timestamp[timestamp] = {}
                        if image_file.startswith('front_'):
                            images_by_timestamp[timestamp]['front'] = image_file
                        elif image_file.startswith('back_'):
                            images_by_timestamp[timestamp]['back'] = image_file

                # Fuse images with the same timestamp
                for timestamp, images in images_by_timestamp.items():
                    if 'front' in images and 'back' in images:
                        front_image = images['front']
                        back_image = images['back']
                        
                        # Open the front and back images
                        front_img = Image.open(os.path.join(month_path, front_image))
                        back_img = Image.open(os.path.join(month_path, back_image))
                        
                        # Ensure both images have the same size
                        if front_img.size != back_img.size:
                            print(f"Skipping {front_image} and {back_image} due to size mismatch.")
                            continue
                        
                        # Create a new image by concatenating front and back images horizontally
                        fused_img = Image.new('RGB', (front_img.width + back_img.width, front_img.height))
                        fused_img.paste(front_img, (0, 0))
                        fused_img.paste(back_img, (front_img.width, 0))
                        
                        # Save the fused image
                        fused_img.save(os.path.join(output_dir, f"{timestamp}_fused.jpg"))


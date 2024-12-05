#!/bin/bash

# Directory containing the files
DIRECTORY="/Users/adamtsou/clothingquality/demo"

# URL of the API endpoint
URL="http://127.0.0.1:5000/api/upload"

# Output file to save the combined API responses
OUTPUT_FILE="combined_api_responses.json"

# Temporary directory to save individual API responses
TEMP_DIR="temp_responses"

# Ensure the temporary directory exists
mkdir -p $TEMP_DIR

# Loop through all files in the directory
for FILE in "$DIRECTORY"/*; do
  # Extract the base name of the file (without the directory path)
  BASENAME=$(basename "$FILE")
  
  # Construct the temporary output file path
  TEMP_FILE="$TEMP_DIR/${BASENAME%.*}_response.json"
  
  # Send the file to the API endpoint and save the response
  curl -X POST -F "file=@$FILE" $URL > $TEMP_FILE
  
  echo "API response for $BASENAME saved to $TEMP_FILE"
done

# Combine all individual response files into a single file
jq -s '.' $TEMP_DIR/*.json > $OUTPUT_FILE

echo "Combined API responses saved to $OUTPUT_FILE"

# Clean up temporary files
rm -r $TEMP_DIR
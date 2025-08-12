# dataset_creator.py

import pandas as pd
import os

# Create the dataset as shown above
data = [
    ["PCB001", "dataset/single_sided/PCB001.jpg", "single_sided", "low", 1, "basic", "CE", "none", "consumer_electronics", "FR-4", "none"],
    ["PCB002", "dataset/single_sided/PCB002.jpg", "single_sided", "medium", 1, "enhanced", "CE;RoHS", "solder_bridge", "consumer_electronics", "FR-4", "none"],
    # ... continue with all 30 entries from the CSV above
    ["PCB030", "dataset/double_sided/PCB030.jpg", "double_sided", "low", 2, "basic", "CE;RoHS", "none", "consumer_electronics", "FR-4", "none"]
]

# Define column names
columns = [
    "image_id", "image_path", "pcb_type", "components_density", "layer_count", 
    "quality_check_required", "certification_needed", "defect_type", 
    "intended_application", "material_type", "special_features"
]

# Create DataFrame
df = pd.DataFrame(data, columns=columns)

# Ensure the output directory exists
os.makedirs("data", exist_ok=True)

# Save to CSV
df.to_csv("data/pcb_dataset.csv", index=False)

print("Dataset CSV created successfully!")

# Also create the necessary directory structure for organizing images
directories = [
    "dataset/single_sided",
    "dataset/double_sided",
    "dataset/multilayer",
    "dataset/flexible",
    "dataset/rigid_flex",
    "dataset/high_frequency",
    "dataset/high_power"
]

for directory in directories:
    os.makedirs(directory, exist_ok=True)

print("Directory structure created successfully!")

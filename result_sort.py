import os
import shutil
import pandas as pd

# Paths
base_dir = 'images'  # relative to your current working directory
results_dir = os.path.join(base_dir, 'results')
predictions_csv = 'sorted_predictions.csv'  # or predictions_v3.csv if you want

# Create result directories
for label in ['pool', 'no_pool']:
    os.makedirs(os.path.join(results_dir, label), exist_ok=True)

# Load predictions
df = pd.read_csv(predictions_csv)

# Index all available images under images/{zip}/
image_index = {}
for root, dirs, files in os.walk(base_dir):
    for f in files:
        if f.endswith('.png'):
            image_index[f] = os.path.join(root, f)

# Copy based on prediction
missing = []

for _, row in df.iterrows():
    filename = row['filename'].strip()
    prediction = row['prediction'].strip().lower()

    if filename in image_index:
        src_path = image_index[filename]
        dest_path = os.path.join(results_dir, prediction, filename)
        shutil.copy2(src_path, dest_path)
    else:
        print(f"⚠️ Warning: {filename} not found in any ZIP folder.")
        missing.append(filename)

# Optional: Save missing files to debug later
if missing:
    pd.DataFrame(missing, columns=["missing_filename"]).to_csv("missing_predictions.csv", index=False)
    print(f"\nMissing files saved to missing_predictions.csv")

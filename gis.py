import requests
import csv
import json
import os
import math
from statistics import mean

# Configuration
zipcode = "37205"  # <-- change this to the ZIP you want

DOWNLOAD_IMAGES = True
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")  # Or hardcode if needed

# Create output directories
image_dir = f"images/{zipcode}"
os.makedirs(image_dir, exist_ok=True)

# Step 1: Query Nashville GIS for parcels in the given ZIP code
url = "https://maps.nashville.gov/arcgis/rest/services/Cadastral/Parcels/MapServer/0/query"

data = {
    "where": f"PropZip='{zipcode}'",
    "outFields": "*",
    "returnGeometry": "true",
    "returnZ": "false",
    "returnM": "false",
    "f": "json",
    "resultRecordCount": 100
}

response = requests.post(url, data=data)
results = response.json()["features"]

# Save raw GIS response
json_output_path = f"parcel_results_{zipcode}.json"
with open(json_output_path, "w") as json_out:
    json.dump(results, json_out, indent=2)

# Step 2: Helper to compute centroid from polygon ring
def compute_centroid(rings):
    all_coords = rings[0]
    xs = [pt[0] for pt in all_coords]
    ys = [pt[1] for pt in all_coords]
    return mean(xs), mean(ys)

# Step 3: Convert from Web Mercator to lat/lng
def web_mercator_to_latlng(x, y):
    lon = x * 180 / 20037508.34
    lat = math.degrees(math.atan(math.exp(y / 6378137)) * 2 - math.pi / 2)
    return round(lat, 6), round(lon, 6)

# Step 4: Save CSV and download satellite images
csv_output_path = f"parcel_centroids_{zipcode}.csv"
with open(csv_output_path, "w", newline="") as f:
    fieldnames = [
        "ParcelID", "PropAddr", "PropZip", "Owner", "Acres",
        "AppraisedValue", "Latitude", "Longitude", "ImageFilename"
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    for feat in results:
        attr = feat["attributes"]
        rings = feat.get("geometry", {}).get("rings")
        if not rings:
            continue

        centroid_x, centroid_y = compute_centroid(rings)
        lat, lon = web_mercator_to_latlng(centroid_x, centroid_y)

        parcel_id = attr.get("ParID")
        image_filename = f"{image_dir}/{parcel_id}.png"

        if DOWNLOAD_IMAGES and GOOGLE_MAPS_API_KEY:
            image_url = (
                f"https://maps.googleapis.com/maps/api/staticmap"
                f"?center={lat},{lon}&zoom=20&size=600x600&maptype=satellite&key={GOOGLE_MAPS_API_KEY}"
            )
            try:
                img = requests.get(image_url)
                with open(image_filename, "wb") as img_out:
                    img_out.write(img.content)
            except Exception as e:
                print(f"Error downloading {image_filename}: {e}")
                image_filename = ""

        writer.writerow({
            "ParcelID": parcel_id,
            "PropAddr": attr.get("PropAddr"),
            "PropZip": attr.get("PropZip"),
            "Owner": attr.get("Owner"),
            "Acres": attr.get("Acres"),
            "AppraisedValue": attr.get("TotlAppr"),
            "Latitude": lat,
            "Longitude": lon,
            "ImageFilename": image_filename if DOWNLOAD_IMAGES else ""
        })

print(f"✅ Done for ZIP {zipcode}. CSV and images saved.")

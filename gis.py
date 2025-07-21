import requests
import csv
import os
from statistics import mean

# Set this if you want to download images
DOWNLOAD_IMAGES = True
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")  # Or hardcode if needed

# Step 1: Query Nashville GIS for 37205 parcels
url = "https://maps.nashville.gov/arcgis/rest/services/Cadastral/Parcels/MapServer/0/query"

data = {
    "where": "PropZip='37205'",
    "outFields": "*",
    "returnGeometry": "true",
    "returnZ": "false",
    "returnM": "false",
    "f": "json",
    "resultRecordCount": 300
}

response = requests.post(url, data=data)
results = response.json()["features"]

# Step 2: Helper to compute centroid from polygon ring
def compute_centroid(rings):
    all_coords = rings[0]  # Use first ring
    xs = [pt[0] for pt in all_coords]
    ys = [pt[1] for pt in all_coords]
    return mean(xs), mean(ys)

# Step 3: Convert from Web Mercator to lat/lng
# Source: https://developers.arcgis.com/documentation/core-concepts/rest-api/spatial-references/
import math
def web_mercator_to_latlng(x, y):
    lon = x * 180 / 20037508.34
    lat = math.degrees(math.atan(math.exp(y / 6378137)) * 2 - math.pi / 2)
    return round(lat, 6), round(lon, 6)

# Step 4: Save CSV and images
os.makedirs("images", exist_ok=True)

with open("parcel_centroids_37205.csv", "w", newline="") as f:
    fieldnames = ["ParcelID", "PropAddr", "PropZip", "Owner", "Acres", "AppraisedValue", "Latitude", "Longitude", "ImageFilename", "resultRecordCount"]
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
        image_filename = f"images/{parcel_id}.png"

        # Optional: Download satellite image
        if DOWNLOAD_IMAGES and GOOGLE_MAPS_API_KEY:
            image_url = f"https://maps.googleapis.com/maps/api/staticmap?center={lat},{lon}&zoom=20&size=600x600&maptype=satellite&key={GOOGLE_MAPS_API_KEY}"
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

print("✅ Done. CSV and images saved.")

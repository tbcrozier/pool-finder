"""
Overlay parcel boundaries on satellite images.

This script takes satellite images and their corresponding parcel geometry,
then draws the parcel boundary on top of the image so it's clear which
property is being classified.

Usage:
    python overlay_boundaries.py --zipcode 37205
    python overlay_boundaries.py --image images/37205/12345.png --geometry parcel_results_37205.json
"""

import os
import json
import math
import argparse
from pathlib import Path
from PIL import Image, ImageDraw

# Google Maps Static API parameters (must match gis.py)
ZOOM = 20
IMAGE_SIZE = 600  # pixels (600x600)


def web_mercator_to_latlng(x: float, y: float) -> tuple[float, float]:
    """Convert Web Mercator coordinates to lat/lng."""
    lon = x * 180 / 20037508.34
    lat = math.degrees(math.atan(math.exp(y / 6378137)) * 2 - math.pi / 2)
    return lat, lon


def latlng_to_pixel(lat: float, lng: float, center_lat: float, center_lng: float, zoom: int, image_size: int) -> tuple[int, int]:
    """
    Convert lat/lng to pixel coordinates relative to image center.

    Based on Google Maps tile math:
    - At zoom 0, the world is 256 pixels
    - Each zoom level doubles the pixels
    """
    # World coordinates (Mercator projection)
    def lat_to_y(lat):
        sin_lat = math.sin(lat * math.pi / 180)
        return 0.5 - math.log((1 + sin_lat) / (1 - sin_lat)) / (4 * math.pi)

    def lng_to_x(lng):
        return (lng + 180) / 360

    # Scale factor for this zoom level
    scale = 256 * (2 ** zoom)

    # Convert center and point to world pixels
    center_x = lng_to_x(center_lng) * scale
    center_y = lat_to_y(center_lat) * scale
    point_x = lng_to_x(lng) * scale
    point_y = lat_to_y(lat) * scale

    # Convert to image pixels (relative to center)
    pixel_x = int((point_x - center_x) + image_size / 2)
    pixel_y = int((point_y - center_y) + image_size / 2)

    return pixel_x, pixel_y


def draw_parcel_boundary(
    image_path: str,
    rings: list,
    center_lat: float,
    center_lng: float,
    output_path: str = None,
    boundary_color: tuple = (255, 0, 0),  # Red
    boundary_width: int = 3,
    fill_color: tuple = None,  # Optional fill (e.g., (255, 0, 0, 50) for transparent red)
) -> str:
    """
    Draw parcel boundary polygon on a satellite image.

    Args:
        image_path: Path to the satellite image
        rings: Parcel polygon rings from GIS (in Web Mercator coordinates)
        center_lat: Center latitude of the image
        center_lng: Center longitude of the image
        output_path: Where to save the result (default: adds '_boundary' suffix)
        boundary_color: RGB tuple for boundary line color
        boundary_width: Width of boundary line in pixels
        fill_color: Optional RGBA tuple for polygon fill

    Returns:
        Path to the output image
    """
    # Load image
    img = Image.open(image_path).convert("RGBA")

    # Create overlay for semi-transparent fill if needed
    if fill_color:
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)

    draw = ImageDraw.Draw(img)

    # Process each ring (outer boundary is first, holes would be subsequent)
    for ring_idx, ring in enumerate(rings):
        # Convert ring coordinates to image pixels
        pixel_coords = []
        for point in ring:
            x, y = point[0], point[1]
            lat, lng = web_mercator_to_latlng(x, y)
            px, py = latlng_to_pixel(lat, lng, center_lat, center_lng, ZOOM, IMAGE_SIZE)
            pixel_coords.append((px, py))

        if len(pixel_coords) < 3:
            continue

        # Draw fill if specified (only for outer ring)
        if fill_color and ring_idx == 0:
            overlay_draw.polygon(pixel_coords, fill=fill_color)

        # Draw boundary
        draw.polygon(pixel_coords, outline=boundary_color, width=boundary_width)

    # Composite overlay if we used one
    if fill_color:
        img = Image.alpha_composite(img, overlay)

    # Convert back to RGB for saving as JPEG/PNG
    img = img.convert("RGB")

    # Determine output path
    if output_path is None:
        path = Path(image_path)
        output_path = str(path.parent / f"{path.stem}_boundary{path.suffix}")

    img.save(output_path)
    return output_path


def process_zipcode(zipcode: str, output_dir: str = None):
    """
    Process all images for a zipcode, adding parcel boundaries.

    Requires:
        - images/{zipcode}/*.png - satellite images
        - parcel_results_{zipcode}.json - GIS data with geometry
        - parcel_centroids_{zipcode}.csv - centroid coordinates
    """
    import csv

    image_dir = Path(f"images/{zipcode}")
    json_path = f"parcel_results_{zipcode}.json"
    csv_path = f"parcel_centroids_{zipcode}.csv"

    if output_dir is None:
        output_dir = Path(f"images/{zipcode}_boundaries")
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load GIS data
    if not os.path.exists(json_path):
        print(f"ERROR: {json_path} not found. Run gis.py first.")
        return

    with open(json_path) as f:
        features = json.load(f)

    # Build lookup by parcel ID
    geometry_lookup = {}
    for feat in features:
        parcel_id = feat.get("attributes", {}).get("ParID")
        rings = feat.get("geometry", {}).get("rings")
        if parcel_id and rings:
            geometry_lookup[str(parcel_id)] = rings

    # Load centroids
    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} not found. Run gis.py first.")
        return

    centroid_lookup = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            parcel_id = row.get("ParcelID")
            lat = row.get("Latitude")
            lng = row.get("Longitude")
            if parcel_id and lat and lng:
                centroid_lookup[str(parcel_id)] = (float(lat), float(lng))

    # Process each image
    image_files = list(image_dir.glob("*.png"))
    print(f"Processing {len(image_files)} images...")

    processed = 0
    for img_path in image_files:
        parcel_id = img_path.stem  # filename without extension

        if parcel_id not in geometry_lookup:
            print(f"  Skipping {img_path.name}: no geometry found")
            continue

        if parcel_id not in centroid_lookup:
            print(f"  Skipping {img_path.name}: no centroid found")
            continue

        rings = geometry_lookup[parcel_id]
        center_lat, center_lng = centroid_lookup[parcel_id]

        output_path = output_dir / img_path.name

        try:
            draw_parcel_boundary(
                str(img_path),
                rings,
                center_lat,
                center_lng,
                str(output_path),
                boundary_color=(255, 50, 50),  # Bright red
                boundary_width=4,
                fill_color=(255, 0, 0, 30),  # Very light red fill
            )
            processed += 1
            print(f"  ✓ {img_path.name}")
        except Exception as e:
            print(f"  ✗ {img_path.name}: {e}")

    print(f"\nDone! Processed {processed}/{len(image_files)} images.")
    print(f"Output saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Overlay parcel boundaries on satellite images")
    parser.add_argument("--zipcode", type=str, help="Process all images for a ZIP code")
    parser.add_argument("--image", type=str, help="Single image to process")
    parser.add_argument("--geometry", type=str, help="JSON file with parcel geometry (for single image)")
    parser.add_argument("--output-dir", type=str, help="Output directory for processed images")

    args = parser.parse_args()

    if args.zipcode:
        process_zipcode(args.zipcode, args.output_dir)
    elif args.image:
        print("Single image mode not yet implemented. Use --zipcode for batch processing.")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

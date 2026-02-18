1# Pool Finder

A machine learning project to detect swimming pools in residential parcels using satellite imagery, targeting Nashville/Davidson County properties.

## Project Overview

This project uses a MobileNetV2 CNN (transfer learning) to classify aerial/satellite images as either containing a pool (`pool_in_parcel`) or not (`no_pool`). The workflow involves:
1. Fetching parcel data from Nashville GIS
2. Downloading satellite imagery from Google Maps Static API
3. Training/using a binary image classifier

## File Structure

```
pool-finder/
├── gis.py              # Fetches parcel data and downloads satellite images
├── train_classifier.py # Trains the MobileNetV2 pool classifier
├── predict_images.py   # Runs inference on new images
├── result_sort.py      # Organizes prediction results into folders
├── data/               # Training/validation/test datasets (gitignored)
│   ├── train/
│   ├── valid/
│   └── test/
├── images/             # Downloaded satellite images by ZIP code (gitignored)
├── etc/                # Reference data (davidson-zips.csv)
└── pool_classifier.pth # Trained model weights (gitignored)
```

## Key Components

### gis.py
- Queries Nashville GIS ArcGIS REST API for parcels by ZIP code
- Computes parcel centroids from polygon geometry
- Converts Web Mercator coordinates to lat/lng
- Downloads satellite images via Google Maps Static API
- Outputs CSV with parcel metadata and image paths

### train_classifier.py
- Uses PyTorch with MobileNetV2 pretrained backbone
- Trains on ImageFolder datasets (train/valid/test splits)
- Uses MPS (Apple Silicon GPU) when available
- Saves best model to `pool_classifier.pth`

### predict_images.py
- Loads trained model and runs inference on image folders
- Outputs predictions with confidence scores to CSV
- Class names: `['no_pool', 'pool']`

### result_sort.py
- Copies images into `pool/` or `no_pool/` folders based on predictions

## Tech Stack

- **Python 3** with PyTorch, torchvision
- **MobileNetV2** for efficient image classification
- **Nashville Open Data** (ArcGIS REST services) for parcel boundaries
- **Google Maps Static API** for satellite imagery

## Environment

- Requires `GOOGLE_MAPS_API_KEY` environment variable for image downloads
- Uses MPS (Metal Performance Shaders) on Apple Silicon, falls back to CPU
- Virtual environment: `.venv-pools/`

## Dataset Classes

- `pool_in_parcel` - Aerial images showing a pool within the parcel
- `no_pool` - Aerial images with no pool visible

## Common Commands

```bash
# Activate virtual environment
source .venv-pools/bin/activate

# Train the classifier
python train_classifier.py

# Run predictions on images
python predict_images.py

# Fetch parcel data and images for a ZIP code (edit zipcode in gis.py)
python gis.py

# Sort prediction results into folders
python result_sort.py
```

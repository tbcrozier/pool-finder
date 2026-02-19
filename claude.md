# Pool Finder

Detect swimming pools in residential parcels using satellite imagery and Claude Vision API. Targets Nashville/Davidson County properties.

## Goal

**Input**: A ZIP code
**Output**: CSV of addresses with pools, owner names, and verification links

Example output:
```
Address,Owner,ParcelID,HasPool,Confidence,ImagePath,GoogleMapsURL
123 Main St,John Doe,12345,pool,high,images/37205_boundaries/12345.png,https://google.com/maps/...
```

**Use case**: Sell pool owner lists to pool cleaning companies.

---

## The Pipeline

```
ZIP Code → Download Images → Draw Boundaries → Claude Vision → Report CSV
              (gis.py)     (overlay_boundaries.py)  (claude_vision_classify.py)
```

| Step | Script | What it does |
|------|--------|--------------|
| 1 | `gis.py` | Fetches parcel data from Nashville GIS + downloads satellite images |
| 2 | `overlay_boundaries.py` | Draws red parcel boundary on each image |
| 3 | `claude_vision_classify.py` | Asks Claude "is there a pool inside the red boundary?" |
| 4 | Manual | Generate report CSV, spot-check results via links |

---

## Quick Start

```bash
# 1. Activate environment
source .venv-pools/bin/activate

# 2. Download parcels + images for a ZIP code (edit zipcode in gis.py)
python gis.py

# 3. Add boundary overlays
python overlay_boundaries.py --zipcode 37205

# 4. Run Claude Vision classification
python claude_vision_classify.py --image-folder images/37205_boundaries --output predictions_37205.csv

# 5. Generate report (see "Generating the Report" below)
```

---

## File Structure

```
pool-finder/
├── gis.py                      # Downloads parcel data + satellite images
├── overlay_boundaries.py       # Draws boundary lines on images
├── claude_vision_classify.py   # Claude Vision pool detection
├── etc/davidson-zips.csv       # List of Nashville ZIP codes
├── images/
│   ├── {zipcode}/              # Raw satellite images
│   └── {zipcode}_boundaries/   # Images with red boundary overlay
├── parcel_centroids_{zip}.csv  # Parcel metadata (address, owner, lat/lng)
├── parcel_results_{zip}.json   # Raw GIS geometry data
├── predictions_{zip}.csv       # Claude's predictions
├── pool_report_{zip}.csv       # Full report with links
└── pools_only_{zip}.csv        # Just parcels with pools
```

---

## Environment Setup

Required environment variables in `~/.zshrc`:
```bash
export GOOGLE_MAPS_API_KEY='your-google-api-key'
export ANTHROPIC_API_KEY='your-anthropic-api-key'
```

Python packages:
```bash
pip install anthropic Pillow
```

---

## Generating the Report

After running predictions, join with parcel data to create the final report:

```python
import pandas as pd

predictions = pd.read_csv('predictions_37205.csv')
parcels = pd.read_csv('parcel_centroids_37205.csv')

predictions['ParcelID'] = predictions['filename'].str.replace('.png', '')
merged = predictions.merge(parcels, on='ParcelID', how='left')

merged['google_maps_url'] = merged.apply(
    lambda r: f"https://www.google.com/maps/@{r['Latitude']},{r['Longitude']},20z/data=!3m1!1e3",
    axis=1
)
merged['image_path'] = 'images/37205_boundaries/' + merged['filename']

# Save
merged.to_csv('pool_report_37205.csv', index=False)
merged[merged['prediction'] == 'pool'].to_csv('pools_only_37205.csv', index=False)
```

---

## Costs

| Service | Cost |
|---------|------|
| Google Maps Static API | ~$2 per 1000 images |
| Claude Vision API | ~$0.01-0.02 per image |

For a ZIP code with 1000 parcels: ~$15-25 total.

---

## Validation

The output CSV includes:
- **ImagePath**: Click to open the boundary image locally
- **GoogleMapsURL**: Click to verify on Google Maps satellite view

Spot-check 10-20 results to verify accuracy before delivering to customer.

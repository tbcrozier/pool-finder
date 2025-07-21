import requests

url = "https://maps.nashville.gov/arcgis/rest/services/Cadastral/Parcels/MapServer/0/query"

data = {
    "where": "PropZip='37205'",
    "outFields": "*",  
    "returnGeometry": "true",
    "returnZ": "false",
    "returnM": "false",
    "f": "json"
}

r = requests.post(url, data=data)
parcel_data = r.json()

# Preview one parcel
print(parcel_data["features"][0])


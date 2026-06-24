# Real coordinates extracted from the NISAR HDF5 boundingPolygon metadata
# Min Lon: 75.5382, Max Lon: 78.3604
# Min Lat: 11.4843, Max Lat: 14.1690

lon1, lat1 = 75.5382, 11.4843  # Bottom-Left
lon2, lat2 = 78.3604, 11.4843  # Bottom-Right
lon3, lat3 = 78.3604, 14.1690  # Top-Right
lon4, lat4 = 75.5382, 14.1690  # Top-Left

aoi = {
    "type": "Polygon",
    "coordinates": [[
        [lon1, lat1],
        [lon2, lat2],
        [lon3, lat3],
        [lon4, lat4],
        [lon1, lat1]
    ]]
}
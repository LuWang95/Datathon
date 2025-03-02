import pandas as pd
import geopandas as gpd
from geodatasets import get_path
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point, Polygon

def find_avg(df):
    avgs = {}
    for i in range(1, 26):
        if i < 10:
            temp_str = '0' + str(i)
        else:
            temp_str = str(i)
        temp = df[df["AREA_SHORT_CODE"] == temp_str]
        avgs[temp_str] = temp["price"].mean()
    return avgs

# read geojson, filter, and flatten from 3D to 2D
map_df = gpd.read_file('City Wards Data - 4326.geojson')
map_df = map_df[['AREA_SHORT_CODE', 'geometry']]

# import data files, filter, and combine into one dataframe
sdss_data = pd.read_csv('real-estate-data.csv')
sdss_data = sdss_data[(sdss_data['beds'] == 1) & (sdss_data['baths'] == 1)]
own_data = pd.read_csv('properties.csv')
own_data = own_data[own_data['Address'].str.contains("Toronto, ON")]
own_df = own_data[['Price ($)', 'lat', 'lng']]
sd_df = sdss_data[['price', 'lt', 'lg']]
own_df.columns = ['price', 'lt', 'lg']
# df = pd.concat([own_df, sd_df], axis=0)
df = sd_df

# converet into geodataframe (lat lon into points)
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['lg'], df['lt']), crs="EPSG:4326")
gdf = gdf[['price', 'geometry']]

combined = gpd.sjoin(map_df, gdf)

map_df = map_df.sort_values('AREA_SHORT_CODE')

map_df['mean'] = find_avg(combined).values()

# make map
fig, ax = plt.subplots(1, 1, figsize=(12, 10))
map_df.plot(
    column="mean",
    ax=ax,
    cmap="OrRd",
    legend=True,
    legend_kwds={"label": "Mean Price (CAD $)", "orientation": "vertical"},
    missing_kwds={"color": "lightgrey", "edgecolor": "red", "label": "No data"}
)
ax.set_title("Heat Map of House Prices by Ward in Toronto\n(Combined Data: real_estate + properties)", fontsize=14)
ax.set_axis_off()
plt.show()

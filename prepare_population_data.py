import osmnx as ox
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point, box
from functions import project_nxpos, initplot, nx_to_ig, greedy_triangulation_routing, mst_routing, fill_holes, extract_relevant_polygon, osm_to_ig, nxdraw, ox_to_csv,csv_to_ox, csv_to_ig
import igraph as ig
import networkx as nx
import shapely
import numpy as np 
from matplotlib import cm
import pandas as pd
from tqdm.notebook import tqdm
from pathlib import Path
from matplotlib.patches import Polygon as MplPolygon

data_pth = Path(".\Data")
network_data_pth = data_pth/"network_data"
poi_data_pth = data_pth/"poi_data"
population_data_pth = data_pth/"population_data"
population_density_file = Path(".\population_density_ro.csv")

placeid = "Bucharest"
place_name = "Bucharest, Romania"
location = ox.geocoder.geocode_to_gdf(place_name)
location = fill_holes(extract_relevant_polygon(place_name, shapely.geometry.shape(location['geometry'][0])))
G_carall = csv_to_ox(network_data_pth, "Bucharest", "carall")
largest_cc = max(nx.weakly_connected_components(G_carall), key=len)
G_carall_largest = G_carall.subgraph(largest_cc).copy()
G_carall_largest.graph['crs'] = 'EPSG:4326'

pd_df = pd.read_csv(population_density_file)
pd_df.columns = ["x", "y", "density"]
grid_centers = pd_df.to_dict(orient="records")
grid_centers_gdf = gpd.GeoDataFrame(
    grid_centers, 
    geometry=[Point(d["x"], d["y"]) for d in grid_centers], 
    crs="EPSG:4326"
)
grid_centers_gdf["is_within"] = grid_centers_gdf.geometry.within(location)
bucharest_grid_centers = grid_centers_gdf[grid_centers_gdf['is_within'] == True]
bucharest_grid_centers_nnids = [ox.distance.nearest_nodes(G_carall_largest, row.geometry.x, row.geometry.y) for _, row in bucharest_grid_centers.iterrows()]
with open(population_data_pth/f'Bucharest_pdgridcenters_nnidscarall.csv', 'w') as f:
    for item in bucharest_grid_centers_nnids:
        f.write("%s\n" % item)
bucharest_grid_centers["osmid"] = bucharest_grid_centers_nnids
bucharest_grid_centers.to_csv(population_data_pth/"Bucharest_population_density_centers_nodes.csv")


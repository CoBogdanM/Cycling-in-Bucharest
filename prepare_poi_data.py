import osmnx as ox
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
from functions import nx_to_ig, greedy_triangulation_routing, mst_routing, fill_holes, extract_relevant_polygon, osm_to_ig, nxdraw, ox_to_csv,csv_to_ox
import igraph as ig
import networkx as nx
import shapely
import numpy as np 
from matplotlib import cm
import pandas as pd
from tqdm.notebook import tqdm
from pathlib import Path

data_pth = Path(".\Data")
network_data_pth = data_pth/"network_data"
poi_data_pth = data_pth/"poi_data"


place_name = "Bucharest, Romania"
G_carall = csv_to_ox(network_data_pth, "Bucharest", "carall")
largest_cc = max(nx.weakly_connected_components(G_carall), key=len)
G_carall_largest = G_carall.subgraph(largest_cc).copy()
G_carall_largest.graph['crs'] = 'EPSG:4326'


top_stations = ["Gara de Nord 1", "Basarab 1 - M1", "Obor", "Piața Sudului", "Titan", "Piața Unirii 1", "Dristor 1", "Eroilor", "Crângași", "Piața Victoriei 1", "Eroii Revoluției", "Politehnica", "Lujerului", "București Progresul"]

metro_stations_gdf = ox.features_from_place(
    place_name, tags={"railway": "station", "subway": "yes"}
)
metro_stations_gdf = metro_stations_gdf[metro_stations_gdf["subway"] == "yes"]
metro_stations_gdf = metro_stations_gdf[metro_stations_gdf.geometry.apply(lambda geom: isinstance(geom, Point))]
metro_stations_gdf = metro_stations_gdf[metro_stations_gdf["public_transport"] == "station"]
metro_stations_gdf = metro_stations_gdf[metro_stations_gdf["name"].isin(top_stations)]

points_of_interest = [
    {"name": "Bucuresti Progresul", "lat": 44.365661, "lon": 26.091669},
    {"name": "Pipera", "lat": 44.50585, "lon": 26.13702},
    {"name": "Baneasa", "lat": 44.49456, "lon": 26.07914},
    #{"name": "Piata Victoriei", "lat": 44.45247, "lon": 26.08583},
    #{"name": "Politehnica", "lat": 44.44437, "lon": 26.05265},
    {"name": "Jiului", "lat": 44.48249, "lon": 26.04104},
    {"name": "Palatul Parlamentului", "lat": 44.42754, "lon": 26.08785},
    {"name": "Chiajna", "lat": 44.45782, "lon": 25.97450},
    {"name": "Institutul de fizica", "lat": 44.348957, "lon": 26.03074},
    {"name": "NordEst Logistic Park", "lat": 44.48592, "lon": 26.21884},
    {"name": "Pantelimon", "lat": 44.464273, "lon": 26.208355},
    {"name": "Icme Ecab", "lat": 44.42034, "lon": 26.21877},
    {"name": "Danubiana", "lat": 44.36317, "lon": 26.19406},
    {"name": "Anticorosiv", "lat":44.406449, "lon":26.201620},
    {"name": "Statie epurare", "lat": 44.394378, "lon":26232730},
    {"name": "Vulcan SA", "lat": 44.358109, "lon":26.140956},
    {"name": "Depozit Petrolier Petrom", "lat": 44.341158, "lon":26.091196},
    {"name": "Universitatea din Bucuresti", "lat": 44.43553, "lon": 26.10222},
    {"name": "Universitatea Politehnica", "lat": 44.43855, "lon": 26.04958},
    {"name": "ASE", "lat": 44.44475, "lon": 26.09778},
    {"name": "Bucharest Mall", "lat":44.419860, "lon":26.126013},
    #{"name": "Sun Plaza", "lat": 44.395257, "lon":26.121031},
    {"name": "AFI PALACE", "lat": 44.43099, "lon": 26.05433},
    {"name": "Baneasa Shopping City", "lat": 44.50794, "lon": 26.09133},
    {"name": "Plaza Romania", "lat": 44.42854, "lon": 26.03352},
]

point_of_interest_gdf = gpd.GeoDataFrame(
    points_of_interest, 
    geometry=[Point(d["lon"], d["lat"]) for d in points_of_interest], 
    crs="EPSG:4326"
)

metro_stations_nodes = [ox.distance.nearest_nodes(G_carall_largest, row.geometry.x, row.geometry.y) for _, row in metro_stations_gdf.iterrows()]
poi_nodes = [ox.distance.nearest_nodes(G_carall_largest, d["lon"], d["lat"]) for d in points_of_interest]
nnids = metro_stations_nodes + poi_nodes
with open(poi_data_pth/f'Bucharest_poi_nnidscarall.csv', 'w') as f:
    for item in nnids:
        f.write("%s\n" % item)
print(nnids)
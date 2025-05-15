import osmnx as ox
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
from functions import fill_holes, extract_relevant_polygon, ox_to_csv, csv_to_ig
import igraph as ig
import networkx as nx
import shapely
import numpy as np 
from matplotlib import cm
import pandas as pd
from tqdm.notebook import tqdm
from pathlib import Path
from haversine import haversine, haversine_vector
import pickle

data_pth = Path(".\Data")
network_data_pth = data_pth/"network_data"
poi_data_pth = data_pth/"poi_data"
result_pth = data_pth/"results"
plots_pth = data_pth/"plots"

placeid = "Bucharest"
poi_source = "PlanBucuresti"


G_carall = csv_to_ig(network_data_pth, placeid, "carall")
routes_layers = ["existing_routes","riders_preferences_routes", "transport_hubs_routes", "employment_hubs_routes", "commercial_hubs_routes", "connectivity_routes"]
Gs_layer = []
for layer in routes_layers:
        G = csv_to_ig(network_data_pth, placeid, f"SUMP_{layer}")
        G.vs["name"] = [str(id) for id in G.vs["id"]]
        Gs_layer.append(G)
        G_simplified = csv_to_ig(network_data_pth, placeid, f"SUMP_{layer}_simplified")
        G_simplified.vs["name"] = [str(id) for id in G_simplified.vs["id"]]

Gs_bikenetwork = []
for i in range(len(Gs_layer)):
    G_final = ig.union(Gs_layer[:i+1], byname=True)
    Gs_bikenetwork.append(G_final)

G_SUMP = Gs_bikenetwork[-1]

with open(result_pth/"Bucharest_poi_SUMP_6quantiles_Bq.pickle", 'rb') as f:
    res = pickle.load(f)

GT = res["GTs"][-1]
MST = res["MST"]

print(f"Street network: number of nodes: {len(G_carall.vs)}, number of edges: {len(G_carall.es)}")
print(f"SUMP network: number of nodes: {len(G_SUMP.vs)}, number of edges: {len(G_SUMP.es)}")
print(f"GT network: number of nodes: {len(GT.vs)}, number of edges: {len(GT.es)}")
print(f"MST network: number of nodes: {len(MST.vs)}, number of edges: {len(MST.es)}")


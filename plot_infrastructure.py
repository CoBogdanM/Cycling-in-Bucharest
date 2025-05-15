import osmnx as ox
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
from functions import nx_to_ig, greedy_triangulation_routing, mst_routing, fill_holes, extract_relevant_polygon, osm_to_ig, nxdraw, ox_to_csv,csv_to_ox, csv_to_ig, write_result, nodesize_from_pois, initplot
import igraph as ig
import networkx as nx
import shapely
import numpy as np 
from matplotlib import cm
import pandas as pd
from tqdm.notebook import tqdm
from pathlib import Path
import pickle




def plot_street_network(network_data_pth, plots_pth, placeid, map_center, plotparam):
    G_carall = csv_to_ig(network_data_pth, placeid, "carall")
    fig = initplot()
    nxdraw(G_carall, "carall", map_center)
    plt.savefig(plots_pth/f'{placeid}_carall.pdf', bbox_inches="tight")
    plt.savefig(plots_pth/f'{placeid}_carall.png', bbox_inches="tight", dpi=plotparam["dpi"])
    plt.close()

def plot_bicycle_network_plan(network_data_pth, plots_pth, placeid, map_center, plotparam, routes_layers,nnids, nodesize_poi ,plot_all = True):
    G_carall = csv_to_ig(network_data_pth, placeid, "carall")
    G_layers = {}
    for layer in routes_layers:
        G_layers[layer] = csv_to_ig(network_data_pth, placeid, f"SUMP_{layer}")
    fig = initplot()
    nxdraw(G_carall, "carall", map_center)
    i = 1
    for layer in routes_layers:
        nxdraw(G_layers[layer], f"SUMP_{layer}", map_center)
        if plot_all:
            plt.savefig(plots_pth/f'{placeid}_SUMP_{i}.pdf', bbox_inches="tight")
            plt.savefig(plots_pth/f'{placeid}_SUMP_{i}.png', bbox_inches="tight", dpi=plotparam["dpi"])
            i += 1
        else:
            plt.savefig(plots_pth/f'{placeid}_SUMP_{layer}.pdf', bbox_inches="tight")
            plt.savefig(plots_pth/f'{placeid}_SUMP_{layer}.png', bbox_inches="tight", dpi=plotparam["dpi"])
            plt.close()
            fig = initplot()
            nxdraw(G_carall, "carall", map_center)
    plt.close()


def plot_pois_on_G_carall(network_data_pth, plots_pth, placeid, poi_source, nodesize_poi, map_center, nnids, plotparam):
    G_carall = csv_to_ig(network_data_pth, placeid, "carall")
    fig = initplot()
    nxdraw(G_carall, "carall", map_center)
    nxdraw(G_carall, "poi_unreached", map_center, nnids, "nx.draw_networkx_nodes", nodesize_poi)
    plt.savefig(plots_pth/f'{placeid}_carall_poi_{poi_source}.pdf', bbox_inches="tight")
    plt.savefig(plots_pth/f'{placeid}_carall_poi_{poi_source}.png', bbox_inches="tight", dpi=plotparam["dpi"])
    plt.close()

if __name__ == "__main__":
    data_pth = Path(".\Data")
    network_data_pth = data_pth/"network_data"
    poi_data_pth = data_pth/"poi_data"
    result_pth = data_pth/"results"
    plots_pth = data_pth/"plots"

    placeid = "Bucharest"
    poi_source = "SUMP"

    plotparam = {"bbox": (1280,1280),
    			"dpi": 96,
    			"carall": {"width": 0.5, "edge_color": '#999999'},
    			# "biketrack": {"width": 1.25, "edge_color": '#2222ff'},
                "biketrack": {"width": 1, "edge_color": '#000000'},
    			"biketrack_offstreet": {"width": 0.75, "edge_color": '#00aa22'},
    			"bikeable": {"width": 0.75, "edge_color": '#222222'},
    			# "bikegrown": {"width": 6.75, "edge_color": '#ff6200', "node_color": '#ff6200'},
    			# "highlight_biketrack": {"width": 6.75, "edge_color": '#0eb6d2', "node_color": '#0eb6d2'},
                "bikegrown": {"width": 3.75, "edge_color": '#0eb6d2', "node_color": '#0eb6d2'},
                "highlight_biketrack": {"width": 3.75, "edge_color": '#2222ff', "node_color": '#2222ff'},
    			"highlight_bikeable": {"width": 3.75, "edge_color": '#222222', "node_color": '#222222'},
    			"poi_unreached": {"node_color": '#ff7338', "edgecolors": '#ffefe9'},
    			"poi_reached": {"node_color": '#0b8fa6', "edgecolors": '#f1fbff'},
    			"abstract": {"edge_color": '#000000', "alpha": 0.75}
    			}
    
    with open(poi_data_pth/f'{placeid}_poi_nnidscarall.csv') as f:
        nnids = [int(line.rstrip()) for line in f]
    nodesize_poi = nodesize_from_pois(nnids)
    routes_layers = ["existing_routes","riders_preferences_routes", "transport_hubs_routes", "employment_hubs_routes", "commercial_hubs_routes", "connectivity_routes"]
    G_carall = csv_to_ig(network_data_pth, placeid, "carall")
    map_center = nxdraw(G_carall, "carall")
    plot_bicycle_network_plan(network_data_pth, plots_pth, placeid, map_center, plotparam, routes_layers,nnids, nodesize_poi,True)
    plot_pois_on_G_carall(network_data_pth, plots_pth, placeid, poi_source, nodesize_poi, map_center, nnids, plotparam)
    plot_street_network(network_data_pth, plots_pth, placeid, map_center, plotparam)
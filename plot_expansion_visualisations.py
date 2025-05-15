import osmnx as ox
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point, Polygon
from functions import create_pop_density_proj, greedy_triangulation_routing, mst_routing, fill_holes, extract_relevant_polygon, osm_to_ig, nxdraw, ox_to_csv,csv_to_ox, csv_to_ig, write_result, nodesize_from_pois, initplot, cov_to_patchlist
import igraph as ig
import networkx as nx
import shapely
import numpy as np 
from matplotlib import cm
import pandas as pd
from tqdm.notebook import tqdm
from pathlib import Path
import pickle
from matplotlib.collections import PatchCollection
import shapely.ops as ops

def plot_MST_visualizations(map_center, G_carall, res, plots_pth, plotparam, nnids, nodesize_poi, placeid, poi_source, nodesize_grown):
    # PLOT abstract MST
    fig = initplot()
    nxdraw(G_carall, "carall", map_center)
    nxdraw(res["MST_abstract"], "abstract", map_center, weighted = 6)
    nxdraw(G_carall, "poi_unreached", map_center, nnids, "nx.draw_networkx_nodes", nodesize_poi)
    nxdraw(G_carall, "poi_reached", map_center, list(set([v["id"] for v in res["MST"].vs]).intersection(set(nnids))), "nx.draw_networkx_nodes", nodesize_poi)
    plt.savefig(plots_pth/f"{placeid}_MSTabstract_poi_{poi_source}.pdf", bbox_inches="tight")
    plt.savefig(plots_pth/f"{placeid}_MSTabstract_poi_{poi_source}.png", bbox_inches="tight", dpi=plotparam["dpi"])
    plt.close()

    # PLOT MST all together
    fig = initplot()
    nxdraw(G_carall, "carall")
    nxdraw(res["MST"], "bikegrown", map_center, nodesize = nodesize_grown)
    nxdraw(G_carall, "poi_unreached", map_center, nnids, "nx.draw_networkx_nodes", nodesize_poi)
    nxdraw(G_carall, "poi_reached", map_center, list(set([v["id"] for v in res["MST"].vs]).intersection(set(nnids))), "nx.draw_networkx_nodes", nodesize_poi)
    plt.savefig(plots_pth/f"{placeid}_MSTall_poi_{poi_source}.pdf", bbox_inches="tight")
    plt.savefig(plots_pth/f"{placeid}_MSTall_poi_{poi_source}.png", bbox_inches="tight", dpi=plotparam["dpi"])
    plt.close()

    # PLOT MST all together with abstract
    fig = initplot()
    nxdraw(G_carall, "carall", map_center)
    nxdraw(res["MST"], "bikegrown", map_center, nodesize = 0)
    nxdraw(res["MST_abstract"], "abstract", map_center, weighted = 6)
    nxdraw(G_carall, "poi_unreached", map_center, nnids, "nx.draw_networkx_nodes", nodesize_poi)
    nxdraw(G_carall, "poi_reached", map_center, list(set([v["id"] for v in res["MST"].vs]).intersection(set(nnids))), "nx.draw_networkx_nodes", nodesize_poi)
    plt.savefig(plots_pth/f"{placeid}_MSTabstractall_poi_{poi_source}.pdf", bbox_inches="tight")
    plt.savefig(plots_pth/f"{placeid}_MSTabstractall_poi_{poi_source}.png", bbox_inches="tight", dpi=plotparam["dpi"])
    plt.close()

def plot_GT_visualizations(map_center, G_carall, res, plots_pth, plotparam, nnids, nodesize_poi, placeid, poi_source, prune_measure, weight_abstract, nodesize_grown):
    # PLOT abstract greedy triangulation (this can take some minutes)
    plots_pth = plots_pth/f"{len(res["prune_quantiles"])}quantiles"
    for GT_abstract, prune_quantile in tqdm(zip(res["GT_abstracts"], res["prune_quantiles"]), "Abstract triangulation", total=len(res["prune_quantiles"])):
        fig = initplot()
        nxdraw(G_carall, "carall")
        try:
            GT_abstract.es["weight"] = GT_abstract.es["width"]
        except:
            pass
        nxdraw(GT_abstract, "abstract", map_center, drawfunc = "nx.draw_networkx_edges", nodesize = 0, weighted = weight_abstract, maxwidthsquared = nodesize_poi)
        nxdraw(G_carall, "poi_unreached", map_center, nnids, "nx.draw_networkx_nodes", nodesize_poi)
        nxdraw(G_carall, "poi_reached", map_center, list(set([v["id"] for v in GT_abstract.vs]).intersection(set(nnids))), "nx.draw_networkx_nodes", nodesize_poi)
        plt.savefig(plots_pth/f"{placeid}_GTabstract_poi_{poi_source}_{prune_measure}{prune_quantile:.3f}.png", bbox_inches="tight", dpi=plotparam["dpi"])
        plt.close()

    # PLOT all together (this can take some minutes)
    for GT, prune_quantile in tqdm(zip(res["GTs"], res["prune_quantiles"]), "Routed triangulation", total=len(res["prune_quantiles"])):
        fig = initplot()
        nxdraw(G_carall, "carall")
        nxdraw(GT, "bikegrown", map_center, nodesize = nodesize_grown)
        nxdraw(G_carall, "poi_unreached", map_center, nnids, "nx.draw_networkx_nodes", nodesize_poi)
        nxdraw(G_carall, "poi_reached", map_center, list(set([v["id"] for v in GT.vs]).intersection(set(nnids))), "nx.draw_networkx_nodes", nodesize_poi)
        plt.savefig(plots_pth/f"{placeid}_GTall_poi_{poi_source}_{prune_measure}{prune_quantile:.3f}.png", bbox_inches="tight", dpi=plotparam["dpi"])
        plt.close()

def plot_GT_covs(plots_pth, map_center, G_carall, placeid, nnids, nodesize_poi, res, covs, cov_car, prune_measure, squares, pc_population = None):
    plots_pth = plots_pth/f"{len(res["prune_quantiles"])}quantiles"
    # Construct and plot patches from covers
    patchlist_car, patchlist_car_holes = cov_to_patchlist(cov_car, map_center)
    for GT, prune_quantile, cov in tqdm(zip(res["GTs"], res["prune_quantiles"], covs.values()), "Covers", total=len(res["prune_quantiles"])):
        fig = initplot()
        pc_pop = PatchCollection(pc_population[0])
        pc_pop.set_facecolors(pc_population[1])
        # Covers
        axes = fig.add_axes([0, 0, 1, 1]) # left, bottom, width, height (range 0 to 1)
        patchlist_bike, patchlist_bike_holes = cov_to_patchlist(cov, map_center)
        
        # We have this contrived order due to alphas, holes, and matplotlib's inability to draw polygon patches with holes. This only works because the car network is a superset of the bike network.
        # car orange, bike white, bike blue, bike holes white, bike holes orange, car holes white
        if not pc_population:
            patchlist_combined = patchlist_car + patchlist_bike + patchlist_bike + patchlist_bike_holes+ patchlist_bike_holes + patchlist_car_holes
            colors = np.array([[255/255,115/255,56/255,0.2] for _ in range(len(patchlist_car))]) # car orange
        else:
            patchlist_combined = patchlist_bike + patchlist_bike
            colors = np.empty((0, 4))
        pc = PatchCollection(patchlist_combined)
        if len(patchlist_bike):
            if not pc_population:
                colors = np.append(colors, [[1,1,1,1] for _ in range(len(patchlist_bike))], axis = 0) # bike white
                colors = np.append(colors, [[86/255,220/255,244/255,0.4] for _ in range(len(patchlist_bike))], axis = 0) # bike blue
            else:
                colors = np.append(colors, [[86/255,220/255,244/255,0] for _ in range(len(patchlist_bike))], axis = 0) # bike blue
        if len(patchlist_bike_holes):
            if not pc_population:
                colors = np.append(colors, [[1,1,1,1] for _ in range(len(patchlist_bike_holes))], axis = 0) # bike holes white
        if len(patchlist_bike_holes):
            if not pc_population:
                colors = np.append(colors, [[255/255,115/255,56/255,0.2] for _ in range(len(patchlist_bike_holes))], axis = 0) # bike holes orange
        if len(patchlist_car_holes):
            if not pc_population:
                colors = np.append(colors, [[1,1,1,1] for _ in range(len(patchlist_car_holes))], axis = 0) # car holes white
        pc.set_facecolors(colors)
        pc.set_edgecolors(np.array([[0,0,0,0.4] for _ in range(len(patchlist_combined))])) # remove this line if the outline of the full coverage should remain
        axes.add_collection(pc)
        if pc_population:
            axes.add_collection(pc_pop)
        axes.set_aspect('equal')
        axes.set_xmargin(0.01)
        axes.set_ymargin(0.01)
        axes.plot()
        # Networks
        nxdraw(G_carall, "carall", map_center)
        nxdraw(GT, "bikegrown", map_center, nodesize = nodesize_grown)
        if not pc_population:
            nxdraw(G_carall, "poi_unreached", map_center, nnids, "nx.draw_networkx_nodes", nodesize_poi)
            nxdraw(G_carall, "poi_reached", map_center, list(set([v["id"] for v in GT.vs]).intersection(set(nnids))), "nx.draw_networkx_nodes", nodesize_poi)
        plt.savefig(plots_pth/f"{placeid}_GTallcover_poi_{poi_source}_{prune_measure}{prune_quantile:.3f}.png", bbox_inches="tight", dpi=plotparam["dpi"])
        plt.close()

def plot_MST_covs(plots_pth, map_center, G_carall, placeid, nnids, nodesize_poi, res, covs, cov_car, prune_measure,squares ,pc_population = None):
    # Construct and plot patches from covers
    MST = res["MST"]
    cov = covs
    patchlist_car, patchlist_car_holes = cov_to_patchlist(cov_car, map_center)
    fig = initplot()
    pc_pop = PatchCollection(pc_population[0])
    pc_pop.set_facecolors(pc_population[1])
    # Covers
    axes = fig.add_axes([0, 0, 1, 1]) # left, bottom, width, height (range 0 to 1)
    patchlist_bike, patchlist_bike_holes = cov_to_patchlist(cov, map_center)
    
    # We have this contrived order due to alphas, holes, and matplotlib's inability to draw polygon patches with holes. This only works because the car network is a superset of the bike network.
    # car orange, bike white, bike blue, bike holes white, bike holes orange, car holes white
    if not pc_population:
        patchlist_combined = patchlist_car + patchlist_bike + patchlist_bike + patchlist_bike_holes+ patchlist_bike_holes + patchlist_car_holes
        colors = np.array([[255/255,115/255,56/255,0.2] for _ in range(len(patchlist_car))]) # car orange
    else:
        patchlist_combined = patchlist_bike + patchlist_bike + patchlist_bike_holes+ patchlist_bike_holes + patchlist_car_holes
        colors = np.empty((0, 4))
    pc = PatchCollection(patchlist_combined)
    if len(patchlist_bike):
        if not pc_population:
            colors = np.append(colors, [[1,1,1,1] for _ in range(len(patchlist_bike))], axis = 0) # bike white
            colors = np.append(colors, [[86/255,220/255,244/255,0.4] for _ in range(len(patchlist_bike))], axis = 0) # bike blue
        else:
            colors = np.append(colors, [[86/255,220/255,244/255,0] for _ in range(len(patchlist_bike))], axis = 0) # bike blue
    if len(patchlist_bike_holes) and not pc_population:
        colors = np.append(colors, [[1,1,1,1] for _ in range(len(patchlist_bike_holes))], axis = 0) # bike holes white
    if len(patchlist_bike_holes) and not pc_population:
        colors = np.append(colors, [[255/255,115/255,56/255,0.2] for _ in range(len(patchlist_bike_holes))], axis = 0) # bike holes orange
    if len(patchlist_car_holes) and not pc_population:
        colors = np.append(colors, [[1,1,1,1] for _ in range(len(patchlist_car_holes))], axis = 0) # car holes white
    pc.set_facecolors(colors)
    pc.set_edgecolors(np.array([[0,0,0,0.4] for _ in range(len(patchlist_combined))])) # remove this line if the outline of the full coverage should remain
    axes.add_collection(pc)
    if pc_population:
            axes.add_collection(pc_pop)

    axes.set_aspect('equal')
    axes.set_xmargin(0.01)
    axes.set_ymargin(0.01)
    axes.plot() 
    # Networks
    nxdraw(G_carall, "carall", map_center)
    nxdraw(MST, "bikegrown", map_center, nodesize = nodesize_grown)
    if not pc_population:
        nxdraw(G_carall, "poi_unreached", map_center, nnids, "nx.draw_networkx_nodes", nodesize_poi)
        nxdraw(G_carall, "poi_reached", map_center, list(set([v["id"] for v in MST.vs]).intersection(set(nnids))), "nx.draw_networkx_nodes", nodesize_poi)
    plt.savefig(plots_pth/f"{placeid}_MSTcovers_{poi_source}_{prune_measure}.png", bbox_inches="tight", dpi=plotparam["dpi"])
    plt.close()

def plot_SUMP_covs(plots_pth,map_center, G_carall, placeid, nnids, nodesize_poi, Gs_sump, covs, cov_car, prune_measure, prune_quantiles,squares ,pc_population = None):

    # Construct and plot patches from covers
    patchlist_car, patchlist_car_holes = cov_to_patchlist(cov_car, map_center)
    for SUMP, prune_quantile, cov in tqdm(zip(Gs_sump, prune_quantiles, covs.values()), "Covers", total=len(prune_quantiles)):
        fig = initplot()
        pc_pop = PatchCollection(pc_population[0])
        pc_pop.set_facecolors(pc_population[1])
        # Covers
        axes = fig.add_axes([0, 0, 1, 1]) # left, bottom, width, height (range 0 to 1)
        patchlist_bike, patchlist_bike_holes = cov_to_patchlist(cov, map_center)
        
        # We have this contrived order due to alphas, holes, and matplotlib's inability to draw polygon patches with holes. This only works because the car network is a superset of the bike network.
        # car orange, bike white, bike blue, bike holes white, bike holes orange, car holes white
        if not pc_population:
            patchlist_combined = patchlist_car + patchlist_bike + patchlist_bike + patchlist_bike_holes+ patchlist_bike_holes + patchlist_car_holes
            colors = np.array([[255/255,115/255,56/255,0.2] for _ in range(len(patchlist_car))]) # car orange
        else:
            patchlist_combined = patchlist_bike + patchlist_bike + patchlist_bike_holes+ patchlist_bike_holes + patchlist_car_holes
            colors = np.empty((0, 4))
        pc = PatchCollection(patchlist_combined)
        if len(patchlist_bike):
            if not pc_population:
                colors = np.append(colors, [[1,1,1,1] for _ in range(len(patchlist_bike))], axis = 0) # bike white
                colors = np.append(colors, [[86/255,220/255,244/255,0.4] for _ in range(len(patchlist_bike))], axis = 0) # bike blue
            else:
                 colors = np.append(colors, [[86/255,220/255,244/255,0] for _ in range(len(patchlist_bike))], axis = 0) # bike blue
        if len(patchlist_bike_holes) and not pc_population:
            colors = np.append(colors, [[1,1,1,1] for _ in range(len(patchlist_bike_holes))], axis = 0) # bike holes white
        if len(patchlist_bike_holes) and not pc_population:
            colors = np.append(colors, [[255/255,115/255,56/255,0.2] for _ in range(len(patchlist_bike_holes))], axis = 0) # bike holes orange
        if len(patchlist_car_holes) and not pc_population:
            colors = np.append(colors, [[1,1,1,1] for _ in range(len(patchlist_car_holes))], axis = 0) # car holes white
        pc.set_facecolors(colors)
        pc.set_edgecolors(np.array([[0,0,0,0.4] for _ in range(len(patchlist_combined))])) # remove this line if the outline of the full coverage should remain
        if pc_population:
            axes.add_collection(pc_pop)
        axes.add_collection(pc)
        axes.set_aspect('equal')
        axes.set_xmargin(0.01)
        axes.set_ymargin(0.01)
        axes.plot()
        pois_indices = set()
        for poi in nnids:
            pois_indices.add(G_carall.vs.find(id = poi).index)

        poiscovered = []
        for poi in pois_indices:
            v = G_carall.vs[poi]
            if Point(v["x"], v["y"]).within(cov):
                poiscovered.append(v["id"])
        
        # Networks
        nxdraw(G_carall, "carall", map_center)
        nxdraw(SUMP, "bikegrown", map_center, nodesize = nodesize_grown)
        if not pc_population:
            nxdraw(G_carall, "poi_unreached", map_center, nnids, "nx.draw_networkx_nodes", nodesize_poi)
            nxdraw(G_carall, "poi_reached", map_center, poiscovered, "nx.draw_networkx_nodes", nodesize_poi)
        plt.savefig(plots_pth/f"{placeid}_SUMPcover_poi_{poi_source}_{prune_measure}{prune_quantile:.3f}.png", bbox_inches="tight", dpi=plotparam["dpi"])
        plt.close()

def plot_population_density_squares(plots_pth, squares, map_center, G_carall, placeid, poi_source, prune_measure):
    fig = initplot()
    squares_densities = []
    for square in squares:
        squares_densities.append(squares[square]['density'])
    squares_densities = sorted(squares_densities)
    q1 = np.percentile(squares_densities, 33)
    q2 = np.percentile(squares_densities, 66)
    max = squares_densities[-1]
    colors = np.empty((0, 4))
    for i, square in enumerate(squares):
        patchlist_population, patchlist_population_holes = cov_to_patchlist(squares[square]["square_wgs84"], map_center)
        if i == 0:
            all_patchlist = patchlist_population
        else:
            all_patchlist += patchlist_population
        if squares[square]['density'] < q1:
            colors = np.append(colors, [[255/255, 0/255, 0/255, 0.1]], axis = 0) # car orange
        elif squares[square]['density'] < q2:
            colors = np.append(colors, [[255/255, 255/255, 0/255, 0.1]], axis = 0) # car orange
        elif squares[square]['density'] < max:
            colors = np.append(colors, [[0/255, 255/255, 0/255, 0.1]], axis = 0)
    pc_population = PatchCollection(all_patchlist)
    pc_population.set_facecolors(colors)
    #pc.set_edgecolors(np.array([[0,0,0,0.4] for _ in range(len(all_patchlist))]))
    axes = fig.add_axes([0, 0, 1, 1]) # left, bottom, width, height (range 0 to 1)
    axes.add_collection(pc_population)
    axes.set_aspect('equal')
    axes.set_xmargin(0.01)
    axes.set_ymargin(0.01)
    axes.plot()
    nxdraw(G_carall, "carall", map_center)
    plt.savefig(plots_pth/f"population_density_units.png", bbox_inches="tight", dpi=plotparam["dpi"])
    return all_patchlist, colors

if __name__ == "__main__":
    data_pth = Path(".\Data")
    network_data_pth = data_pth/"network_data"
    poi_data_pth = data_pth/"poi_data"
    result_pth = data_pth/"results"
    plots_pth = data_pth/"plots"
    analysis_results_pth = data_pth/"analysis_results"
    population_data_pth = data_pth/"population_data"

    placeid = "Bucharest"
    poi_source = "SUMP"


    prune_measure = "Bq"
    prune_quantiles = [[x/40 for x in list(range(1, 41))], [x/6 for x in list(range(1, 7))]]
    if prune_measure == "Bq":
        weight_abstract = True
    else:
        weight_abstract = 6

    #PLOT_PARAMETERS
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
    nodesize_grown = 7.5

    G_carall = csv_to_ig(network_data_pth, "Bucharest", "carall")
    map_center = nxdraw(G_carall, "carall")
    
    with open(poi_data_pth/f'Bucharest_poi_nnidscarall.csv') as f:
        nnids = [int(line.rstrip()) for line in f]
    nodesize_poi = nodesize_from_pois(nnids)
    
    results = []
    for pq in prune_quantiles:
        with open(result_pth/f"{placeid}_poi_{poi_source}_{len(pq)}quantiles_{prune_measure}.pickle", 'rb') as f:
            res = pickle.load(f)
            results.append(res)
    plot_MST_visualizations(map_center, G_carall, res, plots_pth, plotparam, nnids, nodesize_poi, placeid, poi_source, nodesize_grown)

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
    prune_quantiles_SUMP = [x/6 for x in list(range(1,7))]
    # Load covers
    
    with open(analysis_results_pth/f"{placeid}_poi_{poi_source}_Bq_covers.pickle",'rb') as f:
        covs_GT = pickle.load(f)
    with open(analysis_results_pth/f"{placeid}_poi_{poi_source}_Bq_cover_mst.pickle",'rb') as f:
        covs_MST = pickle.load(f)
    with open(analysis_results_pth/f"{placeid}_poi_{poi_source}_Bq_covers_SUMP.pickle",'rb') as f:
        covs_SUMP = pickle.load(f)
    with open(analysis_results_pth/f"{placeid}_carall_covers.pickle",'rb') as f:
        cov_car = pickle.load(f)['carall']
    
    G_population_centers = csv_to_ig(population_data_pth, placeid, 'population_density_centers')
    squares = create_pop_density_proj(G_carall, G_population_centers, 500)
    pc_population = plot_population_density_squares(plots_pth, squares, map_center, G_carall, placeid, poi_source, prune_measure)
    for res in results:
        plot_GT_visualizations(map_center, G_carall, res, plots_pth, plotparam, nnids, nodesize_poi, placeid, poi_source, prune_measure, weight_abstract, nodesize_grown)
        plot_GT_covs(plots_pth, map_center, G_carall, placeid, nnids,nodesize_poi, res, covs_GT, cov_car, prune_measure,squares ,pc_population)
    plot_MST_covs(plots_pth, map_center, G_carall, placeid, nnids, nodesize_poi, res, covs_MST, cov_car, prune_measure,squares ,pc_population)
    plot_SUMP_covs(plots_pth,map_center, G_carall, placeid, nnids, nodesize_poi, Gs_bikenetwork, covs_SUMP, cov_car, prune_measure, prune_quantiles_SUMP,squares ,pc_population) 
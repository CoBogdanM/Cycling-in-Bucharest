
import osmnx as ox
import matplotlib.pyplot as plt
import geopandas as gpd
from functions import write_result, calculate_metrics, csv_to_ig, calculate_metrics_additively, set_analysissubplot, create_pop_density_proj, initplot, nxdraw, fill_holes, extract_relevant_polygon, ox_to_csv
import igraph as ig
import networkx as nx
import shapely
import numpy as np 
from matplotlib import cm
import pandas as pd
from tqdm.notebook import tqdm
from pathlib import Path
from matplotlib.patches import Polygon as MplPolygon
import copy
import os
import itertools
import pickle
from matplotlib.ticker import MaxNLocator
from shapely.geometry import Point


def analyse_SUMP(Gs,G_existing ,G_population_centers,nnids, analysis_results_pth, placeid):
    population_squares = create_pop_density_proj(G_carall, G_population_centers, 500)
    empty_metrics = {
                     "length":0,
                     "length_lcc":0,
                     "coverage": 0,
                     "poi_coverage": 0,
                     "components": 0,
                     "directness_all_linkwise": 0,
                     "directness_all_linkwise_reachability": 0,
                     "population_coverage" : 0,
                     "directness_bicycle_car" : 0,
                     "directness_bicycle_car_reachability": 0
                    }
    output_place = {}
    output_place["carall"] = copy.deepcopy(empty_metrics)  

    covs = {}
    try:
        print("ANALYSING G_CARALL")
        metrics, cov = calculate_metrics(Gs["carall"], Gs["carall_simplified"], Gs['carall'],population_squares,nnids, empty_metrics)
        for key, val in metrics.items():
            output_place["carall"][key] = val
        covs["carall"] = cov
    except Exception as e:
        print("carall" + " is empty")
        raise(e)
    write_result(analysis_results_pth,covs, "pickle", placeid, "", "", "carall_covers.pickle")
    write_result(analysis_results_pth, output_place, "dictnested", placeid, "", "", "carall.csv", empty_metrics)

    prune_quantiles = range(len(Gs["bikenetwork"]))
    empty_metrics = {
                     "length":[],
                     "length_lcc":[],
                     "coverage": [],
                     "poi_coverage": [],
                     "components": [],
                     "directness_all_linkwise": [],
                     "directness_all_linkwise_reachability": [],
                     "population_coverage" : [],
                     "directness_bicycle_car" : [],
                     "directness_bicycle_car_reachability": []
                    }
    print("ANLYSING SUMP NETWORK")
    output, covs = calculate_metrics_additively(Gs["bikenetwork"], Gs["bikenetwork_simplified"],population_squares,prune_quantiles, Gs["carall"], nnids, return_cov = True, Gexisting=G_existing, output=empty_metrics)
    
    #Manully solve double edges for same street that is affecting the length
    false_one_way_streets = ['Strada Brașov' , 'Calea Dorobanților', 'Șoseaua Mihai Bravu',  'Șoseaua Colentina', 'Drumul Taberei', 'Bulevardul Nicolae Bălcescu', 'Pasaj Băneasa', 'Bulevardul Gloriei', 'Bulevardul Iuliu Maniu', 'Bulevardul Unirii', 'Strada Liviu Rebreanu', 'Calea 13 Septembrie', 'Bulevardul Geniului', 'Strada Barbu Văcărescu', 'Bulevardul Dimitrie Cantemir', 'Șoseaua Grozăvești', 'Strada Sergent Nițu Vasile', 'Bulevardul Camil Ressu', 'Podul Ciurel', 'Bulevardul Lascăr Catargiu', 'Bulevardul Alexandru Obregia', 'Bulevardul Doina Cornea', 'Bulevardul Profesor Doctor Gheorghe Marinescu', 'Strada Turda', 'Șoseaua Pipera', 'Strada Căpitan Aviator Alexandru Șerbănescu','Strada Doamna Ghica', 'Șoseaua Pantelimon', 'Podul Grant', 'Strada Berzei', 'Calea Moșilor', 'Bulevardul Iancu de Hunedoara', 'Calea Dudești', 'Pasajul Basarab', 'Bulevardul Chișinău', 'Calea Șerban Vodă', 'Bulevardul Basarabia', 'Șoseaua Giurgiului', 'Bulevardul General Gheorghe Magheru', 'Calea Griviței', 'Bulevardul Dacia',  'Șoseaua Ștefan cel Mare', 'Calea Rahovei', 'Bulevardul Nicolae Grigorescu', 'Bulevardul Tudor Vladimirescu', 'Splaiul Independenței', 'Șoseaua Alexandria', 'Bulevardul Poligrafiei', 'Șoseaua Virtuții', 'Calea Vitan',  'Șoseaua Cotroceni', 'Bulevardul Bucureștii Noi', 'Strada Răzoare', 'Piața Romană', 'Strada Buzești', 'Bulevardul Theodor Pallady',  'Bulevardul Corneliu Coposu', 'Calea Călărașilor', 'Bulevardul Timișoara', 'Bulevardul Mareșal Alexandru Averescu', 'Șoseaua Viilor', 'Șoseaua Chitilei', 'Șoseaua Berceni', 'Bulevardul Ion Constantin Brătianu', 'Șoseaua Panduri', 'Splaiul Unirii', 'Piața Victoriei']
    for i, Graph in enumerate(Gs['bikenetwork']):
        cl = Graph.clusters()
        LCC = cl.giant()
        length = 0
        length_lcc = 0
        for edge in Graph.es:
            if edge['oneway'] == True and edge["name"] in false_one_way_streets:
                length += edge["weight"]
        for edge in LCC.es:
            if edge['oneway'] == True and edge["name"] in false_one_way_streets:
                length_lcc += edge["weight"]
        output['length'][i] = output['length'][i] - length/2
        output['length_lcc'][i] = output['length_lcc'][i] - length_lcc/2
    
    write_result(analysis_results_pth, covs, "pickle", placeid, poi_source, "Bq", "_covers_SUMP.pickle")
    write_result(analysis_results_pth, output, "dict", placeid, poi_source, "Bq", "SUMP.csv")

def analyse_GT_MST(res, nnids, G_existing, G_carall, G_population_centers,analysis_results_pth, placeid, poi_source):
    empty_metrics = {
                     "length":[],
                     "length_lcc":[],
                     "coverage": [],
                     "poi_coverage": [],
                     "components": [],
                     "directness_all_linkwise": [],
                     "directness_all_linkwise_reachability": [],
                     "population_coverage" : [],
                     "directness_bicycle_car" : [],
                     "directness_bicycle_car_reachability": []
                    }
    analysis_results_pth = analysis_results_pth/f"{len(res["prune_quantiles"])}quantiles"
    population_squares = create_pop_density_proj(G_carall, G_population_centers, 500)
    print(f"ANALYSING GT for{len(res["prune_quantiles"])}quantiles")
    output, covs = calculate_metrics_additively(res["GTs"], res["GT_abstracts"],population_squares,res["prune_quantiles"], G_carall, nnids, return_cov = True, Gexisting=G_existing, output=empty_metrics)
    print("ANALYSING MST")
    output_MST, cov_MST = calculate_metrics(res["MST"], res["MST_abstract"], G_carall,population_squares,nnids, output, return_cov=True, Gexisting = G_existing)

    write_result(analysis_results_pth, covs, "pickle", placeid, poi_source, "Bq", "_covers.pickle")
    write_result(analysis_results_pth, cov_MST, "pickle", placeid, poi_source, "Bq", "_cover_mst.pickle")

    write_result(analysis_results_pth, output, "dict", placeid, poi_source, "Bq", ".csv")
    write_result(analysis_results_pth, output_MST, "dict", placeid, poi_source, "", "mst.csv")

def plot_analysis(placeid, poi_source, prune_measure, prune_quantiles, analysis_results_pth):
    keys_metrics = {"length": "Length [km]", "length_lcc": "Length of LCC [km]","coverage": "Coverage [km$^2$]","components": "Components","poi_coverage": "POIs covered","directness_all_linkwise": "Link-wise Directness", "directness_bicycle_car": "Directness bicycle-car", "population_coverage" : "Population Covered"}
    plotparam_analysis = {
			"bikegrown": {"linewidth": 3.75, "color": '#0eb6d2', "linestyle": "solid", "label": "GT network"},
			"bikegrown_abstract": {"linewidth": 3.75, "color": '#000000', "linestyle": "solid", "label": "Grown network (unrouted)", "alpha": 0.75},
			"mst": {"linewidth": 2, "color": '#0eb6d2', "linestyle": "dashed", "label": "MST network"},
			"mst_abstract": {"linewidth": 2, "color": '#000000', "linestyle": "dashed", "label": "MST (unrouted)", "alpha": 0.75},
			"biketrack": {"linewidth": 2, "color": 'red', "linestyle": "solid", "label": "SUMP network"},
			"bikeable": {"linewidth": 1, "color": '#222222', "linestyle": "dashed", "label": "Bikeable"},
			"constricted": {"linewidth": 3.75, "color": '#D22A0E', "linestyle": "solid", "label": "Street network"},
            "constricted_SI": {"linewidth": 2, "color": '#D22A0E', "linestyle": "solid", "label": "Street network"},
			"constricted_3": {"linewidth": 2, "color": '#D22A0E', "linestyle": "solid", "label": "Top 3%"},
			"constricted_5": {"linewidth": 2, "color": '#a3210b', "linestyle": "solid", "label": "Top 5%"},
			"constricted_10": {"linewidth": 2, "color": '#5a1206', "linestyle": "solid", "label": "Top 10%"},
            "bikegrown_betweenness": {"linewidth": 2.5, "color": '#0eb6d2', "linestyle": "solid", "label": "Betweenness"},
            "bikegrown_closeness": {"linewidth": 2, "color": '#186C7A', "linestyle": "dashed", "label": "Closeness"},
            "bikegrown_random": {"linewidth": 1.5, "color": '#222222', "linestyle": "dotted", "label": "Random"}
			}
    filename = f"{placeid}_carall.csv"
    Carall_analysis = np.genfromtxt(analysis_results_pth/filename, delimiter=',', names=True)
    filename = f"{placeid}_poi_{poi_source}_{prune_measure}.csv"
    GT_analysis_results = np.genfromtxt(analysis_results_pth/f"{len(prune_quantiles)}quantiles"/filename, delimiter=',', names=True)
    filename = f"{placeid}_poi_{poi_source}_mst.csv"
    MST_analysis_results = np.genfromtxt(analysis_results_pth/f"{len(prune_quantiles)}quantiles"/filename, delimiter=',', names=True)
    filename = f"{placeid}_poi_{poi_source}_{prune_measure}_SUMP.csv"
    Existing_analysis_results = np.genfromtxt(analysis_results_pth/filename, delimiter=',', names=True)
    min_diff = abs(GT_analysis_results["length"][0] - MST_analysis_results["length"])
    quantile_inx = 0
    prune_quantiles = [0] + prune_quantiles
    for i, value in enumerate(GT_analysis_results["length"]):
        diff = abs(value - MST_analysis_results["length"])
        if diff < min_diff:
            min_diff = diff
            quantile_inx = i
    nc = 4
    nr = 2
    index = 0
    fig, axes = plt.subplots(nrows= nr, ncols = nc, figsize = (16,6))
    for j in range(nr):
        for i, ax in enumerate(axes[j]):
            key = list(keys_metrics.keys())[index]
            index += 1
            GT_values = np.insert(GT_analysis_results[key], 0, 0)
            Existing_values = np.insert(Existing_analysis_results[key], 0, 0)
            if key in ["length", "length_lcc"]:
                ax.plot(prune_quantiles, GT_values/1000, **plotparam_analysis["bikegrown"])
                xmin, xmax = ax.get_xlim()
                ax.plot([xmin, xmax], [MST_analysis_results[key]/1000, MST_analysis_results[key]/1000], **plotparam_analysis["mst"])
                ax.plot(prune_quantiles, Existing_values/1000, **plotparam_analysis["biketrack"])
            if key == "poi_coverage":
                ax.plot(prune_quantiles, GT_values, **plotparam_analysis["bikegrown"])
                xmin, xmax = ax.get_xlim()
                ax.plot([xmin, xmax], [MST_analysis_results[key], MST_analysis_results[key]], **plotparam_analysis["mst"])
                ax.plot(prune_quantiles, Existing_values, **plotparam_analysis["biketrack"])
            if key == "population_coverage":
                ax.plot(prune_quantiles, GT_values/1000, **plotparam_analysis["bikegrown"])
                xmin, xmax = ax.get_xlim()
                ax.plot([xmin, xmax], [MST_analysis_results[key]/1000, MST_analysis_results[key]/1000], **plotparam_analysis["mst"])
                ax.plot(prune_quantiles, Existing_values/1000, **plotparam_analysis["biketrack"])
                ax.set_ylim(top = Carall_analysis[key]/1000)
            if key == "coverage":
                ax.plot(prune_quantiles, GT_values, **plotparam_analysis["bikegrown"])
                xmin, xmax = ax.get_xlim()
                ax.plot([xmin, xmax], [MST_analysis_results[key], MST_analysis_results[key]], **plotparam_analysis["mst"])
                ax.plot(prune_quantiles, Existing_values, **plotparam_analysis["biketrack"])
            if key == "components":
                ax.plot(prune_quantiles, GT_values, **plotparam_analysis["bikegrown"])
                xmin, xmax = ax.get_xlim()
                ax.plot([xmin, xmax], [MST_analysis_results[key], MST_analysis_results[key]], **plotparam_analysis["mst"])
                ax.plot(prune_quantiles, Existing_values, **plotparam_analysis["biketrack"])
            if key in ["directness_all_linkwise", "directness_bicycle_car"]:
                ax.plot(prune_quantiles, GT_values, **plotparam_analysis["bikegrown"])
                xmin, xmax = ax.get_xlim()
                ax.plot([xmin, xmax], [MST_analysis_results[key], MST_analysis_results[key]], **plotparam_analysis["mst"])
                ax.plot(prune_quantiles, Existing_values, **plotparam_analysis["biketrack"])
                y1 = float(MST_analysis_results[key])
                ax.plot([xmin, xmax], [1, 1], linewidth = 1, linestyle = "dotted", color = "black")
                ax.tick_params(axis='y', which='minor', length=2, color='gray', labelsize = 7.5)
            ax.set_ylim(bottom = 0)
            ax.set_xlim(0, 1)
            ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
            ymin, ymax = ax.get_ylim()
            #ax.plot([prune_quantiles[quantile_inx], prune_quantiles[quantile_inx]], [ymin, ymax], linewidth = 1, linestyle = "dotted", label = "Equal lengths point", color = "orange")
            if j == 0 and i == 0:
                ax.legend()
            ax.set_title(keys_metrics[key])
            ax.set_xlabel("Quantile")
    plt.subplots_adjust(top = 0.87, bottom = 0.09, left = 0.05, right = 0.97, wspace = 0.25, hspace = 0.4)
    fig.savefig(analysis_results_pth/f"{len(prune_quantiles)}quantiles"/f"Analysis_plots.png")
        
if __name__ == "__main__":
    placeid = "Bucharest"
    place_name = "Bucharest, Romania"
    poi_source = "SUMP"
    prune_measure = "Bq"
    prune_quantiles = [[x/40 for x in list(range(1, 41))], [x/6 for x in list(range(1, 7))]]
    data_pth = Path(".\Data")
    network_data_pth = data_pth/"network_data"
    poi_data_pth = data_pth/"poi_data"
    analysis_results_pth = data_pth/"analysis_results"
    result_pth = data_pth/"results"
    population_data_pth = data_pth/"population_data"
    
    with open(poi_data_pth/f'{placeid}_poi_nnidscarall.csv') as f:
        nnids = [int(line.rstrip()) for line in f]
    
    routes_layers = ["existing_routes","riders_preferences_routes", "transport_hubs_routes", "employment_hubs_routes", "commercial_hubs_routes", "connectivity_routes"]
    Gs_layer = []
    Gs_layer_simplified = []
    for layer in routes_layers:
        G = csv_to_ig(network_data_pth, placeid, f"SUMP_{layer}")
        G.vs["name"] = [str(id) for id in G.vs["id"]]
        Gs_layer.append(G)
        G_simplified = csv_to_ig(network_data_pth, placeid, f"SUMP_{layer}_simplified")
        G_simplified.vs["name"] = [str(id) for id in G_simplified.vs["id"]]
        Gs_layer_simplified.append(G_simplified)
    
    G_population_centers = csv_to_ig(population_data_pth, placeid, 'population_density_centers')
    G_carall = csv_to_ig(network_data_pth, placeid, "carall")
    G_carall_simplified = csv_to_ig(network_data_pth, placeid, "carall_simplified")
    
    Gs_bikenetwork = []
    Gs_bikenetwork_simplified = []
    for i in range(len(Gs_layer)):
        G_final = ig.union(Gs_layer[:i+1], byname=True)
        Gs_bikenetwork.append(G_final)
        G_final_simplified = ig.union(Gs_layer_simplified[:i+1], byname= True)
        Gs_bikenetwork_simplified.append(G_final_simplified)
    Gs = {}
    Gs["carall"] = G_carall
    Gs["carall_simplified"] = G_carall_simplified
    Gs["bikenetwork"] = Gs_bikenetwork
    Gs["bikenetwork_simplified"] = Gs_bikenetwork_simplified
    G_existing = {}
    G_existing["biketrack"] = Gs_layer[0]
    population_squares = create_pop_density_proj(G_carall, G_population_centers, 500)
    
    analyze_SUMP = True
    if analyze_SUMP:
        analyse_SUMP(Gs,G_existing,G_population_centers, nnids, analysis_results_pth, placeid)
    
    results = []
    for pq in prune_quantiles:
        with open(result_pth/f"{placeid}_poi_{poi_source}_{len(pq)}quantiles_{prune_measure}.pickle", 'rb') as f:
            res = pickle.load(f)
            results.append(res)
    
    GT_MST_analysis = True
    if GT_MST_analysis:
        for res in results:
            analyse_GT_MST(res, nnids, G_existing,G_carall, G_population_centers,analysis_results_pth, placeid, poi_source)
    for pq in prune_quantiles:
        plot_analysis(placeid,poi_source, prune_measure, pq, analysis_results_pth)
        
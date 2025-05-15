import osmnx as ox
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
from functions import fill_holes, extract_relevant_polygon, ox_to_csv
import igraph as ig
import networkx as nx
import shapely
import numpy as np 
from matplotlib import cm
import pandas as pd
from tqdm.notebook import tqdm
from pathlib import Path
from haversine import haversine, haversine_vector

def construct_street_networks(Gs, network_data_pth, place_name, placeid):
    osmnxparameters = {'car30': {'network_type':'drive', 'custom_filter':'["maxspeed"~"^30$|^20$|^15$|^10$|^5$|^20 mph|^15 mph|^10 mph|^5 mph"]', 'export': True, 'retain_all': True},
                       'carall': {'network_type':'drive', 'custom_filter': None, 'export': True, 'retain_all': False},
                       'toproutes' :{'network_type':'bike', 'custom_filter':None, 'export': False, 'retain_all': True},
                       'bike_cyclewaytrack': {'network_type':'bike', 'custom_filter':'["cycleway"~"track"]', 'export': False, 'retain_all': True},
                       'bike_highwaycycleway': {'network_type':'bike', 'custom_filter':'["highway"~"cycleway"]', 'export': False, 'retain_all': True},
                       'bike_designatedpath': {'network_type':'all', 'custom_filter':'["highway"~"path"]["bicycle"~"designated"]', 'export': False, 'retain_all': True},
                       'bike_cyclewayrighttrack': {'network_type':'bike', 'custom_filter':'["cycleway:right"~"track"]', 'export': False, 'retain_all': True},
                       'bike_cyclewaylefttrack': {'network_type':'bike', 'custom_filter':'["cycleway:left"~"track"]', 'export': False, 'retain_all': True},
                       'bike_cyclestreet': {'network_type':'bike', 'custom_filter':'["cyclestreet"]', 'export': False, 'retain_all': True},
                       'bike_bicycleroad': {'network_type':'bike', 'custom_filter':'["bicycle_road"]', 'export': False, 'retain_all': True},
                       'bike_livingstreet': {'network_type':'bike', 'custom_filter':'["highway"~"living_street"]', 'export': False, 'retain_all': True}
                      }  
    networktypes = ["biketrack", "carall", "bikeable", "biketrackcarall", "biketrack_onstreet", "bikeable_offstreet"] # Existing infrastructures to analyze

    
    location = ox.geocoder.geocode_to_gdf(place_name)
    location = fill_holes(extract_relevant_polygon(place_name, shapely.geometry.shape(location['geometry'][0])))

    for parameterid, parameterinfo in tqdm(osmnxparameters.items(), desc = "Networks", leave = False):
        for i in range(0,10): # retry
            try:
                Gs[parameterid] = ox.graph_from_polygon(location, 
                                       network_type = parameterinfo['network_type'],
                                       custom_filter = (parameterinfo['custom_filter']),
                                       retain_all = parameterinfo['retain_all'],
                                       simplify = False)
            except ValueError:
                Gs[parameterid] = nx.empty_graph(create_using = nx.MultiDiGraph)
                print(f"No OSM data for graph {parameterid}. Created empty graph.")
                break
            except ConnectionError or UnboundLocalError:
                print("ConnectionError or UnboundLocalError. Retrying.")
                continue
            except:
                print("Other error. Retrying.")
                continue
            break
        if parameterinfo['export']: ox_to_csv(Gs[parameterid], network_data_pth,placeid,parameterid)

    # Compose special cases biketrack, bikeable, biketrackcarall
    parameterid = 'biketrack'
    Gs[parameterid] = nx.compose_all([Gs['bike_cyclewaylefttrack'], Gs['bike_cyclewaytrack'], Gs['bike_highwaycycleway'], Gs['bike_bicycleroad'], Gs['bike_cyclewayrighttrack'], Gs['bike_designatedpath'], Gs['bike_cyclestreet']])
    ox_to_csv(Gs[parameterid], network_data_pth,placeid ,parameterid)
    parameterid = 'bikeable'
    Gs[parameterid] = nx.compose_all([Gs['biketrack'], Gs['car30'], Gs['bike_livingstreet']]) 
    ox_to_csv(Gs[parameterid], network_data_pth, placeid, parameterid)
    parameterid = 'biketrackcarall'
    Gs[parameterid] = nx.compose(Gs['biketrack'], Gs['carall']) # Order is important
    ox_to_csv(Gs[parameterid], network_data_pth, placeid, parameterid)

    for parameterid in networktypes[:-2]:
        #G_temp = nx.MultiDiGraph(ox.utils_graph.get_digraph(ox.simplify_graph(Gs[parameterid]))) # This doesnt work - cant get rid of multiedges
        ox_to_csv(ox.simplify_graph(Gs[parameterid]), network_data_pth,placeid ,parameterid, "_simplified")

def construct_SUMP_network(Gs, network_data_pth, placeid, place_name):
    routes = {"existing_routes": ["Calea Victoriei"],
        "riders_preferences_routes":["Bulevardul General Gheorghe Magheru","Piața Romană","Piața Victoriei","Piața Victor Babeș","Piața Universității","Strada Răzoare","Piața Danny Huwe",'Drumul Taberei', "Drumul Sării", "Bulevardul Iuliu Maniu", "Splaiul Independenței", "Șoseaua Colentina", "Bulevardul Carol I", "Strada Barbu Văcărescu", "Strada Căpitan Aviator Alexandru Șerbănescu","Pasaj Băneasa", "Bulevardul Unirii", "Splaiul Unirii", 'Bulevardul Lascăr Catargiu', 'Bulevardul Nicolae Bălcescu', 'Bulevardul Ion Constantin Brătianu', 'Piața Unirii', 'Bulevardul Iancu de Hunedoara', 'Șoseaua Ștefan cel Mare', 'Șoseaua Mihai Bravu', 'Bulevardul Regina Elisabeta', "Calea Moșilor", "Strada Bărăției"],
        "transport_hubs_routes" : ['Șoseaua Olteniței', 'Bulevardul Dimitrie Cantemir', 'Bulevardul Corneliu Coposu', 'Strada Matei Basarab', 'Calea Călărașilor', 'Bulevardul Basarabia', 'Strada Industriilor', 'Bulevardul Ferdinand I', 'Calea Griviței', 'Bulevardul Dacia', 'Șoseaua Giurgiului', "Calea Șerban Vodă", "Piața Eroii Revoluției", "Piața Pache Protopopescu"],
        "employment_hubs_routes" : ["Bulevardul Profesor Doctor Gheorghe Marinescu","Calea 13 Septembrie","Piața Regina Maria","Strada Gara Herăstrău","Podul Ciurel", 'Bulevardul Geniului', 'Șoseaua Panduri', 'Calea 13 Septembri', 'Șoseaua Cotroceni', 'Bulevardul Eroii Sanitari', 'Șoseaua Grozăvești', 'Bulevardul Doina Cornea', "Pasajul Basarab", 'Șoseaua Virtuții', 'Bulevardul Uverturii', 'Bulevardul Alexandru Ioan Cuza', 'Bulevardul Bucureștii Noi', 'Calea Griviței', 'Șoseaua Pipera', 'Bulevardul Dimitrie Pompeiu', 'Strada Vasile Pârvan', 'Strada Berzei', 'Strada Buzești', 'Șoseaua Colentina', 'Șoseaua Pantelimon', 'Bulevardul Tudor Vladimirescu', 'Șoseaua Olteniței', 'Șoseaua Berceni', 'Șoseaua Giurgiului', 'Bulevardul Națiunile Unite', "Calea Rahovei", "Bulevardul George Coșbuc", "Șoseaua București-Măgurele", "Strada Bogdan Petriceicu Hașdeu", "Piața Delfinului"],
        "commercial_hubs_routes" : ['Bulevardul Burebista', 'Piața Alba Iulia', 'Calea Dudești', 'Bulevardul Camil Ressu', 'Bulevardul Theodor Pallady', 'Calea Vitan', 'Strada Sergent Nițu Vsile', 'Bulevardul Alexandru Obregia', 'Strada Lujerului', 'Bulevardul Timișoara', 'Strada Odoarei', 'Șoseaua Viilor', 'Șoseaua Chitilei', "Strada Sergent Nițu Vasile", "Strada Lucian Blaga"],
        "connectivity_routes" : ["Bulevardul Gloriei","Piața Charles de Gaulle","Bulevardul Mareșal Constantin Prezan","Piața Arcul de Triumf","Bulevardul Mareșal Alexandru Averescu","Strada Turda","Podul Grant","Calea Crângași",'Șoseaua București-Târgoviște', 'Șoseaua Alexandria', 'Calea Dorobanților', 'Strada Liviu Rebreanu', 'Șoseaua Antiaeriană', 'Drumul Sării', 'Strada Brașov', 'Strada General Constantin Budișteanu', 'Strada Doamna Ghica', 'Bulevardul Chișinău', 'Bulevardul Tudor Vladimirescu', 'Șoseaua Mihai Bravu', 'Strada Jiului', 'Bulevardul Poligrafiei', 'Bulevardul Nicolae Grigorescu', 'Bulevardul Constantin Prezan', "Calea Crângași"]
        }
    location = ox.geocoder.geocode_to_gdf(place_name)
    location = fill_holes(extract_relevant_polygon(place_name, shapely.geometry.shape(location['geometry'][0])))
    street_network = ox.graph_from_polygon(location, network_type="drive", retain_all=True, simplify= False)
    for layer, streets in routes.items():
        G_filtered = nx.MultiDiGraph()
        kept_nodes = set()
        edges_kept = set()
        for u, v, data in street_network.edges(data=True):
            # Check if 'name' exists and matches (OSMnx can have 'name' as a list sometimes)
            edge_name = data.get("name")
            if isinstance(edge_name, list):
                for name in edge_name:
                    if name in streets:
                        edges_kept.update([name])
                        G_filtered.add_edge(u, v, **data)
                        kept_nodes.update([u, v])
                        break
            elif edge_name in streets:
                edges_kept.update([edge_name])
                G_filtered.add_edge(u, v, **data)
                kept_nodes.update([u, v])
        for node in kept_nodes:
            if node in street_network.nodes:
                G_filtered.add_node(node, **street_network.nodes[node])
        if layer == "riders_preferences_routes":
            #BUG ON SOSEAUA MIHAI BRAVU
            point1 = Point(26.134948, 44.417998)
            point2 = Point(26.135307, 44.417903)
            node1, node2  = [ox.distance.nearest_nodes(street_network, d.x, d.y) for d in [point1, point2]]
            G_filtered.add_edge(node1, node2, name = "Șoseaua Mihai Bravu", length = haversine_vector([point1.y, point1.x], [point2.y, point2.x], unit = "m")[0])
            #BUG ON SOSEAUA STEFAN CEL MARE
            point1 = Point(26.120106, 44.452358)
            point2 = Point(26.119976, 44.452041)
            node1, node2  = [ox.distance.nearest_nodes(street_network, d.x, d.y) for d in [point1, point2]]
            G_filtered.add_edge(node1, node2, name = "Șoseaua Ștefan cel Mare", length = haversine_vector([point1.y, point1.x], [point2.y, point2.x], unit = "m")[0])
            #Drumul taberei
            G_filtered.remove_nodes_from([287125492, 287125381, 287125491])
        if layer == "transport_hubs_routes":
            #PROBLEM ON CALEA GRIVITEI WITH A ROAD UNDER CONSTRUCTION ON OSM DATABASE-NOT UPDATED
            point1 = Point(26.049467, 44.471738)
            point2 = Point(26.048556, 44.472870)
            node1, node2  = [ox.distance.nearest_nodes(street_network, d.x, d.y) for d in [point1, point2]]
            G_filtered.add_edge(node1, node2, name = "Calea Griviței", length = haversine_vector([point1.y, point1.x], [point2.y, point2.x], unit = "m")[0])
            #Bug on strada MATEI BASARAB on the two roundabouts
            point1 = Point(26.114530, 44.430491)
            point2 = Point(26.115180, 44.430494)
            point3 = Point(26.120079, 44.430418)
            point4 = Point(26.120768, 44.430422)
            node1, node2, node3, node4 = [ox.distance.nearest_nodes(street_network, d.x, d.y) for d in [point1, point2, point3, point4]]
            G_filtered.add_edge(node1, node2, name = "Strada Matei Basarab", length = haversine_vector([point1.y, point1.x], [point2.y, point2.x], unit = "m")[0])
            G_filtered.add_edge(node3, node4, name = "Strada Matei Basarab", length = haversine_vector([point3.y, point3.x], [point4.y, point4.x], unit = "m")[0])
            #Bug on soseaua giurgiului
            point1 = Point(26.091845, 44.386622)
            point2 = Point(26.092073, 44.386583)
            node1, node2  = [ox.distance.nearest_nodes(street_network, d.x, d.y) for d in [point1, point2]]
            G_filtered.add_edge(node1, node2, name = "Șoseaua Giurgiului", length = haversine_vector([point1.y, point1.x], [point2.y, point2.x], unit = "m")[0])
            #Bulevardul Basarabia
            G_filtered.remove_nodes_from([2109297930, 3883878892, 2109297928])
            #Bug Calea Calarasilor
            point1 = Point(26.132528, 44.431956)
            point2 = Point(26.132558, 44.431479)
            node1, node2  = [ox.distance.nearest_nodes(street_network, d.x, d.y) for d in [point1, point2]]
            G_filtered.add_edge(node1, node2, name = "Calea Călărașilor", length = haversine_vector([point1.y, point1.x], [point2.y, point2.x], unit = "m")[0])
        if layer == "employment_hubs_routes":
            #PROBLEM ON CALEA 13 SEPTEMBRIE
            point1 = Point(26.076844, 44.424707)
            point2 = Point(26.078196, 44.425057)
            node1, node2  = [ox.distance.nearest_nodes(street_network, d.x, d.y) for d in [point1, point2]]
            G_filtered.add_edge(node1, node2, name = "Calea 13 Septembrie", length = haversine_vector([point1.y, point1.x], [point2.y, point2.x], unit = "m")[0])
            #Problem on Calea RAHOVEI
            point1 = Point(26.069761, 44.410960)
            point2 = Point(26.070125, 44.410705)
            node1, node2  = [ox.distance.nearest_nodes(street_network, d.x, d.y) for d in [point1, point2]]
            G_filtered.add_edge(node1, node2, name = "Calea Rahovei", length = haversine_vector([point1.y, point1.x], [point2.y, point2.x], unit = "m")[0])
        if layer == "commercial_hubs_routes":
            #PROBLEM ON INTERSECTION STRADA LUJERULUI AND BULEVARDUL TIMISOARA
            point1 = Point(26.033405, 44.427969)
            point2 = Point(26.033584, 44.427945)
            node1, node2  = [ox.distance.nearest_nodes(street_network, d.x, d.y) for d in [point1, point2]]
            G_filtered.add_edge(node1, node2, name = "Strada Lujerului", length = haversine_vector([point1.y, point1.x], [point2.y, point2.x], unit = "m")[0])
            #Bulevardul Burebista
            point1 = Point(26.131341, 44.423816)
            point2 = Point(26.130850, 44.423502)
            node1, node2  = [ox.distance.nearest_nodes(street_network, d.x, d.y) for d in [point1, point2]]
            G_filtered.add_edge(node1, node2, name = "Bulevardul Burebista", length = haversine_vector([point1.y, point1.x], [point2.y, point2.x], unit = "m")[0])
            #Chitilei
            G_filtered.remove_nodes_from([3578035830, 256476955, 256476712, 2446698839])
        if layer == "connectivity_routes":
            #Liviu Rebreanu
            G_filtered.remove_nodes_from([4076840698, 10778576121, 2160418283, 4076840699, 2160418277, 10701116773, 2160418289, 2160418280, 3869978549, 3869978516, 2160418284, 2160418276, 2160418279])
            #Industriilor
            G_filtered.remove_nodes_from(['2272935601', '6316739933', '2272935600'])
        #Unknown street but small components
        G_filtered.remove_nodes_from([6316739933, 2272935601, 2272935600, 6528635006, 6528635007, 2994825800, 12616315921, 6248196073, 2474857804, 256683575, 256683812])
        Gs[layer] = G_filtered
        ox_to_csv(Gs[layer], network_data_pth, placeid, "SUMP", postfix= f"_{layer}")
    for layer in list(routes.keys()):
        #G_temp = nx.MultiDiGraph(ox.utils_graph.get_digraph(ox.simplify_graph(Gs[parameterid]))) # This doesnt work - cant get rid of multiedges
        ox_to_csv(ox.simplify_graph(Gs[layer]),network_data_pth,placeid ,f"SUMP_{layer}", "_simplified")

if __name__ == "__main__":
    #Create DATA STORAGE INFRASTRUCTURE
    data_pth = Path(".\Data")
    data_pth.mkdir(exist_ok=True)
    network_data_pth = data_pth/"network_data"
    network_data_pth.mkdir(exist_ok=True)
    poi_data_pth = data_pth/"poi_data"
    poi_data_pth.mkdir(exist_ok=True)
    analysis_results_pth = data_pth/"analysis_results"
    analysis_results_pth.mkdir(exist_ok=True)
    result_pth = data_pth/"results"
    result_pth.mkdir(exist_ok=True)
    population_data_pth = data_pth/"population_data"
    population_data_pth.mkdir(exist_ok=True)
    plots_pth = data_pth/"plots"
    plots_pth.mkdir(exist_ok=True)
    prune_quantiles = [[x/40 for x in list(range(1, 41))], [x/6 for x in list(range(1, 7))]]
    for pq in prune_quantiles:
        (analysis_results_pth/f"{len(pq)}quantiles").mkdir(exist_ok = True)
        (plots_pth/f"{len(pq)}quantiles").mkdir(exist_ok= True)

    #CONSTRUC NETWORKS
    place_name = "Bucharest, Romania"
    placeid = "Bucharest"
    Gs = {}
    construct_street_networks(Gs, network_data_pth, place_name, placeid)
    construct_SUMP_network(Gs, network_data_pth, placeid, place_name)
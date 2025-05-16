# Cycling-in-Bucharest

This is the source code for the Bachelor's Thesis paper "Cycling in Bucharest: Expanding a transit network" by Amalia-Elena Chirila and Bogdan-Mihai Cobzaru. The code downloads and pre-processes data from OpenStreetMap, prepares points of interest, runs expansion algorithms, measures and saves the results, and creates visualisations.

## Instructions
### 1. Clone the repository
Run from your terminal:
```
git clone https://github.com/CoBogdanM/Cycling-in-Bucharest.git
```
### 2. Run the code locally
Run the files in the following order:
1. [`prepare_street_network_data.py`](prepare_street_network_data.py) -> The file creates the data storage infrastructure and populates the Data/network_data folder. To get the paper's results, overwrite the folder's contents with the zip files from the [`old_network_data`](old_network_data/) folder. 
2. [`prepare_poi_data.py`](prepare_poi_data.py) -> Data/poi_data
3. [`prepare_population_data.py`](prepare_population_data.py) -> Data/population_data
4. [`routing.py`](routing.py) -> Data/results
5. [`analysis.py`](analysis.py) -> Data/analysis_results
6. [`plot_infrastructure.py`](plot_infrastructure.py) -> Data/plots
7. [`plot_expansion_visualisations.py`](plot_expansion_visualisations.py) Data/plots
8. [`eda.py`](eda.py) -> Terminal

### The data storage infrastructure is not uploaded to the repository due to its size; It is created locally in the prepare_street_network_data.py file.

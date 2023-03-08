# PPI network simulation under study bias

Simulates the measurement process of observed PPI networks under study bias as described in the manuscript 
"Emergence of power law distributions in protein-protein interaction networks through study bias".

- To generate the plots shown in the manuscript, execute the workflow in the notebook `plot_results.ipynb`.
- To reproduce the results of the simulation study,run script_simulation.py; use the following command: `python3 script_simulation.py`. **Attention: this will take several weeks.**
The first part of the script generates parameter files (.json), which are saved in `parameter_settings/all_param_combinations/` folder.
The second part of the script reads the parameter files and runs the simulation for each set of parameters using parallel processing. The results are saved in `output_results/` folder.


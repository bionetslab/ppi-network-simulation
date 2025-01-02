# PPI network simulation under study bias

Simulates the measurement process of observed PPI networks under study bias as described in: 

Blumenthal DB, Lucchetta M, Kleist L, Fekete SP, List M, Schaefer MH. Emergence of power-law distributions in protein-protein interaction networks through study bias. eLife. 2024;13:e99951. doi: [10.7554/eLife.99951](https://doi.org/10.7554/eLife.99951).

- To generate the plots shown in the manuscript, execute the workflow in the notebook `plot_results.ipynb`.
- To reproduce the results of the simulation study, run the `script_simulation.py` as follows: 

  ```sh
  python3 script_simulation.py
  ```
  
- The first part of the script generates parameter files (.json), which are saved in `parameter_settings/all_param_combinations/` folder. 
- The second part of the script reads the parameter files and runs the simulation for each set of parameters using parallel processing. 
- The results are saved in `output_results/` folder.
- **Attention: Reproducing the results might take several weeks depending on the computational infrastructure.**

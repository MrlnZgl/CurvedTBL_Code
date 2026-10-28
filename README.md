# CurvedTBL_Code
This is a repo with all codes I used for my master thesis about DNS of curved TBL with zero-pressure gradient conditions. 


Workflow: 
1. Mesh generation with gmsh
2. run neko simulation with base configuration (Dirichlet BC with (1,0,0) velocity profile)
3. extract nu, omega and XX at certain streamwise position and store them as .dat-files
    - OF_interpolation_pp_script.py
    - TBL_pp_function.py
4. set up openfaom simulation with extracted quantities as inflow condition
5. rum openfoam optimisation
    - 
6. extract velocity slice
    - extract_streamwise_slice.py
7. run neko simulation with optimised configuration
8. postprocess statistics
    - TBL_pp_script.py
    - TBL_pp_function.py
    - TBL_turbtrans_rebuild.py

# vision-cortex-hierarchy

Author: Alexandros Tzalavras (at1159@rutgers.edu)

Includes the Python (3.9.7) Jupyter notebook vision_cortex_hierarchy.ipynb, which contains the necessary code to run the core analyses described below. HCP 7T retinotopy fMRI data are used (see details below). The remaining files included with the repository are required for the function to run (see details below).
Analyses include: 
  1. Functional connectivity estimation at the parcel & vertex level.
  2. Communicability estimation at the parcel and vertex level.
  3. Representational similarity matrix creation.
  4. Representational distance estimation.
  5. V1-initiated empirical neural network.
  6. pRF modeling.
  7. In silico lesion experiments.
  8. Dimensionality.
  9. Receptive field size-dependent pathways routing.
  10. Permutations and null-hypothesis testing for all the above. 

The top notebook provides a detailed description of the code. 

DEPENDENCIES:

Connectome workbench must be installed and accessible to your system path, including the command line version (wb_command), https://www.humanconnectome.org/software/get-connectome-workbench
(Included with the repo) Relevant files from Matlab Gifti toolbox https://github.com/gllmflndn/gifti, allowing for opening (ciftiopen) and saving (ciftisave and ciftisavereset) HCP CIFTI files 
Function files (.py) featuring custom Python code to run the core analyses, called within the notebook.  
(Included with the repo) CAB-NP network partition .dlabel file (CortexSubcortex_ColeAnticevic_NetPartition_wSubcorGSR_parcels_LR.dlabel.nii) from https://github.com/ColeLab/ColeAnticevicNetPartition that links vertex/voxel data to affiliated Glasser atlas regions
Example data to test code execution: i) fMRI vertexwise task activation data for one subject, and ii) intermediate files containing group-level results. Both need to be stored in the data/ subdirectory and need to be downloaded first from this access link https://rutgers.box.com/s/8abw0n2ydeau2688rdozozzrqgq5hf37.

Note that running the function on the supplied example data for one subject on the Rutgers Amarel HPC cluster (2x Intel Xeon Gold 6338 Processors (48MB cache, 2.0GHz), 1 CPU requested, 80GB memory) takes ~114 seconds for execution.

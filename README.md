# vision-cortex-hierarchy

Author: Alexandros Tzalavras (at1159@rutgers.edu)

Includes the Python (3.9.7) Jupyter notebook vision_cortex_hierarchy.ipynb, which contains the necessary code to run the core analyses described below. HCP 7T retinotopy fMRI data are used (see details below). The remaining files included with the repository are required for the code to run (see details below).
Analyses include: 
  1. Functional connectivity estimation.
  2. Communicability estimation at the parcel and vertex level.
  3. Representational similarity matrix creation.
  4. Representational distance estimation.
  5. V1-initiated empirical neural network.
  6. pRF modeling.
  7. In silico lesion experiments.
  8. Dimensionality.
  9. Receptive field size-dependent pathways routing.
  10. Null-hypothesis testing for all the above. 

The top notebook provides a detailed description of the code. 

DEPENDENCIES:

(Included with the repo) Relevant files from Matlab Gifti toolbox https://github.com/gllmflndn/gifti, allowing for opening (ciftiopen) and saving (ciftisave and ciftisavereset) HCP CIFTI files 
Function files (.py) featuring custom Python code to run the core analyses, called within the notebook.  

(Included with the repo) CAB-NP network partition .dlabel file (CortexSubcortex_ColeAnticevic_NetPartition_wSubcorGSR_parcels_LR.dlabel.nii) from https://github.com/ColeLab/ColeAnticevicNetPartition that links vertex/voxel data to affiliated Glasser atlas regions.

(Included with the repo) graphicalLassoCV.py function from https://github.com/ColeLab/ActflowToolbox/connectivity_estimation that is used to estimate functional connectivity at the parcel level (Peterson et al., 2025). Also, addNetColors_Seaborn from https://github.com/ColeLab/ActflowToolbox/tools, which is used for functional connectivity (parcel-level) graph visualization. Lastly, cortex_parcel_network_assignments.txt from https://github.com/ColeLab/ActflowToolbox/dependencies/ColeAnticevicNetPartition, which links parcels to affiliated networks based on the CAB-NP network partition. 

Example data to test code execution: i) fMRI vertexwise task activation data for one subject, and ii) intermediate files containing group-level results. Both need to be stored in the same folder as the functions from this repository and need to be downloaded first from this access link 
https://rutgers.box.com/s/0uitqsy5bwzmxlh5l2cs3nbqzmmfjgmb.

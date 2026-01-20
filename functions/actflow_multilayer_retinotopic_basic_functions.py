import numpy as np
import pickle
from sklearn.metrics import r2_score
import os
import nibabel as nib
from scipy import signal
import time
from joblib import Parallel, delayed

subj_here = ['100610', '102311', '102816', '104416', '105923', '108323', '109123', '111514', '114823', '115017', '115825', '116726', 
             '118225', '125525', '126426', '814649', '818859', '825048', '826353', '833249', '859671', '861456', '871762', '872764', 
             '878776', '878877', '898176', '899885', '901139', '901442', '905147', '910241', '926862', '927359', '942658', '943862', 
             '958976', '966975', '971160', '995174']

# all subjects for this project (Apr 2025)
subj_here = ['100610', '102311', '102816', '104416', '105923', '108323', '109123', '111514', '114823', '115017', '115825', '116726', '118225', '125525', '126426', '467351', '525541', '825048', '826353', '833249', '859671', '861456', '871762', '872764', '878776', '878877', '898176', '899885', '901139', '901442', '905147', '910241', '926862', '927359', '942658', '943862', '958976', '966975', '971160', '995174', '128935', '131722', '137128', '140117', '144226', '283543', '318637', '320826', '330324', '346137', '541943', '547046', '562345', '572045', '573249', '581450', '601127', '617748', '627549', '638049', '644246', '654552', '671855', '680957', '690152', '706040', '724446', '725751', '732243', '751550', '757764', '765864', '770352', '771354', '782561', '783462', '789373', '814649', '818859', '951457'] # this is the full dataset, 80 subs. 0-39 (included) is group A,
            # 40-79 (included) is group B (control)

subj_control = ['128935', '131722', '137128', '140117', '144226', '145834', '467351', '525541', '536647', '541943', '547046', '552241', 
                '562345', '527045', '573249', '581450', '585256', '601127', '617748', '627549', '638049', '644246', '654552', '671855', 
                '680957', '690152', '706040', '724446', '725751', '732243', '745555', '751550', '757764', '765864', '770352', '771354', 
                '782561', '783462', '789373', '814649'] 

ret_runs = [
        'tfMRI_RETCCW_7T_AP_Atlas_MSMAll_hp2000_clean.dtseries.nii',
        'tfMRI_RETCW_7T_PA_Atlas_MSMAll_hp2000_clean.dtseries.nii',
        'tfMRI_RETEXP_7T_AP_Atlas_MSMAll_hp2000_clean.dtseries.nii',
        'tfMRI_RETCON_7T_PA_Atlas_MSMAll_hp2000_clean.dtseries.nii',
        'tfMRI_RETBAR1_7T_AP_Atlas_MSMAll_hp2000_clean.dtseries.nii',
        'tfMRI_RETBAR2_7T_PA_Atlas_MSMAll_hp2000_clean.dtseries.nii'
    ]
nVertices = 59412 # cortex only
basedir = '/projects/f_mc1689_1/MovieActFlow/'
basedir2 = '/projects/f_mc1689_1/MovieActFlow2/'
scriptDir2 = basedir2 + 'docs/scripts/'
scriptDir = basedir + 'docs/scripts/HCP_7T_Movie/'
rest_folder = basedir2 + 'docs/scripts/resting_state_parallel/'
groupDirAmarel = '/projects/f_mc1689_1/'
analysis_tools_dir = groupDirAmarel + 'AnalysisTools/ActflowToolbox/dependencies/ColeAnticevicNetPartition/'
#dlabelfile = scriptDir2 + 'dlabels_files/CortexSubcortex_ColeAnticevic_NetPartition_wSubcorGSR_parcels_LR_official_newest.dlabel.nii' # new dlabels file
dlabelfile = analysis_tools_dir + 'CortexSubcortex_ColeAnticevic_NetPartition_wSubcorGSR_parcels_LR.dlabel.nii' # old dlabels file
dlabels = np.squeeze(nib.load(dlabelfile).get_fdata()) # whole brain, 91282 vertices, 1 based not 0 based
dlabels_full = dlabels
dlabels = dlabels[:nVertices]
#dlabels_LGN_Pulvinar_path = os.path.join(rest_folder, 'combined_LGN_pulvinar_mask.dscalar.nii')
#dlabels_LGN_Pulvinar_img = nib.load(dlabels_LGN_Pulvinar_path)
#dlabels_LGN_Pulvinar = dlabels_LGN_Pulvinar_img.get_fdata().squeeze()
#dlabels_LGN_Pulvinar_final = np.concatenate((np.zeros((59412)), dlabels_LGN_Pulvinar))
filename = rest_folder + 'CortexSubcortex_ColeAnticevic_NetPartition_wSubcorGSR_parcels_LR_LabelKey.txt'
labels = np.genfromtxt(filename, delimiter="\t", names=True, dtype=None, encoding='utf-8')
# LGN_left = 1, LGN_right = 2, Pulvinar = 3
#indices_struc1 = np.where((dlabels_LGN_Pulvinar == 1) | (dlabels_LGN_Pulvinar == 2))[0]
#indices_struc1 = np.where(dlabels_LGN_Pulvinar == 1)[0]
#indices_struc2 = np.where(dlabels_LGN_Pulvinar == 2)[0]
#indices_struc3 = np.where(dlabels_LGN_Pulvinar == 3)[0]
'''
#dlabels_indices_pulvinar = dlabels_full[indices_struc3]
pulv_labels = []
for idx in np.unique(dlabels_indices_pulvinar):
    pulv_labels.append(labels['LABEL'][np.where(labels['KEYVALUE']==int(idx))[0]])

leftPulvVertices = []
rightPulvVertices = []

for label_of_interest in pulv_labels:
    real_label_string = label_of_interest[0]  # extract the string
    # Skip the ambiguous Brainstem label if you like:
    if 'Brainstem' in real_label_string:  
        continue

    # Figure out if this label is left or right based on substring '_L-' or '_R-'
    # (Adjust if your naming uses something slightly different.)
    is_left_label = '_L-' in real_label_string
    is_right_label = '_R-' in real_label_string

    # 1) Find the KEYVALUE for the label_of_interest
    matching_indices = np.where(labels['LABEL'] == label_of_interest)[0]
    if len(matching_indices) == 0:
        continue

    keyval_for_label = labels['KEYVALUE'][matching_indices][0]
    # 2) All vertices with this keyvalue
    vertices_for_label = np.where(dlabels_full == keyval_for_label)[0]
    # 3) Only keep those that also appear in pulvinar_correct_indices
    common_vertices = np.intersect1d(indices_struc3, vertices_for_label)

    # Store them in the appropriate list
    if is_left_label:
        leftPulvVertices.append(common_vertices)
    else:
        rightPulvVertices.append(common_vertices)

# 2) HANDLE THE AMBIGUOUS LABEL (ASSIGN TO RIGHT)
ambiguous_label = "Visual1-10_LR-Brainstem"
matching_indices = np.where(labels['LABEL'] == ambiguous_label)[0]
if len(matching_indices) > 0:
    keyval_for_ambiguous = labels['KEYVALUE'][matching_indices][0]
    ambiguous_vertices = np.where(dlabels_full == keyval_for_ambiguous)[0]
    ambiguous_vertices = np.intersect1d(indices_struc3, ambiguous_vertices)
    if len(ambiguous_vertices) > 0:
        rightPulvVertices.append(ambiguous_vertices)

# 3) CONCATENATE INTO FINAL ARRAYS
leftPulvVertices  = np.concatenate(leftPulvVertices)  if len(leftPulvVertices)  > 0 else np.array([])
rightPulvVertices = np.concatenate(rightPulvVertices) if len(rightPulvVertices) > 0 else np.array([])
'''

def load_ret(sub,retinotopy_ts_data):
    
    subj = subj_here[sub]
    subj_folder = retinotopy_ts_data + subj + '/MNINonLinear/Results/'
    os.chdir(subj_folder)
    run_folders = [
        'tfMRI_RETCCW_7T_AP',
        'tfMRI_RETCW_7T_PA',
        'tfMRI_RETEXP_7T_AP',
        'tfMRI_RETCON_7T_PA',
        'tfMRI_RETBAR1_7T_AP',
        'tfMRI_RETBAR2_7T_PA'
    ]
    ret_runs = [
        'tfMRI_RETCCW_7T_AP_Atlas_MSMAll_hp2000_clean.dtseries.nii',
        'tfMRI_RETCW_7T_PA_Atlas_MSMAll_hp2000_clean.dtseries.nii',
        'tfMRI_RETEXP_7T_AP_Atlas_MSMAll_hp2000_clean.dtseries.nii',
        'tfMRI_RETCON_7T_PA_Atlas_MSMAll_hp2000_clean.dtseries.nii',
        'tfMRI_RETBAR1_7T_AP_Atlas_MSMAll_hp2000_clean.dtseries.nii',
        'tfMRI_RETBAR2_7T_PA_Atlas_MSMAll_hp2000_clean.dtseries.nii'
    ]
    # load data
    data = {}
    for r,folder in enumerate(run_folders):
        r_f = subj_folder + folder + '/'
        os.chdir(r_f)
        inputfile = ret_runs[r]
        data[r] = np.squeeze(nib.load(inputfile).get_fdata()).T
        data[r] = data[r][0:59412,:]
        # Demean each run
        data[r] = signal.detrend(data[r],axis=1,type='constant')
        # Detrend each run
        data[r] = signal.detrend(data[r],axis=1,type='linear')
        
    return data

def load_ret_LGN_pul(sub,retinotopy_ts_data):
    
    subj = subj_here[sub]
    subj_folder = retinotopy_ts_data + subj + '/'
    os.chdir(subj_folder)
    ret_runs = [
        'tfMRI_RETCCW_7T_AP_Atlas_MSMAll_hp2000_clean.dtseries.nii',
        'tfMRI_RETCW_7T_PA_Atlas_MSMAll_hp2000_clean.dtseries.nii',
        'tfMRI_RETEXP_7T_AP_Atlas_MSMAll_hp2000_clean.dtseries.nii',
        'tfMRI_RETCON_7T_PA_Atlas_MSMAll_hp2000_clean.dtseries.nii',
        'tfMRI_RETBAR1_7T_AP_Atlas_MSMAll_hp2000_clean.dtseries.nii',
        'tfMRI_RETBAR2_7T_PA_Atlas_MSMAll_hp2000_clean.dtseries.nii'
    ]
    # load data
    data = {}
    for r,inputfile in enumerate(ret_runs):
        data[r] = np.squeeze(nib.load(inputfile).get_fdata()).T
        # Create two extra subcortical parcels
        struc1_ts = np.mean(data[r][indices_struc1, :], axis=0)
        struc2_ts = np.mean(data[r][indices_struc2, :], axis=0)
        struc3_ts = np.mean(data[r][leftPulvVertices, :], axis=0)
        struc4_ts = np.mean(data[r][rightPulvVertices, :], axis=0)
        data[r] = data[r][0:59412,:]
        data[r] = np.vstack((data[r], struc1_ts[np.newaxis, :], struc2_ts[np.newaxis, :], struc3_ts[np.newaxis, :], struc4_ts[np.newaxis, :]))
    
        # Demean each run
        data[r] = signal.detrend(data[r],axis=1,type='constant')
        # Detrend each run
        data[r] = signal.detrend(data[r],axis=1,type='linear')
        
    return data


def calc_score(ts1, ts2):
    
    if np.std(ts2) == 0:
        correlation = 0
        r_squared = 0
    else:
        correlation = np.corrcoef(ts1, ts2)[0, 1]
        r_squared = r2_score(ts1, ts2)
    
    return correlation, r_squared
    

def ret_ts(label, data, dlabels):
    if label < 181:
        label_for_flipped = label + 180
    else:
        label_for_flipped = label - 180
        
    original_vert_idx = np.where(dlabels==label)[0]
    flipped_vert_idx = np.where(dlabels==label_for_flipped)[0]

    original_ts = {}
    flipped_ts = {}
    #l_ts[r] = data_ret[run][l_vert_idx,:]
    original_ts = data[original_vert_idx,:]
    #r_ts[r] = data_ret[run][r_vert_idx,:]
    flipped_ts = data[flipped_vert_idx,:]
    
    return original_ts, flipped_ts



def load_parcel_dict(pickle_dir, target_parcel, dilation, LGN, LGN_but_V1):
    """
    Load a single parcel dictionary from a pickle file.
    
    Args:
        pickle_dir (str): The directory containing the pickle files for each parcel.
        target_parcel (int): The parcel number to load (e.g., 1, 4, 5, etc. 1-based).
        
    Returns:
        parcel_dict (dict): The loaded parcel dictionary containing connectivity data.
    """
    if (LGN) or (LGN_but_V1):
        file_name = f'fc_pc_glasso_cross_final_fast_full_solver_LGN_sep_pulvinar_sep_no_dilation_cortex_single_vert_target_z_scored_{target_parcel-1}.pkl'
        pickle_file = os.path.join(pickle_dir, file_name)
        with open(pickle_file, 'rb') as f:
            connectivity_dict = pickle.load(f)
    else:
        if dilation:
            file_prefix = '_5mm_dilation_'
        else:
            file_prefix = '_no_dilation_'
        file_name = f'fc_pc_glasso_cross_final_fast_full_solver{file_prefix}cortex_single_vert_target_z_scored_{target_parcel-1}.pkl'
        pickle_file = os.path.join(pickle_dir, file_name)
        with open(pickle_file, 'rb') as f:
            connectivity_dict = pickle.load(f)
    
    if (target_parcel==361) or (target_parcel==362) or (target_parcel==363) or (target_parcel==364):
        connectivity_dictt = connectivity_dict
    else:
        # Extract the actual connectivity data for each vertex (removing the metadata)
        connectivity_dictt = {item[0]: item[1:] for item in connectivity_dict}    
    
    
    return connectivity_dictt


def get_source_indices(dlabels, selected_parcels):
    """
    Get the list of source indices that belong to the selected parcels.
    
    Args:
        dlabels (np.ndarray): Array of labels where each index is a vertex and each value is the parcel number.
        selected_parcels (list): List of parcel numbers to process.
        
    Returns:
        list: Indices of vertices that belong to the selected parcels.
    """
    source_indices = []
    for src_parcel in selected_parcels:
        source_indices_here = np.where(dlabels == src_parcel)[0]
        source_indices.extend(source_indices_here)
    return source_indices


def fill_connectivity_matrix(connectivity_matrix, parcel_dict, source_indices, parcel_indices):
    
    for vertex in parcel_indices:
        if vertex in parcel_dict:
            full_conn_vector = parcel_dict[vertex][0]
            filtered_conn_vector = full_conn_vector[:, 0][source_indices]
            connectivity_matrix[vertex, source_indices] = filtered_conn_vector


def create_square_connectivity_matrix(pickle_dir, dlabels, selected_parcels, dilation, LGN, LGN_but_V1):
    """
    Constructs a square connectivity matrix where both rows and columns correspond to vertices
    in the selected parcels.
    """
    # Get indices of vertices in selected parcels
    source_indices = get_source_indices(dlabels, selected_parcels)
    source_indices = np.array(source_indices)
    num_vertices = len(source_indices)
    connectivity_matrix = np.zeros((num_vertices, num_vertices))

    # Initialize the row offset
    row_offset = 0

    # Load and fill the connectivity matrix
    for parcel in selected_parcels:
        parcel_dict = load_parcel_dict(pickle_dir, parcel, dilation, LGN, LGN_but_V1)

        # Get the vertices for the current parcel
        parcel_vertices = np.where(dlabels == parcel)[0]

        num_parcel_vertices = len(parcel_vertices)

        # Iterate over the vertices in the current parcel
        for i, vertex in enumerate(parcel_vertices):
            # Get the corresponding row index in the connectivity matrix
            row_idx = row_offset + i

            #if vertex in parcel_dict:
            # Load the connectivity vector for this vertex
            if (parcel==361) or (parcel==362) or (parcel==363) or (parcel==364):
                full_conn_vector = parcel_dict[1]
            else:
                full_conn_vector = parcel_dict[vertex][0]  # Assuming it's a 1D array  
            # Filter to selected vertices
            filtered_conn_vector = full_conn_vector[:, 0][source_indices]

            # Assign to the connectivity matrix
            connectivity_matrix[row_idx, :] = filtered_conn_vector

        # Update the row offset for the next parcel
        row_offset += num_parcel_vertices
        
    is_all_zero = np.all(connectivity_matrix == 0)
    if is_all_zero:
        print('Connectivity matrix is all zero')
        
    return connectivity_matrix

def permute_all_inter_parcel_connections(connectivity_matrix, dlabels, selected_parcels):
    num_vertices = connectivity_matrix.shape[0]
    permuted_matrix = connectivity_matrix.copy()
    source_indices = get_source_indices(dlabels, selected_parcels)
    source_indices = np.array(source_indices)
    dlabels_subset = dlabels[source_indices]

    # Create a mask for all inter-parcel connections
    parcel_labels = dlabels_subset[:, None]
    inter_parcel_mask = parcel_labels != parcel_labels.T

    # Extract all inter-parcel connections
    inter_parcel_values = connectivity_matrix[inter_parcel_mask]

    # Permute all inter-parcel connection values together
    permuted_values = np.random.permutation(inter_parcel_values)

    # Assign the permuted values back to the connectivity matrix
    permuted_matrix[inter_parcel_mask] = permuted_values

    return permuted_matrix


def permute_connectivity_matrix_columns_per_row(connectivity_matrix, dlabels, selected_parcels):
    num_vertices = connectivity_matrix.shape[0]
    source_indices = get_source_indices(dlabels, selected_parcels)
    source_indices = np.array(source_indices)
    dlabels_subset = dlabels[source_indices]
    permuted_matrix = np.zeros_like(connectivity_matrix)

    total_inter_parcel_connections = 0  # Counter for inter-parcel connections
    
    for i in range(num_vertices):
        inter_parcel_indices = np.where((dlabels_subset != dlabels_subset[i]) & (np.arange(num_vertices) != i))[0]
        inter_parcel_values = connectivity_matrix[i, inter_parcel_indices]
        
        # Count non-zero inter-parcel connections
        num_connections = np.count_nonzero(inter_parcel_values)
        total_inter_parcel_connections += num_connections
        
        permuted_values = np.random.permutation(inter_parcel_values)
        permuted_matrix[i, inter_parcel_indices] = permuted_values
    
    return permuted_matrix

def permute_connectivity_matrix_between_source_parcels(connectivity_matrix, dlabels, selected_parcels):
    num_vertices = connectivity_matrix.shape[0]
    source_indices = get_source_indices(dlabels, selected_parcels)
    source_indices = np.array(source_indices)
    dlabels_subset = dlabels[source_indices]
    permuted_matrix = connectivity_matrix.copy()

    for i in range(num_vertices):
        target_parcel_label = dlabels_subset[i]
        # Identify inter-parcel source vertices for the current target vertex
        inter_parcel_indices = np.where(dlabels_subset != target_parcel_label)[0]
        source_parcels = np.unique(dlabels_subset[inter_parcel_indices])

        # Group source indices by their parcels
        source_indices_by_parcel = {}
        for parcel_label in source_parcels:
            indices = np.where((dlabels_subset == parcel_label) & (np.arange(num_vertices) != i))[0]
            if len(indices) > 0:
                source_indices_by_parcel[parcel_label] = indices

        # Get list of source parcels
        parcel_labels = list(source_indices_by_parcel.keys())
        num_parcels = len(parcel_labels)

        # If there's only one source parcel, permute within that parcel
        if num_parcels <= 1:
            for indices in source_indices_by_parcel.values():
                weights = permuted_matrix[i, indices]
                permuted_weights = np.random.permutation(weights)
                permuted_matrix[i, indices] = permuted_weights
            continue

        # Determine the minimum number of vertices among source parcels
        min_parcel_size = min(len(indices) for indices in source_indices_by_parcel.values())

        # Randomly select indices to swap
        selected_indices_by_parcel = {}
        for parcel_label in parcel_labels:
            indices = source_indices_by_parcel[parcel_label]
            selected_indices = np.random.choice(indices, size=min_parcel_size, replace=False)
            selected_indices_by_parcel[parcel_label] = selected_indices

        # Collect weights to be swapped
        weights_by_parcel = {}
        for parcel_label in parcel_labels:
            indices = selected_indices_by_parcel[parcel_label]
            weights = permuted_matrix[i, indices]
            weights_by_parcel[parcel_label] = weights

        # Shuffle parcel labels to swap weights between parcels
        shuffled_parcels = np.random.permutation(parcel_labels)

        # Assign swapped weights back to the permuted matrix
        for idx, parcel_label in enumerate(parcel_labels):
            next_parcel_label = shuffled_parcels[idx]
            indices = selected_indices_by_parcel[parcel_label]
            swapped_weights = weights_by_parcel[next_parcel_label]
            permuted_matrix[i, indices] = swapped_weights

    return permuted_matrix


def calculate_total_vertices(dlabels, selected_parcels):
    """
    Calculate the total number of vertices that belong to the parcels of interest.

    Args:
        dlabels (np.ndarray): Array of vertex-to-parcel mappings (1-based).
        selected_parcels (list): List of parcel numbers to process (1-based).

    Returns:
        int: The total number of vertices that belong to the selected parcels.
    """
    total_vertices = 0

    for parcel in selected_parcels:
        # Count the number of vertices that belong to the current parcel
        vertices_in_parcel = np.sum(dlabels == parcel)
        total_vertices += vertices_in_parcel

    return total_vertices

def convert_to_selected_parcel_dicts(connectivity_matrix, dlabels, selected_parcels, total_vertices=59412):
    """
    Convert the connectivity matrix into dictionaries for the selected parcels, 
    with connectivity vectors that are 60k elements long.

    Args:
        connectivity_matrix (np.ndarray): Swapped matrix with shape (vertices x vertices).
        dlabels (np.ndarray): A NumPy array where each index is a vertex and each value is the parcel number (1-based).
        selected_parcels (list): A list of parcel numbers to process (e.g., [1, 4, 5, 6, 181, 184, 185, 186]).
        total_vertices (int): The total number of vertices in the brain (default is 59,412 for cortex only).

    Returns:
        parcel_dicts (dict): A dictionary for each selected parcel, containing vertex connectivity.
                            The key is the vertex index, and the value is the full 60k connectivity vector.
    """
    parcel_dicts = {}
    row_offset = 0  # This will track the starting row for each parcel

    # Get all the indices for the selected parcels from the dlabels array
    all_selected_indices = []
    for parcel in selected_parcels:
        all_selected_indices.extend(np.where(dlabels == parcel)[0])
    
    all_selected_indices = np.array(all_selected_indices)  # Ensure it's a numpy array

    # Only process the selected parcels
    for parcel in selected_parcels:
        # Store the parcel as 0-based in the dictionary
        parcel_dicts[parcel - 1] = {}

        # Get the vertex indices for the current parcel
        parcel_indices = np.where(dlabels == parcel)[0]

        # For each vertex in the parcel, populate its 60k connectivity vector
        for i, vertex in enumerate(parcel_indices):
            # Create a zero vector of length 60k
            full_conn_vector = np.zeros(total_vertices)

            # Get the connectivity vector from the permuted connectivity matrix (only for selected parcels)
            filtered_conn_vector = connectivity_matrix[row_offset + i, :].flatten()

            # Place the filtered_conn_vector into the correct positions of the full_conn_vector
            full_conn_vector[all_selected_indices] = filtered_conn_vector

            # Store the 60k vector in the parcel dictionary
            parcel_dicts[parcel - 1][vertex] = full_conn_vector

        # Update the row_offset by the number of vertices in this parcel
        row_offset += len(parcel_indices)

    return parcel_dicts


def lesion_connections(connectivity_matrix, brain_areas, lesion_paths):
    """
    Lesions hierarchical connections by zeroing out the relevant parts of the connection vectors.

    Parameters:
        connectivity_matrix (dict): The connectivity structure where keys are brain areas and values are dictionaries of vertices.
        brain_areas (dict): A mapping of brain areas (e.g., 'lv1', 'lv2') to their respective indices in the brain.
        hierarchical_paths (list): List of tuples representing hierarchical connections to lesion (e.g., [('lv1', 'lv2')])

    Returns:
        connectivity_matrix (dict): The modified connectivity structure with hierarchical connections lesioned.
    """
    for (source, target) in lesion_paths:
        # Get indices for the source and target regions
        source_indices = brain_areas[source]
        print(f"Lesioning connections from {source} to {target}")
        # Lesion connections from source to target by zeroing out the relevant parts of the vectors
        for vertex, connections in connectivity_matrix[target].items():
            # Check if connections is a tuple (first time processing this target)
            if isinstance(connections, tuple):
                # Extract the connectivity vector from the tuple
                connections_vector = connections[0][:, 0]
                # Zero out the connections from source to target
                connections_vector[source_indices] = 0
                # Update the connectivity dictionary with the modified vector
                connectivity_matrix[target][vertex] = connections_vector
            else:
                # Connections have already been modified; they are now a NumPy array
                connections_vector = connections
                # Zero out the connections from source to target
                connections_vector[source_indices] = 0
                # Update the connectivity dictionary
                connectivity_matrix[target][vertex] = connections_vector
                
    return connectivity_matrix

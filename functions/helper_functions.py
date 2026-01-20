import numpy as np
import sys
import os
import h5py
import pickle
sys.path.append('/projects/f_mc1689_1/MovieActFlow2/docs/scripts/Act_Flow/')
sys.path.append('/projects/f_mc1689_1/MovieActFlow2/docs/scripts/retinotopic/project_functions/')
import actflow_multilayer_retinotopic_basic_functions as act_functions

selected_parcels = [1, 4, 5, 6, 181, 184, 185, 186]
selected_parcels_output = [4, 5, 6, 184, 185, 186]

# all subjects for this project (Apr 2025)
subj_here = ['100610', '102311', '102816', '104416', '105923', '108323', '109123', '111514', '114823', '115017', '115825', '116726', 
             '118225', '125525', '126426', '467351', '525541', '825048', '826353', '833249', '859671', '861456', '871762', '872764', 
             '878776', '878877', '898176', '899885', '901139', '901442', '905147', '910241', '926862', '927359', '942658', '943862', 
             '958976', '966975', '971160', '995174', '128935', '131722', '137128', '140117', '144226', '283543', '318637', '320826', 
             '330324', '346137', '541943', '547046', '562345', '572045', '573249', '581450', '601127', '617748', '627549', '638049', 
             '644246', '654552', '671855', '680957', '690152', '706040', '724446', '725751', '732243', '751550', '757764', '765864', 
             '770352', '771354', '782561', '783462', '789373', '814649', '818859', '951457'] # this is the full dataset, 80 subs. 0-39 (included) is group A,
            # 40-79 (included) is group B (control)

    
def create_square_connectivity_matrix(pickle_dir, dlabels, selected_parcels, dilation):
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
        parcel_dict = load_parcel_dict(pickle_dir, parcel, dilation)

        # Get the vertices for the current parcel
        parcel_vertices = np.where(dlabels == parcel)[0]

        num_parcel_vertices = len(parcel_vertices)

        # Iterate over the vertices in the current parcel
        for i, vertex in enumerate(parcel_vertices):
            # Get the corresponding row index in the connectivity matrix
            row_idx = row_offset + i

            if vertex in parcel_dict:
                # Load the connectivity vector for this vertex
                full_conn_vector = parcel_dict[vertex][0]  # Assuming it's a 1D array

                # Filter to selected vertices
                filtered_conn_vector = full_conn_vector[:, 0][source_indices]

                # Assign to the connectivity matrix
                connectivity_matrix[row_idx, :] = filtered_conn_vector

        # Update the row offset for the next parcel
        row_offset += num_parcel_vertices

    return connectivity_matrix


def load_parcel_dict(pickle_dir, target_parcel, dilation=False):
    """
    Load a single parcel dictionary from a pickle file.
    
    Args:
        pickle_dir (str): The directory containing the pickle files for each parcel.
        target_parcel (int): The parcel number to load (e.g., 1, 4, 5, etc. 1-based).
        
    Returns:
        parcel_dict (dict): The loaded parcel dictionary containing connectivity data.
    """
    if dilation:
        file_name = f'fc_pc_glasso_cross_final_fast_full_solver_5mm_dilation_cortex_single_vert_target_z_scored_{target_parcel-1}.pkl'
    else:
        file_name = f'fc_pc_glasso_cross_final_fast_full_solver_no_dilation_cortex_single_vert_target_z_scored_{target_parcel-1}.pkl'
    pickle_file = os.path.join(pickle_dir, file_name)
    with open(pickle_file, 'rb') as f:
        connectivity_dict = pickle.load(f)
    
    # Extract the actual connectivity data for each vertex (removing the metadata)
    connectivity_dictt = {item[0]: item[1:] for item in connectivity_dict}    
    
    
    return connectivity_dictt


def fill_connectivity_matrix(connectivity_matrix, parcel_dict, source_indices, parcel_indices):
    
    for vertex in parcel_indices:
        if vertex in parcel_dict:
            full_conn_vector = parcel_dict[vertex][0]
            filtered_conn_vector = full_conn_vector[:, 0][source_indices]
            connectivity_matrix[vertex, source_indices] = filtered_conn_vector


def load_conn_matrix_all_subs(num_subs, pc_folder, selected_parcels, dlabels, dilation=False):
    '''
    num_subs: array of subject indices 0-40 group A, 40-80 group B
    '''
    dlabels_subset = dlabels_subset_creation(dlabels, selected_parcels)
    num_vertices = len(dlabels_subset)
    connectivity_matrix = np.zeros((len(num_subs), num_vertices, num_vertices))
    for i,sub_num in enumerate(num_subs):
        subject_id = subj_here[sub_num]
        pc_res = os.path.join(pc_folder, subject_id, 'ICA_Fix/')
        pickle_dir = pc_res
        connectivity_matrix[i] = create_square_connectivity_matrix(pickle_dir, dlabels, selected_parcels, dilation)
    
    return connectivity_matrix


# load the glasso connectivity matrix across all subs
def create_connectivity_matrix_glasso():
    os.chdir(glasso_results_ICA_all)
    subs = np.arange(0,15)
    glasso_all = np.zeros((15,360,360))
    for sub in subs:
        subj = subjNums[sub]
        print(subj)
        filename = 'glasso_fc_parcels_ICA_'+subj+'.npy'
        glasso_fc_sub = np.load(filename)
        print(glasso_fc_sub.shape)
        glasso_all[sub] = glasso_fc_sub
        
    return glasso_all


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

def map_vertices_to_parcels(dlabels_subset, parcel_labels):
    """
    Maps vertices to their corresponding parcels.

    Returns:
    - parcel_vertex_indices: Dictionary mapping parcel labels to vertex indices.
    """
    parcel_vertex_indices = {}
    for label in parcel_labels:
        indices = np.where(dlabels_subset == label)[0]
        parcel_vertex_indices[label] = indices
        
    return parcel_vertex_indices

def map_vertices_to_hemispheres(parcel_vertex_indices, left_parcels, right_parcels):
    """
    Separates vertex indices into left and right hemispheres.

    Returns:
    - hemisphere_indices: Dictionary with 'Left' and 'Right' keys mapping to vertex indices.
    """
    left_indices = np.concatenate([parcel_vertex_indices[label] for label in left_parcels])
    right_indices = np.concatenate([parcel_vertex_indices[label] for label in right_parcels])
    hemisphere_indices = {'Left': left_indices, 'Right': right_indices}
    
    return hemisphere_indices

# create a mapping between parcels and vertices for certain parcels
def dlabels_subset_creation(dlabels, selected_parcels):
    source_indices = get_source_indices(dlabels, selected_parcels)
    source_indices = np.array(source_indices)
    dlabels_subset = dlabels[source_indices]
    
    return dlabels_subset

def check_parcel_connection(conn_matrix, parcel_labels, parcel_indices):
    parcel_connectivity = np.zeros((len(parcel_labels), len(parcel_labels)))
    for i,parcel in enumerate(parcel_labels):
        for j,parcel2 in enumerate(parcel_labels):
            if conn_matrix[parcel-1,parcel2-1] != 0:
                parcel_connectivity[i,j] = 1
    
    return parcel_connectivity

def weight_parcel_connection(conn_matrix, parcel_labels, parcel_indices):
    parcel_connectivity = np.zeros((len(parcel_labels), len(parcel_labels)))
    for i, parcel in enumerate(parcel_labels):
        for j, parcel2 in enumerate(parcel_labels):
            connections = conn_matrix[np.ix_(parcel_indices[parcel], parcel_indices[parcel2])]
            non_zero_connections = connections[connections != 0]
            if non_zero_connections.size > 0:
                parcel_connectivity[i, j] = np.mean(non_zero_connections) * 10**4
            else:
                parcel_connectivity[i, j] = 0
                
    return parcel_connectivity

def check_parcel_connection_all_subs(conn_matrix, parcel_labels, parcel_indices):
    num_subs = conn_matrix.shape[0]
    conn_matrix_parcels = np.zeros((len(parcel_labels), len(parcel_labels)))
    for sub in range(num_subs):
        conn_matrix_parcels += check_parcel_connection(conn_matrix[sub], parcel_labels, parcel_indices)
    conn_matrix_parcels = conn_matrix_parcels*100/num_subs
    
    return conn_matrix_parcels

def weight_parcel_connection_all_subs(conn_matrix, parcel_labels, parcel_indices):
    num_subs = conn_matrix.shape[0]
    conn_matrix_parcels = np.zeros((len(parcel_labels), len(parcel_labels)))
    for sub in range(num_subs):
        conn_matrix_parcels += weight_parcel_connection(conn_matrix[sub], parcel_labels, parcel_indices)
    conn_matrix_parcels = conn_matrix_parcels/num_subs
    
    return conn_matrix_parcels

def rename_files_by_date_and_pattern(folder_path):
    """
    Renames files in the folder if they contain '_5mm_dilation' in their name and were modified
    within the last two months. The new name replaces '_5mm_dilation' with '_no_dilation'.

    Parameters:
    - folder_path: Path to the folder containing the files.
    """
    # Calculate the date two months ago from today
    date_threshold = datetime.now() - timedelta(days=90)
    
    for filename in os.listdir(folder_path):
        # Check if the filename contains the target pattern
        if '_5mm_dilation' in filename:
            file_path = os.path.join(folder_path, filename)
            
            # Ensure we're working with a file, not a directory
            if os.path.isfile(file_path):
                # Get the file's last modification time
                mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                
                # Check if the modification time is within the last two months
                if mod_time >= date_threshold:
                    # Rename the file by replacing '_5mm_dilation' with '_no_dilation'
                    new_filename = filename.replace('5mm_dilation', 'no_dilation')
                    new_file_path = os.path.join(folder_path, new_filename)
                    
                    # Rename the file
                    os.rename(file_path, new_file_path)
                    print(f"Renamed: {filename} -> {new_filename}")
                    
def symmetrize_matrix(conn_matrix):
    
    sym_matrix = (conn_matrix + conn_matrix.T) / 2
    
    return sym_matrix

def remove_negative_weights(conn_matrix):
    
    conn_matrix[conn_matrix < 0] = 0
    
    return conn_matrix

def invert_weights(conn_matrix):
    
    with np.errstate(divide='ignore'):
        distance_matrix = 1 / conn_matrix
    distance_matrix[conn_matrix == 0] = 10**6
    np.fill_diagonal(distance_matrix, 0)  # Distance from node to itself is zero
    
    return distance_matrix


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

def activation_matrix(data_run, dlabels, selected_parcels):
    source_indices = get_source_indices(dlabels, selected_parcels)
    source_indices = np.array(source_indices)
    num_vertices = len(source_indices)
    num_tps = data_run.shape[1]
    act_matrix = np.zeros((num_vertices, num_tps))

    # Initialize the row offset
    row_offset = 0

    # Load and fill the connectivity matrix
    for parcel in selected_parcels:

        # Get the vertices for the current parcel
        parcel_vertices = np.where(dlabels == parcel)[0]

        num_parcel_vertices = len(parcel_vertices)

        # Iterate over the vertices in the current parcel
        for i, vertex in enumerate(parcel_vertices):
            # Get the corresponding row index in the connectivity matrix
            row_idx = row_offset + i

            # Assign to the connectivity matrix
            act_matrix[row_idx, :] = data_run[vertex, :]

        # Update the row offset for the next parcel
        row_offset += num_parcel_vertices

    return act_matrix

def compute_model_differences(mean_corr_original, mean_corr_lesioned):
    """
    Computes the difference in mean correlations between the original and lesioned models.

    Returns:
    - diff_corr: Difference in mean correlations per step.
    """
    diff_corr = mean_corr_original - mean_corr_lesioned
    
    return diff_corr

# function that takes data for each subject and prepares them for the prf model

def prepare_actual_for_prf(sub, retinotopy_ts_data, dlabels):
    
    data = act_functions.load_ret(sub, retinotopy_ts_data)
    lv2_data = {}
    lv3_data = {}
    lv4_data = {}
    rv2_data = {}
    rv3_data = {}
    rv4_data = {}
    lv2_size = np.where(dlabels==4)[0].shape[0]
    lv3_size = np.where(dlabels==5)[0].shape[0]
    lv4_size = np.where(dlabels==6)[0].shape[0]
    rv2_size = np.where(dlabels==184)[0].shape[0]
    rv3_size = np.where(dlabels==185)[0].shape[0]
    rv4_size = np.where(dlabels==186)[0].shape[0]
    total_size = lv2_size + lv3_size + lv4_size + rv2_size + rv3_size + rv4_size
    size_until_rv4 = total_size - rv4_size
    size_left_hem = lv2_size + lv3_size + lv4_size
    for r in range(6):
        lv2_data[r] = data[r][np.where(dlabels==4)[0], :]
        lv3_data[r] = data[r][np.where(dlabels==5)[0], :]
        lv4_data[r] = data[r][np.where(dlabels==6)[0], :]
        rv2_data[r] = data[r][np.where(dlabels==184)[0], :]
        rv3_data[r] = data[r][np.where(dlabels==185)[0], :]
        rv4_data[r] = data[r][np.where(dlabels==186)[0], :]
    # Prepare the early visual system 
    actual_data_early_vis_no_zscore = {}
    for r in range(6):
        num_tps = lv2_data[r].shape[1]
        actual_data_early_vis_no_zscore[r] = np.zeros((total_size, num_tps))
        actual_data_early_vis_no_zscore[r][0:(lv2_size), :] = lv2_data[r]
        actual_data_early_vis_no_zscore[r][lv2_size:(lv2_size+lv3_size), :] = lv3_data[r]
        actual_data_early_vis_no_zscore[r][(lv2_size+lv3_size):(size_left_hem), :] = lv4_data[r]
        actual_data_early_vis_no_zscore[r][(size_left_hem):(size_left_hem+rv2_size), :] = rv2_data[r]
        actual_data_early_vis_no_zscore[r][(size_left_hem+rv2_size):(size_until_rv4), :] = rv3_data[r]
        actual_data_early_vis_no_zscore[r][(size_until_rv4):(total_size), :] = rv4_data[r]
    
    return actual_data_early_vis_no_zscore

def prepare_actflow_for_prf(sub, data, dlabels):
    '''
    sub: int denoting the subject whose data are to be prepared, not the group int
    data: data of all subjects, in shape: (subjects, steps, runs, vertices, timepoints), vertices here is only lv1-rv4 (~2565, depending on dlabels)
    '''
    data_sub = data[sub,5] # we extract the last step, here its 6th (5th in 0-based)
    num_tps = data.shape[4]
    num_runs = data.shape[2]
    lv2_size = np.where(dlabels==4)[0].shape[0]
    lv3_size = np.where(dlabels==5)[0].shape[0]
    lv4_size = np.where(dlabels==6)[0].shape[0]
    rv2_size = np.where(dlabels==184)[0].shape[0]
    rv3_size = np.where(dlabels==185)[0].shape[0]
    rv4_size = np.where(dlabels==186)[0].shape[0]
    total_size = lv2_size + lv3_size + lv4_size + rv2_size + rv3_size + rv4_size
    lv1_size = np.where(dlabels==1)[0].shape[0]
    rv1_size = np.where(dlabels==181)[0].shape[0]
    size_until_rv4 = total_size - rv4_size
    size_left_hem = lv2_size + lv3_size + lv4_size
    actflow_data_early_vis_no_zscore = {}
    for r in range(num_runs):
        actflow_data_early_vis_no_zscore[r] = np.zeros((total_size, num_tps))
        actflow_data_early_vis_no_zscore[r][0:(size_left_hem), :] = data_sub[r][(lv1_size):(size_left_hem+lv1_size), :]
        actflow_data_early_vis_no_zscore[r][(size_left_hem):(total_size), :] = data_sub[r][(size_left_hem+lv1_size+rv1_size):,:]
    
    return actflow_data_early_vis_no_zscore


# meas = 0 - ang, 1-ecc
def process_all_subjects(prf_model_data_folder, dlabels, parcels, subs, meas, group):
    actual_results = {}
    generated_results = {}
    for sub in subs:
        print(sub)
        subject_id = subj_here[sub]
        # Load data for the subject
        all_results_actual, all_results_generated = load_subject_data(prf_model_data_folder, subject_id, group)
        all_results_actual = all_results_actual[meas, :]
        all_results_generated = all_results_generated[meas, :]
        # Assign actual data to full brain matrix
        full_brain_actual = assign_values_to_full_brain(all_results_actual, dlabels, parcels)
        
        # Assign generated data to full brain matrix
        full_brain_generated = assign_values_to_full_brain(all_results_generated, dlabels, parcels)
        
        actual_results[sub] = full_brain_actual
        generated_results[sub] = full_brain_generated
    
    return actual_results, generated_results


def assign_values_to_full_brain(early_visual_matrix, dlabels, parcels):
    full_brain_matrix = np.zeros((91282))
    early_visual_index = 0

    for parcel in parcels:
        parcel_indices = np.where(dlabels == parcel)[0]
        parcel_values = early_visual_matrix[early_visual_index : early_visual_index + len(parcel_indices)]
        full_brain_matrix[parcel_indices] = parcel_values
        early_visual_index += len(parcel_indices)

    return full_brain_matrix


def load_subject_data(prf_model_data_folder, subject_id, group):
    #gen_results_fold = os.path.join(prf_model_data_folder, 'actflow/V1_initiated/not_zscore/orig_conn/comb_hems/5_layers/')
    os.chdir(prf_model_data_folder)
    
    # Load actual data
    if group == 'A':
        actual_file = f'actual_data_early_vis_no_zscore_all_seeds_{subject_id}_myPRFResults_new.mat'
        generated_file = f'V1_initiated_actflow_all_seeds_{subject_id}_myPRFResults_new.mat'
    else:
        actual_file = f'actual_data_early_vis_{subject_id}_myPRFResults_new.mat'
        generated_file = f'actflow_data_early_vis_{subject_id}_myPRFResults_new.mat'
    h5f_actual = h5py.File(actual_file, 'r')
    all_results_actual = h5f_actual['allresults']
    all_results_actual = all_results_actual[0, 0, :, :]  # Assuming shape [0, 0, num_vertices, num_measurements]
    h5f_actual.close()
    print('result shape = ',all_results_actual.shape)
    #os.chdir(gen_results_fold)
    # Load generated data
    h5f_generated = h5py.File(generated_file, 'r')
    all_results_generated = h5f_generated['allresults']
    all_results_generated = all_results_generated[0, 0, :, :]  # Assuming shape [0, 0, num_vertices, num_measurements]
    h5f_generated.close()

    return all_results_actual, all_results_generated
    
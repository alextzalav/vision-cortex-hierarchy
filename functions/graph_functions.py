import igraph as ig
import numpy as np
import os
import sys

basedir2 = '/projects/f_mc1689_1/MovieActFlow2/'
scripts_dir2 = basedir2 + 'docs/scripts/'
final_results_dir = basedir2 + 'data/final_results/'
ret_scripts = scripts_dir2 + 'retinotopic/'
functions_dir = ret_scripts + 'project_functions/'
sys.path.append(functions_dir)

import helper_functions
import plotting_functions
import V1_initiated_model
import stats_functions

selected_parcels = [1, 4, 5, 6, 181, 184, 185, 186]
selected_parcels_output = [4, 5, 6, 184, 185, 186]


def generate_null_fc_original(fc_matrix, dlabels_subset, random_state=None):
    """
    Generate one null FC matrix by permuting inter-parcel connections within each row (same parcel vertices excluded)

    Parameters
    ----------
    fc_matrix : np.ndarray, shape (n, n)
        Original FC matrix with zeros on diagonal and within-parcel.
    parcel_labels : array-like, shape (n,)
        Integer parcel label for each vertex (e.g., 1–8).
    random_state : int or None
        Seed for reproducibility.

    Returns
    -------
    null_fc : np.ndarray, shape (n, n)
        Permuted FC matrix (one null).
    """
    rng = np.random.RandomState(random_state)
    n = fc_matrix.shape[0]
    null_fc = np.zeros_like(fc_matrix)

    for i in range(n):
        # mask of allowed columns (different parcel)
        allowed = dlabels_subset != dlabels_subset[i]
        idx_allowed = np.where(allowed)[0]
        # extract and shuffle
        values = fc_matrix[i, idx_allowed].copy()
        rng.shuffle(values)
        # assign back
        null_fc[i, idx_allowed] = values

    return null_fc


def generate_null_fc_everything(fc_matrix, dlabels_subset, random_state=None):
    """
    Generate one null FC matrix by permuting every connection within each row.

    Parameters
    ----------
    fc_matrix : np.ndarray, shape (n, n)
        Original FC matrix with zeros on diagonal and within-parcel.
    parcel_labels : array-like, shape (n,)
        Integer parcel label for each vertex (e.g., 1–8).
    random_state : int or None
        Seed for reproducibility.

    Returns
    -------
    null_fc : np.ndarray, shape (n, n)
        Permuted FC matrix (one null).
    """
    rng = np.random.RandomState(random_state)
    n = fc_matrix.shape[0]
    null_fc = np.zeros_like(fc_matrix)

    for i in range(n):
        # extract and shuffle
        values = fc_matrix[i, :].copy()
        rng.shuffle(values)
        # assign back
        null_fc[i, :] = values

    return null_fc


def generate_null_fc_zeros_in_place(fc_matrix, dlabels_subset, random_state=None):
    """
    Generate one null FC matrix by permuting non-zero inter-parcel connections within each row.

    Parameters
    ----------
    fc_matrix : np.ndarray, shape (n, n)
        Original FC matrix with zeros on diagonal and within-parcel.
    parcel_labels : array-like, shape (n,)
        Integer parcel label for each vertex (e.g., 1–8).
    random_state : int or None
        Seed for reproducibility.

    Returns
    -------
    null_fc : np.ndarray, shape (n, n)
        Permuted FC matrix (one null).
    """
    rng = np.random.RandomState(random_state)
    n = fc_matrix.shape[0]
    null_fc = np.zeros_like(fc_matrix)

    for i in range(n):
        # mask of allowed columns (different parcel)
        allowed = fc_matrix[i, :] != 0
        idx_allowed = np.where(allowed)[0]
        # extract and shuffle
        values = fc_matrix[i, idx_allowed].copy()
        rng.shuffle(values)
        # assign back
        null_fc[i, idx_allowed] = values

    return null_fc


# normalize it to run communicability calculation 
def normalize_adjacency_matrix(A):
    
    A = A.astype(float)
    # Compute the strengths (weighted degrees) of each node
    strengths = A.sum(axis=1)
    
    # Initialize the inverse square root of strengths
    inv_sqrt_strengths = np.zeros_like(strengths)
    
    # Avoid division by zero by handling nodes with zero strength
    nonzero_indices = strengths > 0
    inv_sqrt_strengths[nonzero_indices] = 1.0 / np.sqrt(strengths[nonzero_indices])
    print(strengths)
    print('Inverted strengths sqrt = ', inv_sqrt_strengths)
    # For nodes with zero strength, the inv_sqrt_strength remains zero

    # Create the diagonal matrix D^{-1/2}
    D_inv_sqrt = np.diag(inv_sqrt_strengths)
    print(D_inv_sqrt)
    
    # Compute the normalized adjacency matrix using matrix multiplication
    A_normalized = D_inv_sqrt @ A @ D_inv_sqrt  # D^{-1/2} A D^{-1/2}
    
    return A_normalized

# normalize it to run communicability calculation 
def normalize_adjacency_matrix_symmetric(A):
    
    A = A.astype(float)
    A_sym = 0.5 * (A + A.T)
    # Compute the strengths (weighted degrees) of each node
    strengths = A_sym.sum(axis=1)
    
    # Initialize the inverse square root of strengths
    inv_sqrt_strengths = np.zeros_like(strengths)
    
    # Avoid division by zero by handling nodes with zero strength
    nonzero_indices = strengths > 0
    inv_sqrt_strengths[nonzero_indices] = 1.0 / np.sqrt(strengths[nonzero_indices])
    print(strengths)
    print('Inverted strengths sqrt = ', inv_sqrt_strengths)
    # For nodes with zero strength, the inv_sqrt_strength remains zero

    # Create the diagonal matrix D^{-1/2}
    D_inv_sqrt = np.diag(inv_sqrt_strengths)
    print(D_inv_sqrt)
    
    # Compute the normalized adjacency matrix using matrix multiplication
    A_normalized = D_inv_sqrt @ A_sym @ D_inv_sqrt  # D^{-1/2} A D^{-1/2}
    
    return A_normalized


def calc_communicability_parcel(comm_matrix_vertex, parcel_labels, parcel_indices):
    
    parcel_communicability = np.zeros((len(parcel_labels), len(parcel_labels)))
    for i, parcel in enumerate(parcel_labels):
        for j, parcel2 in enumerate(parcel_labels):
            connections = comm_matrix_vertex[np.ix_(parcel_indices[parcel], parcel_indices[parcel2])]
            non_zero_connections = connections[connections != 0]
            if non_zero_connections.size > 0:
                parcel_communicability[i, j] = np.mean(non_zero_connections) 
            else:
                parcel_communicability[i, j] = 0
                
    return parcel_communicability


def threshold_matrix_graph_analysis(conn_matrix, weight_handling='positive'):
    
    # Handle the weights according to the specified method
    if weight_handling == 'positive':
        weights = conn_matrix.copy()
        weights[weights < 0] = 0  # Set negative weights to zero
    elif weight_handling == 'negative':
        weights = conn_matrix.copy()
        weights[weights > 0] = 0  # Set positive weights to zero
        weights = np.abs(weights)  # Convert to positive values
    elif weight_handling == 'absolute':
        weights = np.abs(conn_matrix)
    elif weight_handling == 'transformed':
        # Example transformation; adjust as needed
        weights = ((conn_matrix + 1) / 2) ** 12
    else:
        raise ValueError("Invalid weight_handling method")
        
    return weights


def threshold_conn_matrix(connectivity_matrix, threshold):
    """
    Retains the top 10% of weights in the connectivity matrix, setting all other values to zero.

    Parameters:
    - connectivity_matrix: numpy array of shape (vertices, vertices)
    - threshold: value from 0 to 100. percentile to be removed

    Returns:
    - filtered_matrix: numpy array with only the top 10% of weights retained
    """
    # Flatten the matrix and find the threshold for the top 10% of weights
    flattened_matrix = connectivity_matrix.flatten()
    threshold = np.percentile(flattened_matrix, threshold)
    
    # Create a copy of the matrix and keep only values above the threshold
    filtered_matrix = np.where(connectivity_matrix >= threshold, connectivity_matrix, 0)
    
    return filtered_matrix

def threshold_conn_matrix_all_subs(connectivity_matrix_all_subs, threshold):
    numsubs = connectivity_matrix_all_subs.shape[0]
    numvert = connectivity_matrix_all_subs.shape[1]
    filtered_matrix_all_subs = np.zeros((numsubs,numvert,numvert))
    for sub in range(numsubs):
        conn_matrix = connectivity_matrix_all_subs[sub]
        thres_conn_matrix = threshold_conn_matrix(conn_matrix, threshold)
        filtered_matrix_all_subs[sub] = thres_conn_matrix
    
    return filtered_matrix_all_subs


def run_null_model(seed, data_all, connectivity_matrix_all_subs, subjects, dlabels, null_method=None):
    '''
    data_all: all the retinotopic data of the subjects
    verify that v1 initiated model and this function in general works independently of the subjects provided 
    '''

    num_subs = len(data_all)
    dlabels_subset_output = helper_functions.dlabels_subset_creation(dlabels, selected_parcels_output)    
    dlabels_subset = helper_functions.dlabels_subset_creation(dlabels, selected_parcels)

    sub_nums = subjects
    num_vertices = len(dlabels_subset)
    steps = 6
    sources = [1, 181]

    FC_shuffle = np.zeros((num_subs, num_vertices, num_vertices))
    for sub in range(num_subs):
        fc = connectivity_matrix_all_subs[sub]
        if null_method == 'everything':
            FC_shuffle[sub] = generate_null_fc_everything(fc, dlabels_subset, random_state=seed)
        elif null_method == 'zeros':
            FC_shuffle[sub] = generate_null_fc_zeros_in_place(fc, dlabels_subset, random_state=seed)
        else:
            FC_shuffle[sub] = generate_null_fc_original(fc, dlabels_subset, random_state=seed)


    v1_pred_act_all, corr_all_subjects, r2_all_subjects = V1_initiated_model.run_model_parallel_across_subjects(data_all,
        sub_nums, steps, sources, selected_parcels, dlabels,
        lesion_type=None, dilation=False, threshold=None, LGN=False, LGN_but_V1=False,
        FC_shuffle=FC_shuffle
    )

    # Forgiving method
    corr_final_original, r2_final_original = V1_initiated_model.model_comp_parcel_level(v1_pred_act_all, data_all, selected_parcels, dlabels)
    r2_final_original_last_step = r2_final_original[:, 5, :]  # extract last step
    r2_final_original_last_step_avg_subs = np.mean(r2_final_original_last_step, axis=0)
    r2_forgiving_max = np.max(r2_final_original_last_step_avg_subs)

    # Unforgiving method
    r2_avg_vert = np.zeros((num_subs, 6, 6, 6))
    for sub in range(num_subs):
        for step in range(6):
            for run in range(6):
                for parcel in range(6):
                    indices_here = np.where(dlabels_subset_output == selected_parcels_output[parcel])[0]
                    r2_here = r2_all_subjects[sub, step, run, indices_here]
                    r2_avg_vert[sub, step, run, parcel] = np.mean(r2_here)
    r2_avg_vert_avg_subs_avg_runs = np.mean(r2_avg_vert, axis=(0, 2))
    r2_avg_vert_avg_subs_avg_runs_last_step = r2_avg_vert_avg_subs_avg_runs[5]
    r2_unforgiving_max = np.max(r2_avg_vert_avg_subs_avg_runs_last_step)

    # Vertex-wise method
    r2_all_subjects_avg_runs = np.mean(r2_all_subjects, axis=2)
    r2_all_subjects_avg_runs_last_step = r2_all_subjects_avg_runs[:, 5, :]
    r2_all_subjects_avg_runs_last_step_avg_subs = np.mean(r2_all_subjects_avg_runs_last_step, axis=0)
    r2_vertexwise_max = np.max(r2_all_subjects_avg_runs_last_step_avg_subs)

    return {
        'seed': seed,
        'r2_forgiving_max': r2_forgiving_max,
        'r2_unforgiving_max': r2_unforgiving_max,
        'r2_vertexwise_max': r2_vertexwise_max
    }


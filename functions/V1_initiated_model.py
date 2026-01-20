import sys
import os
sys.path.append('/projects/f_mc1689_1/MovieActFlow2/docs/scripts/Act_Flow/')
sys.path.append('/projects/f_mc1689_1/MovieActFlow2/docs/scripts/retinotopic/project_functions/')
import actflow_multilayer_retinotopic_basic_functions as act_functions
import helper_functions
import graph_functions
import numpy as np
from joblib import Parallel, delayed
import time
from sklearn.metrics import r2_score, mean_squared_error

basedir2 = '/projects/f_mc1689_1/MovieActFlow2/'
retinotopy_ts_data = basedir2 + 'data/Retinotopy/HCPMSMALL/' # folder that contains the preprocessed timeseries data
pc_folder = basedir2 + 'data/Rest/fc/pc_fc/'
V1 = [1, 181]
V2 = [4, 184]
V3 = [5, 185]
V4 = [6, 186]

# all subjects for this project (Apr 2025)
subj_here = ['100610', '102311', '102816', '104416', '105923', '108323', '109123', '111514', '114823', '115017', '115825', '116726', 
             '118225', '125525', '126426', '467351', '525541', '825048', '826353', '833249', '859671', '861456', '871762', '872764', 
             '878776', '878877', '898176', '899885', '901139', '901442', '905147', '910241', '926862', '927359', '942658', '943862', 
             '958976', '966975', '971160', '995174', '128935', '131722', '137128', '140117', '144226', '283543', '318637', '320826', 
             '330324', '346137', '541943', '547046', '562345', '572045', '573249', '581450', '601127', '617748', '627549', '638049', 
             '644246', '654552', '671855', '680957', '690152', '706040', '724446', '725751', '732243', '751550', '757764', '765864', 
             '770352', '771354', '782561', '783462', '789373', '814649', '818859', '951457'] # this is the full dataset, 80 subs. 0-39 (included) is group A,
            # 40-79 (included) is group B (control)
    

def calc_score_avg_ts_for_each_parcel_runs_for_each_step_sinlge_sub(data, pred_data, dlabels_subset):
    steps = pred_data.shape[0]
    corr = np.zeros((steps,6,len(np.unique(dlabels_subset))))
    r2 = np.zeros((steps,6,len(np.unique(dlabels_subset))))
    for step in range(steps):
        for r in range(6):
            corr[step][r], r2[step][r] = calc_score_avg_ts_parcel(data[r], pred_data[step][r], dlabels_subset)
    corr_avg_runs = np.mean(corr, axis=1)
    r2_avg_runs = np.mean(r2, axis=1)
    # Define which indices to keep (include)
    include_indices = [1, 2, 3, 5, 6, 7]
    # Create a full range of possible indices
    indices = np.arange(r2_avg_runs.shape[1])
    # Create a mask of booleans that is True only for indices we want to keep
    mask = np.isin(indices, include_indices)
    # Apply the mask to select the columns
    r2_avg_runs_final = r2_avg_runs[:, mask]
    corr_avg_runs_final = corr_avg_runs[:, mask]
    
    return corr_avg_runs_final, r2_avg_runs_final


# calculate correlation and r2 based on the average time series for each parcel. more forgiving method. 
def calc_score_avg_ts_parcel(act_orig, act_pred, dlabels_subset):
    parcels = np.sort(np.unique(dlabels_subset))
    corr_avg_parcels = np.zeros((len(parcels)))
    r2_avg_parcels = np.zeros((len(parcels)))
    for idx, parcel in enumerate(parcels):
        indices = np.where(dlabels_subset==parcel)[0]
        avg_orig = np.mean(act_orig[indices, :], axis=0)
        avg_pred = np.mean(act_pred[indices, :], axis=0)
        if np.std(avg_orig) == 0 or np.std(avg_pred) == 0:
            corr_coef = 0
            r2 = 0
        else:
            corr_coef = np.corrcoef(avg_orig, avg_pred)[0, 1]
            r2 = r2_score(avg_orig, avg_pred)
        
        corr_avg_parcels[idx] = corr_coef
        r2_avg_parcels[idx] = r2
        
    return corr_avg_parcels, r2_avg_parcels


def extract_early_vis_sub(data, dlabels, selected_parcels):
    dlabels_subset = helper_functions.dlabels_subset_creation(dlabels, selected_parcels)
    parcels_here = np.unique(dlabels_subset)
    num_tps = data[0].shape[1]
    num_runs = len(data)
    vertices_visual = len(dlabels_subset)
    actual_data_early_vis_no_zscore = np.zeros((num_runs, vertices_visual, num_tps))
    for r in range(num_runs):
        index = 0
        for parcel in parcels_here:
            vert_ind_here = np.where(dlabels==parcel)[0]
            num_vert_here = len(vert_ind_here)
            actual_data_early_vis_no_zscore[r][index:index+num_vert_here ,:]= data[r][vert_ind_here, :]
            index = index+num_vert_here
    
    return actual_data_early_vis_no_zscore


def model_comp_parcel_level(model_pred, data_all, selected_parcels, dlabels):
    corr_avg = np.zeros((len(data_all), model_pred.shape[1], len(selected_parcels)-2)) # subjects x steps
    r2_avg = np.zeros((len(data_all), model_pred.shape[1], len(selected_parcels)-2))
    dlabels_subset = helper_functions.dlabels_subset_creation(dlabels, selected_parcels)
    num_subs = data_all.keys()
    for i,sub in enumerate(num_subs):
        pred_data = model_pred[i]
        data = extract_early_vis_sub(data_all[sub], dlabels, selected_parcels)
        corr_avg[i], r2_avg[i] = calc_score_avg_ts_for_each_parcel_runs_for_each_step_sinlge_sub(data, pred_data, dlabels_subset)
        
    return corr_avg, r2_avg
    

def calc_score(act_orig, act_pred):
    num_vert = act_orig.shape[0]
    correlation = np.zeros((num_vert))
    r_squared = np.zeros((num_vert))
    for vertex in range(num_vert):
        ts1 = act_orig[vertex, :]
        ts2 = act_pred[vertex, :]
        if np.std(ts2) == 0:
            correlation[vertex] = 0
            r_squared[vertex] = 0
        else:
            correlation[vertex] = np.corrcoef(ts1, ts2)[0, 1]
            r_squared[vertex] = r2_score(ts1, ts2)
     
    return correlation, r_squared


def get_indices(parcels, dlabels_form):
    parcel_indices = []
    for parcel in parcels:
        parcel_indices_here = np.where(dlabels_form==parcel)[0]
        parcel_indices.append(parcel_indices_here)
    parcel_indices = np.concatenate(parcel_indices)
    
    return parcel_indices


def actflow_early_vis_model(data_run, conn_matrix, sources, targets, orig_data_run, dlabels_here, selected_parcels_here):
    num_tps = data_run.shape[1]
    #if 361 in sources:
    #    selected_parcels = 
    dlabels_subset = helper_functions.dlabels_subset_creation(dlabels_here, selected_parcels_here)
    # dlabels_subset : for each vertex in the small matrix,
    # tells us the original parcel it belongs to, v1, v2 etc
    source_indices = get_indices(sources, dlabels_subset)
    target_indices = get_indices(targets, dlabels_subset)
    #target_indices = get_indices(targets, dlabels_subset_output) # for logic 2565 x 4183 instead of the original 4183 x 4183. 
    predicted_activation = np.dot(conn_matrix[np.ix_(target_indices, source_indices)] , data_run[source_indices, :])
    correlation, r_squared = calc_score(orig_data_run[target_indices, :], predicted_activation)
        
    return  predicted_activation, correlation, r_squared


def actflow_all_runs(data, conn_matrix, sources, targets, num_target_vert, orig_data, dlabels_here, selected_parcels_here):
    num_tps = data[0].shape[1]
    num_runs = data.shape[0]
    correlation = np.zeros((num_runs, num_target_vert))
    r_squared = np.zeros((num_runs, num_target_vert))
    pred_act = np.zeros((num_runs, num_target_vert, num_tps))
    for r in range(num_runs):
        pred_act[r, :, :], correlation[r,:], r_squared[r,:] = actflow_early_vis_model(data[r], conn_matrix, sources, targets, orig_data[r], dlabels_here, selected_parcels_here)
        
    return pred_act, correlation, r_squared


def v1_initiated_model(data, conn_matrix, dlabels_here, selected_parcels_here, steps, sources):
    '''
    sources: list with source parcels
    '''
    
    time1 = time.time()
    num_tps = data.shape[2]
    dlabels_subset = helper_functions.dlabels_subset_creation(dlabels_here, selected_parcels_here)
    sources_orig = [1, 181]
    sources_orig = sources
    if sources_orig == [1, 181]:
        targets = [4, 5, 6, 184, 185, 186]
    else:
        targets = [1, 4, 5, 6, 181, 184, 185, 186]
    sources_orig_indices = get_indices(sources_orig, dlabels_subset)
    target_indices = get_indices(targets, dlabels_subset)
    num_target_vertices = target_indices.shape[0]
    pred_act_steps = np.zeros((steps, 6, data.shape[1], num_tps))
    correlation = np.zeros((steps, 6, num_target_vertices))
    r_squared = np.zeros((steps, 6, num_target_vertices))
    pred_act_steps[0, :, :, :] = np.copy(data)
    pred_act, corr_here, r2_here = actflow_all_runs(data, conn_matrix, sources_orig, targets, num_target_vertices, data, dlabels_here, selected_parcels_here)
    pred_act_transposed = pred_act.transpose(1, 0, 2)  # Shape: (2565, 6, 300)
    pred_act_steps[0, :, target_indices, :] = pred_act_transposed
    correlation[0, :, :] = corr_here
    r_squared[0, :, :] = r2_here
    if sources == [1, 181]:
        sources = [1, 4, 5, 6, 181, 184, 185, 186]
    else:
        sources = (np.concatenate((targets, sources))).tolist()
    for step in range(1,steps):
        pred_act_steps[step, :, sources_orig_indices, :] = np.copy(data[:, sources_orig_indices, :].transpose(1, 0, 2))
        pred_act, corr_here, r2_here = actflow_all_runs(pred_act_steps[step - 1], conn_matrix, sources, targets, num_target_vertices, data, dlabels_here, selected_parcels_here)
        pred_act_transposed = pred_act.transpose(1, 0, 2)
        pred_act_steps[step, :, target_indices, :] = pred_act_transposed
        correlation[step, :, :] = corr_here
        r_squared[step, :, :] = r2_here
    time2 = time.time()
    print('Time it took to run the model for 1 sub = ', time2-time1)
    
    return pred_act_steps, correlation, r_squared   


def lesion_connections(conn_matrix, dlabels_subset, lesion_pairs):
    """
    Lesion connections in conn_matrix by setting specified connections to zero.

    Parameters:
    - conn_matrix: The original connectivity matrix (numpy array).
    - dlabels_subset: Array mapping each vertex in the matrix to its parcel label.
    - lesion_pairs: List of tuples, each containing:
        (source_labels, target_labels), where source_labels and target_labels
        are lists of parcel labels.

    Returns:
    - conn_matrix_lesioned: A copy of conn_matrix with specified connections set to zero.
    """
    # Make a copy to avoid modifying the original connectivity matrix
    conn_matrix_lesioned = conn_matrix.copy()
    for source_labels, target_labels in lesion_pairs:
        # Get indices for sources and targets
        source_indices = get_indices(source_labels, dlabels_subset)
        target_indices = get_indices(target_labels, dlabels_subset)
        # Set the connections from sources to targets to zero
        conn_matrix_lesioned[np.ix_(target_indices, source_indices)] = 0
        
    return conn_matrix_lesioned
    
def lesion_hierarchical(conn_matrix, dlabels_subset, lesion_feedback=False):
    """
    Performs hierarchical lesions by setting connections between V1-V2, V2-V3, and V3-V4 to zero.
    This includes all combinations across both hemispheres.
    lesion pairs are in the form source to target

    Returns:
    - conn_matrix_hierarchical_lesioned: Connectivity matrix after hierarchical lesions.
    """
    # Define the lesion pairs for hierarchical lesions
    lesion_pairs = [
        #(V1, V2),  
        (V2, V3),  
        (V3, V4)   
    ]
    
    if lesion_feedback:
        lesion_pairs += [
    #(V2, V1),  
    (V3, V2),  
    (V4, V3)   
]
    # Perform the lesions
    conn_matrix_hierarchical_lesioned = lesion_connections(conn_matrix, dlabels_subset, lesion_pairs)
    
    return conn_matrix_hierarchical_lesioned

def lesion_direct(conn_matrix, dlabels_subset, lesion_feedback=False):
    """
    Performs direct lesions by setting connections between V1-V3, V1-V4, and V2-V4 to zero.
    This includes all combinations across both hemispheres.

    Returns:
    - conn_matrix_direct_lesioned: Connectivity matrix after direct lesions.
    """
    # Define the lesion pairs for direct lesions
    lesion_pairs = [
        (V1, V3),  
        (V1, V4)  
        #(V2, V4)   
    ]
    
    if lesion_feedback:
        lesion_pairs += [
    (V3, V1),  
    (V4, V1)  
    #(V4, V2)   
]
        
    # Perform the lesions
    conn_matrix_direct_lesioned = lesion_connections(conn_matrix, dlabels_subset, lesion_pairs)
    
    return conn_matrix_direct_lesioned


def lesion_hierarchical_extra(conn_matrix, dlabels_subset):
    """
    Performs hierarchical lesions by setting connections between V1-V2, V2-V3, and V3-V4 to zero.
    This includes all combinations across both hemispheres.
    lesion pairs are in the form source to target

    Returns:
    - conn_matrix_hierarchical_lesioned: Connectivity matrix after hierarchical lesions.
    """
    # Define the lesion pairs for hierarchical lesions
    lesion_pairs = [
        (V1, V2),  
        (V2, V3),  
        (V3, V4)   
    ]
    # Perform the lesions
    conn_matrix_hierarchical_lesioned = lesion_connections(conn_matrix, dlabels_subset, lesion_pairs)
    
    return conn_matrix_hierarchical_lesioned

def lesion_direct_extra(conn_matrix, dlabels_subset):
    """
    Performs hierarchical lesions by setting connections between V1-V2, V2-V3, and V3-V4 to zero.
    This includes all combinations across both hemispheres.
    lesion pairs are in the form source to target

    Returns:
    - conn_matrix_hierarchical_lesioned: Connectivity matrix after hierarchical lesions.
    """
    # Define the lesion pairs for hierarchical lesions
    lesion_pairs = [
        (V1, V3),  
        (V1, V4),  
        (V2, V4)   
    ]
    # Perform the lesions
    conn_matrix_hierarchical_lesioned = lesion_connections(conn_matrix, dlabels_subset, lesion_pairs)
    
    return conn_matrix_hierarchical_lesioned

def run_model_for_subject(data, sub_num, i, steps, sources, selected_parcels, dlabels, lesion_type=None, lesion_feedback=False, dilation=True, threshold=None, LGN=False, LGN_but_V1=False, FC_shuffle=None):
    """
    Loads data and connectivity matrix for a subject, runs the model, and returns the results.

    Parameters:
    - data_here: data for the subject running right now
    - sub_num: identifier for the subject to extract it from subj_here
    - i: identifier for the subject for the current batch of subjects
    - steps: Number of steps to run the model
    - lesion_type: none=no lesions, 'hier' = hierarchical lesions, 'dir' = direct lesions
    - FC_shuffle: shuffled FC for all subjects for a single permutation

    Returns:
    - v1_pred_act: Predicted activations (steps x runs x vertices x timepoints)
    - corr_all: Correlations (steps x runs x vertices)
    - r2_all: R-squared values (steps x runs x vertices)
    """
    nVertices = 59412 # cortex only
    dlabels_subset = helper_functions.dlabels_subset_creation(dlabels, selected_parcels)
    num_vertices = dlabels_subset.shape[0]
    dlabels_LGN_pul = np.concatenate((dlabels, [361, 362, 363, 364]))
    if LGN:
        LGN_pul_parcels = [361,362,363,364]
        selected_parcels_here = np.concatenate((selected_parcels, LGN_pul_parcels)).tolist() 
        data_matrix_runs = np.zeros((6, num_vertices+4, 300))
        dlabels_here = dlabels_LGN_pul
    else:
        data_matrix_runs = np.zeros((6, num_vertices, 300))
        dlabels = dlabels[:nVertices]
        dlabels_here = dlabels
        selected_parcels_here = selected_parcels
    subject_id = subj_here[int(sub_num)]
    pc_res = os.path.join(pc_folder, subject_id, 'ICA_Fix/')
    pickle_dir = pc_res
    num_runs = 6
    for r in range(num_runs):
        data_matrix_runs[r,:,:] = helper_functions.activation_matrix(data[r], dlabels_here, selected_parcels_here)    
        
    if FC_shuffle is not None:
        connectivity_matrix = FC_shuffle[i]
    else:
        connectivity_matrix = act_functions.create_square_connectivity_matrix(pickle_dir, dlabels_here, selected_parcels_here, dilation, LGN, LGN_but_V1)
    
    if threshold is not None:
        connectivity_matrix = graph_functions.threshold_conn_matrix(connectivity_matrix, threshold)
    
    if lesion_type == 'hier':
        connectivity_matrix = lesion_hierarchical(connectivity_matrix, dlabels_subset, lesion_feedback)
    elif lesion_type == 'dir':
        connectivity_matrix = lesion_direct(connectivity_matrix, dlabels_subset, lesion_feedback)
    elif lesion_type == 'hier_extra':
        connectivity_matrix = lesion_hierarchical_extra(connectivity_matrix, dlabels_subset)
    elif lesion_type == 'dir_extra':
        connectivity_matrix = lesion_direct_extra(connectivity_matrix, dlabels_subset)

    # Run the model
    v1_pred_act, corr_all, r2_all = v1_initiated_model(data_matrix_runs, connectivity_matrix, dlabels_here, selected_parcels_here, steps=steps, sources=sources)

    return v1_pred_act, corr_all, r2_all

def run_model_parallel_across_subjects(data_all, sub_nums, steps, sources, selected_parcels, dlabels, lesion_type, lesion_feedback=False, dilation=True, threshold=None, LGN=False, LGN_but_V1=False, FC_shuffle=None):
    """
    Runs the model in parallel across subjects using joblib.

    Parameters:
    - sub_nums: List of subject identifiers
    - steps: Number of steps to run the model

    Returns:
    - v1_pred_act_all: numpy array of shape (subjects, steps, runs, vertices, timepoints)
    - corr_all_subjects: numpy array of shape (subjects, steps, runs, vertices)
    - r2_all_subjects: numpy array of shape (subjects, steps, runs, vertices)
    """
    # Use joblib's Parallel and delayed to run the model across subjects
    results = Parallel(n_jobs=-1)(
        delayed(run_model_for_subject)(data_all[sub_num], sub_num, i, steps, sources, selected_parcels, dlabels, lesion_type, lesion_feedback, dilation, threshold, LGN, LGN_but_V1, FC_shuffle) for i,sub_num in enumerate(sub_nums)
    )

    # Unpack the results and collect them into lists
    v1_pred_act_list = []
    corr_all_list = []
    r2_all_list = []

    for v1_pred_act, corr_all, r2_all in results:
        v1_pred_act_list.append(v1_pred_act)
        corr_all_list.append(corr_all)
        r2_all_list.append(r2_all)

    # Convert lists to numpy arrays
    v1_pred_act_all = np.array(v1_pred_act_list)        # Shape: (subjects, steps, runs, vertices, timepoints)
    corr_all_subjects = np.array(corr_all_list)         # Shape: (subjects, steps, runs, vertices)
    r2_all_subjects = np.array(r2_all_list)             # Shape: (subjects, steps, runs, vertices)

    return v1_pred_act_all, corr_all_subjects, r2_all_subjects

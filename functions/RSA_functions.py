import numpy as np
from scipy.special import gamma
from scipy.interpolate import CubicSpline
import os
import h5py
from scipy.stats import shapiro, pearsonr, spearmanr

run_names = ['CCW', 'CW', 'EXP', 'CON', 'BAR']
num_subs = 40
runs_to_use = [0,1,2,3,4]
# last two runs which are the bar, are identical. therefore we can average the whole runs and 
# keep all the time points
T = np.arange(22, 278)


def getDimensionality(data):
    """
    data: square, symmetric matrix (e.g., an RSM)
    Returns the effective dimensionality.
    """
    # For a symmetric matrix, eigenvalues are real.
    data = np.nan_to_num(data, nan=0.0)
    eigenvalues, _ = np.linalg.eig(data)
    eigenvalues = np.real(eigenvalues)
    
    dimensionality_nom = np.sum(eigenvalues)
    dimensionality_denom = np.sum(eigenvalues**2)
    
    dimensionality = (dimensionality_nom**2) / dimensionality_denom
    
    return dimensionality


def RSM_post_processing(RSM, first_halves, second_halves):
    '''
    some values are NaN because there are rest periods where all pixels are 0. correlation of 
    something steady with anything else will result to a nan. 
    Distinguish between constant-vs-constant and constant-vs-non-constant
    '''
    big_stim_first = np.concatenate(first_halves, axis=1)
    big_stim_second = np.concatenate(second_halves, axis=1)
    is_constant_first = np.all(big_stim_first == 0, axis=0)
    is_constant_second = np.all(big_stim_second == 0, axis=0)
    for i in range(RSM.shape[0]):
        for j in range(RSM.shape[1]):
            # Both timepoints are constant
            if is_constant_first[i] and is_constant_second[j]:
                RSM[i, j] = 1.0 # previosly had that as 0. lets test what happens when it is 1. 
            # One constant, one not => correlation = 0
            elif is_constant_first[i] != is_constant_second[j]:
                RSM[i, j] = 0.0
    
    return RSM


def create_stimulus_RSM(basedir2, total_time, convolve_hrf=False, convert_to_grayscale=False, conv_type="full"):
    """
    Creates a stimulus RSM (timepoints x timepoints) from 5 runs 
    of retinotopy stimuli, optionally convolving each run's data 
    with a canonical HRF after downsampling, and optionally converting 
    the images to grayscale.
    
    Parameters:
      basedir2: Base directory for the data.
      convolve_hrf: Boolean flag whether to convolve with an HRF.
      convert_to_grayscale: Boolean flag whether to convert color images to grayscale.
    
    Returns: 
      stimulus_RSM (numpy array), and the concatenated stimulus data.
    """
    # Directory containing your run .mat files
    stimuli_dir = os.path.join(basedir2, 'data', 'Retinotopy', 'stimuli', 'full_stimuli')

    processed_run_list = []
    first_halves = []
    second_halves = []
    # Precompute HRF if we want to convolve
    if convolve_hrf:
        # hrf_vec = spm_hrf(tr=1.0)  # 1 Hz sampling
        ttime, hrf_vec = kay_hrf_1s_response(total_time=total_time)
        print(f"HRF vector shape: {hrf_vec.shape}")
    else:
        hrf_vec = None

    for run_num in runs_to_use:
        run_name = run_names[run_num]
        mat_file = os.path.join(stimuli_dir, f"Stimulus_{run_name}.mat")
        
        if not os.path.exists(mat_file):
            print(f"Warning: File does not exist: {mat_file}")
            continue
        
        # 1) Load the 4D data
        data_4d = load_stimulus_run(mat_file, varname='downsampledStimulus')
        print('Data shape = ', data_4d.shape)
        # data_4d shape: [height, width, 3, timepoints_for_this_run]
        
        # If desired, convert the data to grayscale.
        if convert_to_grayscale:
            # Using standard luminance weights: 0.2989*R + 0.5870*G + 0.1140*B.
            # This will result in data_4d_gray of shape: [height, width, timepoints]
            data_4d = np.tensordot(data_4d, [0.2989, 0.5870, 0.1140], axes=([2],[0]))
            print('data shape after grayscale = ', data_4d.shape)
            # Now data_4d is 3D: [height, width, timepoints]
        
        # 2) Flatten the spatial dimensions => [n_features, n_timepoints]
        data_2d = flatten_stimulus_data(data_4d)
        data_2d_raw = data_2d.copy()
        print('Data 2d has dimensions = ', data_2d.shape)
        
        # 3) Optionally convolve each row with the HRF
        if convolve_hrf and hrf_vec is not None:
            data_2d = convolve_with_hrf(data_2d, hrf_vec, conv_type="full")
        
        # 4) Slice time if run 1..4 => keep T_128, else keep all
        if run_num in [0, 1, 2, 3]:
            data_2d = data_2d[:, T] # shape: [n_features, 256]
            data_2d_raw = data_2d_raw[:, T] # shape: [n_features, 256]
            first_half, second_half = split_run_in_half(data_2d)
        else:
            # For special runs, keep all timepoints (or apply another slicing)
            first_half = data_2d
            second_half = data_2d
        
        first_halves.append(first_half)
        second_halves.append(second_half)
        processed_run_list.append(data_2d)
        
    stimulus_RSM = cross_validated_rsm_via_corrcoef(first_halves, second_halves)

    print(f"Stimulus RSM shape: {stimulus_RSM.shape}")
    
    return stimulus_RSM, first_halves, second_halves


def split_run_in_half(run_data):
    """
    Split a run's data into two halves.
    run_data: [n_vertices, n_timepoints]
    returns: (first_half, second_half)
    """
    n_vertices, n_timepoints = run_data.shape
    half = n_timepoints // 2
    return run_data[:, :half], run_data[:, half:]


def cross_validated_rsm_via_corrcoef(half1_runs, half2_runs):
    """
    half1_runs, half2_runs: lists of (n_vertices, L) arrays
    Returns M×M RSM, where M = R·L
    """
    # 1) Concatenate all first halves & second halves along time
    f1 = np.concatenate(half1_runs, axis=1)  # shape (V, M)
    f2 = np.concatenate(half2_runs, axis=1)  # shape (V, M)
    M = f1.shape[1]

    # 2) Build data matrix: rows = patterns, cols = vertices
    X = np.vstack([f1.T, f2.T])             # shape (2M, V)

    # 3) Full (2M×2M) correlation matrix
    C = np.corrcoef(X, rowvar=True)          # rowvar=True is default

    # 4) Extract first‐half vs second‐half block
    return C[:M, M:]     


def first_second_half_each_run(data_here, regions, T):
    '''
    data_here: data of one subject (runs x cortical vertices x timepoints)
    '''
    
    all_regions_first_half = []
    all_regions_second_half = []
    for region_name, region_indices in regions.items():
    # List to hold the region's data from each run, shape: [n_region_vertices, 128 timepoints]
        region_run_list = []
        first_halves = []
        second_halves = []
        for run_num in runs_to_use:
            data_run = data_here[run_num]  # [n_vertices, 128 timepoints]
            region_data = data_run[region_indices,:]  # [n_region_vertices, 128]
            if (run_num==4):
                first_half = region_data
                second_half = data_here[run_num+1][region_indices,:]
            else:
                region_data_period = region_data[:, T]
                first_half, second_half = split_run_in_half(region_data_period)
            first_halves.append(first_half)
            second_halves.append(second_half)
        all_regions_first_half.append(first_halves)
        all_regions_second_half.append(second_halves)
        
    return all_regions_first_half, all_regions_second_half


def vectorize_RSM(RSM):
    """
    Extract the upper triangle of the RSM and flatten it into a vector.
    This is often done before correlation comparisons.
    """
    idx = np.triu_indices(RSM.shape[0], k=0) # k=0 -> include diagonal
    return RSM[idx]


def visual_function_distance(RSM_ref, RSM_other, correlation_type):
    """
    Compute distance as (1 - r) where r is the correlation between
    the upper triangle of RSM_ref and RSM_other.
    """
    vec_ref = vectorize_RSM(RSM_ref)
    vec_other = vectorize_RSM(RSM_other)
    if correlation_type == 'pearson':
        r, p = pearsonr(vec_ref, vec_other)
    elif correlation_type == 'spearman':
        # spearmanr returns (rho, pval)
        r, p = spearmanr(vec_ref, vec_other)
    else:
        raise ValueError("correlation_type must be either 'pearson' or 'spearman'")
    
    distance = 1 - r
    return distance, r, p


def subject_region_reliability(rsm):
    # rsm: M×M split-half RSM for one subject & region
    diag_r = np.diag(rsm)                            # shape (M,)
    eps    = 1e-9
    z      = np.arctanh(np.clip(diag_r, -1+eps, 1-eps))
    return z.mean()                                  # mean in z-space


def group_region_reliability(z_vals):
    """
    z_vals: array of shape (n_subjects,)
    returns: dict with keys
      'r_mean'   : mean reliability (back-transformed)
      'CI_r'     : (low, high) 95% CI on the r-scale
    """
    M      = z_vals.size
    mean_z = z_vals.mean()
    se_z   = z_vals.std(ddof=1) / np.sqrt(M)

    # 95% CI in z-space
    lo_z, hi_z = mean_z - 1.96*se_z, mean_z + 1.96*se_z

    # back-transform
    r_mean = np.tanh(mean_z)
    ci_r   = (np.tanh(lo_z), np.tanh(hi_z))

    return {'r_mean': r_mean, 'CI_r': ci_r}


def distances_to_fisher_z(distances):
    """
    Convert raw distances (d = 1 - r) to Fisher-z scores of r.

    Parameters
    ----------
    distances : array-like, shape (N,)
        Raw distances between RSMs, where distance = 1 - r.

    Returns
    -------
    z_scores : np.ndarray, shape (N,)
        Fisher-z transformed correlations.
    """
    # 1) Recover correlations
    r = 1.0 - np.array(distances)

    # 2) Clip so we never hit exactly ±1
    eps = 1e-6
    r_clipped = np.clip(r, -1 + eps, 1 - eps)

    # 3) Fisher transform
    z_scores = np.arctanh(r_clipped)

    return z_scores


def convolve_with_hrf(data_2d, hrf, conv_type="full"):
    """
    Convolve each row of data_2d with the given 1D HRF.
    data_2d: shape [n_features, n_timepoints]
    hrf: 1D array
    conv_type: same or full. full is preferred.
    Returns a new array of same shape [n_features, n_timepoints].
    """
    n_features, n_timepoints = data_2d.shape
    convolved_data = np.zeros_like(data_2d)
    if conv_type=="same":
        for i in range(n_features):
            convolved_data[i, :] = np.convolve(data_2d[i, :], hrf, mode='same')
    else:
        for i in range(n_features):
            convolved_data[i, :] = np.convolve(data_2d[i, :], hrf, mode='full')[:n_timepoints]
    return convolved_data
    


def kay_hrf_impulse(t):
    """
    Compute the Kay et al. (2013) impulse response at times t (in seconds).
    Returns an array of f(t) values.
    """
    # Paper parameters
    p1, p2, p3, p4, p5, delta_t, a, b = (6.68, 14.66, 1.82, 3.15, 3.08, 0.1/16, 160, -16)
    
    # Derived gamma parameters
    alpha1 = p1/p3
    beta1  = delta_t/p3
    alpha2 = p2/p4
    beta2  = delta_t/p4
    
    # x(t) = a*t + b
    x = a * t + b
    
    fvals = np.zeros_like(t, dtype=float)
    mask = (x > 0)
    
    # term1 = (beta1^alpha1 * x^(alpha1-1) * e^(-beta1*x)) / Gamma(alpha1)
    # term2 = p5 * (beta2^alpha2 * x^(alpha2-1) * e^(-beta2*x)) / Gamma(alpha2)
    if np.any(mask):
        x_pos = x[mask]
        term1 = (beta1**alpha1) * (x_pos**(alpha1-1)) * np.exp(-beta1 * x_pos) / gamma(alpha1)
        term2 = (beta2**alpha2) * (x_pos**(alpha2-1)) * np.exp(-beta2 * x_pos) / gamma(alpha2)
        fvals[mask] = term1 - term2/p5

    return fvals


def kay_hrf_1s_response(total_time=32):
    """
    Return the HRF for a 1-s stimulus using the Kay et al. (2013) impulse response.
    1) Convolve with a 1-s boxcar
    2) Resample to 1-s intervals
    3) Normalize the peak to 1
    """
    # 1) We'll define a fine time-grid, e.g. 0.1 s steps, for the impulse response
    dt_fine = 0.1
    t_fine = np.arange(0, total_time, dt_fine)
    
    # Impulse response
    ir = kay_hrf_impulse(t_fine)  # shape = (len(t_fine),)
    
    # 1-s boxcar (in same fine resolution)
    boxcar = np.zeros_like(t_fine)
    # For first 1 second, we want it "on". That means indices from 0 up to 1/dt_fine
    boxcar[0 : int(1.0/dt_fine)] = 1.0
    
    # Convolution
    conv_response = np.convolve(ir, boxcar, mode='full')[:len(t_fine)]
    
    # 2) Resample to 1-s intervals using cubic interpolation
    t_coarse = np.arange(0, total_time, 1.0)  # 1-s steps
    spline = CubicSpline(t_fine, conv_response)
    response_1s = spline(t_coarse)
    
    # 3) Normalize peak to 1
    peak = np.max(response_1s)
    if peak > 0:
        response_1s /= peak
    
    return t_coarse, response_1s


def load_stimulus_run(mat_filepath, varname='downsampledStimulus'):
    """
    Load a single run's stimulus data from a .mat file (in HDF5 format).
    The data shape is [768, 768, 3, n_timepoints].
    """
    with h5py.File(mat_filepath, 'r') as f:
        data_4d = np.array(f[varname])  # shape [768, 768, 3, timepoints]
    data_4d = data_4d.transpose(3, 2, 1, 0)
    return data_4d


def flatten_stimulus_data(data_4d):
    """
    Flatten the spatial/rgb dimensions into one 'feature' dimension
    so the shape becomes [n_features, n_timepoints].
    Original shape: [768, 768, 3, timepoints].
    """
    spatial_dims = data_4d.shape[:-1]   # (768, 768, 3)
    n_features = np.prod(spatial_dims)  # 768*768*3
    n_timepoints = data_4d.shape[-1]
    
    data_2d = data_4d.reshape((n_features, n_timepoints))
    return data_2d  # shape: [n_features, n_timepoints]


def compute_temporal_RSM(data_2d):
    """
    Computes the correlation matrix across the time dimension.
    data_2d: [n_features, n_timepoints]
    Returns an RSM: [n_timepoints, n_timepoints].
    """
    return np.corrcoef(data_2d, rowvar=False)

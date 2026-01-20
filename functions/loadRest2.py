import os
import numpy as np
import nibabel as nib
from scipy import signal
basedir2 = '/projects/f_mc1689_1/MovieActFlow2/'
scriptDir2 = basedir2 + 'docs/scripts/'
rest_3 = basedir2 + 'docs/scripts/resting_state_parallel/'
rest_data = basedir2 + 'data/Rest/'
nVertices = 59412
dlabelfile = scriptDir2 + 'dlabels_files/CortexSubcortex_ColeAnticevic_NetPartition_wSubcorGSR_parcels_LR_official_newest.dlabel.nii' # new dlabels file
dlabels = np.squeeze(nib.load(dlabelfile).get_fdata()) # whole brain, 91282 vertices, 1 based not 0 based
dlabels = dlabels[:nVertices]
n_parcels = 360

subjNums = ['100610', '102311', '102816', '104416', '105923', '108323', '109123', '111514', '114823', '115017', '115825', '116726', '118225', '125525', '126426', '128935', '130114', '130518', '131217', '131722', '132118', '134627', '134829', '135124', '137128', '140117', '144226', '145834', '146129', '146432', '146735', '146937', '148133', '150423', '155938', '156334', '157336', '158035', '158136', '159239', '162935', '164131', '164636', '165436', '167036', '167440', '169040', '169343', '169444', '169747', '171633', '172130', '173334', '175237', '176542', '177140', '177645', '177746', '178142', '178243', '178647', '180533', '181232', '182436', '182739', '185442', '186949', '187345', '191033', '191336', '191841', '192439', '192641', '193845', '195041', '196144', '197348', '198653', '199655', '200210', '200311', '200614', '201515', '203418', '204521', '205220', '209228', '212419', '214019', '214524', '221319', '233326', '239136', '246133', '249947', '251833', '257845', '263436', '283543', '318637'];

subj_here = ['100610', '102311', '102816', '104416', '105923', '108323', '109123', '111514', 
             '114823', '115017', '115825', '116726', '118225', '125525', '126426']

# all subjects for this project (Apr 2025)
subj_here = ['100610', '102311', '102816', '104416', '105923', '108323', '109123', '111514', '114823', '115017', '115825', '116726', 
             '118225', '125525', '126426', '467351', '525541', '825048', '826353', '833249', '859671', '861456', '871762', '872764', 
             '878776', '878877', '898176', '899885', '901139', '901442', '905147', '910241', '926862', '927359', '942658', '943862', 
             '958976', '966975', '971160', '995174', '128935', '131722', '137128', '140117', '144226', '283543', '318637', '320826', 
             '330324', '346137', '541943', '547046', '562345', '572045', '573249', '581450', '601127', '617748', '627549', '638049', 
             '644246', '654552', '671855', '680957', '690152', '706040', '724446', '725751', '732243', '751550', '757764', '765864', 
             '770352', '771354', '782561', '783462', '789373', '814649', '818859', '951457'] # this is the full dataset, 80 subs. 0-39 (included) is group A,
            # 40-79 (included) is group B (control)
    
def parcelate_data(data):
    X = np.zeros((n_parcels, data.shape[1]), dtype=float)
    for par in range(n_parcels):
        idx = np.where(dlabels == par + 1)[0]
        if idx.size:
            X[par] = np.mean(data[idx], axis=0)
    return X

def loadRestResiduals(sub, cortex_only = True):
    '''
    sub:    integer to choose subject from
    '''

    # Load the rest time series with ICA fix
    subj = subj_here[sub]
    subj_folder = rest_data + subj + '/MNINonLinear/Results/'
    os.chdir(subj_folder)
    run_folders = [
        'rfMRI_REST1_7T_PA',
        'rfMRI_REST2_7T_AP',
        'rfMRI_REST3_7T_PA',
        'rfMRI_REST4_7T_AP'
    ]
    runs = [
        'rfMRI_REST1_7T_PA_Atlas_MSMAll_hp2000_clean.dtseries.nii',
        'rfMRI_REST2_7T_AP_Atlas_MSMAll_hp2000_clean.dtseries.nii',
        'rfMRI_REST3_7T_PA_Atlas_MSMAll_hp2000_clean.dtseries.nii',
        'rfMRI_REST4_7T_AP_Atlas_MSMAll_hp2000_clean.dtseries.nii'
    ]
    # load data
    data = {}
    data_cortex = {}
    data_subcortex = {}
    for r,folder in enumerate(run_folders):
        r_f = subj_folder + folder + '/'
        os.chdir(r_f)
        inputfile = runs[r]
        data[r] = np.squeeze(nib.load(inputfile).get_fdata()).T
        if cortex_only:
            data[r] = data[r][:nVertices,:]
            print(data[r].shape)
            # Demean each run
            data[r] = signal.detrend(data[r],axis=1,type='constant')
            # Detrend each run
            data[r] = signal.detrend(data[r],axis=1,type='linear')
        else:
            data_cortex[r] = data[r][:nVertices, :]
            data_subcortex[r] = data[r][nVertices:, :]
            # Demean each run
            data_cortex[r] = signal.detrend(data_cortex[r],axis=1,type='constant')
            # Detrend each run
            data_cortex[r] = signal.detrend(data_cortex[r],axis=1,type='linear')
            # Demean each run
            data_subcortex[r] = signal.detrend(data_subcortex[r],axis=1,type='constant')
            # Detrend each run
            data_subcortex[r] = signal.detrend(data_subcortex[r],axis=1,type='linear')
            # Concatenate time series of all 4 rest runs

    if cortex_only: 
        # Concatenate time series of all 4 rest runs
        data_conc = np.concatenate(list(data.values()), axis=1)
        return data_conc
    else:
        # Concatenate time series of all 4 rest runs
        data_conc_cortex = np.concatenate(list(data_cortex.values()), axis=1)
        data_conc_subcortex = np.concatenate(list(data_subcortex.values()), axis=1)
        data_full = np.concatenate([data_conc_cortex, data_conc_subcortex], axis=0)
        return data_full, data_conc_cortex, data_conc_subcortex
  


    
    
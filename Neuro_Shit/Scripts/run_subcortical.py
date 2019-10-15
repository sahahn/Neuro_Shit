import numpy as np
import Neuro_Shit as NS

affine = np.array([[-2.,  0.,    0.,   90.],
                  [0.,    2.,    0., -126.],
                  [0.,    0.,    2.,  -72.],
                  [0.,    0.,    0.,    1.]])

name_to_ind_map = {}
name_to_ind_map['SST'] =\
    ['SST_correct_stop_vs_correct_go_beta_reg2mni_censored5.0.nii.gz',
     'SST_incorrect_stop_vs_correct_go_beta_reg2mni_censored5.0.nii.gz',
     'SST_correct_stop_vs_incorrect_stop_sem_reg2mni_censored5.0.nii.gz']

name_to_ind_map['nBack'] =\
    ['nBack_0_back_beta_reg2mni_censored5.0.nii.gz',
     'nBack_2_back_beta_reg2mni_censored5.0.nii.gz',
     'nBack_2_back_vs_0_back_beta_reg2mni_censored5.0.nii.gz',
     'nBack_face_vs_place_beta_reg2mni_censored5.0.nii.gz']

name_to_ind_map['MID'] =\
    ['MID_antic_large_reward_vs_neutral_beta_reg2mni_censored5.0.nii.gz',
     'MID_antic_large_loss_vs_neutral_beta_reg2mni_censored5.0.nii.gz',
     'MID_reward_pos_vs_neg_feedback_beta_reg2mni_censored5.0.nii.gz',
     'MID_loss_pos_vs_neg_feedback_beta_reg2mni_censored5.0.nii.gz']

data_dr = '/users/s/a/sahahn/ABCD_Data/'
main_dr = data_dr + 'raw_voxel_data'
save_dr = data_dr + 'new_merged_voxel_data'
dof_dr = data_dr + 'dofs_by_subject'
sub_mask_loc = data_dr + 'sub_mask.nii'

NS.Merge_SubCortical(main_dr, save_dr, dof_dr, name_to_ind_map,
                     sub_mask_loc, affine)

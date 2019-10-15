import numpy as np
import os
from Neuro_Shit.Base_Loader import Base_Task_Loader


class Reliability():

    def __init__(self, task_to_covar_dict, rely_dr, cortical_dr,
                 subcortical_dr, run_name=''):

        self.task_to_covar_dict = task_to_covar_dict
        self.run_name = run_name
        self.rely_dr = rely_dr
        
        self.prep_data(cortical_dr, subcortical_dr)

    def prep_data(self, cortical_dr, subcortical_dr):

        for task in list(self.task_to_covar_dict.keys()):
            covars_loc = self.task_to_covar_dict[task]
 
            Prep_Task_Rely(self.rely_dr, cortical_dr, subcortical_dr,
                           covars_loc, task, end='.npy',
                           run_name=self.run_name)


class Prep_Task_Rely(Base_Task_Loader):

    def __init__(self, rely_dr, cortical_dr, subcortical_dr, covars_loc, task,
                 end='.npy', run_name=''):

        super().__init__(cortical_dr, subcortical_dr, covars_loc, task, end)

        self.rely_dr = rely_dr
        self.run_name = run_name

        self.load_covars()
        self.prep_dr()
        self.get_num_inds()
        self.make_data_files()

    def prep_dr(self):

        self.data_dr = os.path.join(self.rely_dr, 'data' + self.run_name)
        os.makedirs(self.data_dr, exist_ok=True)

        loc = os.path.join(self.data_dr, self.task + '_final_subjects.txt')
        with open(loc, 'w') as f:
            for subject in self.subject_order:
                f.write(subject)
                f.write('\n')

    def load_cortical_hemi(self, hemi, ind):

        all_data, affine = super().load_cortical_hemi(hemi, ind)
        all_data = np.transpose(np.squeeze(all_data), (1, 0))

        loc = self.save_data(all_data, 'cortical_' + hemi + '_' + str(ind),
                             self.data_dr, affine)

    def load_subcortical(self, ind):

        all_data, affine = super().load_subcortical(ind)
        all_data = np.transpose(np.squeeze(all_data), (1, 0))

        loc = self.save_data(all_data, 'subcortical_' + str(ind), self.data_dr,
                             affine)

import pandas as pd
import numpy as np
import nibabel as nib
import os
import gc
from Neuro_Shit.Base_Loader import Base_Task_Loader


class Palm(Base_Task_Loader):

    def __init__(self, palm_dr, task, covars_loc=None,
                 cortical_dr=None, subcortical_dr=None, performance_covars=[],
                 run_name='', end='.mgz', specific_ind=None):

        super().__init__(cortical_dr, subcortical_dr, covars_loc, task, end, specific_ind)

        self.palm_dr = palm_dr
        self.performance_covars = performance_covars
        self.run_name = run_name
        self.and_run = False

    def prep(self):

        self.load_covars()
        self.prep_dr()
        self.get_num_inds()
        self.make_files()
        self.make_data_files()

    def prep_and_run(self):

        self.and_run = True
        self.prep()

    def run_just_palm(self):

        self.palm_task_dr = os.path.join(self.palm_dr, self.task + self.run_name)

        f_check = self.end
        if self.specific_ind is not None:
            f_check = str(self.specific_ind) + '_' + f_check

        data_files = [file for file in os.listdir(self.palm_task_dr) if f_check in file]

        data_locs = [os.path.join(self.palm_task_dr, data_file)
                     for data_file in data_files]

        self.output_dr = os.path.join(self.palm_task_dr, 'output')

        for loc in data_locs:
            self.run_palm(loc)

    def prep_dr(self):

        self.palm_task_dr = os.path.join(self.palm_dr,
                                         self.task + self.run_name)
        os.makedirs(self.palm_task_dr, exist_ok=True)

        self.output_dr = os.path.join(self.palm_task_dr, 'output')
        os.makedirs(self.output_dr, exist_ok=True)

        loc = os.path.join(self.palm_task_dr, 'final_subjects.txt')
        with open(loc, 'w') as f:
            for subject in self.subject_order:
                f.write(subject)
                f.write('\n')

    def load_cortical_hemi(self, hemi, ind):

        all_data, affine = super().load_cortical_hemi(hemi, ind)

        loc = self.save_data(all_data, 'cortical_' + hemi + '_' + str(ind),
                             self.palm_task_dr, affine)

        if self.and_run:
            self.run_palm(loc)

    def load_subcortical(self, ind):

        all_data, affine = super().load_subcortical(ind)

        loc = self.save_data(all_data, 'subcortical_' + str(ind),
                             self.palm_task_dr, affine)

        if self.and_run:
            self.run_palm(loc)

    def make_files(self):

        base_covars = self.covars.drop(self.performance_covars, axis=1)
        self.make_base_files(base_covars)

        base_covar_names = list(base_covars)
        for i in range(len(self.performance_covars)):

            covar_names = base_covar_names + [self.performance_covars[i]]
            self.make_performance_files(self.covars[covar_names], i)

    def make_base_files(self, covars):

        loc = os.path.join(self.palm_task_dr, 'activation.con')
        with open(loc, 'w') as f:
            f.write('/ContrastName1\tttest_vs_0\n')
            f.write('\n')
            f.write('/NumWaves\t')
            f.write(str(covars.shape[1] + 1))
            f.write('\n')
            f.write('\n')
            f.write('/NumContrasts\t1\n')
            f.write('\n')
            f.write('/Matrix\n')
            f.write('1 ')
            zeros = ' '.join(['0' for i in range(covars.shape[1])])
            f.write(zeros)
            f.write('\n')

        loc = os.path.join(self.palm_task_dr, 'activation.mat')
        self.make_cov_mat_file(loc, covars)

    def make_performance_files(self, covars, i):
        '''The performance covar should be last'''

        loc = os.path.join(self.palm_task_dr, 'performance' + str(i) + '.con')
        with open(loc, 'w') as f:
            f.write('/ContrastName1\tposcorr\n')
            f.write('/ContrastName2\tnegcorr\n')
            f.write('\n')
            f.write('/NumWaves\t')
            f.write(str(covars.shape[1] + 1))
            f.write('\n')
            f.write('/NumContrasts\t2\n')
            f.write('\n')
            f.write('/Matrix\n')

            zeros = ' '.join(['0' for i in range(covars.shape[1])])

            f.write(zeros)
            f.write(' 1')
            f.write('\n')

            f.write(zeros)
            f.write(' -1')
            f.write('\n')

        loc = os.path.join(self.palm_task_dr, 'performance' + str(i) + '.mat')
        self.make_cov_mat_file(loc, covars)

    def make_cov_mat_file(self, loc, covars):

        covars = np.array(covars)

        with open(loc, 'w') as f:

            f.write('/NumWaves\t')
            f.write(str(covars.shape[1] + 1))
            f.write('\t\t\t\t\t\t\t\t\t\t\t\t\t\n')
            f.write('/NumPoints\t')
            f.write(str(len(covars)))
            f.write('\t\t\t\t\t\t\t\t\t\t\t\t\t\t\n')
            f.write('\n')
            f.write('/Matrix\t\n')

            for row in covars:
                f.write('1\t')
                for entry in row[:-1]:
                    f.write(str(entry))
                    f.write('\t')
                f.write(str(row[-1]))
                f.write('\n')

    def run_palm(self, data_loc):

        self.run_activation(data_loc)
        self.run_performance(data_loc)

    def run_activation(self, data_loc):

        command = 'palm -i ' + data_loc + ' -d '
        command += os.path.join(self.palm_task_dr, 'activation.mat')
        command += ' -n 2 -t '
        command += os.path.join(self.palm_task_dr, 'activation.con')
        command += ' -saveparametric -saveglm'

        out_name = os.path.basename(data_loc).replace('.mgz', '')
        command += ' -o ' + os.path.join(self.output_dr, out_name)

        self.run_command(command)

    def run_performance(self, data_loc):

        for i in range(len(self.performance_covars)):

            command = 'palm -i ' + data_loc + ' -d '
            command += os.path.join(self.palm_task_dr, 'performance' + str(i) + '.mat')
            command += ' -n 2 -t '
            command += os.path.join(self.palm_task_dr, 'performance' + str(i) + '.con')
            command += ' -saveparametric -saveglm'

            out_name = os.path.basename(data_loc).replace('.mgz', '')
            out_name += '_performance' + str(i)

            command += ' -o ' + os.path.join(self.output_dr, out_name)
            self.run_command(command)

    def run_command(self, command):

        os.system(command)

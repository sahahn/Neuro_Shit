import pandas as pd
import numpy as np
import nibabel as nib
import os
import gc


class Base_Task_Loader():

    def __init__(self, cortical_dr, subcortical_dr, covars_loc, task,
                 end='.mgz', specific_ind=None):

        self.cortical_dr = cortical_dr
        self.subcortical_dr = subcortical_dr
        self.covars_loc = covars_loc
        self.task = task
        self.end = end
        self.specific_ind = specific_ind

    def load_covars(self):

        self.covars = pd.read_csv(self.covars_loc, index_col='src_subject_id')
        data_subjects = self.get_data_subjects()

        keep = set(self.covars.index).intersection(data_subjects)
        keep = list(keep)
        keep.sort() 

        self.covars = self.covars.loc[keep]
        self.subject_order = self.covars.index

    def get_data_subjects(self):

        subjects = None

        if self.cortical_dr is not None:
            subjects = self.get_subjects(self.cortical_dr)

        if self.subcortical_dr is not None:

            subcort_subjects = self.get_subjects(self.subcortical_dr)

            if subjects:
                subjects = subjects.intersection(subcort_subjects)
            else:
                subjects = subcort_subjects

        return subjects

    def get_subjects(self, dr):

        files = os.listdir(dr)
        files = [file for file in files if self.task in file]

        subjects = ['NDAR_' + file.split('_')[0] for file in files]

        return set(subjects)

    def get_num_inds(self):

        if self.cortical_dr is not None:
            data_file = [f for f in os.listdir(self.cortical_dr)
                         if self.task in f][0]
            loc = os.path.join(self.cortical_dr, data_file)
        else:
            data_file = [f for f in os.listdir(self.subcortical_dr)
                         if self.task in f][0]
            loc = os.path.join(self.subcortical_dr, data_file)

        sample_data = nib.load(loc)
        self.num_inds = sample_data.shape[-1]

    def make_data_files(self):

        if self.specific_ind is not None:
            if self.cortical_dr is not None:
                self.load_cortical_hemi('lh', self.specific_ind)
                gc.collect()

                self.load_cortical_hemi('rh', self.specific_ind)
                gc.collect()

            if self.subcortical_dr is not None:
                self.load_subcortical(self.specific_ind)
                gc.collect()

        else:

            for ind in range(self.num_inds):

                if self.cortical_dr is not None:
                    self.load_cortical_hemi('lh', ind)
                    gc.collect()

                    self.load_cortical_hemi('rh', ind)
                    gc.collect()

                if self.subcortical_dr is not None:
                    self.load_subcortical(ind)
                    gc.collect()

    def load_cortical_hemi(self, hemi, ind):
        return self.load_and_stack_data(ind, self.cortical_dr, hemi)

    def load_subcortical(self, ind):
        return self.load_and_stack_data(ind, self.subcortical_dr)

    def load_and_stack_data(self, ind, dr, hemi=None):

        all_data = []

        for subject in self.subject_order:

            file_name = subject.replace('NDAR_', '') + '_' + self.task

            if hemi:
                file_name = file_name + '_' + hemi
            file_name = file_name + '.mgz'

            subject_path = os.path.join(dr, file_name)

            data, affine = self.load_data_ind(subject_path, ind)
            all_data.append(data)

        all_data = np.stack(all_data, axis=-1)
        return all_data, affine

    def load_data_ind(self, path, ind):

        data = nib.load(path)
        affine = data.affine
        data = data.get_fdata()

        return data[:, :, :, ind], affine

    def save_data(self, data, name, dr, affine=None):

        loc = os.path.join(dr, name + self.end)

        if self.end == '.npy':
            np.save(loc, img)
        else:

            if affine is None:
                affine = np.eye(4)

            img = nib.Nifti1Image(data, affine)
            nib.save(img, loc)

        return loc

import os
import nibabel as nib
import scipy
import scipy.io
import numpy as np
from random import shuffle


class Merge():

    def __init__(self, main_dr, save_dr, dof_dr):

        self.main_dr = main_dr
        self.save_dr = save_dr
        self.dof_dr = dof_dr

        self.load_dof_dict()
        os.makedirs(self.save_dr, exist_ok=True)

    def load_dof_dict(self):

        self.dof_dict = {}

        for subject_name in os.listdir(self.dof_dr):

            subject_key = subject_name.split('_')[2]
            subject_loc = os.path.join(self.dof_dr, subject_name)

            for dof_loc in os.listdir(subject_loc):
                dof_file = File(os.path.join(subject_loc, dof_loc))
                dof_file.load_dof()

                ind = subject_key + '_' + dof_file.get_ind()
                self.dof_dict[ind] = dof_file.dof

    def merge(self):

        all_subjects = os.listdir(self.main_dr)
        shuffle(all_subjects)

        for subject_name in all_subjects:
            self.proc_subject(subject_name)

    def merge_runs_data(self, run1, run2, dof1, dof2):

        new_data = np.sum([run1 * dof1, run2 * dof2], axis=0) / (dof1 + dof2)
        return new_data

    def save_img(self, data, path):

        if not os.path.exists(path + self.end):

            if self.affine is None:
                affine = np.eye(4)
            else:
                affine = self.affine

            if data.shape[-1] == 1:
                data = np.squeeze(data, -1)

            img = nib.Nifti1Image(data, affine)
            nib.save(img, path + self.end)

    def save_merged(self, subject_name, task, merged_runs):

        save_path = os.path.join(self.save_dr, subject_name + '_' + task)

        if self.split_hemi:
            n_vert_per_hemi = merged_runs.shape[0] // 2

            self.save_img(merged_runs[:n_vert_per_hemi],
                          save_path + '_' + 'lh')

            self.save_img(merged_runs[n_vert_per_hemi:],
                          save_path + '_' + 'rh')

        else:
            self.save_img(merged_runs, save_path)

    def check_exists(self, subject_name, task):

        save_path = os.path.join(self.save_dr, subject_name + '_' + task)

        if self.split_hemi:
            save_path_lh = save_path + '_' + 'lh'
            save_path_rh = save_path + '_' + 'rh'

            if os.path.exists(save_path_lh + self.end):
                if os.path.exists(save_path_rh + self.end):
                    return True
            return False

        if os.path.exists(save_path + self.end):
            return True
        return False


class Merge_Cortical(Merge):

    def __init__(self, main_dr, save_dr, dof_dr, task_names, task_inds,
                 split_hemi=False):

        super().__init__(main_dr, save_dr, dof_dr)

        self.task_names = task_names
        self.task_inds = task_inds
        self.split_hemi = split_hemi
        self.affine = None
        self.end = '.mgz'

        self.merge()

    def proc_subject(self, subject_name):

        subject_path = os.path.join(self.main_dr, subject_name)
        subject = Subject_Cortical(subject_path)

        for task in subject.avaliable_tasks:
            if task in self.task_names:
                if not self.check_exists(subject.subject_name, task):

                    t_inds = self.task_inds[self.task_names.index(task)]

                    run1, run2 = subject.get_runs_lr(task)
                    run1, run2 = run1[:, :, :, t_inds], run2[:, :, :, t_inds]

                    ind1, ind2 = subject.get_runs_ind(task)

                    try:
                        dof1, dof2 = self.dof_dict[ind1], self.dof_dict[ind2]
                        merged_runs = self.merge_runs_data(run1, run2, dof1, dof2)
                        self.save_merged(subject.subject_name, task, merged_runs)
                    except KeyError:
                        print('No dof for', ind1, ind2)


class Merge_SubCortical(Merge):

    def __init__(self, main_dr, save_dr, dof_dr, name_to_ind_map,
                 sub_mask_loc, affine, save_full=False):

        super().__init__(main_dr, save_dr, dof_dr)

        self.name_to_ind_map = name_to_ind_map
        self.task_names = list(self.name_to_ind_map)
        self.sub_mask_loc = sub_mask_loc
        self.affine = affine
        self.split_hemi = False
        self.end = '.mgz'
        self.save_full = save_full

        self.load_sub_mask()
        self.merge()

    def load_sub_mask(self):
        self.submask = np.squeeze(nib.load(self.sub_mask_loc).get_fdata())

    def proc_subject(self, subject_name):

        subject_path = os.path.join(self.main_dr, subject_name)
        subject = Subject_SubCortical(subject_path)

        for task in subject.avaliable_tasks:
            if task in self.task_names:
                if not self.check_exists(subject.subject_name, task):
               
                    files = self.name_to_ind_map[task]
                    run1, run2 = subject.get_runs_data(task, files, self.submask, self.save_full)

                    ind1, ind2 = subject.get_runs_ind(task)

                    try:
                        dof1, dof2 = self.dof_dict[ind1], self.dof_dict[ind2]
                        merged_runs = self.merge_runs_data(run1, run2, dof1, dof2)
                        self.save_merged(subject.subject_name, task, merged_runs)
                    except KeyError:
                        print('No dof for', ind1, ind2)


class Subject():

    def __init__(self, loc):

        self.loc = loc
        self.set_subject_name()
        self.determine_files()
        self.process_files()
        self.set_avaliable_tasks()

    def set_subject_name(self):

        raw_name = os.path.basename(self.loc)
        self.subject_name = raw_name.split('_')[2]

    def determine_files(self):

        file_names = os.listdir(self.loc)
        self.files = []

        for name in file_names:
            file = File(os.path.join(self.loc, name))
            self.files.append(file)

    def process_files(self):
        pass

    def set_avaliable_tasks(self):

        self.avaliable_tasks = {}

        for file in self.files:
            try:
                self.avaliable_tasks[file.task].append(file)
            except KeyError:
                self.avaliable_tasks[file.task] = [file]

        tasks = list(self.avaliable_tasks.keys())
        for task in tasks:
            if len(self.avaliable_tasks[task]) == 1:
                self.avaliable_tasks.pop(task)

    def get_runs_ind(self, task):

        avaliable = self.avaliable_tasks[task]

        ind1 = self.subject_name + '_' + avaliable[0].get_ind()
        ind2 = self.subject_name + '_' + avaliable[1].get_ind()

        return ind1, ind2


class Subject_Cortical(Subject):

    def process_files(self):

        for file in self.files:
            file.load_hemi_locs()

    def get_runs_lr(self, task):

        avaliable = self.avaliable_tasks[task]

        run1 = avaliable[0].get_lr()
        run2 = avaliable[1].get_lr()

        return run1, run2


class Subject_SubCortical(Subject):

    def process_files(self):

        for file in self.files:
            file.load_subcort_locs()

        self.files = [file for file in self.files if len(file.data_files) > 0]

    def get_runs_data(self, task, files, submask, save_full):

        avaliable = self.avaliable_tasks[task]

        run1 = avaliable[0].load_files(files, submask, save_full)
        run2 = avaliable[1].load_files(files, submask, save_full)

        return run1, run2


class File():

    def __init__(self, loc):

        self.loc = loc
        self.proc_raw_name()

    def proc_raw_name(self):

        raw_name = os.path.basename(self.loc)
        split_name = raw_name.split('_')

        self.task = split_name[1]
        self.scan_num = split_name[3]

    def load_hemi_locs(self):

        data_files = os.listdir(self.loc)

        for data_file in data_files:
            if 'lh.mgz' in data_file:
                self.lh = os.path.join(self.loc, data_file)
            elif 'rh.mgz' in data_file:
                self.rh = os.path.join(self.loc, data_file)

        try:
            self.lh, self.rh
        except NameError:
            print('No lh or rh for', self.loc)

    def load_subcort_locs(self):

        self.data_dr = os.path.join(self.loc, 'contrasts_mni')

        try:
            self.data_files = os.listdir(self.data_dr)
        except FileNotFoundError:
            self.data_files = []

    def get_lr(self):

        lh = nib.load(self.lh).get_fdata()
        rh = nib.load(self.rh).get_fdata()

        lr = np.concatenate([lh, rh])
        return lr

    def load_files(self, names, submask, save_full):

        files = [os.path.join(self.data_dr, name) for name in names]

        if save_full:
            data = [nib.load(file).get_fdata() for file in files]

        else:
            data = [nib.load(file).get_fdata()[submask == 1] for file in files]
            data = [d[:, np.newaxis, np.newaxis] for d in data]

        return np.stack(data, axis=-1)

    def load_dof(self):

        self.dof = scipy.io.loadmat(self.loc)['dof'][0][0]
        self.dof = float(self.dof)

    def get_ind(self):

        return self.task + '_' + self.scan_num

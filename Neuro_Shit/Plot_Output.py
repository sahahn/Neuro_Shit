import nibabel as nib
import nilearn
import nilearn.plotting
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import os


class Plotting():

    def __init__(self, labels, palm_dr, output_dr, perf_labels={}, run_name='',
                 sub_mask_loc=None, plot_key=' cohen'):

        self.labels = labels
        self.palm_dr = palm_dr
        self.output_dr = output_dr
        self.perf_labels = perf_labels
        self.run_name = run_name
        self.sub_mask_loc = sub_mask_loc
        self.plot_key = plot_key

        self.load_sub_mask()
        self.fs_avg = nilearn.datasets.fetch_surf_fsaverage(mesh='fsaverage5')
        self.prep_output_dr()
        self.make_plots()

    def prep_output_dr(self):

        os.makedirs(self.output_dr, exist_ok=True)

        self.activation_dr = os.path.join(self.output_dr, 'Activations')
        self.perf_dr = os.path.join(self.output_dr, 'Performance_Corrs')

        os.makedirs(self.activation_dr, exist_ok=True)
        os.makedirs(self.perf_dr, exist_ok=True)

    def load_sub_mask(self):
        self.submask_affine = nib.load(self.sub_mask_loc).affine
        self.submask = np.squeeze(nib.load(self.sub_mask_loc).get_fdata())

    def make_plots(self):

        tasks = list(self.labels.keys())

        for task in tasks:
            print('task', task)
            self.plot_task(task)

    def plot_task(self, task):

        self.palm_output_dr = os.path.join(self.palm_dr, task + self.run_name,
                                           'output')

        cortical_data, subcortical_data = self.load_data(task)
        cortical_perf, subcortical_perf = self.load_perf_data(task)

        self.plot(cortical_data, subcortical_data, task, performance=False)

        if len(cortical_perf) > 0:
            self.plot(cortical_perf, subcortical_perf, task, performance=True)

    def load_data(self, task):

        cort_names = []
        subcort_names = []

        for ind in range(len(self.labels[task])):

            for hemi in ['lh', 'rh']:

                name = 'cortical_' + hemi + '_' + str(ind) + '_dpx_cohen.mgz'
                cort_names.append(name)

            name = 'subcortical_' + str(ind) + '_dpx_cohen.mgz'
            subcort_names.append(name)
        
        print('cortical_names', cort_names)
        print('subcortical_names', subcort_names)

        cortical_data = self.load_cortical(cort_names)
        subcortical_data = self.load_subcortical(subcort_names)

        return cortical_data, subcortical_data

    def load_perf_data(self, task):

        cort_names = []
        subcort_names = []

        for perf_ind in range(len(self.perf_labels[task])):
            for ind in range(len(self.labels[task])):

                for hemi in ['lh', 'rh']:

                    name = 'cortical_' + hemi + '_' + str(ind) + '_performance'
                    name += str(perf_ind) + '_dpx_cohen_c1.mgz'
                    cort_names.append(name)

                name = 'subcortical_' + str(ind) + '_performance'
                name += str(perf_ind) + '_dpx_cohen_c1.mgz'
                subcort_names.append(name)

        print('cortical_perf_names', cort_names)
        print('subcortical_perf_names', subcort_names)

        cortical_data = self.load_cortical(cort_names)
        subcortical_data = self.load_subcortical(subcort_names)

        return cortical_data, subcortical_data

    def load_cortical(self, names):

        try:
            paths = [os.path.join(self.palm_output_dr, n) for n in names]
            data = [np.squeeze(nib.load(p).get_fdata()) for p in paths]

            return data
        except:
            return []

    def load_subcortical(self, names):
        
        try:
            paths = [os.path.join(self.palm_output_dr, n) for n in names]
            data = [np.squeeze(nib.load(p).get_fdata()) for p in paths]

            data_3d = []
            for d in data:
                new = np.copy(self.submask)
                new[self.submask == 1] = d
                data_3d.append(new)

            return data_3d
        except:
            return []

    def plot(self, cortical_data, subcortical_data, task, performance=False):

        plot_dr = self.activation_dr
        threshold = .2

        if performance:
            plot_dr = self.perf_dr
            threshold = .000001

        task_plot_dr = os.path.join(plot_dr, task)
        os.makedirs(task_plot_dr, exist_ok=True)
        
        if len(cortical_data) > 0:
            vmax_c = np.max(np.abs(np.array(cortical_data)))
        else:
            vmax_c = 0

        if len(subcortical_data) > 0:
            vmax_s = np.max(np.abs(np.array(subcortical_data)))
        else:
            vmax_s = 0

        vmax  = max([vmax_c, vmax_s])

        cort_names, subcort_names = self.get_plot_names(task, performance)

        print('Final cort plot names', performance, cort_names)
        print('Final subcort plot names', performance, subcort_names)

        for data, base_name in zip(cortical_data, cort_names):
            self.plot_cortical(data, base_name, threshold, task_plot_dr, vmax)

        for data, base_name in zip(subcortical_data, subcort_names):
            self.plot_subcortical(data, base_name, threshold, task_plot_dr, vmax)

    def get_plot_names(self, task, performance):

        cort_names = []
        subcort_names = []

        for contrast in self.labels[task]:

            name = contrast
            subcort_names.append(name)

            for hemi in ['left', 'right']:
                cort_names.append((name + ' ' + hemi, hemi))

        if performance:

            subcort_names = [perf + ' pos corr with ' + s for perf in self.perf_labels[task] for s in subcort_names]
            cort_names = [(perf + ' pos corr with ' + s[0], s[1]) for perf in self.perf_labels[task] for s in cort_names]

        return cort_names, subcort_names

    def plot_cortical(self, data, base_name, threshold, task_plot_dr, vmax):

        cortical_dr = os.path.join(task_plot_dr, 'Cortical')
        os.makedirs(cortical_dr, exist_ok=True)

        name = base_name[0]
        hemi = base_name[1]

        #cmap = 'seismic'
        cmap = 'cold_hot'
        plot_surfaces = [self.fs_avg.infl_left, self.fs_avg.pial_left]
        background = self.fs_avg.sulc_left
        
        if hemi == 'right':
            plot_surfaces = [self.fs_avg.infl_right, self.fs_avg.pial_right]
            background = self.fs_avg.sulc_right
        
        for surface_name, plot_surface in zip(['Inflated', 'Pial'],
                                              plot_surfaces):

            int_cortical_dr = os.path.join(cortical_dr, surface_name)
            os.makedirs(int_cortical_dr, exist_ok=True)

            # Make static images
            static_dr = os.path.join(int_cortical_dr, 'Static')
            os.makedirs(static_dr, exist_ok=True)

            for view in ['lateral', 'medial']:

                fig = plt.figure()

                nilearn.plotting.plot_surf_stat_map(plot_surface,
                                                    data,
                                                    hemi=hemi,
                                                    view=view,
                                                    colorbar=True,
                                                    threshold=threshold,
                                                    alpha=1,
                                                    figure=fig,
                                                    vmax=vmax,
                                                    cmap=cmap,
                                                    bg_map=background,
                                                    bg_on_data=True)

                title = name + ' ' + view + self.plot_key
                fig.suptitle(title)

                save_name = title.replace(' ', '_') + '.png'
                save_spot = os.path.join(static_dr, save_name)
                fig.savefig(save_spot, dpi=100)
                plt.close(fig)

            # Make HTML interative images
            html_dr = os.path.join(int_cortical_dr, 'HTML')
            os.makedirs(html_dr, exist_ok=True)

            html_img = nilearn.plotting.view_surf(plot_surface,
                                                  data,
                                                  vmax=vmax,
                                                  bg_map=background,
                                                  threshold=threshold)

            title = name + self.plot_key

            save_name = title.replace(' ', '_') + '.html'
            save_spot = os.path.join(html_dr, save_name)
            html_img.save_as_html(save_spot)

            # Re-save with added title
            with open(save_spot, 'r') as f:
                lines = f.readlines()

            body_start = lines.index('<body>\n')
            new_line = '<B><h1> <p style="text-align:center;"> ' + title
            new_line = new_line + ' </p></h1></B>' + lines[body_start + 1]
            lines[body_start + 1] = new_line

            with open(save_spot, 'w') as f:
                for line in lines:
                    f.write(line)

    def plot_subcortical(self, data, base_name, threshold, task_plot_dr, vmax):
        
        data = nib.Nifti1Image(data, affine=self.submask_affine)

        subcortical_dr = os.path.join(task_plot_dr, 'Subcortical')
        os.makedirs(subcortical_dr, exist_ok=True)

        name = base_name
        #cmap = 'seismic'
        cmap = 'cold_hot'

        range_dict = {'x': [[-50, 50], [-40, 40], [-30, 30], [-20, 20], [-10, 10], [-5, 5]],
                      'y': [[-85, 20], [-75, 10], [-65, 0], [-55, -10], [-45, -20], [-35, -30]],
                      'z': [[-50, 25], [-40, 15], [-30, 5], [-20, -5], [-15, -10]]}

        for view in ['x', 'y', 'z']:

            view_dr = os.path.join(subcortical_dr, view)
            os.makedirs(view_dr, exist_ok=True)

            for views in range_dict[view]:

                fig = plt.figure()
                nilearn.plotting.plot_stat_map(data,
                                               cmap=cmap,
                                               symmetric_cbar=True,
                                               vmax=vmax,
                                               figure=fig,
                                               draw_cross=False,
                                               threshold=threshold,
                                               display_mode=view,
                                               cut_coords=views)

                title = name + ' slices ' + str(views[0]) + ' ' + str(views[1]) + self.plot_key
                fig.suptitle(title)

                save_name = title.replace(' ', '_') + '.png'
                save_spot = os.path.join(view_dr, save_name)
                fig.savefig(save_spot, dpi=100)
                plt.close(fig)

        fig = plt.figure()
        nilearn.plotting.plot_glass_brain(data,
                                          symmetric_cbar=True,
                                          plot_abs=False,
                                          colorbar=True,
                                          cmap=cmap,
                                          figure=fig,
                                          vmax=vmax,
                                          threshold=threshold)

        title = name + ' glass corr coef'
        fig.suptitle(title)

        save_name = title.replace(' ', '_') + '.png'
        save_spot = os.path.join(subcortical_dr, save_name)
        fig.savefig(save_spot, dpi=100)
        plt.close(fig)


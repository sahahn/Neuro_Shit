import numpy as np
import matplotlib.pyplot as plt
import nilearn
import nilearn.plotting
import os
from matplotlib import cm
import nibabel.freesurfer.io as io
from nilearn import datasets
from Neuro_Shit.plot_surf import plot_surf, add_collage_colorbar
from nilearn.plotting.img_plotting import _crop_colorbar
import matplotlib.gridspec as gridspec


def load_mapping(loc):

    mapping = {}

    with open(loc, 'r') as f:
        for line in f.readlines():
            line = line.split(',')
            mapping[line[0]] = line[1].rstrip()

    return mapping


def get_chunk_from_df(data, name_col, value_col, keys, d_keys=[]):

    if isinstance(keys, str):
        keys = [keys]

    chunk = {}

    for i in data.index:
        name = data[name_col].loc[i]

        if all([key in name for key in keys]) and not any([key in name for key in d_keys]):
            chunk[name] = data[value_col].loc[i]

    return chunk


def get_surface(chunk, ref_surface, mapping, label_names):

    surface = np.zeros(np.shape(ref_surface))

    for name in chunk:

        value = chunk[name]
        name = name.lower()

        for key in mapping:
            if key in name:
                name = name.replace(key, mapping[key])

        for l in range(len(label_names)):
            if label_names[l] in name:
                ind = l
                break

        surface = np.where(ref_surface == ind, value, surface)

    return surface


def get_hemi_surfaces(chunk, lh_key, rh_key, lh_ref_surface, rh_ref_surface,
                      label_names, mapping={}):

    surfaces = []
    for key, ref_surface in zip([lh_key, rh_key],
                                [lh_ref_surface, rh_ref_surface]):

        hemi_chunk = {k: chunk[k] for k in chunk if key in k}
        surface = get_surface(hemi_chunk, ref_surface, mapping, label_names)
        surfaces.append(surface)

    return surfaces


def get_setup(fs_home, destr=True, fs5=False):

    if destr:
        lh_name = 'lh.aparc.a2009s.annot'
        rh_name = 'rh.aparc.a2009s.annot'

    else:
        lh_name = 'lh.aparc.annot'
        rh_name = 'rh.aparc.annot'

    if fs5:
        fs_type = 'fsaverage5'
        fs_avg = nilearn.datasets.fetch_surf_fsaverage(mesh='fsaverage5')
    else:
        fs_type = 'fsaverage'
        fs_avg = nilearn.datasets.fetch_surf_fsaverage(mesh='fsaverage')

    lh_loc = os.path.join(fs_home, 'subjects', fs_type, 'label', lh_name)
    rh_loc = os.path.join(fs_home, 'subjects', fs_type, 'label', rh_name)

    lh = io.read_annot(lh_loc)
    rh = io.read_annot(rh_loc)

    label_names = [label.decode('UTF-8') for label in lh[2]]

    lh = lh[0]
    rh = rh[0]

    return lh, rh, label_names, fs_avg


def base_surf_plot(data, hemi, inflate, fs_avg, dist=6, **kwargs):

    if hemi == 'lh':

        hemi = 'left'
        if inflate:
            surf_mesh = fs_avg.infl_left
        else:
            surf_mesh = fs_avg.pial_left
        bg_map = fs_avg.sulc_left

    else:
        hemi = 'right'
        if inflate:
            surf_mesh = fs_avg.infl_right
        else:
            surf_mesh = fs_avg.pial_right
        bg_map = fs_avg.sulc_right

    figure, surf_map_faces = plot_surf(surf_mesh,
                                       surf_map=data,
                                       bg_map=bg_map,
                                       hemi=hemi,
                                       dist=dist,
                                       **kwargs)

    return figure, surf_map_faces


def base_surf_collage(data, inflate, fs_avg, colorbar=True, figsize=(20, 20),
                      title='Collage', fontsize=30, hspace=0, wspace=0,
                      vmin=None, vmax=None,
                      title_y_adjust=.1, midpoint=None, dist=8, **kwargs):

    figure, ax = plt.subplots(nrows=2, ncols=2,
                              subplot_kw={'projection': '3d'},
                              figsize=figsize)

    if vmin is None:
        vmin = np.nanmin(np.nanmin(data))
    if vmax is None:
        vmax = np.nanmax(np.nanmax(data))

        if np.abs(vmin) > vmax:
            vmax = np.abs(vmin)
        else:
            vmin = -vmax

    smfs = []
    hemis = [['lh', 'rh'], ['lh', 'rh']]
    views = ['lateral', 'medial']

    for i in range(2):
        for j in range(2):
            figure, smf = base_surf_plot(data[j], hemis[i][j], inflate, fs_avg,
                                         figure=figure, axes=ax[i, j],
                                         view=views[i], vmin=vmin, vmax=vmax,
                                         midpoint=midpoint, dist=dist,
                                         **kwargs)
            smfs.append(smf)

    figure.subplots_adjust(hspace=0, wspace=0)

    if colorbar:
        add_collage_colorbar(figure, ax, smfs, vmax, vmin,
                             midpoint=midpoint, **kwargs)

    y = ax[0][0].get_position().y1 - title_y_adjust

    plt.suptitle(title, fontsize=fontsize, y=y,
                 x=ax[0][0].get_position().x1)


def Collages(all_data,
             inflate,
             fs_avg,
             titles=None,
             figsize=(15, 10),
             outer_wspace=.1,
             outer_hspace=.1,
             vmin=None,
             vmax=None,
             midpoint=None,
             colorbar=True,
             cbar_2_fig_ratio=.5,
             cbar_fraction=.25,
             cbar_shrink=1,
             cbar_aspect=20,
             cbar_pad=.1,
             dist=6,
             **kwargs):

    '''Data as list of [lh, rh] lists. Must be 3 deep.
       Titles should be 2 nested list'''

    if vmin is None:
        vmin = np.nanmin(np.nanmin(all_data))
    if vmax is None:
        vmax = np.nanmax(np.nanmax(all_data))

        if np.abs(vmin) > vmax:
            vmax = np.abs(vmin)
        else:
            vmin = -vmax

    figure = plt.figure(figsize=figsize)

    n_rows = len(all_data)
    n_cols_by_row = [len(all_data[r]) for r in range(n_rows)]
    n_cols = max(n_cols_by_row)

    if colorbar is True:

        widths = [1 for i in range(n_cols)] + [cbar_2_fig_ratio]

        outer_gs = gridspec.GridSpec(n_rows, n_cols+1,
                                     wspace=outer_wspace,
                                     hspace=outer_hspace,
                                     width_ratios=widths)

        colorbar_ax = figure.add_subplot(outer_gs[:, n_cols])
        colorbar_ax.set_axis_off()

    else:

        outer_gs = gridspec.GridSpec(n_rows, n_cols,
                                     wspace=outer_wspace,
                                     hspace=outer_hspace)

    smfs = []

    for row in range(n_rows):
        for col in range(n_cols_by_row[row]):

            # This is fixed as collage of 4
            gs = gridspec.GridSpecFromSubplotSpec(2, 2,
                                                  subplot_spec=outer_gs[row,
                                                                        col],
                                                  hspace=0, wspace=0)

            if titles is not None:
                title_ax = figure.add_subplot(outer_gs[row, col])
                title_ax.set_title(titles[row][col])
                title_ax.set_axis_off()

            hemis = [['lh', 'rh'], ['lh', 'rh']]
            views = ['lateral', 'medial']
            data = all_data[row][col]

            for i in range(2):
                for j in range(2):
                    ax = figure.add_subplot(gs[i, j], projection='3d')

                    figure, smf = base_surf_plot(data[j], hemis[i][j], inflate,
                                                 fs_avg, figure=figure,
                                                 axes=ax, view=views[i],
                                                 vmin=vmin, vmax=vmax,
                                                 midpoint=midpoint, dist=6,
                                                 **kwargs)
                    smfs.append(smf)

    figure.subplots_adjust(hspace=0, wspace=0)

    if colorbar is True:
        add_collage_colorbar(figure=figure, ax=colorbar_ax, smfs=smfs,
                             vmax=vmax, vmin=vmin,
                             midpoint=midpoint,
                             multicollage=True,
                             cbar_shrink=cbar_shrink,
                             cbar_aspect=cbar_aspect,
                             cbar_pad=cbar_pad,
                             **kwargs)

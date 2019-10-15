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


def load_mapping(loc):

    mapping = {}

    with open(loc, 'r') as f:
        for line in f.readlines():
            line = line.split(',')
            mapping[line[0]] = line[1].rstrip()

    return mapping


def get_chunk_from_df(data, name_col, value_col, keys):

    if isinstance(keys, str):
        keys = [keys]

    chunk = {}

    for i in data.index:
        name = data[name_col].loc[i]

        if all([key in name for key in keys]):
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


def base_surf_plot(data, hemi, inflate, fs_avg, **kwargs):

    if hemi == 'lh':

        hemi = 'left'
        if inflate:
            surf_mesh = fs_avg.infl_left
        else:
            surf_mesh = fs_avg.sulc_left
        bg_map = fs_avg.sulc_left

    else:
        hemi = 'right'
        if inflate:
            surf_mesh = fs_avg.infl_right
        else:
            surf_mesh = fs_avg.sulc_right
        bg_map = fs_avg.sulc_right

    figure, surf_map_faces = plot_surf(surf_mesh,
                                       surf_map=data,
                                       bg_map=bg_map,
                                       hemi=hemi,
                                       **kwargs)

    return figure, surf_map_faces


def base_surf_collage(data, inflate, fs_avg, figsize=(20, 20), title='Collage',
                      fontsize=30, hspace=0, wspace=0, title_y_adjust=.1,
                      **kwargs):

    figure, ax = plt.subplots(nrows=2, ncols=2,
                              subplot_kw={'projection': '3d'},
                              figsize=figsize)

    if 'vmin' not in kwargs:
        vmin1 = np.nanmin(data[0])
        vmin2 = np.nanmin(data[1])
        vmin = min([vmin1, vmin2])
    else:
        vmin = kwargs.pop('vmin')

    if 'vmax' not in kwargs:
        vmax1 = np.nanmax(data[0])
        vmax2 = np.nanmax(data[1])
        vmax = min([vmax1, vmax2])

        if np.abs(vmin) > vmax:
            vmax = np.abs(vmin)
        else:
            vmin = -vmax

    else:
        vmin = kwargs.pop('vmax')

    smfs = []
    hemis = [['lh', 'rh'], ['lh', 'rh']]
    views = ['lateral', 'medial']

    for i in range(2):
        for j in range(2):
            figure, smf = base_surf_plot(data[i], hemis[i][j], inflate, fs_avg,
                                         figure=figure, axes=ax[i, j],
                                         view=views[i], vmin=vmin, vmax=vmax,
                                         **kwargs)
            smfs.append(smf)

    figure.subplots_adjust(hspace=0, wspace=0)
    add_collage_colorbar(figure, ax, smfs, vmax, vmin, **kwargs)

    y = ax[0][0].get_position().y1 - title_y_adjust

    plt.suptitle(title, fontsize=fontsize, y=y,
                 x=ax[0][0].get_position().x1)

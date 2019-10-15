import Neuro_Shit as NS

main_dr = '/users/s/a/sahahn/raw_vertex_data'
save_dr = '/users/s/a/sahahn/new_merged_vertex_data'
dof_dr = '/users/s/a/sahahn/dofs_by_subject'
task_names = ['MID', 'SST', 'nBack']
task_inds = [[39, 45, 35, 37], [19, 21, 25], [19, 21, 27, 29]]

NS.Merge_Cortical(main_dr, save_dr, dof_dr, task_names, task_inds, True)

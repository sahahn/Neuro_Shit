import Neuro_Shit

tasks = ['nBack', 'SST', 'MID']
perf_covars = {'nBack': "['dprime_2back', 'dprime_0back']",
               'SST': "['tfmri_sst_all_beh_total_meanrt']",
               'MID': "[]"}

base_args =\
 ["import Neuro_Shit",
  "data_dr = '/users/s/a/sahahn/ABCD_Data/'",
  "palm_dr = '/users/s/a/sahahn/Process_ABCD_Data/Palm'",
  "cortical_dr = data_dr + 'new_merged_vertex_data'",
  "subcortical_dr = None",
  "run_name = ''"
  ]


rest =\
 ["covars_loc = palm_dr + '/Covars_' + task + '.csv'",
  "palm_obj = Neuro_Shit.Palm(palm_dr, task, covars_loc=covars_loc, cortical_dr=cortical_dr, subcortical_dr=subcortical_dr, performance_covars=performance_covars, run_name=run_name)"]

content = []
for task in tasks:

    content += base_args

    content.append("task = '" + task + "'")
    content.append("performance_covars = " + perf_covars[task])

    content += rest
    content.append("palm_obj.prep()")

    Neuro_Shit.VACC(base_script_contents=content, ppn='2', mem='200gb',
                    vmem='220gb', fs_import=True)

    content = []

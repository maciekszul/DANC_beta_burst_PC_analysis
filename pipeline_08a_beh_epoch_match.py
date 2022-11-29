import sys
import json
from mne import read_epochs, read_events, set_log_level
import os.path as op
from os import sep
from utilities import files
import pandas as pd
import numpy as np

set_log_level(verbose=False)

# parsing command line arguments
try:
    index = int(sys.argv[1])
except:
    print("incorrect arguments")
    sys.exit()

try:
    json_file = sys.argv[2]
    print("USING:", json_file)
except:
    json_file = "settings.json"
    print("USING:", json_file)

# opening a json file
with open(json_file) as pipeline_file:
    parameters = json.load(pipeline_file)

path = parameters["dataset_path"]
sfreq = parameters["downsample_dataset"]
hi_pass = parameters["hi_pass_filter"]
sub_path = op.join(path, "data")

der_path = op.join(path, "derivatives")
files.make_folder(der_path)
proc_path = op.join(der_path, "processed")
files.make_folder(proc_path)

subjects = files.get_folders_files(proc_path)[0]
subjects.sort()
subject = subjects[index]
subject_id = subject.split("/")[-1]

sub_path = op.join(proc_path, subject_id)
files.make_folder(sub_path)

qc_folder = op.join(sub_path, "QC")
files.make_folder(qc_folder)

mot_epo_paths = files.get_files(sub_path, "sub", "motor-epo.fif")[2]
mot_epo_paths.sort()
vis_epo_paths = files.get_files(sub_path, "sub", "visual-epo.fif")[2]
vis_epo_paths.sort()
beh_paths = files.get_files(sub_path, "sub", "-beh.csv")[2]
beh_paths.sort()
eve_paths = files.get_files(sub_path, "sub", "-eve.fif")[2]
eve_paths.sort()

match_out_dict = {}
# for epo in epo_paths:
for beh_path in beh_paths[1:]:
    numero = beh_path.split(sep)[-1].split("-")[2]
    mot_epo_path = [i for i in mot_epo_paths if numero in i][0]
    vis_epo_path = [i for i in vis_epo_paths if numero in i][0]
    eve_path = [i for i in eve_paths if numero in i][0]
    print(
        numero, "\n",
        beh_path.split(sep)[-1], "\n",
        mot_epo_path.split(sep)[-1], "\n",
        vis_epo_path.split(sep)[-1], "\n",
        eve_path.split(sep)[-1]
    )

    beh = pd.read_csv(beh_path, index_col=False)
    beh.dropna(inplace=True)
    fixation = read_events(eve_path, include=20)
    dots = read_events(eve_path, include=30)
    feedback = read_events(eve_path, include=70)
    
    range_trial = list(zip(list(fixation[:,0][:-1]), list(fixation[:,0][1:])))
    range_trial.append((fixation[-1,0], fixation[-1,0] + 7000))

    mot_ixs = []
    vis_ixs = []
    beh_ixs = beh.trial_in_block.to_numpy()
    for tr_ix, tr_rg in enumerate(range_trial):
        if any([(tr_rg[0] < dot) and (dot < tr_rg[1]) for dot in dots[:,0]]):
            vis_ixs.append(tr_ix)
        if any([(tr_rg[0] < fdb) and (fdb < tr_rg[1]) for fdb in feedback[:,0]]):
            mot_ixs.append(tr_ix)

    mot = read_epochs(mot_epo_path, verbose=False, preload=False)
    vis = read_epochs(vis_epo_path, verbose=False, preload=False)

    match_out_dict[mot_epo_path.split(sep)[-1]] = mot_ixs
    match_out_dict[vis_epo_path.split(sep)[-1]] = vis_ixs

    print("VIS", len(vis_ixs) == len(vis))
    print("MOT", len(mot_ixs) == len(mot))

# print(match_out_dict)
json_file_name = "{}-beh-match.json".format(subject_id)
json_out_path = op.join(sub_path, json_file_name)

with open(json_out_path, "w") as fp:
    json.dump(match_out_dict, fp, indent=4)
print("FILE SAVED:", json_out_path)
import json
import math
import copy
import sys
import warnings
from os import sep
import os.path as op
import numpy as np
from fooof import FOOOF
from extra.tools import extract_bursts
from mne import read_epochs
from utilities import files


warnings.filterwarnings("ignore")
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

#setting the paths and extracting files
slt_mot_paths  = [i for i in files.get_folders_files(sub_path)[0] if "motor" in i]
slt_vis_paths = [i for i in files.get_folders_files(sub_path)[0] if "visual" in i]
epo_mot_paths  = files.get_files(sub_path, "sub", "motor-epo.fif")[2]
epo_vis_paths = files.get_files(sub_path, "sub", "visual-epo.fif")[2]
beh_match_path = files.get_files(sub_path, "sub", "beh-match.json")[2][0]
with open(beh_match_path) as f:
    beh_match = json.load(f)
slt_mot_paths.sort()
slt_vis_paths.sort()
epo_mot_paths.sort()
epo_vis_paths.sort()
epo_slt_mot_vis = list(zip(epo_mot_paths, epo_vis_paths, slt_mot_paths, slt_vis_paths))

info = read_epochs(epo_mot_paths[0], verbose=False)
info.pick_types(meg=True, ref_meg=False, misc=False)
info = info.info
freqs = np.linspace(1,120, num=400)
search_range = np.where((freqs >= 10) & (freqs <= 33))[0]
beta_lims = [13, 30]

vis_output = {}
mot_output = {}

# for block, (epo_mot_p, epo_vis_p, slt_mot_p, slt_vis_p) in enumerate(epo_slt_mot_vis[:2]):
for block, (epo_mot_p, epo_vis_p, slt_mot_p, slt_vis_p) in enumerate(epo_slt_mot_vis):
    beh_match_vis = beh_match[epo_vis_p.split(sep)[-1]]
    beh_match_mot = beh_match[epo_mot_p.split(sep)[-1]]
    slt_vis_nps = files.get_files(slt_vis_p, "", ".npy")[2]
    slt_vis_nps.sort()
    slt_mot_nps = files.get_files(slt_mot_p, "", ".npy")[2]
    slt_mot_nps.sort()
    epo_vis = read_epochs(epo_vis_p, verbose=False, preload=False)
    epo_mot = read_epochs(epo_mot_p, verbose=False, preload=False)
    print("vis", subject_id, block, len(beh_match_vis)==len(slt_vis_nps), len(beh_match_vis)==len(epo_vis), len(slt_vis_nps)==len(epo_vis), len(beh_match_vis), len(slt_vis_nps), len(epo_vis))
    print("mot", subject_id, block, len(beh_match_mot)==len(slt_mot_nps), len(beh_match_mot)==len(epo_mot), len(slt_mot_nps)==len(epo_mot), len(beh_match_mot), len(slt_mot_nps), len(epo_mot))
print("\n")







import sys
import json
import numpy as np
import pandas as pd
import mne
import os.path as op
from utilities import files
import matplotlib.pylab as plt
from matplotlib import gridspec
import meegkit as mk
from tools import dump_the_dict


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
sub_path = op.join(path, "data")
der_path = op.join(path, "derivatives")
files.make_folder(der_path)
proc_path = op.join(der_path, "processed")
files.make_folder(proc_path)

subjects = files.get_folders_files(proc_path)[0]
subjects.sort()
subject = subjects[index]
subject_id = subject.split("/")[-1]

meg_path = op.join(subject, "ses-01", "meg")

sub_path = op.join(proc_path, subject_id)
files.make_folder(sub_path)

qc_folder = op.join(sub_path, "QC")
files.make_folder(qc_folder)

raw_paths = files.get_files(sub_path, "zapline-" + subject_id, "-raw.fif")[2]
raw_paths.sort()
event_paths = files.get_files(sub_path, subject_id, "-eve.fif")[2]
event_paths.sort()

raw_eve_list = list(zip(raw_paths, event_paths))

ica_json = dict()

for raw_path, eve_path in raw_eve_list:
    print("INPUT RAW FILE:", raw_path)
    print("EVE_RAW MATCH:", raw_path.split("-")[-2] == eve_path.split("-")[-2])
    numero = str(raw_path.split("-")[-2]).zfill(3)
    
    raw = mne.io.read_raw_fif(raw_path, verbose=False, preload=False)

    raw_filtered = raw.copy()
    raw_filtered.load_data().filter(
        l_freq=1.,
        h_freq=60, 
        n_jobs=-1
    )
    
    ica = mne.preprocessing.ICA(
        method="infomax",
        fit_params=dict(extended=True),
        n_components=25,
        max_iter=5000
    )
    ica.fit(raw_filtered)

    ica_name = "{}-{}-ica.fif".format(subject_id, numero)

    ica_file = op.join(
        sub_path,
        ica_name
    )

    ica.save(ica_file)

    ica_json[ica_name] = []

ica_json_path = op.join(
    sub_path, 
    "{}-ICA_to_reject.json".format(subject_id)
)
if not op.exists(ica_json_path):
    dump_the_dict(
        ica_json_path,
        ica_json
    )

import sys
import json
import numpy as np
import pandas as pd
import mne
import os
import os.path as op
import subprocess as sp
from utilities import files
import matplotlib.pylab as plt

# parsing command line arguments
try:
    index = int(sys.argv[1])
except:
    print("incorrect subject index")
    sys.exit()

try:
    file_index = int(sys.argv[2])
except:
    print("incorrect file index")
    sys.exit()

try:
    json_file = sys.argv[3]
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
subject_id = subject.split(os.sep)[-1]

meg_path = op.join(subject, "ses-01", "meg")

sub_path = op.join(proc_path, subject_id)
files.make_folder(sub_path)

qc_folder = op.join(sub_path, "QC")
files.make_folder(qc_folder)

raw_paths = files.get_files(sub_path, "zapline-" + subject_id, "-raw.fif")[2]
raw_paths.sort()
raw_path = raw_paths[file_index]

ica_paths = files.get_files(sub_path, subject_id, "-ica.fif")[2]
ica_paths.sort()
ica_path = ica_paths[file_index]

ica_json_file = op.join(
    sub_path,
    "{}-ICA_to_reject.json".format(subject_id)
)


print("SUBJ: {}".format(subject_id), index, file_index)
print("INPUT RAW FILE:", raw_path.split(os.sep)[-1])
print("INPUT ICA FILE:", ica_path.split(os.sep)[-1])
print("INPUT JSON FILE", ica_json_file.split(os.sep)[-1])

raw = mne.io.read_raw_fif(
    raw_path, preload=True, verbose=False
)
ica = mne.preprocessing.read_ica(
    ica_path, verbose=False
)

raw.filter(1,20, verbose=False)
raw.close()

title_ = "sub:{}, file: {}".format(subject_id, ica_path.split(os.sep)[-1])

ica.plot_components(inst=raw, show=False, title=title_)

ica.plot_sources(inst=raw, show=False, title=title_)

plt.show(block=False)

sp.Popen(
    ["notepad", str(ica_json_file)], 
    stdout=sp.DEVNULL, 
    stderr=sp.DEVNULL
)

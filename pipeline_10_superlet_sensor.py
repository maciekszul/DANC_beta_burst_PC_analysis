import sys
import json
from mne import read_epochs, set_log_level, pick_types
import os.path as op
from os import sep
from utilities import files
import numpy as np
from mne_superlet import superlets_do_single_epoch
from time import gmtime, strftime


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
subject_id = subject.split(sep)[-1]

sub_path = op.join(proc_path, subject_id)
files.make_folder(sub_path)

qc_folder = op.join(sub_path, "QC")
files.make_folder(qc_folder)

epo_paths = files.get_files(sub_path, "sub", "-epo.fif")[2]
epo_paths.sort()

for epo_path in epo_paths:
    filename = epo_path.split(sep)[-1].split(".")[0]
    npz_path = op.join(subject, "{}.npz".format(filename))
    output_folder = op.join(subject, filename)
    files.make_folder(output_folder)
    if not op.exists(npz_path):
        print(strftime("%a, %d %b %Y %H:%M:%S", gmtime()), "STARTED", output_folder)
        epochs = read_epochs(epo_path)
        epochs = epochs.pick_types(meg=True, ref_meg=False)
        for ix, epoch in enumerate(epochs.__iter__()):
            fname = "{}-{}.npy".format(str(ix).zfill(3), filename)
            output_array = superlets_do_single_epoch(epoch, epochs.info, max_freq=100, num=20, n_jobs=-1)
            output_path = op.join(output_folder,fname)
            np.save(output_path, output_array)
            del output_array
        print(strftime("%a, %d %b %Y %H:%M:%S", gmtime()), "FINISHED:", output_folder)
    if op.exists(npz_path):
        print(strftime("%a, %d %b %Y %H:%M:%S", gmtime()), "EXISTS", output_folder)
    
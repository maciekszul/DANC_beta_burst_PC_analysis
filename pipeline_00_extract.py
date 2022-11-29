import sys
import json
import numpy as np
import pandas as pd
import mne
import os.path as op
from utilities import files
import subprocess as sp


try:
    json_file = sys.argv[1]
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

subjects = files.get_folders_files(sub_path)[0]
subjects.sort()
for subject in subjects:
    subject_id = subject.split("/")[-1]

    meg_path = op.join(subject, "ses-01", "meg")

    if len(files.get_folders_files(meg_path)[0]) > 0:
        continue
    else:
        sub_path = op.join(proc_path, subject_id)
        files.make_folder(sub_path)

        zip_file = files.get_files(meg_path, "MEG", ".zip")[2][0]

        print(files.get_files(meg_path, "MEG", ".zip")[1])

        sp.call(["unzip", zip_file, "-d" , meg_path])

        unzip_path_part = op.join(meg_path, "scans", "MEGSCAN_CTF")
        unzip_path = files.get_folders_files(unzip_path_part)[0][0]
        print(unzip_path, op.exists(unzip_path))
        path_to_move = "mv -v " + unzip_path + "/* " + meg_path
        print(path_to_move)
        sp.call(path_to_move, shell=True)
        sp.call(["rm", "-rf", op.join(meg_path, "scans")])
        print(subject_id, "DONE")
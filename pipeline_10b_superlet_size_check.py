import sys
import json
import os.path as op
from os import sep
from utilities import files
import pandas as pd
import numpy as np

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

all_folders = files.get_folders_files(sub_path)[0]
all_folders = [i for i in all_folders if "-epo" in i]
all_folders.sort()

all_visual = [i for i in all_folders if "visual" in i]
all_motor = [i for i in all_folders if "motor" in i]

mot_size = 658038528
vis_size = 789558528

for folder in all_visual:
    all_files = files.get_files(folder, "", ".npy")[2]
    for file in all_files:
        if op.getsize(file) < vis_size:
            print(file, op.getsize(file))

for folder in all_motor:
    all_files = files.get_files(folder, "", ".npy")[2]
    for file in all_files:
        if op.getsize(file) < mot_size:
            print(file, op.getsize(file))
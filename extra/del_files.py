import sys
import json
import os.path as op
from os import sep
from utilities import files
import pandas as pd
import numpy as np
import subprocess as sp


# parsing command line arguments
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
hi_pass = parameters["hi_pass_filter"]
sub_path = op.join(path, "data")

der_path = op.join(path, "derivatives")
files.make_folder(der_path)
proc_path = op.join(der_path, "processed")
files.make_folder(proc_path)

files_2_del_path = op.join(der_path, "missing_superlet_files_24_02_2022.csv")
files_2_del = pd.read_csv(files_2_del_path, names=["files", "bytes"])

for path in files_2_del.files.to_numpy():
    print(path)
    sp_call = [
        "rm",
        path
    ]

    sp.call(sp_call)
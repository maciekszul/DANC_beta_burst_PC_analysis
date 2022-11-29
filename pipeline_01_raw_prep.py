import sys
import json
import numpy as np
import pandas as pd
import mne
import os.path as op
from utilities import files


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

subjects = files.get_folders_files(sub_path)[0]
subjects.sort()
subject = subjects[index]
subject_id = subject.split("/")[-1]

meg_path = op.join(subject, "ses-01", "meg")

sub_path = op.join(proc_path, subject_id)
files.make_folder(sub_path)

dss = files.get_folders_files(meg_path)[0]
dss = [i for i in dss if "ds" in i]
dss.sort()

for ds in dss:
    print("INPUT RAW FILE:", ds)
    numero = int(ds.split(".")[0][-2:])
    raw = mne.io.read_raw_ctf(
        ds, 
        clean_names=True,
        verbose=False
    )
    raw_events = mne.find_events(
        raw,
        stim_channel="UPPT002",
        min_duration=0.002,
        verbose="DEBUG",
        consecutive=True
    )

    diode_events = mne.pick_events(
        raw_events,
        include=[30, 50]
    )

    raw = raw.crop(
        tmin=raw.times[raw_events[0,0]] - 0.1,
        tmax=raw.times[raw_events[-1,0]] + 0.1
    )

    raw, events = raw.copy().resample(
        sfreq, 
        npad="auto", 
        events=raw_events,
        n_jobs=-1,
    )

    f_n = str(numero).zfill(3) # file number

    raw_path = op.join(
        sub_path, 
        "{}-{}-raw.fif".format(subject_id, f_n)
    )
    eve_path = op.join(
        sub_path, 
        "{}-{}-eve.fif".format(subject_id, f_n)
    )

    raw.save(
        raw_path,
        fmt="single",
        overwrite=True)

    print("RAW SAVED:", raw_path)
    
    raw.close()

    mne.write_events(
        eve_path,
        events
    )

    print("EVENTS SAVED:", eve_path)
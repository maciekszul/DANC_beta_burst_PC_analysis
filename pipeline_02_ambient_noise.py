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
from zapline_iterator.zapline_iter import zapline_until_gone


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

raw_paths = files.get_files(sub_path, subject_id, "-raw.fif")[2]
raw_paths.sort()
event_paths = files.get_files(sub_path, subject_id, "-eve.fif")[2]
event_paths.sort()

raw_eve_list = list(zip(raw_paths, event_paths))

# for raw_path, eve_path in [raw_eve_list[0]]:
for raw_path, eve_path in raw_eve_list:
    print("INPUT RAW FILE:", raw_path)
    print("EVE_RAW MATCH:", raw_path.split("-")[-2] == eve_path.split("-")[-2])
    numero = str(raw_path.split("-")[-2]).zfill(3)
    
    raw = mne.io.read_raw_fif(raw_path, verbose=False)
    # raw = raw.apply_gradient_compensation(2, verbose=True)
    raw = raw.pick_types(meg=True, eeg=False, ref_meg=True)
    fig = raw.plot_psd(
        tmax=np.inf, fmax=260, average=True, show=False, picks="meg"
    )
    fig.suptitle(subject_id)
    plt.savefig(
        op.join(qc_folder, "{}-{}-raw-psd.png".format(subject_id, numero)),
        dpi=150, bbox_inches="tight"
    )
    plt.close("all")
    
    info = raw.info
    raw = raw.get_data()
       
    nsplit = 20
    if numero == "001":
        nsplit = 5

    raw = np.array_split(raw, nsplit, axis=1)
    # ZAPLINE
    output = []
    for chunk in raw:

        zapped, iterations = zapline_until_gone(
            chunk,
            50.0,
            600.0,
            win_sz=7.5
        )

        output.append(zapped)

        zapped, iterations = zapline_until_gone(
            zapped,
            60.0,
            600.0,
            win_sz=7.5
        )
        output.append(zapped)

    # recreating the data structure
    output = [i.transpose() for i in output]
    raw = np.concatenate(output)
    del output
    raw = mne.io.RawArray(
        raw.transpose(),
        info
    )

    fig = raw.plot_psd(tmax=np.inf, fmax=260, average=True, show=False)
    fig.suptitle(subject_id)
    plt.savefig(
        op.join(qc_folder, "{}-{}-zapline-raw-psd.png".format(subject_id, numero)),
        dpi=150, 
        bbox_inches="tight"
    )
    plt.close("all")

    f_n = str(numero).zfill(3) # file number
    out_path = op.join(
        sub_path, 
        "zapline-{}-{}-raw.fif".format(subject_id, f_n)
    )

    raw.save(
        out_path,
        overwrite=True
    )
    del raw

    

import sys
import json
import os.path as op
from os import sep
import pandas as pd
import numpy as np
from mne import read_epochs, read_events, set_log_level, pick_channels
from utilities import files
from autoreject import AutoReject
import matplotlib.pylab as plt

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

epo_paths = files.get_files(sub_path, "sub", "-epo.fif")[2]
epo_paths.sort()
beh_paths = files.get_files(sub_path, "sub", "-beh.csv")[2]
eve_paths = files.get_files(sub_path, "sub", "-eve.fif")[2]


for epo in epo_paths:
# for epo in [epo_paths[0]]:
    numero = epo.split(sep)[-1].split("-")[2]
    beh_path = [i for i in beh_paths if numero in i][0]
    eve_path = [i for i in eve_paths if numero in i][0]

    print("BEH:", beh_path.split(sep)[-1])
    print("EVE:", eve_path.split(sep)[-1])
    print("EPO:", epo.split(sep)[-1])

    epochs = read_epochs(epo, verbose=False)
    print("AMOUNT OF EPOCHS:", len(epochs))
    beh = pd.read_csv(beh_path)
    beh_ixs = beh.dropna().trial_in_block.to_numpy()
    print("AMOUNT OF BEH TRIALS AFTER REMOVAL:", len(beh_ixs))

    fix = read_events(eve_path, include=20)
    dts = read_events(eve_path, include=30)
    fdb = read_events(eve_path, include=70)

    print("FIX EVE:", fix.shape[0])
    print("DOTS EVE:", dts.shape[0])
    print("FEEDBACK EVE:", fdb.shape[0])

    tr_ranges = list(zip(list(fix[:,0][:-1]), list(fix[:,0][1:])))
    tr_ranges.append((fix[-1,0], fix[-1,0] + 10000))

    fix_ixs = np.arange(fix.shape[0])
    beh_ixs = np.intersect1d(fix_ixs, beh_ixs)
    
    dots_ixs = []
    for ix, tr in enumerate(tr_ranges):
        if any([(tr[0] < dt) and (dt < tr[1]) for dt in dts[:,0]]):
            dots_ixs.append(ix)
    dots_ixs = np.array(dots_ixs)
    dots_ixs = np.intersect1d(dots_ixs, fix_ixs)

    fdb_ixs = []
    for ix, tr in enumerate(tr_ranges):
        if any([(tr[0] < fb) and (fb < tr[1]) for fb in fdb[:,0]]):
            fdb_ixs.append(ix)
    fdb_ixs = np.array(fdb_ixs)
    fdb_ixs = np.intersect1d(fdb_ixs, fix_ixs)

    ixs = None
    if "motor" in epo:
        ixs = fdb_ixs
    elif "visual" in epo:
        ixs = dots_ixs

    intersect_ixs = np.intersect1d(beh_ixs, ixs)
    intersect_order = np.intersect1d(beh_ixs, ixs, return_indices=True)[2]
    epoch_order = np.arange(ixs.shape[0])
    epochs_2_drop = [i for i in epoch_order if i not in intersect_order]
    
    # print(beh_ixs, "beh")
    # print(ixs, "meg")
    # print(np.intersect1d(beh_ixs, ixs), "isct")

    # print(intersect_order, "ix intersect")
    # print(epoch_order, "ix base")
    # print(epochs_2_drop, "drop")
    epochs.load_data()
    epochs = epochs.drop(epochs_2_drop, reason="bad behaviour")
    epochs.save(
        op.join(sub_path, "clean-" + epo.split(sep)[-1]),
        overwrite=True
    )
    print("AMOUNT OF EPOCHS AFTER MATCHING WITH BEH:", len(epochs))
    print("DOES IT MATCH?", len(beh_ixs)==len(epochs))
    print("\n")

    if len(beh_ixs)==len(epochs):
        ar = AutoReject(
            consensus=np.linspace(0, 1.0, 27),
            n_interpolate=np.array([1, 4, 32]),
            thresh_method="bayesian_optimization",
            cv=10,
            n_jobs=-1,
            random_state=42,
            verbose="progressbar"
        )
        ar.fit(epochs)

        epo_type = epo.split(sep)[-1].split("-")[3]
        name = "{}-{}-{}".format(subject_id, numero, epo_type)
        ar_fname = op.join(
            qc_folder,
            "{}-autoreject.h5".format(name)
        )
        ar.save(ar_fname, overwrite=True)
        epochs_ar, rej_log = ar.transform(epochs, return_log=True)
        rej_log.plot(show=False)
        plt.savefig(op.join(qc_folder, "{}-autoreject-log.png".format(name)))
        plt.close("all")
        epo.split(sep)[-1]
        cleaned = op.join(sub_path, "autoreject-" + epo.split(sep)[-1])
        epochs.save(
            op.join(sub_path, "autoreject-" + epo.split(sep)[-1]),
            overwrite=True
        )
        print("CLEANED EPOCHS SAVED:", cleaned)
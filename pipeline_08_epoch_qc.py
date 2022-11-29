import sys
import json
from mne import read_epochs, read_events, set_log_level
from matplotlib import colors
import os.path as op
from os import sep
from utilities import files
import matplotlib.pylab as plt
from autoreject import compute_thresholds
import pandas as pd
import numpy as np

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

cmap = colors.ListedColormap(["#FFFFFF", "#CFEEFA", "#FFDE00", "#FF9900", "#FF0000", "#000000"])
boundaries = [-0.9, -0.1, 1.1, 10, 100, 1000, 10000]
norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)

# for epo in epo_paths:
for epo in epo_paths[1:]:
    numero = epo.split(sep)[-1].split("-")[2]
    epochs = read_epochs(epo, verbose=False)
    epochs = epochs.decimate(3)
    print("INPUT FILE:", epo)
    beh_path = [i for i in beh_paths if numero in i][0]
    eve_path = [i for i in eve_paths if numero in i][0]
    beh = pd.read_csv(beh_path)
    beh_ixs = beh.dropna().trial_in_block.to_numpy()
    fix = read_events(eve_path, include=20)
    dts = read_events(eve_path, include=30)
    fdb = read_events(eve_path, include=70)

    tr_ranges = list(zip(list(fix[:,0][:-1]), list(fix[:,0][1:])))
    tr_ranges.append((fix[-1,0], fix[-1,0] + 7000))

    fix_ixs = np.arange(fix.shape[0])
    beh_ixs = np.intersect1d(fix_ixs, beh_ixs)

    dots_ixs = []
    for ix, tr in enumerate(tr_ranges):
        if any([(tr[0] < dt) and (dt < tr[1]) for dt in dts[:,0]]):
            dots_ixs.append(ix)
    dots_ixs = np.array(dots_ixs)
    dots_ixs = np.intersect1d(fix_ixs, dots_ixs)

    fdb_ixs = []
    for ix, tr in enumerate(tr_ranges):
        if any([(tr[0] < fb) and (fb < tr[1]) for fb in fdb[:,0]]):
            fdb_ixs.append(ix)
    fdb_ixs = np.array(fdb_ixs)
    fdb_ixs = np.intersect1d(fix_ixs, fdb_ixs)
    
    ixs = []
    if "motor" in epo:
        ixs = fdb_ixs
    elif "visual" in epo:
        ixs = dots_ixs
    
    ch_thr = compute_thresholds(
        epochs,
        random_state=42,
        method="bayesian_optimization",
        verbose="progressbar",
        n_jobs=-1,
        augment=False
    )
    # save the thresholds in JSON
    ch_list = list(ch_thr.keys())
    ch_list.sort()
    results = np.zeros((len(ch_list), 56))
    results = results - 1
    for ix, ch in enumerate(ch_list):
        thr = ch_thr[ch]
        ch_tr = epochs.copy().pick_channels([ch]).get_data()
        res = [np.where(ch_tr[i][0] > thr)[0].shape[0] for i in range(len(epochs))]
        res = np.array(res)
        # res = np.sign(res)
        mask = np.zeros(56)
        mask[ixs] = res
        beh_xx = np.intersect1d(beh_ixs, ixs)
        results[ix, beh_xx] = mask[beh_xx]
    epo_type = epo.split(sep)[-1].split("-")[3]
    name = "{}-{}-{}".format(subject_id, numero, epo_type)
    npy_path = op.join(qc_folder, name + ".npy")
    np.save(npy_path, results)
    img_path = op.join(qc_folder, name + "-epo-QC.png")
    print(results[:15, :15])
    print(np.min(results), np.max(results))
    print(np.unique(results))

    plt.rcParams.update({'font.size': 5})
    f, ax = plt.subplots(
            figsize=(20,20), 
            dpi=200
        )
    
    im = ax.imshow(
        results,
        aspect="auto",
        cmap=cmap,
        interpolation="none",
        norm=norm
    )
    f.colorbar(im, ax=ax, fraction=0.01, pad=0.01)
    ax.set_xlabel("Trials")
    ax.set_ylabel("Channels")
    ax.set_xticks(list(range(56)))
    ax.set_xticklabels([str(i) for i in range(1, 57)])
    ax.set_yticks(list(range(len(ch_list))))
    ax.set_yticklabels(ch_list)
    ax.grid(color='w', linestyle='-', linewidth=0.2)
    ax.set_title(name)
    plt.savefig(
        img_path,
        bbox_inches="tight"
    )
    plt.close("all")
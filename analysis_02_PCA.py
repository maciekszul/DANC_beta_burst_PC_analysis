import sys
import json
import pickle
import warnings
import os.path as op
import numpy as np
import pandas as pd
from os import sep
import itertools as it
from copy import deepcopy
from utilities import files
from functools import partial
from tqdm import trange, tqdm
import matplotlib.pylab as plt
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from mne import read_epochs, pick_channels
from extra.tools import many_is_in, cat, shuffle_array, shuffle_array_range, consecutive_margin_ix, dump_the_dict

warnings.filterwarnings("ignore", category=RuntimeWarning)

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
der_path = op.join(path, "derivatives")
files.make_folder(der_path)
proc_path = op.join(der_path, "processed")
files.make_folder(proc_path)
PCA_path = op.join(der_path, "PCA_results")
files.make_folder(PCA_path)

sensors = [
    'MLC21', 'MLC22', 'MLC23', 'MLC24', 
    'MLC31', 'MLC41', 'MLC51', 'MLC52', 
    'MLC53', 'MLC61', 'MLC62'
]

path = "/home/mszul/datasets/explicit_implicit_beta/derivatives/processed"
subs = files.get_folders_files(path)[0]
subs = [op.join(i, "bursts") for i in subs]
subs = [i for i in subs if op.exists(i)]
subs.sort()

json_files = {}
for sub in subs:
    sub_id = sub.split(sep)[-2]
    subs_bursts = files.get_files(sub, "", ".json")[2]
    subs_bursts = [i for i in subs_bursts if many_is_in(sensors, i)]
    subs_bursts.sort()
    json_files[sub_id] = subs_bursts

all_files = [y for i in json_files.keys() for y in json_files[i]]
waveforms = []
for json_file in tqdm(all_files):
    with open(json_file) as pipeline_file:
        bs = json.load(pipeline_file)
    am_bs = len(bs["waveform"])
    bs_samp_ix = np.random.choice(np.arange(am_bs), int(am_bs*0.2))
    bs_samp = np.vstack(bs['waveform'])[bs_samp_ix, :]
    waveforms.append(bs_samp)

waveforms = np.vstack(waveforms)

waveforms_medians = np.median(waveforms, axis=1)
clean_ixs = np.where((waveforms_medians > np.percentile(waveforms_medians, 10)) & (waveforms_medians < np.percentile(waveforms_medians, 90)))[0]
waveforms_clean = waveforms[clean_ixs]

npy_path = op.join(PCA_path, "waveforms_clean_subset_fit.npy")
np.save(npy_path, waveforms_clean)

scaler = RobustScaler().fit(waveforms_clean)
waveforms_scaled = scaler.transform(waveforms_clean)
scaler_path = op.join(PCA_path, "scaler_MEG.pkl")
pickle.dump(scaler, open(scaler_path, "wb"))

pca_obj = PCA(n_components=20)
pca_obj.fit(waveforms_scaled)
pca_pkl_path = op.join(PCA_path, "pca_solution_MEG.pkl")
pickle.dump(pca_obj, open(pca_pkl_path, "wb"))

sub_metrics = {}
for sub in tqdm(json_files.keys()):
    print(sub)
    all_files = json_files[sub]
    metrics = {
        "vis": {
            "waveform": [],
            "peak_time": [],
            "peak_freq": [],
            "peak_amp_base": [],
            "fwhm_freq": [],
            "fwhm_time": [],
            "peak_adjustment": [],
            "polarity": [],
            "trial": [],
            "pp_ix": [],
            "block" : []
        },
        "mot": {
            "waveform": [],
            "peak_time": [],
            "peak_freq": [],
            "peak_amp_base": [],
            "fwhm_freq": [],
            "fwhm_time": [],
            "peak_adjustment": [],
            "polarity": [],
            "trial": [],
            "pp_ix": [],
            "block" : []
        }
    }

    for json_file in all_files:
        big_key = [i for i in ["mot", "vis"] if i in json_file][0]
        with open(json_file) as pipeline_file:
            bs = json.load(pipeline_file)
        wf = np.array(bs['waveform'])
        wf_median = np.median(wf, axis=1)
        wf_ixs = np.where(
            (wf_median > np.percentile(waveforms_medians, 1)) & 
            (wf_median < np.percentile(waveforms_medians, 99))
        )[0]

        wf = wf[wf_ixs,:]
        metrics[big_key]["waveform"].append(wf)
        for k in ["peak_time", "peak_amp_base", "fwhm_freq", "fwhm_time", "peak_freq", "trial", "pp_ix", "block"]:
            metrics[big_key][k].append(np.array(bs[k])[wf_ixs])      

    for i in ["mot", "vis"]:
        metrics[i]["waveform"] = np.vstack(metrics[i]["waveform"])
        for k in ["peak_time", "peak_amp_base", "fwhm_freq", "fwhm_time", "peak_freq", "trial", "pp_ix", "block"]:
            metrics[i][k] = np.hstack(metrics[i][k])
    sub_metrics[sub] = metrics

sub_metrics_path = op.join(PCA_path, "sub_metrics.pkl")
pickle.dump(sub_metrics, open(sub_metrics_path, "wb"))

import sys
import json
import numpy as np
import pandas as pd
import subprocess
import mne
import os.path as op
from utilities import files
import matplotlib.pylab as plt
from pyedfread import edf
from tools import update_key_value, dump_the_dict, resamp_interp
from ecgdetectors import Detectors
from scipy.interpolate import interp1d
from scipy.stats import pearsonr
import gc


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


mne.set_log_level(verbose=None)


path = parameters["dataset_path"]
sfreq = parameters["downsample_dataset"]

der_path = op.join(path, "derivatives")
files.make_folder(der_path)
proc_path = op.join(der_path, "processed")
files.make_folder(proc_path)

subjects = files.get_folders_files(proc_path)[0]
subjects.sort()
subject = subjects[index]
subject_id = subject.split("/")[-1]

beh_path = op.join(path, "data", subject_id, "ses-01", "behaviour")

edf_paths = files.get_files(beh_path, "", ".edf")[2]
edf_paths.sort()
edf_paths = [i for i in edf_paths if "0.edf" not in i]

sub_path = op.join(proc_path, subject_id)
files.make_folder(sub_path)

qc_folder = op.join(sub_path, "QC")
files.make_folder(qc_folder)

raw_paths = files.get_files(sub_path, "zapline-" + subject_id, "-raw.fif")[2]
raw_paths.sort()

ica_json_file = op.join(
    sub_path,
    "{}-ICA_to_reject.json".format(subject_id)
)

with open(ica_json_file) as ica_file:
    ica_files = json.load(ica_file)

ica_keys = list(ica_files.keys())
ica_keys.sort()

raw_ica_edf = list(zip(raw_paths, ica_keys, edf_paths))

ecg_out = dict()
eog_out = dict()

eog_file_path = op.join(
    sub_path, 
    "{}-eog-stats.json".format(subject_id)
)
ecg_file_path = op.join(
    sub_path, 
    "{}-ecg-stats.json".format(subject_id)
)

ds = Detectors(sfreq)


# # for (raw_path, ica_key, edf_path) in [raw_ica_edf[2]]:
for (raw_path, ica_key, edf_path) in raw_ica_edf:
    ica_path = op.join(
        sub_path,
        ica_key
    )
    numero = str(raw_path.split("-")[-2]).zfill(3)
    

    try:
        if numero == "001":
            raise Exception
    except Exception:
        continue

    print("INPUT RAW FILE:", raw_path)
    print("INPUT ICA FILE:", ica_path)
    print("INPUT EDF FILE:", edf_path)
    
    raw = mne.io.read_raw_fif(
        raw_path,
        verbose=False,
        preload=True
    )

    ica = mne.preprocessing.read_ica(
        ica_path,
        verbose=False
    )
    raw.filter(1,20)

    raw.close()

    ica_com = ica.get_sources(raw)
    raw = None
    gc.collect()
    ica_times = ica_com.times
    ica_data = ica_com.get_data()
    ica_com.close()
    ica_com = None
    gc.collect()


    # https://github.com/berndporr/py-ecg-detectors
    # variance of the distance between detected R peaks
    # if the variance is not distinct enough from the 1 percentile,
    # signal has to be found manually, indicated as 666 in the first item
    # of the list.
    r_hr = [ds.hamilton_detector(ica_data[i]) for i in range(ica_data.shape[0])]
    r_hr = [np.var(np.diff(i)) for i in r_hr]
    ecg_out[ica_key] = r_hr
    r_hr = np.array(r_hr)

    if (np.percentile(r_hr, 1) - np.min(r_hr)) > 500:
        hr = list(np.where(r_hr < np.percentile(r_hr, 1))[0])
    else:
        hr = [666]

    samples, events, messages = edf.pread(edf_path)
    eye = ["left", "right"][events.eye.unique()[0]]
    
    del events
    del messages

    # cleaning the eyetracking datas
    samples = samples.loc[samples.time != 0.0]
    start = samples.loc[samples.input == 252.]
    end = samples.loc[samples.input == 253.]
    start_ix = start.index[0] - 100
    end_ix = end.index[-1] + 100
    samples = samples.iloc[start_ix:end_ix]
    samples.reset_index(inplace=True)
    samples.time = samples.time - samples.time.iloc[0]
    
    # picking the relevant piece of data    
    gx = samples["gx_{}".format(eye)]
    gy = samples["gy_{}".format(eye)]
    samples_times = samples.time/1000
    
    del samples

    # resampling to meg sampling rate
    gx = resamp_interp(samples_times, gx, ica_times)
    gy = resamp_interp(samples_times, gy, ica_times)
    
    # gx, gy is a gaze screen position, EyeLink recorded blinks as position way
    # outside of the screen, thus safe threshold to detect blinks. 
    # dependent on the screen resolution.
    blink_ix = np.where(gy > 1500)[0]

    clean_gx = np.copy(gx)
    clean_gy = np.copy(gy)
    gx_iqr = np.percentile(gx, [25, 50])
    gy_iqr = np.percentile(gx, [25, 50])
    gx_iqr_med = np.median(gx[np.where((gx>gx_iqr[0]) & (gx<gx_iqr[1]))[0]])
    gy_iqr_med = np.median(gy[np.where((gy>gy_iqr[0]) & (gy<gy_iqr[1]))[0]])

    clean_gx[blink_ix] = gx_iqr_med
    clean_gy[blink_ix] = gy_iqr_med

    clean_gx = pd.Series(clean_gx).interpolate().to_numpy()
    clean_gy = pd.Series(clean_gy).interpolate().to_numpy()

    # x, y, clean_x, clean_y
    # ICA comp N x 4x(r, p)
    ica_eog = []
    comp = []
    for i in range(ica_data.shape[0]):
        out = [pearsonr(j, ica_data[i]) for j in [gx, gy, clean_gx, clean_gy]]
        comp.append(out)
        results = np.array(out)
        if np.average(np.abs(results[:,0]) > 0.15) >=0.25:
            ica_eog.append(i)
    eog_out[ica_key] = comp


    # all the numbers have to be integers
    ica_eog = hr + ica_eog
    ica_eog = [int(i) for i in ica_eog]

    # update of the key values
    update_key_value(ica_json_file, ica_key, ica_eog)

# dump the stats in json files

for i in (ecg_file_path, eog_file_path):
    if not op.exists(i):
        subprocess.run(["touch", i])

dump_the_dict(ecg_file_path, ecg_out)
dump_the_dict(eog_file_path, eog_out)

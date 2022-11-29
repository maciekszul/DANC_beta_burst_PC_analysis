import h5py
import mne
import sys
import json
import os.path as op
from utilities import files
from scipy.interpolate import interp1d
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from matplotlib import gridspec
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


def resamp_interp(x, y, new_x):
    """
    returns resampled an interpolated data
    """
    resamp = interp1d(x, y, kind='slinear', fill_value='extrapolate')
    new_data = resamp(new_x)
    return new_data


path = parameters["dataset_path"]
sfreq = parameters["downsample_head"]
sub_path = op.join(path, "data")
subjects = files.get_folders_files(sub_path)[0]
subject = subjects[index]
subject_id = subject.split("/")[-1]
beh_path = op.join(subject, "ses-01", "behaviour")
der_path = op.join(path, "derivatives")
head_path = op.join(der_path, "head_motion", subject_id)

beh_files = files.get_files(beh_path, "block", ".mat")[2]
beh_files.sort()
head_csvs = files.get_files(head_path, subject_id, ".csv")[2]
head_csvs.sort()
head_evnts = files.get_files(head_path, subject_id, "-eve.fif")[2]
head_evnts.sort()

b_h_e = list(zip(beh_files, head_csvs, head_evnts))

for (beh_f, hpi_f, ev_f) in b_h_e[1:]:
    print(beh_f)
    print(hpi_f)
    print(ev_f)

    begin = mne.read_events(
        ev_f,
        include=[50]
    )

    end = mne.read_events(
        ev_f,
        include=[70]
    )

    hpi = pd.read_csv(
        hpi_f
    )

    beh_data = h5py.File(beh_f, "r")

    x = np.array(beh_data["run_data"]["trajectory"])[0].transpose()
    y = np.array(beh_data["run_data"]["trajectory"])[1].transpose()
    t = np.array(beh_data["run_data"]["trajectory"])[2].transpose()

    good_ix = np.argwhere(~np.isnan(x[:,0])).flatten()
    x = x[good_ix][:,:500]
    y = y[good_ix][:,:500]
    t = t[good_ix][:,:500]

    if begin.shape[0] != good_ix.shape[0]:
        raise Exception("Triggers do not match the behaviour")
    
    meg_ix = zip(begin[:,0], end[:,0])

    hpi["X_res"] = np.nan
    hpi["Y_res"] = np.nan
    hpi["VELOCITY"] = np.nan

    for ix, (b, e) in enumerate(meg_ix):
        print(ix, b, e)
        meg_times = hpi.times.iloc[b:e].to_numpy()
        joy_times = t[ix][~np.isnan(t[ix])]
        joy_times = ((joy_times-joy_times[0])/1000) + meg_times[0]
        x_o = x[ix][~np.isnan(x[ix])]
        y_o = y[ix][~np.isnan(y[ix])]
        x_n = resamp_interp(joy_times, x_o, meg_times)
        y_n = resamp_interp(joy_times, y_o, meg_times)
        velocity = np.sqrt(np.diff(x_n)**2 + np.diff(y_n)**2)/np.diff(meg_times)
        # velocity in px/s
        velocity = np.insert(velocity, 0, 0)
        hpi.X_res.iloc[b:e] = x_n
        hpi.Y_res.iloc[b:e] = y_n
        hpi.VELOCITY.iloc[b:e] = velocity
    
    hpi.to_csv(hpi_f, index=False)
    print("SAVED:", hpi_f)

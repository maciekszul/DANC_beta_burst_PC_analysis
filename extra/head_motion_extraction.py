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


def norm_vec(vec):
    """
    returns unit vector
    """
    mag = np.sqrt((vec[0]**2 + vec[1]**2 + vec[2]**2))
    unit_vector = np.array([vec[0] / mag, vec[1] / mag, vec[2] / mag])
    return unit_vector


path = parameters["dataset_path"]
sfreq = parameters["downsample_head"]
sub_path = op.join(path, "data")
der_path = op.join(path, "derivatives")
files.make_folder(der_path)
head_path = op.join(der_path, "head_motion")
files.make_folder(head_path)

subjects = files.get_folders_files(sub_path)[0]
subject = subjects[index]
subject_id = subject.split("/")[-1]

meg_path = op.join(subject, "ses-01", "meg")

dss = files.get_folders_files(meg_path)[0]
dss = [i for i in dss if "ds" in i]
dss.sort()


for ds in dss:
    print("PRINT:", ds)
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

    raw, events = raw.resample(
        sfreq, 
        n_jobs=-1,
        events=raw_events,
        verbose=False
    )

    hpi_coils = ["HLC0011", "HLC0012", "HLC0013", 
    "HLC0021", "HLC0022", "HLC0023", 
    "HLC0031", "HLC0032", "HLC0033"]

    hpi = {
    i: raw.get_data(i)[0] for i in hpi_coils
    }

    hpi = pd.DataFrame.from_dict(hpi)

    # EULER ANGLES EXTRACTION

    hpi["roll"] = None
    hpi["pitch"] = None
    hpi["yaw"] = None
    hpi["X"] = None
    hpi["Y"] = None
    hpi["Z"] = None

    for i, row in hpi.iterrows():
        xyz = row.to_numpy()[:9].astype("float")
        center_point = np.array([
            (xyz[3] + xyz[6]) / 2,
            (xyz[4] + xyz[7]) / 2,
            (xyz[5] + xyz[8]) / 2,
        ])
        P1 = xyz[:3]
        P2 = xyz[6:]
        P3 = xyz[3:6]
        Pm = center_point
        xv = P1 - Pm
        Xv = norm_vec(xv)
        v1 = P1 - P3
        v2 = P1 - P2
        Zv = np.cross(v1, v2)
        Zv = norm_vec(Zv)
        Yv = np.cross(Zv, Xv)
        roll = np.rad2deg(np.arctan2(-Zv[1], Zv[2]))
        pitch = np.rad2deg(np.arcsin(Zv[0]))
        yaw = np.rad2deg(np.arctan2(-Yv[0], Xv[0]))
        Xp = center_point[0]
        Yp = center_point[1]
        Zp = center_point[2]
        hpi.at[i, "roll"] = roll
        hpi.at[i, "pitch"] = pitch
        hpi.at[i, "yaw"] = yaw
        hpi.at[i, "X"] = Xp
        hpi.at[i, "Y"] = Yp
        hpi.at[i, "Z"] = Zp

    hpi["times"] = raw.times

    # saving the goodness

    out_path = op.join(head_path, subject_id)
    files.make_folder(out_path)

    f_n = str(numero).zfill(3) # file number

    raw.save(
        op.join(out_path, "{}-125-{}-raw.fif".format(subject_id, f_n)),
        fmt="single",
        overwrite=True
    )

    mne.write_events(
        op.join(out_path, "{}-{}-eve.fif".format(subject_id, f_n)),
        events
    )

    csv_hpi = op.join(
        out_path, "{}-{}-hpi.csv".format(subject_id, f_n)
    )

    hpi.to_csv(csv_hpi, index=False)
    print("SAVED:", csv_hpi)

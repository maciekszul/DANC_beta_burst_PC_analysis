import sys
import json
import h5py
import numpy as np
import pandas as pd
import os
import os.path as op
from tools import dump_the_dict, cart2pol
from utilities import visang, files
import matplotlib.pylab as plt


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
sub_path = op.join(path, "data")

der_path = op.join(path, "derivatives")
files.make_folder(der_path)
proc_path = op.join(der_path, "processed")
files.make_folder(proc_path)

subjects = files.get_folders_files(sub_path)[0]
subjects.sort()
subject = subjects[index]
subject_id = subject.split("/")[-1]

print("ID:", subject_id)

beh_path = op.join(subject, "ses-01", "behaviour")

beh_files = files.get_files(beh_path, "block", ".mat")[2]
beh_files.sort()

sub_path = op.join(proc_path, subject_id)
files.make_folder(sub_path)

qc_folder = op.join(sub_path, "QC")
files.make_folder(qc_folder)

# for beh_file in [beh_files[4]]:
for beh_file in beh_files:
    numero = beh_file.split(os.sep)[-1].split("_")[-1].split(".")[0].zfill(3)

    # try:
    #     if numero == "001":
    #         raise Exception
    # except Exception:
    #     continue

    raw = h5py.File(beh_file, "r")

    id = "sub-" + str(int(raw["run_params"]["subj_id"][0]))
    trial_n = int(np.array(raw["run_params"]["n_trials"]))

    time_ix = np.array(raw["run_data"]["trial_time_idx"]).flatten().astype(int)
    
    # taking into account MATLAB indexing
    # zero is a trial without a response
    time_ix[time_ix != 0] = time_ix[time_ix != 0] - 1

    traj_array = np.array(raw["run_data"]["trajectory"])[:,:np.max(time_ix)+9,:]

    w_px, h_px = np.array(raw["display"]["resolution"]).flatten().astype(int)
    w_cm = np.array(raw["display"]["width"]).flatten().astype(int)[0]
    h_cm = np.array(raw["display"]["height"]).flatten().astype(int)[0]
    d_cm = np.array(raw["display"]["dist"]).flatten().astype(int)[0]

    va = visang.VisualAngle(w_px, h_px, w_cm, h_cm, d_cm)

    targets = np.array(raw["targets"]["positions"]).transpose()
    targets[:,0] = np.abs(np.rad2deg(targets[:,0]))
    trial_target = np.array(raw["run_data"]["trial_target"]).flatten().astype(int) - 1

    aim_angle = []
    reach_angle = []
    auc_from_target = []
    auc_from_min = []
    aim_pos = []
    reach_pos = []
    target_angle = []
    traj_xyt_tr = dict()

    for tr_ix in range(trial_n):
        # taking care of MATLAB indexing
        tr_targ = trial_target[tr_ix]
        x = traj_array[0,:,tr_ix][:time_ix[tr_ix]]
        y = -traj_array[1,:,tr_ix][:time_ix[tr_ix]]
        t = traj_array[2,:,tr_ix][:time_ix[tr_ix]]
        
        radius, angle = cart2pol(x, y)
        angle = np.rad2deg(angle)

        target_radius = targets[0,1] * va.degPix()
        aim_radius = 50

        rea_ang = np.nan
        aim_ang = np.nan

        try:
            reach_ix = np.max(np.where(radius <= target_radius)[0])
            aim_ix = np.min(np.where(radius >= 50)[0])
            rea_ang = angle[reach_ix] - targets[tr_targ,0]
            aim_ang = angle[aim_ix] - targets[tr_targ,0]

        except:
            rea_ang = np.nan
            aim_ang = np.nan
        if angle.shape[0] == 0:
            aim_pos.append(np.nan)
            reach_pos.append(np.nan)
        elif angle.shape[0] > 0:
            aim_pos.append(angle[aim_ix])
            reach_pos.append(angle[reach_ix])
        traj_xyt_tr[tr_ix] = {
            "x":list(x), 
            "y":list(y), 
            "t":list(t),
            "angle":list(angle),
            "radius":list(radius)
        }
        aim_angle.append(aim_ang)
        reach_angle.append(rea_ang)
        target_angle.append(targets[tr_targ,0])

    data = {
        "subject_id": np.full(trial_n, id),
        "group": np.full(trial_n, np.array(raw["run_params"]["group"]).flatten()[0], dtype=int),
        "block": np.full(trial_n, np.array(raw["run_params"]["block"]).flatten()[0], dtype=int),
        "trial_in_block": np.arange(trial_n),
        "trial_coherence": np.array(raw["run_data"]["trial_coherence"]).flatten(),
        "trial_perturb": np.array(raw["run_data"]["trial_perturb"]).flatten(),
        "trial_type": np.full(trial_n, np.array(raw["run_params"]["trial_type"]).flatten()[0], dtype=int),
        "reach_dur": np.array(raw["run_data"]["reach_dur"]).flatten(),
        "reach_rt": np.array(raw["run_data"]["reach_rt"]).flatten(),
        "trial_directions": np.array(raw["run_data"]["trial_directions"]).flatten().astype(int),
        "trial_target": trial_target,
        "aim_target": np.array(aim_angle),
        "reach_target": np.array(reach_angle),
        "aim_real_angle": np.array(aim_pos),
        "reach_real_angle": np.array(reach_pos),
        "true_target_angle": np.array(target_angle)
    }

    data = pd.DataFrame.from_dict(data)

    coherences = data.trial_coherence.unique()
    coherences.sort()
    categories = ["zero", "low", "med", "high"]
    coh_cat = {i[0]: i[1] for i in zip(coherences, categories)}

    def cat_func(row, dictionary):
        return dictionary[row]

    data["coh_cat"] = data.trial_coherence.apply(lambda x: cat_func(x, coh_cat))

    data["perturb_cat"] = np.rad2deg(data.trial_perturb)

    data_path = op.join(
        sub_path, "{}-{}-beh.csv".format(subject_id, numero)
    )

    data.to_csv(data_path, index=False)

    xyz_path = op.join(
        sub_path, "{}-{}-beh-traj.json".format(subject_id, numero)
    )

    dump_the_dict(xyz_path, traj_xyt_tr)

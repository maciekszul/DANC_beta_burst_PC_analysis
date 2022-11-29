import json
import math
import copy
import sys
import warnings
from os import sep
import os.path as op
import numpy as np
import time
from fooof import FOOOF
from extra.tools import extract_bursts, many_is_in
from mne import read_epochs
from utilities import files


warnings.filterwarnings("ignore")
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

start_time = time.time()

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

#setting the paths and extracting files
slt_mot_paths  = [i for i in files.get_folders_files(sub_path)[0] if "motor" in i]
slt_vis_paths = [i for i in files.get_folders_files(sub_path)[0] if "visual" in i]
epo_mot_paths  = files.get_files(sub_path, "sub", "motor-epo.fif")[2]
epo_vis_paths = files.get_files(sub_path, "sub", "visual-epo.fif")[2]
beh_match_path = files.get_files(sub_path, "sub", "beh-match.json")[2][0]
with open(beh_match_path) as f:
    beh_match = json.load(f)
slt_mot_paths.sort()
slt_vis_paths.sort()
epo_mot_paths.sort()
epo_vis_paths.sort()
epo_slt_mot_vis = list(zip(epo_mot_paths, epo_vis_paths, slt_mot_paths, slt_vis_paths))

info = read_epochs(epo_mot_paths[0], verbose=False)
info.pick_types(meg=True, ref_meg=False, misc=False)
info = info.info
sfreq = info["sfreq"]
freqs = np.linspace(1,120, num=400)
search_range = np.where((freqs >= 10) & (freqs <= 33))[0]
beta_lims = [13, 30]

vis_burst_block = {}
mot_burst_block = {}

sens = ["MLC1", "MLC25", "MLC32", "MLC42", "MLC54", "MLC55", "MLC63"]
channels_used = [i for i in info.ch_names if many_is_in(["MLC"], i)]
channels_used = [i for i in channels_used if not many_is_in(sens, i)]
# channels_used = info.ch_names

burst_path = op.join(subject, "bursts")
if op.exists(burst_path):
    print(subject_id, "burst folder exists, abandon operation")
    sys.exit()

for block, (epo_mot_p, epo_vis_p, slt_mot_p, slt_vis_p) in enumerate(epo_slt_mot_vis):
    vis_burst_block[block] = {i:{} for i in channels_used}
    mot_burst_block[block] = {i:{} for i in channels_used}
    fooof_THR = {i:[] for i in channels_used}
    vis_PSD = {i:[] for i in channels_used}
    mot_TF = {i:[] for i in channels_used}
    vis_TF = {i:[] for i in channels_used}
    beh_match_vis = beh_match[epo_vis_p.split(sep)[-1]]
    beh_match_mot = beh_match[epo_mot_p.split(sep)[-1]]
    slt_vis_nps = files.get_files(slt_vis_p, "", ".npy")[2]
    slt_vis_nps.sort()
    slt_mot_nps = files.get_files(slt_mot_p, "", ".npy")[2]
    slt_mot_nps.sort()
    epo_vis = read_epochs(epo_vis_p, verbose=False)
    epo_vis_times = epo_vis.times
    epo_vis = epo_vis.pick_types(meg=True, ref_meg=False, misc=False)
    epo_vis = epo_vis.get_data()
    epo_mot = read_epochs(epo_mot_p, verbose=False)
    epo_mot_times = epo_mot.times
    epo_mot = epo_mot.pick_types(meg=True, ref_meg=False, misc=False)
    epo_mot = epo_mot.get_data()
    
    for vis_p in slt_vis_nps:
        data = np.load(vis_p)
        # print(vis_p.split(sep)[-1])
        for ch_ix, channel in enumerate(channels_used):
            TF = data[ch_ix, search_range, :]
            psd = np.mean(data[ch_ix, :, :], axis=1)
            vis_PSD[channel].append(psd)
            vis_TF[channel].append(TF)
        del(data)
    print(subject_id, block, "vis load in ", (time.time() - start_time)/60, "minutes")
    
    for ch_ix, channel in enumerate(channels_used):
        vis_psd_avg = np.mean(np.vstack(vis_PSD[channel]), axis=0)
        ff_vis = FOOOF()
        ff_vis.fit(freqs, vis_psd_avg, [1, 120])
        if not ff_vis.has_model:
            fooof_THR[channel] = fooof_THR[channels_used[ch_ix-1]]
        else:
            ap_fit_v = 10 ** ff_vis._ap_fit
            fooof_THR[channel] = ap_fit_v[search_range].reshape(-1, 1)
    
    for mot_p in slt_mot_nps:
        data = np.load(mot_p)
        # print(mot_p.split(sep)[-1])
        for ch_ix, channel in enumerate(channels_used):
            TF = data[ch_ix, search_range, :]
            psd = np.mean(data[ch_ix, :, :], axis=1)
            mot_TF[channel].append(TF)
        del(data)
    print(subject_id, block, "mot load in ", (time.time() - start_time)/60, "minutes")

    for ch_ix, channel in enumerate(channels_used):
        block_vis_burst = extract_bursts(
            epo_vis[:,ch_ix,:], 
            vis_TF[channel],
            epo_vis_times, 
            freqs[search_range], 
            beta_lims, 
            fooof_THR[channel], 
            sfreq,
            beh_ix=beh_match_vis,
            w_size=0.26
        )
        vis_burst_block[block][channel] = block_vis_burst

        block_mot_burst = extract_bursts(
                epo_mot[:,ch_ix,:], 
                mot_TF[channel],
                epo_mot_times, 
                freqs[search_range], 
                beta_lims, 
                fooof_THR[channel], 
                sfreq,
                beh_ix=beh_match_mot,
                w_size=0.26
            )
        
        mot_burst_block[block][channel] = block_mot_burst
        print(subject_id, block, ch_ix+1, "/274")

print(subject_id, "done with bursts in ", (time.time() - start_time)/60, "minutes")
vis_results = {i: {j: [] for j in vis_burst_block[0][channels_used[0]].keys()} for i in channels_used}
mot_results = {i: {j: [] for j in vis_burst_block[0][channels_used[0]].keys()} for i in channels_used}

for ch_ix, i in enumerate(channels_used):
    try:
        vis_results[i]["block"] = []
        vis_results[i]["pp_ix"] = []
        for bl in vis_burst_block.keys():
            for key in vis_burst_block[bl][i]:
                if key == "waveform":
                    vis_results[i][key].extend(vis_burst_block[bl][i][key].astype(float).tolist())
                elif key == "trial":
                    vis_results[i][key].extend(vis_burst_block[bl][i][key].astype(int).tolist())
                elif key == "polarity":
                    vis_results[i][key].extend(vis_burst_block[bl][i][key].astype(int).tolist())
                else:
                    vis_results[i][key].extend(vis_burst_block[bl][i][key])
            vis_results[i]["block"].extend(np.tile(bl, len(vis_burst_block[bl][i]["trial"])).astype(int).tolist())
            vis_results[i]["pp_ix"].extend((bl*56 + np.array(vis_burst_block[bl][i]["trial"])).astype(int).tolist())
        
        vis_json_name = "{}-{}-visual-burst-iter.json".format(i, subject_id)
        vis_json_path = op.join(subject, "bursts")
        files.make_folder(vis_json_path)
        vis_json_path = op.join(vis_json_path, vis_json_name)
        with open(vis_json_path, "w") as fp:
            json.dump(vis_results[i], fp, indent=4)
        
        print(
            "{}|{}|{}|saved in {} min".format(
                subject_id,
                str(ch_ix+1).zfill(3),
                vis_json_path.split(sep)[-1],
                (time.time() - start_time)/60
            )
        )


        mot_results[i]["block"] = []
        mot_results[i]["pp_ix"] = []
        for bl in mot_burst_block.keys():
            for key in mot_burst_block[bl][i]:
                if key == "waveform":
                    mot_results[i][key].extend(mot_burst_block[bl][i][key].astype(float).tolist())
                elif key == "trial":
                    mot_results[i][key].extend(mot_burst_block[bl][i][key].astype(int).tolist())
                elif key == "polarity":
                    mot_results[i][key].extend(mot_burst_block[bl][i][key].astype(int).tolist())
                else:
                    mot_results[i][key].extend(mot_burst_block[bl][i][key])
            mot_results[i]["block"].extend(np.tile(bl, len(mot_burst_block[bl][i]["trial"])).astype(int).tolist())
            mot_results[i]["pp_ix"].extend((bl*56 + np.array(mot_burst_block[bl][i]["trial"])).astype(int).tolist())

        mot_json_name = "{}-{}-motor-burst-iter.json".format(i, subject_id)
        mot_json_path = op.join(subject, "bursts")
        files.make_folder(mot_json_path)
        mot_json_path = op.join(mot_json_path, mot_json_name)
        with open(mot_json_path, "w") as fp:
            json.dump(mot_results[i], fp, indent=4)
        
        print(
            "{}|{}|{}|saved in {} min".format(
                subject_id,
                str(ch_ix+1).zfill(3),
                mot_json_path.split(sep)[-1],
                (time.time() - start_time)/60
            )
        )
    except:
        print("NOT SAVED",subject_id, i)

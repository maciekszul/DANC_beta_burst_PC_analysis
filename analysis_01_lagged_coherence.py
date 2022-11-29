import sys
from utilities import files
from mne import read_epochs
import matlab.engine
import numpy as np
from scipy.io import savemat
import os.path as op

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
lc_path = op.join(der_path, "lagged_coherence")
files.make_folder(lc_path)


subjects = files.get_folders_files(proc_path)[0]
subjects.sort()
subject = subjects[index]
subject_id = subject.split(sep)[-1]

sub_files = files.get_files(subject, "sub-", "-epo.fif")[2]
sub_files.sort()

sensors = [
    'MLC21', 'MLC22', 'MLC23', 'MLC24', 
    'MLC31', 'MLC41', 'MLC51', 'MLC52', 
    'MLC53', 'MLC61', 'MLC62'
]

vis_out = op.join(
    lc_path,
    "{}_vis_lagged_coherence.mat".format(sub)
)

mot_out = op.join(
    lc_path,
    "{}_mot_lagged_coherence.mat".format(sub)
)
epochs_list = {"mot": [], "vis": []}

if all([op.exists(vis_out), op.exists(mot_out)]):
    sys.exit()
else:
    for sub_file in sub_files:
        file_type = None
        pref_out = None
        if "motor-epo" in sub_file:
            file_type = "mot"
            pref_out = mot_out
        elif "visual-epo" in sub_file:
            file_type = "vis"
            pref_out = vis_out
        
        if not op.exists(pref_out):
            epochs = read_epochs(sub_file, verbose=False)
            sfreq = epochs.info["sfreq"]
            epochs = epochs.pick_channels(sensors).get_data()
            epochs = np.moveaxis(epochs, [0,1,2], [2,0,1])
            epochs_list[file_type].append(epochs)
    
    epochs_list = np.concatenate(epochs_list, axis=2)
    
    parasite.run_lagged_coherence(
        matlab.double(epochs_list.tolist()), 
        matlab.double([sfreq]),
        pref_out,
        nargout=0
    )
    
    print(pref_out, "saved")
            
import os
import sys
import json
import os.path as op
import subprocess as sp
from utilities import files
import matlab.engine
import pandas as pd

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


def split_and_eval(x):
    return [eval(i) for i in x.split(",")]


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
print(subject)

raw_meg_dir = op.join(path, "data")
raw_meg_path = op.join(raw_meg_dir, subject_id, "ses-01", "meg")
ds_paths = files.get_folders_files(raw_meg_path)[0]
ds_paths = [i for i in ds_paths if "misc" not in i]
ds_paths.sort()
res4_paths = [files.get_files(i, "", ".res4")[2][0] for i in ds_paths]
res4_paths.sort()

fs_folder = op.join(der_path, "freesurfer", subject_id)
surface_file = op.join(fs_folder, "pial.ds.gii")
mri_file = op.join(fs_folder, "T1_headcast.nii")

src_folder = op.join(der_path, "source")
files.make_folder(src_folder)


#### MODIFY THE FIF SEARCH PATHS ####
fif_paths = files.get_files(subject, "sub", "motor-epo.fif")[2]
fif_paths.sort()

subject_data = files.get_files(raw_meg_dir,"", ".tsv")[2][0]
print(subject_data)

subs = pd.read_csv(subject_data, sep="\t")

subs.lpa = subs.lpa.apply(split_and_eval)
subs.rpa = subs.rpa.apply(split_and_eval)
subs.nas = subs.nas.apply(split_and_eval)

sub_ix = subs.where(subs["Subject ID"] == subject_id.split("-")[1])["Subject ID"].dropna().index[0]
row = subs.iloc[sub_ix]

parasite = matlab.engine.connect_matlab()

fif_res4_paths = list(zip(fif_paths, res4_paths))
for fif, res4 in [fif_res4_paths[0]]:
    print(fif, res4)
    parasite.src_convert_mne_to_spm(res4, fif, 1, nargout=0)

print(surface_file, op.exists(surface_file))
print(mri_file, op.exists(mri_file))
print(row.lpa, row.rpa, row.nas)

spm_converted_paths = files.get_files(subject, "spm_converted", ".mat")[2]
spm_converted_paths.sort()
for spm_path in spm_converted_paths:
    print(spm_path)
    parasite.src_forward_model(
        spm_path, src_folder, surface_file, mri_file,
        matlab.double(row.nas), matlab.double(row.lpa), matlab.double(row.rpa),
        nargout=0
    )

spm_converted_paths = files.get_files(subject, "coreg_spm_converted", ".mat")[2]
spm_converted_paths.sort()
for spm_path in spm_converted_paths:
    print(spm_path)
    filename = spm_path.split("_")[-1].split(".")[0]
    parasite.src_source_reconstruction(
        spm_path, src_folder, filename, nargout=0
    )


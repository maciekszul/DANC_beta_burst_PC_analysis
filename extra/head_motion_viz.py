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


path = parameters["dataset_path"]
sfreq = parameters["sfreq"]
sub_path = op.join(path, "data")
subjects = files.get_folders_files(sub_path)[0]
subject = subjects[index]
subject_id = subject.split("/")[-1]
der_path = op.join(path, "derivatives")
head_path = op.join(der_path, "head_motion", subject_id)

head_csvs = files.get_files(head_path, subject_id, ".csv")[2]
head_csvs.sort()
head_evnts = files.get_files(head_path, subject_id, "-eve.fif")[2]
head_evnts.sort()

h_e = list(zip(head_csvs, head_evnts))

thyme = np.arange(-0.6, 1.2, 0.008)

params = {
    "pitch": (0, 0, "#ffca44"), 
    "roll": (0, 1, "#f344ff"), 
    "yaw": (0, 2, "#445dff"),
    "X": (1, 0, "#ff2600"), 
    "Y": (1, 1, "#8c3fff"), 
    "Z": (1, 2, "#33ff00")
}
order = ["pitch", "roll", "yaw","X", "Y", "Z"]



gs = gridspec.GridSpec(2, 3, wspace=0.2, hspace=0.2)
figure = plt.figure(figsize=(20, 15))

for label in order:
    row, column, colour = params[label]
    ax = figure.add_subplot(gs[row, column], label=label)
    all_data = []
    for hpi_f, ev_f in [h_e[1]]:

        begin = mne.read_events(
        ev_f,
        include=[50]
        )

        hpi = pd.read_csv(
            hpi_f
        )
        for b in begin[:,0]:
            data = hpi[label].iloc[int(b-thyme.size/3):int(b+thyme.size/1.5)]
            data = data - np.median(data)
            all_data.append(data)

            ax.plot(thyme, data, lw=0.5, alpha=0.2, c=colour)
    all_data = np.array(all_data)
    ax.axvline(0, lw=0.5, alpha=0.5, c="#000000")
    ax.plot(thyme, np.median(all_data, axis=0), lw=1.5, alpha=1, c=colour)
    plt.title(label)
    

plt.show(block=False)
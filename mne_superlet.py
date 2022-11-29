from joblib.parallel import delayed
from superlet import superlet, scale_from_period
from mne import time_frequency
from joblib import Parallel, delayed
import numpy as np


def superlets_mne_epochs(epochs, max_freq=120, num=400, n_jobs=1, save_obj=True):
    """
    returns a list of transformed superlets
    """
    def do_superlet(epoch, ix, key, scales, sfreq):
        signal = epoch[ix, :]
        spec = superlet(
            signal,
            samplerate=sfreq,
            scales=scales,
            order_max=40,
            order_min=1,
            c_1=4,
            adaptive=True,
        )
        epoch_dict[key] = np.single(np.abs(spec))
        del spec
    
    foi = np.linspace(1, max_freq, num)
    scales = scale_from_period(1/foi)
    epoch_info = epochs.info
    epochs_list = []

    for epoch in epochs.__iter__():
        epoch_dict = {}
        Parallel(n_jobs=n_jobs, require="sharedmem")(delayed(do_superlet)(epoch, ix, key, scales, epoch_info["sfreq"]) for ix, key in enumerate(epoch_info.ch_names))
        epochs_list.append(np.array([epoch_dict[ch] for ch in epochs.ch_names]))
    
    if save_obj:
        epochs = time_frequency.EpochsTFR(
            epoch_info,
            np.array(epochs_list), 
            epochs.times, 
            foi,
            events=epochs.events,
            comment="Superlet TF"
        )
    else:
        epochs = epochs_list
    return epochs


def superlets_do_single_epoch(epoch, info, max_freq=120, num=400, n_jobs=-1):
    """
    perform a superlet transformation on a
    best to wrap it in:

    for epoch in epochs.__iter__():
        superlets_do_single_epoch(epoch)
    
    """
    def do_superlet(epoch, ix, key, scales, sfreq):
        signal = epoch[ix, :]
        spec = superlet(
            signal,
            samplerate=sfreq,
            scales=scales,
            order_max=40,
            order_min=1,
            c_1=4,
            adaptive=True,
        )
        epoch_dict[key] = np.single(np.abs(spec))
        del spec


    foi = np.linspace(1, max_freq, num)
    scales = scale_from_period(1/foi)

    epoch_dict = {}
    Parallel(n_jobs=n_jobs, require="sharedmem")(delayed(do_superlet)(epoch, ix, key, scales, info["sfreq"]) for ix, key in enumerate(info.ch_names))
    return np.array([epoch_dict[ch] for ch in info.ch_names])
from joblib import Parallel, delayed
import subprocess as sp


def job_to_do(index):
    sp.call([
        "python",
        "/home/mszul/git/DANC_MEG_learning_beta/pipeline_10c_epoch_check.py",
        str(index),
        "settings_local.json"
    ])

Parallel(n_jobs=-1)(delayed(job_to_do)(index) for index in range(39))
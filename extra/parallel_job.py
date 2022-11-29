import subprocess as sp
from joblib import Parallel, delayed


def job(i):
    sp.call([
        "python", 
        "/home/mszul/git/DANC_MEG_learning_beta/pipeline_08_epoch_qc.py",
        str(i),
        "settings_hdd.json"
    ])


Parallel(n_jobs=-1)(delayed(job)(i) for i in range(0,38))
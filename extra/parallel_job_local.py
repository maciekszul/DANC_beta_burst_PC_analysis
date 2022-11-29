import subprocess as sp
from joblib import Parallel, delayed


def job(i):
    sp.call([
        "python", 
        "/home/mszul/git/DANC_MEG_learning_beta/pipeline_10a_burst_extraction.py",
        str(i),
        "settings_local.json"
    ])


Parallel(n_jobs=-1)(delayed(job)(i) for i in range(0,38))
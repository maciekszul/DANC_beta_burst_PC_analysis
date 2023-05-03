from joblib import Parallel, delayed
import subprocess as sp


def job_to_do(index):
    sp.call([
        "python",
        "/home/mszul/git/DANC_beta_burst_PC_analysis/pipeline_10c_burst_extraction_MU.py",
        str(index),
        "settings.json"
    ])

Parallel(n_jobs=10)(delayed(job_to_do)(index) for index in range(39))
import subprocess as sp


for i in range(1,20):
    sp.call([
        "python", 
        "/home/mszul/git/DANC_MEG_learning_beta/pipeline_04_ica_selection.py",
        str(i),
        "settings_hdd.json"
    ])
    print(i)
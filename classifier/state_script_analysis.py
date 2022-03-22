import os
import json

path_analysis = "/Users/josephgmaa/pyuoi/pyuoi/data/features/run_parameters"

sorted_files = os.listdir(path_analysis)
sorted_files.sort()

for file in sorted_files:
    if file >= "20220315-144114.run_parameters.json":
        with open(os.path.join(path_analysis, file), "r") as f:
            res = json.load(f)
            print(res['accuracy'])

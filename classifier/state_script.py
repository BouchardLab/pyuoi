import subprocess
import sys

classifier = "classifier/rat7m_classifier.py"

for seed in range(1, 21):
    print(f"Running seed: {seed}")
    # Call the classifier with a training seed and a model seed up through 20.
    subprocess.call(["python", classifier,
                     "--input_files", "/Users/josephgmaa/pyuoi/pyuoi/data/features/PCs/PCs-mavg-velocity_relative.netcdf",
                     "--column_names",
                    "PCA_mavg_velocity_relative_0", "PCA_mavg_velocity_relative_1", "PCA_mavg_velocity_relative_2" "PCA_mavg_velocity_relative_3", "PCA_mavg_velocity_relative_4",
                     "--training_seed", str(seed)], stdout=sys.stdout, stderr=subprocess.STDOUT)

    subprocess.call(["python", classifier,
                     "--input_files", "/Users/josephgmaa/pyuoi/pyuoi/data/features/PCs/PCs-mavg-velocity_relative.netcdf",
                     "--column_names",
                    "PCA_mavg_velocity_relative_0", "PCA_mavg_velocity_relative_1", "PCA_mavg_velocity_relative_2" "PCA_mavg_velocity_relative_3", "PCA_mavg_velocity_relative_4",
                     "--model_seed", str(seed)], stdout=sys.stdout, stderr=subprocess.STDOUT)

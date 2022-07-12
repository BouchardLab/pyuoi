import os
import subprocess


def main():
    """
    >>> python /Users/josephgmaa/pyuoi/classifier/grid_search.py
    """
    for file in os.listdir("/Users/josephgmaa/pyuoi/pyuoi/data/features/PCs"):
        if file.endswith(".json"):
            file = os.path.join(
                "/Users/josephgmaa/pyuoi/pyuoi/data/features/PCs", file)
            process = subprocess.Popen([
                "python", "classifier/open_predictions.py", "--input_file", file, '--key=accuracy', "--output_graphs", "True"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(subprocess.list2cmdline([
                "python", "classifier/open_predictions.py", "--input_file", file, '--key=accuracy', "--output_graphs"]))
            _, _ = process.communicate()
    print("Finished.")


if __name__ == "__main__":
    main()

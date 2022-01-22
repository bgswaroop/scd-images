from pathlib import Path
from pprint import pprint

if __name__ == '__main__':
    file_paths = sorted(Path('.').glob('*.out'))
    jobs = {}

    for file in file_paths:
        with open(file) as f:
            if [line for line in f if 'FAILED' in line]:

                job_id = str(file)[6:14]
                device_id = str(file)[15:-4]
                if job_id in jobs:
                    jobs[job_id].append(int(device_id))
                else:
                    jobs[job_id] = [int(device_id)]

    for item in jobs:
        jobs[item] = sorted(jobs[item])
    print(jobs)

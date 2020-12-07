import json
from pathlib import Path

fold_id = 5

filename_source = Path(rf'/data/p288722/dresden/test/nat_patches_18_models_128x128_15_Sony_90/fold_{fold_id}.json')
with open(filename_source, 'r') as f:
    source_data = json.load(f)

filename_sony = Path(rf'/data/p288722/dresden/test/nat_patches_sony_models_128x128_90/fold_{fold_id}.json')
with open(filename_sony, 'r') as f:
    sony_data = json.load(f)

for camera in sony_data['file_paths']:
    if camera not in source_data['file_paths']:
        raise ValueError(f'Invalid camera: {camera}')
    else:
        source_data['file_paths'][camera] = sony_data['file_paths'][camera]

with open(filename_source, 'w+') as f:
    json.dump(source_data, f)

"""Discover usable native RUNs instead of assuming consecutive published IDs."""
from pathlib import Path
import re

from .darus_real import _require_h5py, RELEASE25_INITIAL_TIME_GROUP, RELEASE25_PERMEABILITY_DATASET


def discover_reference_runs(dataset_root):
    h5py = _require_h5py()
    root = Path(dataset_root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    records = []
    required = [RELEASE25_PERMEABILITY_DATASET, 'Liquid Pressure [Pa]', 'Material ID', 'Temperature [C]']
    for run in sorted(root.iterdir()):
        if not run.is_dir() or re.fullmatch(r'RUN_[0-9]+', run.name) is None:
            continue
        row = {'reference_id': run.name, 'simulation_output': (run/'pflotran.h5').is_file(),
               'permeability_input': (run/'permeability.h5').is_file(),
               'can_prepare_matched_context': False, 'status': 'missing_data'}
        if row['simulation_output']:
            with h5py.File(run/'pflotran.h5', 'r') as handle:
                group = handle.get(RELEASE25_INITIAL_TIME_GROUP)
                missing = required if group is None else [key for key in required if key not in group]
                row['missing_initial_channels'] = missing
                if not missing:
                    sizes = [group[key].size for key in required]
                    row['native_cell_count'] = int(sizes[0])
                    row['can_prepare_matched_context'] = all(n == 2560*2560 for n in sizes)
                    row['status'] = 'complete_native' if row['can_prepare_matched_context'] else 'different_grid'
                else:
                    row['status'] = 'missing_initial_channels'
        elif row['permeability_input']:
            row['status'] = 'input_only'
        records.append(row)
    return records

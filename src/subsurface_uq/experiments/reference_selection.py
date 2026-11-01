"""Bind a selected permeability source to its matching physical model context."""
from pathlib import Path
import re

import yaml

from ..io import sha256_file


def reference_name(value):
    if not isinstance(value, str) or re.fullmatch(r"RUN_[0-9]+", value) is None:
        raise ValueError("reference ID must be a RUN name such as RUN_2")
    return value


def profile_names(artifact, reference_id):
    reference_name(reference_id)
    if reference_id not in artifact['source_maps']:
        raise ValueError(f"reference {reference_id} is not in this catalog; available: {', '.join(sorted(artifact['source_maps']))}")
    names = [name for name, item in artifact['profiles'].items()
             if item.get('source_map') == reference_id]
    if not names:
        raise ValueError(f"catalog has no profile explicitly bound to source_map={reference_id}")
    return names


def apply_model_reference(release, reference_id):
    """Override the RUN and expand only the documented reference placeholder."""
    result = dict(release)
    if reference_id is not None:
        result['fixed_run_id'] = reference_name(reference_id)
    selected = result.get('fixed_run_id')
    if isinstance(result.get('prepared_pki_dir'), str) and '{reference_id}' in result['prepared_pki_dir']:
        result['prepared_pki_dir'] = result['prepared_pki_dir'].replace('{reference_id}', reference_name(selected))
    return result


def model_context(model_path, asset_root, reference_id, expected_shape=None):
    """Check matching prepared inputs before creating a study or loading CNNs."""
    reference_name(reference_id)
    source = Path(model_path).resolve()
    raw = yaml.safe_load(source.read_text(encoding='utf-8'))
    if not isinstance(raw, dict) or not isinstance(raw.get('release25'), dict):
        raise ValueError("reference selection requires a release25 model configuration")
    release = apply_model_reference(raw['release25'], reference_id)
    value = release.get('prepared_pki_dir')
    if not isinstance(value, str) or not value:
        raise ValueError("declare prepared_pki_dir for the matching RUN")
    directory = Path(value).expanduser()
    if not directory.is_absolute():
        directory = Path(asset_root) / directory
    directory = directory.resolve()
    tensors = [p for p in [directory/'Inputs'/reference_id, directory/'Inputs'/f'{reference_id}.pt'] if p.is_file()]
    info = directory/'info.yaml'
    if not info.is_file() or len(tensors) != 1:
        raise ValueError(f"matching prepared inputs for {reference_id} are missing at {directory}; prepare that RUN, without falling back to another RUN")
    metadata = yaml.safe_load(info.read_text(encoding='utf-8'))
    if not isinstance(metadata,dict):
        raise ValueError('matching prepared metadata must be a mapping')
    if metadata.get('reference_pilot_fixed_run', reference_id) != reference_id:
        raise ValueError("prepared input metadata belongs to a different reference RUN")
    if expected_shape is not None and list(metadata.get('CellsNumber', [])) != list(expected_shape):
        raise ValueError("matching prepared reference has a different grid shape")
    if list(metadata.get('CellsSize', [])[:2]) != [5., 5.]:
        raise ValueError("matching prepared reference must use the native 5 m grid")
    return {'reference_id': reference_id, 'fixed_run_id': reference_id,
            'prepared_pki_dir': str(directory), 'input_tensor': str(tensors[0]),
            'files_sha256': {str(p): sha256_file(p) for p in [source, info, tensors[0]]},
            'qoi_policy': 'use the declared output cells and ROI; no automatic heat-pump-relative relocation'}


def output_directory(root, reference_id):
    root = Path(root).resolve()
    return root if reference_id is None else root/reference_name(reference_id)

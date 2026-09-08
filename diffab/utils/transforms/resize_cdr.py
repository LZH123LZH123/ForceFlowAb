import random
import re
import string
from collections import OrderedDict
from collections.abc import Mapping

import torch

from ..protein import constants
from ._base import register_transform


MIN_CDR_LENGTH = 5
MAX_CDR_LENGTH = 30

CDR_SPECS = {
    'H_CDR1': {
        'short': 'H1',
        'chain_key': 'heavy',
        'seqmap_key': 'heavy_seqmap',
        'sequence_key': 'H1_seq',
        'enum': constants.CDR.H1,
        'number_range': constants.ChothiaCDRRange.H1,
    },
    'H_CDR2': {
        'short': 'H2',
        'chain_key': 'heavy',
        'seqmap_key': 'heavy_seqmap',
        'sequence_key': 'H2_seq',
        'enum': constants.CDR.H2,
        'number_range': constants.ChothiaCDRRange.H2,
    },
    'H_CDR3': {
        'short': 'H3',
        'chain_key': 'heavy',
        'seqmap_key': 'heavy_seqmap',
        'sequence_key': 'H3_seq',
        'enum': constants.CDR.H3,
        'number_range': constants.ChothiaCDRRange.H3,
    },
    'L_CDR1': {
        'short': 'L1',
        'chain_key': 'light',
        'seqmap_key': 'light_seqmap',
        'sequence_key': 'L1_seq',
        'enum': constants.CDR.L1,
        'number_range': constants.ChothiaCDRRange.L1,
    },
    'L_CDR2': {
        'short': 'L2',
        'chain_key': 'light',
        'seqmap_key': 'light_seqmap',
        'sequence_key': 'L2_seq',
        'enum': constants.CDR.L2,
        'number_range': constants.ChothiaCDRRange.L2,
    },
    'L_CDR3': {
        'short': 'L3',
        'chain_key': 'light',
        'seqmap_key': 'light_seqmap',
        'sequence_key': 'L3_seq',
        'enum': constants.CDR.L3,
        'number_range': constants.ChothiaCDRRange.L3,
    },
}

CDR_ALIASES = {}
for _canonical_name, _spec in CDR_SPECS.items():
    _short_name = _spec['short']
    for _alias in (
        _canonical_name,
        _short_name,
        _canonical_name.replace('_', ''),
        'CDR' + _short_name,
    ):
        CDR_ALIASES[_alias.upper().replace('-', '_')] = _canonical_name


def normalize_cdr_name(name):
    key = str(name).strip().upper().replace('-', '_')
    if key not in CDR_ALIASES:
        raise ValueError(
            'Unsupported CDR name {!r}. Expected one of H1, H2, H3, L1, L2, or L3.'.format(name)
        )
    return CDR_ALIASES[key]


def _validate_length(length):
    if isinstance(length, bool):
        raise ValueError('CDR length must be an integer, not a boolean.')
    try:
        length = int(length)
    except (TypeError, ValueError):
        raise ValueError('CDR length must be an integer: {!r}.'.format(length))
    if not MIN_CDR_LENGTH <= length <= MAX_CDR_LENGTH:
        raise ValueError(
            'CDR length must be between {} and {} residues; got {}.'.format(
                MIN_CDR_LENGTH, MAX_CDR_LENGTH, length
            )
        )
    return length


def length_values_from_spec(spec):
    if isinstance(spec, Mapping):
        if 'min' not in spec or 'max' not in spec:
            raise ValueError('CDR length mapping must contain both "min" and "max".')
        start = _validate_length(spec['min'])
        end = _validate_length(spec['max'])
        if start > end:
            raise ValueError('CDR minimum length must not exceed maximum length.')
        return list(range(start, end + 1))

    if isinstance(spec, (list, tuple, set)):
        values = [_validate_length(value) for value in spec]
        if not values:
            raise ValueError('CDR length list must not be empty.')
        return list(OrderedDict.fromkeys(values))

    if isinstance(spec, str):
        text = spec.strip()
        range_match = re.fullmatch(r'(\d+)\s*-\s*(\d+)', text)
        if range_match:
            start = _validate_length(range_match.group(1))
            end = _validate_length(range_match.group(2))
            if start > end:
                raise ValueError('CDR minimum length must not exceed maximum length.')
            return list(range(start, end + 1))
        if re.search(r'[|/]', text):
            values = [
                _validate_length(value.strip())
                for value in re.split(r'[|/]', text)
                if value.strip()
            ]
            if not values:
                raise ValueError('CDR length list must not be empty.')
            return list(OrderedDict.fromkeys(values))
        return [_validate_length(text)]

    return [_validate_length(spec)]


def parse_cdr_lengths_text(text):
    """Parse ``H3:8-16,L3:9`` into a canonical CDR-length mapping."""
    if text is None or not str(text).strip():
        return {}

    parsed = OrderedDict()
    for entry in str(text).split(','):
        entry = entry.strip()
        if not entry:
            continue
        if ':' not in entry:
            raise ValueError(
                'Each CDR length entry must use CDR:SPEC syntax, for example H3:8-16.'
            )
        cdr_name, length_spec = entry.split(':', 1)
        canonical_name = normalize_cdr_name(cdr_name)
        values = length_values_from_spec(length_spec)
        parsed[canonical_name] = values[0] if len(values) == 1 else values
    return parsed


def normalize_cdr_length_config(raw_config, selected_cdrs):
    selected = [normalize_cdr_name(name) for name in selected_cdrs]
    if raw_config is None or raw_config == {} or raw_config == '':
        return {}

    if isinstance(raw_config, str) and ':' in raw_config:
        raw_config = parse_cdr_lengths_text(raw_config)
    elif not isinstance(raw_config, Mapping):
        if len(selected) != 1:
            raise ValueError(
                'A scalar CDR length can only be used when exactly one CDR is selected.'
            )
        raw_config = {selected[0]: raw_config}

    normalized = OrderedDict()
    for name, spec in raw_config.items():
        canonical_name = normalize_cdr_name(name)
        if canonical_name not in selected:
            raise ValueError(
                'CDR length was provided for {}, but the selected CDRs are {}.'.format(
                    canonical_name, ', '.join(selected)
                )
            )
        normalized[canonical_name] = length_values_from_spec(spec)
    return normalized


def _parse_initial_residue(value):
    if isinstance(value, bool):
        raise ValueError('Initial CDR residue must be an amino-acid name, not a boolean.')
    if isinstance(value, str):
        value = value.strip().upper()
    try:
        residue = constants.AA(value)
    except (TypeError, ValueError):
        raise ValueError(
            'Unsupported initial CDR residue {!r}. Use a standard one-letter '
            'or three-letter amino-acid code, such as G or GLY.'.format(value)
        )
    if residue == constants.AA.UNK:
        raise ValueError(
            'UNK/X does not define a residue prior; omit the CDR from '
            'cdr_initial_residues instead.'
        )
    return int(residue)


def _parse_initial_residue_spec(value):
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError('Initial CDR residue sequence must not be empty.')
        return tuple(_parse_initial_residue(item) for item in value)

    if isinstance(value, str):
        text = value.strip().upper()
        try:
            return (_parse_initial_residue(text),)
        except ValueError as single_residue_error:
            if len(text) <= 1:
                raise single_residue_error
            try:
                return tuple(_parse_initial_residue(symbol) for symbol in text)
            except ValueError:
                raise single_residue_error

    return (_parse_initial_residue(value),)


def normalize_cdr_initial_residue_config(raw_config, selected_cdrs):
    """Normalize per-CDR residue priors to canonical names and AA-index tuples."""
    selected = [normalize_cdr_name(name) for name in selected_cdrs]
    if raw_config is None or raw_config == {} or raw_config == '':
        return OrderedDict()

    if not isinstance(raw_config, Mapping):
        if len(selected) != 1:
            raise ValueError(
                'A scalar initial residue can only be used when exactly one CDR is selected.'
            )
        raw_config = {selected[0]: raw_config}

    normalized = OrderedDict()
    for name, residue in raw_config.items():
        canonical_name = normalize_cdr_name(name)
        if canonical_name not in selected:
            raise ValueError(
                'An initial residue was provided for {}, but the selected CDRs are {}.'.format(
                    canonical_name, ', '.join(selected)
                )
            )
        normalized[canonical_name] = _parse_initial_residue_spec(residue)
    return normalized


def native_cdr_lengths(structure, selected_cdrs):
    lengths = OrderedDict()
    for cdr_name in selected_cdrs:
        canonical_name = normalize_cdr_name(cdr_name)
        spec = CDR_SPECS[canonical_name]
        chain_data = structure.get(spec['chain_key'])
        if chain_data is None:
            raise ValueError(
                '{} was selected, but the input structure has no {} chain.'.format(
                    canonical_name, spec['chain_key']
                )
            )
        length = int((chain_data['cdr_flag'] == int(spec['enum'])).sum().item())
        if length == 0:
            raise ValueError('{} was not found in the input structure.'.format(canonical_name))
        lengths[canonical_name] = length
    return lengths


def plan_cdr_length_variants(structure, selected_cdrs, raw_config, num_samples):
    """Return grouped ``(target_lengths, sample_count)`` plans.

    Each requested output independently samples one allowed length for every
    configured CDR. Identical length combinations are grouped so a model batch
    never mixes incompatible patch mappings.
    """
    selected = [normalize_cdr_name(name) for name in selected_cdrs]
    native_lengths = native_cdr_lengths(structure, selected)
    configured = normalize_cdr_length_config(raw_config, selected)
    num_samples = int(num_samples)
    if num_samples < 1:
        raise ValueError('num_samples must be positive.')

    if not configured:
        return [({}, num_samples)], native_lengths

    grouped = OrderedDict()
    for _ in range(num_samples):
        assignment = OrderedDict()
        for cdr_name in selected:
            if cdr_name in configured:
                assignment[cdr_name] = random.choice(configured[cdr_name])
        key = tuple(assignment.items())
        grouped[key] = grouped.get(key, 0) + 1

    plans = [(OrderedDict(key), count) for key, count in grouped.items()]
    return plans, native_lengths


def cdr_length_tag(lengths):
    return '_'.join(
        '{}len{}'.format(CDR_SPECS[name]['short'], length)
        for name, length in lengths.items()
    )


def _cdr_numbering(cdr_name, target_length):
    start, end = CDR_SPECS[cdr_name]['number_range']
    base_numbers = list(range(start, end + 1))
    base_length = len(base_numbers)

    if target_length <= base_length:
        if target_length == 1:
            selected_numbers = [start]
        else:
            selected_numbers = [start]
            selected_numbers.extend(base_numbers[1:-1][:target_length - 2])
            selected_numbers.append(end)
        return selected_numbers, [' '] * len(selected_numbers)

    extra_count = target_length - base_length
    if extra_count > len(string.ascii_uppercase):
        raise ValueError(
            '{} length {} requires more than 26 PDB insertion codes.'.format(
                cdr_name, target_length
            )
        )

    insertion_resseq = end - 1
    numbers = []
    icodes = []
    for number in base_numbers:
        numbers.append(number)
        icodes.append(' ')
        if number == insertion_resseq:
            for insertion_code in string.ascii_uppercase[:extra_count]:
                numbers.append(number)
                icodes.append(insertion_code)
    return numbers, icodes


def _placeholder_backbone(chain_data, first_idx, last_idx, target_length):
    pos = chain_data['pos_heavyatom'].new_zeros(
        (target_length,) + tuple(chain_data['pos_heavyatom'].shape[1:])
    )
    mask = chain_data['mask_heavyatom'].new_zeros(
        (target_length,) + tuple(chain_data['mask_heavyatom'].shape[1:])
    )

    ca_index = int(constants.BBHeavyAtom.CA)
    n_index = int(constants.BBHeavyAtom.N)
    c_index = int(constants.BBHeavyAtom.C)
    if first_idx > 0:
        left_ca = chain_data['pos_heavyatom'][first_idx - 1, ca_index]
    else:
        left_ca = pos.new_tensor([-2.0, 0.0, 0.0])
    if last_idx + 1 < chain_data['aa'].size(0):
        right_ca = chain_data['pos_heavyatom'][last_idx + 1, ca_index]
    else:
        right_ca = pos.new_tensor([2.0, 0.0, 0.0])

    fractions = torch.arange(
        1,
        target_length + 1,
        device=pos.device,
        dtype=pos.dtype,
    ) / float(target_length + 1)
    ca_positions = left_ca[None, :] + fractions[:, None] * (right_ca - left_ca)[None, :]
    pos[:, ca_index] = ca_positions
    pos[:, n_index] = ca_positions + pos.new_tensor([-1.0, 0.0, 0.0])
    pos[:, c_index] = ca_positions + pos.new_tensor([0.0, 1.0, 0.0])
    mask[:, n_index] = True
    mask[:, ca_index] = True
    mask[:, c_index] = True
    return pos, mask


def _replace_tensor_slice(value, first_idx, last_idx, replacement):
    return torch.cat(
        [value[:first_idx], replacement, value[last_idx + 1:]],
        dim=0,
    )


def _replace_list_slice(value, first_idx, last_idx, replacement):
    return list(value[:first_idx]) + list(replacement) + list(value[last_idx + 1:])


@register_transform('resize_cdr')
class ResizeCDR(object):

    def __init__(self, selection, target_length):
        self.selection = normalize_cdr_name(selection)
        self.target_length = _validate_length(target_length)

    def __call__(self, structure):
        spec = CDR_SPECS[self.selection]
        chain_data = structure.get(spec['chain_key'])
        if chain_data is None:
            raise ValueError(
                '{} requires a {} chain, but none was parsed.'.format(
                    self.selection, spec['chain_key']
                )
            )

        cdr_mask = chain_data['cdr_flag'] == int(spec['enum'])
        indices = torch.arange(cdr_mask.size(0), device=cdr_mask.device)[cdr_mask]
        if indices.numel() == 0:
            raise ValueError('{} was not found in the parsed structure.'.format(self.selection))
        first_idx = int(indices.min().item())
        last_idx = int(indices.max().item())
        if int(cdr_mask[first_idx:last_idx + 1].sum().item()) != last_idx - first_idx + 1:
            raise ValueError('{} residues are not contiguous.'.format(self.selection))

        native_length = last_idx - first_idx + 1
        if native_length == self.target_length:
            return structure

        old_chain_length = int(chain_data['aa'].size(0))
        target_length = self.target_length
        resseq_values, icode_values = _cdr_numbering(self.selection, target_length)
        pos_placeholder, mask_placeholder = _placeholder_backbone(
            chain_data, first_idx, last_idx, target_length
        )

        replacement_tensors = {
            'aa': chain_data['aa'].new_full((target_length,), int(constants.AA.UNK)),
            'resseq': chain_data['resseq'].new_tensor(resseq_values),
            'cdr_flag': chain_data['cdr_flag'].new_full(
                (target_length,), int(spec['enum'])
            ),
            'pos_heavyatom': pos_placeholder,
            'mask_heavyatom': mask_placeholder,
        }
        replacement_lists = {
            'chain_id': [chain_data['chain_id'][first_idx]] * target_length,
            'icode': icode_values,
        }

        handled_keys = set(replacement_tensors) | set(replacement_lists) | {'res_nb'}
        for key, replacement in replacement_tensors.items():
            chain_data[key] = _replace_tensor_slice(
                chain_data[key], first_idx, last_idx, replacement
            )
        for key, replacement in replacement_lists.items():
            chain_data[key] = _replace_list_slice(
                chain_data[key], first_idx, last_idx, replacement
            )

        for key, value in list(chain_data.items()):
            if key in handled_keys:
                continue
            if isinstance(value, torch.Tensor) and value.dim() > 0 and value.size(0) == old_chain_length:
                replacement_shape = (target_length,) + tuple(value.shape[1:])
                if key == 'fragment_type':
                    fill_value = value[first_idx].item()
                    replacement = value.new_full(replacement_shape, fill_value)
                else:
                    replacement = value.new_zeros(replacement_shape)
                chain_data[key] = _replace_tensor_slice(
                    value, first_idx, last_idx, replacement
                )
            elif isinstance(value, list) and len(value) == old_chain_length:
                chain_data[key] = _replace_list_slice(
                    value, first_idx, last_idx, [None] * target_length
                )

        new_chain_length = old_chain_length - native_length + target_length
        chain_data['res_nb'] = torch.arange(
            1,
            new_chain_length + 1,
            dtype=chain_data['res_nb'].dtype,
            device=chain_data['res_nb'].device,
        )
        chain_data[spec['sequence_key']] = 'X' * target_length

        seqmap = {}
        for idx, (chain_id, resseq, icode) in enumerate(zip(
            chain_data['chain_id'],
            chain_data['resseq'].tolist(),
            chain_data['icode'],
        )):
            residue_id = (chain_id, int(resseq), icode)
            if residue_id in seqmap:
                raise ValueError(
                    'Duplicate residue identifier generated while resizing {}: {!r}.'.format(
                        self.selection, residue_id
                    )
                )
            seqmap[residue_id] = idx
        structure[spec['seqmap_key']] = seqmap
        return structure

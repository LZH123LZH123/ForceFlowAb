import os
import argparse
import copy
import csv
import json
import math
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

from diffab.datasets.custom import preprocess_antibody_structure
from diffab.models import get_model
from diffab.modules.common.geometry import reconstruct_backbone_partially
from diffab.modules.common.so3 import so3vec_to_rotation
from diffab.utils.inference import RemoveNative
from diffab.utils.protein.constants import AA, CDR, ressymb_to_resindex
from diffab.utils.protein.writers import save_pdb
from diffab.utils.train import recursive_to
from diffab.utils.misc import *
from diffab.utils.data import *
from diffab.utils.transforms import *
from diffab.utils.inference import *
from diffab.tools.renumber import renumber as renumber_antibody

# torch.cuda.set_device(4)

MOE_OUTPUT_AA_FIELDS = [
    'variant_tag', 'variant_name', 'batch_id',
    'layer_name', 'layer_role', 'layer_idx', 'expert_idx',
    'aa_idx', 'aa_name', 'token_weight_sum', 'output_prob_sum',
    'expert_output_total', 'output_prob_frac',
]

MOE_ROUTING_FIELDS = [
    'variant_tag', 'variant_name', 'batch_id',
    'layer_name', 'layer_role', 'layer_idx', 'expert_idx',
    'cdr_token_count', 'hard_count', 'soft_count',
    'hydropathy_sum', 'charge_sum', 'hydropathy_mean', 'charge_mean',
]


def get_moe_transformer(module):
    rectflow_module = getattr(module, 'rectflow_seq_only', None)
    pred_net = getattr(rectflow_module, 'pred_net', None)
    return getattr(pred_net, 'abtransformer', None)


def reset_moe_output_aa_stats(model):
    transformer = get_moe_transformer(model)
    if transformer is not None and hasattr(transformer, 'reset_routing_output_aa_stats'):
        transformer.reset_routing_output_aa_stats()


def reset_moe_routing_stats(model):
    transformer = get_moe_transformer(model)
    if transformer is not None and hasattr(transformer, 'reset_routing_stats'):
        transformer.reset_routing_stats()


def get_energy_guidance_options(config):
    guidance_cfg = config.sampling.get('energy_guidance', {}) or {}
    if 'start_step' in guidance_cfg or 'end_step' in guidance_cfg:
        start_step = guidance_cfg.get('start_step', 90)
        end_step = guidance_cfg.get('end_step', 99)
    else:
        steps = int(guidance_cfg.get('steps', 0))
        start_step = max(0, 100 - steps)
        end_step = 99
    return {
        'energy_guidance': bool(guidance_cfg.get('enabled', False)),
        'energy_guidance_start_step': int(start_step),
        'energy_guidance_end_step': int(end_step),
        'energy_guidance_warmup_steps': int(guidance_cfg.get('warmup_steps', 0)),
    }


def get_classifier_free_guidance_options(config):
    guidance_cfg = config.sampling.get('classifier_free_guidance', {}) or {}
    enabled = bool(guidance_cfg.get('enabled', False))
    scale = float(guidance_cfg.get('scale', 1.0))
    if not math.isfinite(scale) or scale < 0.0:
        raise ValueError('sampling.classifier_free_guidance.scale must be finite and non-negative.')
    return {
        'cfg_scale': scale if enabled else 1.0,
    }


def get_energy_guidance_cdrs_for_variant(variant):
    if variant.get('cdrs') is not None:
        return list(variant['cdrs'])
    if variant.get('cdr') is not None:
        return [variant['cdr']]
    return None


def append_moe_output_aa_rows(csv_path, rows):
    if len(rows) == 0:
        return
    file_exists = os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as handle:
        writer_csv = csv.DictWriter(handle, fieldnames=MOE_OUTPUT_AA_FIELDS)
        if (not file_exists) or os.path.getsize(csv_path) == 0:
            writer_csv.writeheader()
        writer_csv.writerows(rows)


def flush_moe_output_aa_csv(model, csv_path, variant, batch_id):
    transformer = get_moe_transformer(model)
    if transformer is None or not hasattr(transformer, 'pop_routing_output_aa_stats'):
        return 0
    rows = transformer.pop_routing_output_aa_stats()
    for row in rows:
        row['variant_tag'] = variant.get('tag', '')
        row['variant_name'] = variant.get('name', '')
        row['batch_id'] = batch_id
    append_moe_output_aa_rows(csv_path, rows)
    return len(rows)


def append_moe_routing_rows(csv_path, rows):
    if len(rows) == 0:
        return
    file_exists = os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as handle:
        writer_csv = csv.DictWriter(handle, fieldnames=MOE_ROUTING_FIELDS)
        if (not file_exists) or os.path.getsize(csv_path) == 0:
            writer_csv.writeheader()
        writer_csv.writerows(rows)


def flush_moe_routing_csv(model, csv_path, variant, batch_id):
    transformer = get_moe_transformer(model)
    if transformer is None or not hasattr(transformer, 'pop_routing_stats'):
        return 0
    rows = transformer.pop_routing_stats()
    for row in rows:
        row['variant_tag'] = variant.get('tag', '')
        row['variant_name'] = variant.get('name', '')
        row['batch_id'] = batch_id
    append_moe_routing_rows(csv_path, rows)
    return len(rows)


def _apply_target_cdr_lengths(structure, target_lengths, native_lengths):
    for cdr_name, target_length in target_lengths.items():
        if int(target_length) == int(native_lengths[cdr_name]):
            continue
        structure = ResizeCDR(
            selection=cdr_name,
            target_length=target_length,
        )(structure)
    return structure


def _length_metadata(selected_cdrs, target_lengths, native_lengths):
    actual_lengths = {
        cdr_name: int(target_lengths.get(cdr_name, native_lengths[cdr_name]))
        for cdr_name in selected_cdrs
    }
    return {
        'native_cdr_lengths': {
            cdr_name: int(native_lengths[cdr_name])
            for cdr_name in selected_cdrs
        },
        'cdr_lengths': actual_lengths,
        'length_deltas': {
            cdr_name: int(actual_lengths[cdr_name] - native_lengths[cdr_name])
            for cdr_name in selected_cdrs
        },
    }


def _variant_tag(base_tag, target_lengths):
    if not target_lengths:
        return base_tag
    return '{}_{}'.format(base_tag, cdr_length_tag(target_lengths))


def _length_config_for_single_cdr(raw_length_config, cdr_name):
    if raw_length_config is None or raw_length_config == {} or raw_length_config == '':
        return None
    if isinstance(raw_length_config, str) and ':' in raw_length_config:
        raw_length_config = parse_cdr_lengths_text(raw_length_config)
    if isinstance(raw_length_config, dict):
        return {
            name: spec
            for name, spec in raw_length_config.items()
            if normalize_cdr_name(name) == normalize_cdr_name(cdr_name)
        }
    return raw_length_config


def _initial_residue_config_for_single_cdr(raw_config, cdr_name):
    if raw_config is None or raw_config == {} or raw_config == '':
        return None
    if isinstance(raw_config, dict):
        return {
            name: residue
            for name, residue in raw_config.items()
            if normalize_cdr_name(name) == normalize_cdr_name(cdr_name)
        }
    return raw_config


def _apply_cdr_initial_residue_prior(data, selected_cdrs, raw_config):
    configured = normalize_cdr_initial_residue_config(raw_config, selected_cdrs)
    if not configured:
        return {}

    data['aa'] = data['aa'].clone()
    data['cdr_init_aa'] = torch.full_like(data['aa'], fill_value=int(AA.UNK))
    metadata = {}
    aa_index_to_symbol = {
        index: symbol
        for symbol, index in ressymb_to_resindex.items()
        if index < int(AA.UNK)
    }
    for cdr_name, residue_indices in configured.items():
        cdr_enum = getattr(CDR, '{}{}'.format(cdr_name[0], cdr_name[-1]))
        cdr_mask = data['cdr_flag'] == int(cdr_enum)
        if not cdr_mask.any().item():
            raise ValueError('{} was not found after CDR masking.'.format(cdr_name))
        cdr_length = int(cdr_mask.sum().item())
        if len(residue_indices) == 1:
            expanded_indices = [int(residue_indices[0])] * cdr_length
            metadata[cdr_name] = AA(int(residue_indices[0])).name
        elif len(residue_indices) == cdr_length:
            expanded_indices = [int(index) for index in residue_indices]
            metadata[cdr_name] = ''.join(
                aa_index_to_symbol[index] for index in expanded_indices
            )
        else:
            raise ValueError(
                '{} initial residue sequence has length {}, but the target CDR '
                'length is {}. Use one residue type to repeat it, or provide an '
                'exactly matching one-letter sequence.'.format(
                    cdr_name,
                    len(residue_indices),
                    cdr_length,
                )
            )
        data['aa'][cdr_mask] = int(AA.UNK)
        data['cdr_init_aa'][cdr_mask] = data['cdr_init_aa'].new_tensor(
            expanded_indices
        )
    return metadata


def _initial_residue_metadata(configured, strength):
    if not configured:
        return {}
    return {
        'cdr_initial_residues': configured,
        'cdr_initial_residue_strength': float(strength),
    }


def create_data_variants(config, structure_factory):
    structure = structure_factory()
    structure_id = structure['id']
    raw_length_config = config.sampling.get('cdr_lengths', None)
    raw_initial_residue_config = config.sampling.get(
        'cdr_initial_residues',
        config.sampling.get('cdr_initial_residue', None),
    )
    initial_residue_strength = float(
        config.sampling.get('cdr_initial_residue_strength', 1.0)
    )
    if (
        not math.isfinite(initial_residue_strength)
        or initial_residue_strength < 0.0
    ):
        raise ValueError(
            'sampling.cdr_initial_residue_strength must be finite and non-negative.'
        )
    num_samples = int(config.sampling.num_samples)

    data_variants = []
    if config.mode == 'single_cdr':
        cdrs = sorted(list(set(find_cdrs(structure)).intersection(config.sampling.cdrs)))
        for cdr_name in cdrs:
            plans, native_lengths = plan_cdr_length_variants(
                structure=structure,
                selected_cdrs=[cdr_name],
                raw_config=_length_config_for_single_cdr(raw_length_config, cdr_name),
                num_samples=num_samples,
            )
            for target_lengths, variant_num_samples in plans:
                transform = Compose([
                    MaskSingleCDR(cdr_name, augmentation=False),
                    MergeChains(),
                ])
                structure_var = _apply_target_cdr_lengths(
                    structure_factory(), target_lengths, native_lengths
                )
                data_var = transform(structure_var)
                initial_residues = _apply_cdr_initial_residue_prior(
                    data_var,
                    [cdr_name],
                    _initial_residue_config_for_single_cdr(
                        raw_initial_residue_config,
                        cdr_name,
                    ),
                )
                residue_first, residue_last = get_residue_first_last(data_var)
                tag = _variant_tag(cdr_name, target_lengths)
                data_variants.append({
                    'data': data_var,
                    'name': '{}-{}'.format(structure_id, tag),
                    'tag': tag,
                    'cdr': cdr_name,
                    'residue_first': residue_first,
                    'residue_last': residue_last,
                    'num_samples': int(variant_num_samples),
                    **_length_metadata([cdr_name], target_lengths, native_lengths),
                    **_initial_residue_metadata(
                        initial_residues,
                        initial_residue_strength,
                    ),
                })
    elif config.mode == 'multiple_cdrs':
        cdrs = sorted(list(set(find_cdrs(structure)).intersection(config.sampling.cdrs)))
        plans, native_lengths = plan_cdr_length_variants(
            structure=structure,
            selected_cdrs=cdrs,
            raw_config=raw_length_config,
            num_samples=num_samples,
        )
        for target_lengths, variant_num_samples in plans:
            transform = Compose([
                MaskMultipleCDRs(selection=cdrs, augmentation=False),
                MergeChains(),
            ])
            structure_var = _apply_target_cdr_lengths(
                structure_factory(), target_lengths, native_lengths
            )
            data_var = transform(structure_var)
            initial_residues = _apply_cdr_initial_residue_prior(
                data_var,
                cdrs,
                raw_initial_residue_config,
            )
            tag = _variant_tag('MultipleCDRs', target_lengths)
            data_variants.append({
                'data': data_var,
                'name': '{}-{}'.format(structure_id, tag),
                'tag': tag,
                'cdrs': cdrs,
                'residue_first': None,
                'residue_last': None,
                'num_samples': int(variant_num_samples),
                **_length_metadata(cdrs, target_lengths, native_lengths),
                **_initial_residue_metadata(
                    initial_residues,
                    initial_residue_strength,
                ),
            })
    elif config.mode == 'full':
        if raw_length_config:
            raise ValueError('sampling.cdr_lengths is not supported when mode is "full".')
        if raw_initial_residue_config:
            raise ValueError(
                'sampling.cdr_initial_residues is not supported when mode is "full".'
            )
        transform = Compose([
            MaskAntibody(),
            MergeChains(),
        ])
        data_var = transform(structure_factory())
        data_variants.append({
            'data': data_var,
            'name': f'{structure_id}-Full',
            'tag': 'Full',
            'residue_first': None,
            'residue_last': None,
            'num_samples': num_samples,
        })
    elif config.mode == 'abopt':
        if raw_length_config:
            raise ValueError('sampling.cdr_lengths is not supported when mode is "abopt".')
        if raw_initial_residue_config:
            raise ValueError(
                'sampling.cdr_initial_residues is not supported when mode is "abopt".'
            )
        cdrs = sorted(list(set(find_cdrs(structure)).intersection(config.sampling.cdrs)))
        for cdr_name in cdrs:
            transform = Compose([
                MaskSingleCDR(cdr_name, augmentation=False),
                MergeChains(),
            ])
            data_var = transform(structure_factory())
            residue_first, residue_last = get_residue_first_last(data_var)
            for opt_step in config.sampling.optimize_steps:
                data_variants.append({
                    'data': data_var,
                    'name': f'{structure_id}-{cdr_name}-O{opt_step}',
                    'tag': f'{cdr_name}-O{opt_step}',
                    'cdr': cdr_name,
                    'opt_step': opt_step,
                    'residue_first': residue_first,
                    'residue_last': residue_last,
                    'num_samples': num_samples,
                })
    else:
        raise ValueError(f'Unknown mode: {config.mode}.')
    return data_variants


def design_for_pdb(args):
    # Load configs
    config, config_name = load_config(args.config)
    cdr_lengths_override = getattr(args, 'cdr_lengths', None)
    if cdr_lengths_override:
        config.sampling.cdr_lengths = parse_cdr_lengths_text(cdr_lengths_override)
    if config.sampling.get('cdr_lengths', None):
        if not config.sampling.get('sample_structure', True):
            raise ValueError('Variable CDR length requires sampling.sample_structure=true.')
        if not config.sampling.get('sample_sequence', True):
            raise ValueError('Variable CDR length requires sampling.sample_sequence=true.')
    if (
        config.sampling.get('cdr_initial_residues', None)
        or config.sampling.get('cdr_initial_residue', None)
    ) and not config.sampling.get('sample_sequence', True):
        raise ValueError('CDR initial residues require sampling.sample_sequence=true.')
    seed_all(args.seed if args.seed is not None else config.sampling.seed)

    # Structure loading
    data_id = os.path.basename(args.pdb_path)
    if args.no_renumber:
        pdb_path = args.pdb_path
    else:
        in_pdb_path = args.pdb_path
        out_pdb_path = os.path.splitext(in_pdb_path)[0] + '_chothia.pdb'
        heavy_chains, light_chains = renumber_antibody(in_pdb_path, out_pdb_path)
        pdb_path = out_pdb_path

        if args.heavy is None and len(heavy_chains) > 0:
            args.heavy = heavy_chains[0]
        if args.light is None and len(light_chains) > 0:
            args.light = light_chains[0]
    if args.heavy is None and args.light is None:
        raise ValueError("Neither heavy chain id (--heavy) or light chain id (--light) is specified.")
    get_structure = lambda: preprocess_antibody_structure({
        'id': data_id,
        'pdb_path': pdb_path,
        'heavy_id': args.heavy,
        # If the input is a nanobody, the light chain will be ignores
        'light_id': args.light,
    })

    # Logging
    structure_ = get_structure()
    structure_id = structure_['id']
    tag_postfix = '_%s' % args.tag if args.tag else ''
    log_dir = get_new_log_dir(
        os.path.join(args.out_root, config_name + tag_postfix), 
        prefix=data_id
    )
    logger = get_logger('sample', log_dir)
    logger.info(f'Data ID: {structure_["id"]}')
    logger.info(f'Results will be saved to {log_dir}')
    data_native = MergeChains()(structure_)
    save_pdb(data_native, os.path.join(log_dir, 'reference.pdb'))

    # Load checkpoint and model
    logger.info('Loading model config and checkpoints: %s' % (config.model.checkpoint))
    ckpt = torch.load(config.model.checkpoint, map_location='cpu')
    cfg_ckpt = ckpt['config']
    model = get_model(cfg_ckpt.model).to(args.device)
    lsd = model.load_state_dict(ckpt['model'],strict=False)
    logger.info(str(lsd))

    # Make in-memory data variants for the requested CDR lengths and priors.
    data_variants = create_data_variants(
        config = config,
        structure_factory = get_structure,
    )

    # Save metadata
    metadata = {
        'identifier': structure_id,
        'index': data_id,
        'config': args.config,
        'items': [{kk: vv for kk, vv in var.items() if kk != 'data'} for var in data_variants],
    }
    with open(os.path.join(log_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    # Start sampling
    collate_fn = PaddingCollate(eight=False)
    inference_tfm = [ PatchAroundAnchor(), ]
    inference_tfm = Compose(inference_tfm)

    for variant in data_variants:
        os.makedirs(os.path.join(log_dir, variant['tag']), exist_ok=True)
        logger.info(f"Start sampling for: {variant['tag']}")
        if variant.get('cdr_lengths'):
            logger.info(
                'CDR lengths: %s | samples: %d',
                variant['cdr_lengths'],
                variant['num_samples'],
            )
        if variant.get('cdr_initial_residues'):
            logger.info(
                'CDR initial residue priors: %s | strength: %.3f',
                variant['cdr_initial_residues'],
                variant['cdr_initial_residue_strength'],
            )

        save_pdb(data_native, os.path.join(log_dir, variant['tag'], 'REF1.pdb'))       # w/  OpenMM minimization
    
        data_cropped = inference_tfm(
            copy.deepcopy(variant['data'])
        )
        data_list_repeat = [data_cropped] * int(
            variant.get('num_samples', config.sampling.num_samples)
        )
        loader = DataLoader(data_list_repeat, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
        batch_id = 0    #lll
        csv_batch_id = 0
        count = 0
        moe_debug = False
        moe_output_aa_csv = os.path.join(log_dir, variant['tag'], 'moe_cdr_output_aa_infer.csv')
        moe_routing_csv = os.path.join(log_dir, variant['tag'], 'moe_cdr_routing_infer.csv')
        moe_trace_csv = os.path.join(log_dir, variant['tag'], 'moe_cdr_routing_trace.csv')
        for batch in tqdm(loader, desc=variant['name'], dynamic_ncols=True):
            torch.set_grad_enabled(False)
            model.eval()
            batch = recursive_to(batch, args.device)
            if moe_debug:
                reset_moe_output_aa_stats(model)
                reset_moe_routing_stats(model)
            if 'abopt' in config.mode:
                # Antibody optimization starting from native
                traj_batch = model.optimize(batch, opt_step=variant['opt_step'], optimize_opt={
                    'pbar': True,
                    'sample_structure': config.sampling.sample_structure,
                    'sample_sequence': config.sampling.sample_sequence,
                })
            else:
                # De novo design    lll
                traj_batch = model.sample(batch, batch_id, data_cropped, data_variant=variant['data'], log_dir=log_dir,
                                          tag=variant["tag"], sample_opt={
                    'pbar': True,
                    'sample_structure': config.sampling.sample_structure,
                    'sample_sequence': config.sampling.sample_sequence,
                    **get_classifier_free_guidance_options(config),
                    'single': config.sampling.get('single', False),
                    'multi': config.sampling.get('multi', True),
                    'scope': config.sampling.scope,
                    **get_energy_guidance_options(config),
                    'energy_guidance_chains': {
                        'heavy': args.heavy,
                        'light': args.light,
                    },
                    'energy_guidance_cdrs': get_energy_guidance_cdrs_for_variant(variant),
                    'sequence_init_strength': variant.get(
                        'cdr_initial_residue_strength',
                        0.0,
                    ),
                    'trace_moe': moe_debug,
                    'trace_moe_step': 99,
                    'trace_moe_csv': moe_trace_csv,
                    'trace_moe_max_tokens': 4096,
                })

            if moe_debug:
                moe_rows = flush_moe_output_aa_csv(model, moe_output_aa_csv, variant, csv_batch_id)
                if moe_rows > 0:
                    logger.info('[moe-output][infer] batch %04d | saved %d rows to %s' % (
                        csv_batch_id, moe_rows, moe_output_aa_csv
                    ))
                routing_rows = flush_moe_routing_csv(model, moe_routing_csv, variant, csv_batch_id)
                if routing_rows > 0:
                    logger.info('[moe-routing][infer] batch %04d | saved %d rows to %s' % (
                        csv_batch_id, routing_rows, moe_routing_csv
                    ))

            aa_new = traj_batch[0][2]   # 0: Last sampling step. 2: Amino acid.
            pos_atom_new, mask_atom_new = reconstruct_backbone_partially(
                pos_ctx = batch['pos_heavyatom'],
                R_new = so3vec_to_rotation(traj_batch[0][0]),
                t_new = traj_batch[0][1],
                aa = aa_new,
                chain_nb = batch['chain_nb'],
                res_nb = batch['res_nb'],
                mask_atoms = batch['mask_heavyatom'],
                mask_recons = batch['generate_flag'],
            )
            aa_new = aa_new.cpu()
            pos_atom_new = pos_atom_new.cpu()
            mask_atom_new = mask_atom_new.cpu()

            for i in range(aa_new.size(0)):
                data_full = variant['data']
                aa = apply_patch_to_tensor(data_full['aa'], aa_new[i], data_cropped['patch_idx'])
                mask_ha = apply_patch_to_tensor(data_full['mask_heavyatom'], mask_atom_new[i], data_cropped['patch_idx'])
                pos_ha  = (
                    apply_patch_to_tensor(
                        data_full['pos_heavyatom'], 
                        pos_atom_new[i] + batch['origin'][i].view(1, 1, 3).cpu(), 
                        data_cropped['patch_idx']
                    )
                )

                save_path = os.path.join(log_dir, variant['tag'], '%04d.pdb' % (count, ))
                save_pdb({
                    'chain_nb': data_full['chain_nb'],
                    'chain_id': data_full['chain_id'],
                    'resseq': data_full['resseq'],
                    'icode': data_full['icode'],
                    # Generated
                    'aa': aa,
                    'mask_heavyatom': mask_ha,
                    'pos_heavyatom': pos_ha,
                }, path=save_path)
                count += 1
            csv_batch_id += 1

        logger.info('Finished.\n')


def args_from_cmdline():
    parser = argparse.ArgumentParser()
    parser.add_argument('pdb_path', type=str)
    parser.add_argument('--heavy', type=str, default=None, help='Chain id of the heavy chain.')
    parser.add_argument('--light', type=str, default=None, help='Chain id of the light chain.')
    parser.add_argument('--no_renumber', action='store_true', default=False)
    parser.add_argument('-c', '--config', type=str, default='./configs/test/codesign_single.yml')
    parser.add_argument('-o', '--out_root', type=str, default='./results')
    parser.add_argument('-t', '--tag', type=str, default='')
    parser.add_argument('-s', '--seed', type=int, default=None)
    parser.add_argument('-d', '--device', type=str, default='cuda')
    parser.add_argument('-b', '--batch_size', type=int, default=16)
    parser.add_argument(
        '--cdr-lengths',
        '--cdr_lengths',
        dest='cdr_lengths',
        type=str,
        default=None,
        help='Override CDR lengths, for example "H3:8-16" or "H3:10|12|14,L3:9".',
    )
    args = parser.parse_args()
    return args


def args_factory(**kwargs):
    default_args = EasyDict(
        heavy = 'H',
        light = 'L',
        no_renumber = False,
        config = './configs/test/codesign_single.yml',
        out_root = './results',
        tag = '',
        seed = None,
        device = 'cuda',
        batch_size = 16,
        cdr_lengths = None,
    )
    default_args.update(kwargs)
    return default_args


if __name__ == '__main__':
    design_for_pdb(args_from_cmdline())

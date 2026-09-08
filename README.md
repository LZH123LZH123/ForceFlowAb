# ForceFlowAb

ForceFlowAb is a research codebase for antibody sequence and structure design with rectified flow. The repository contains training and inference pipelines for single-CDR and multi-CDR design, variable-length CDR generation, classifier-free guidance (CFG), optional mixture-of-experts (MoE) routing, energy-guided sampling, and antibody-antigen docking workflows.


## Features

- Joint antibody sequence and backbone structure generation with rectified flow.
- Single-CDR and multi-CDR design modes.
- Fixed, ranged, or discrete variable-length CDR generation.
- Classifier-free antigen guidance for sequence, position, and orientation sampling.
- Configurable MoE routing with routed and shared experts.
- Two-stage training for joint sequence-structure learning and sequence-focused fine-tuning.
- Optional energy guidance during sampling.
- Design from test sets, user-provided PDB structures, or HDOCK-generated antibody-antigen poses.

## Repository layout

```text
ForceFlowAb/
|-- configs/                 # Training and inference configurations
|-- diffab/                  # Models, datasets, geometry, sampling, and evaluation code
|-- bin/                     # Optional external docking executables
|-- train.py                 # Main training entry point
|-- train_sec.py             # Second-stage training entry point
|-- train_two_stage.sh       # Two-stage training wrapper
|-- design_pdb.py            # Design from a PDB structure
|-- design_testset.py        # Design on a configured test split
|-- design_dock.py           # Docking followed by antibody design
`-- env.yaml                 # Conda environment specification
```

## Installation

The provided environment targets Python 3.8, PyTorch 1.12.1, and CUDA 11.3. Create it with Conda:

```bash
conda env create -f env.yaml
conda activate ForceFlowAb
```

The repository includes `data/sabdab_summary_all.tsv`, a snapshot of the SAbDab index used by the example configurations. Download the corresponding antibody structure files separately, place them under `data/`, and update the dataset paths in the selected YAML configuration. Large structure datasets remain excluded from version control.

## Template-free data flow

The ForceFlowAb training and inference paths do not read or construct a
separate structural template. Dataset items, padded batches, model interfaces,
and sampling runners pass only the target antibody–antigen structure and its
design masks. Existing checkpoints remain compatible because the removed
template path did not own trainable parameters.

## Pretrained Weights

The pretrained checkpoints are hosted on https://huggingface.co/SherrySherry123/ForceFlowAb

## Training

Run a single training stage with a YAML configuration:

```bash
python train.py configs/train/codesign_muti_rectflow_RF.yml
```

Useful options include:

```text
--logdir PATH       Directory for logs and checkpoints (default: ./logs)
--device DEVICE     Training device (default: cuda)
--num_workers N     Data-loader workers (default: 8)
--resume PATH       Resume from a checkpoint
--finetune PATH     Fine-tune from a checkpoint
--debug             Disable persistent experiment logging
```

For the two-stage workflow:

```bash
ACTIVATE_ENV=0 \
STAGE1_CONFIG=./configs/train/codesign_muti_rectflow_RF.yml \
STAGE2_CONFIG=./configs/train/codesign_muti_rectflow_finetune_RF.yml \
bash ./train_two_stage.sh
```

To resume the two-stage workflow:

```bash
RESUME_CKPT=/path/to/checkpoints/checkpoint_best.pt \
RESUME_STAGE=auto \
ACTIVATE_ENV=0 \
bash ./train_two_stage.sh
```

## Inference

Inference behavior is controlled by files under `configs/test/`. Before running inference, set `model.checkpoint` in the selected configuration to a local trained checkpoint.

### Design on a test split

```bash
python design_testset.py 0 \
  --config ./configs/test/moe/codesign_single_H3_0.4.yml \
  --out_root ./results
```

Here, `0` is the zero-based index of the structure in the configured test split.

### Design from a PDB structure

```bash
python design_pdb.py /path/to/antibody_antigen.pdb \
  --heavy H \
  --light L \
  --config ./configs/test/moe/codesign_single_H3_0.4.yml \
  --out_root ./results
```

`--heavy` and `--light` specify the antibody heavy- and light-chain IDs in the input PDB. All remaining chains are treated as antigen chains. By default, the script applies Chothia renumbering and can detect the first heavy and light chains automatically; specifying the IDs explicitly is recommended. When using `--no_renumber`, at least one antibody chain ID must be provided. For a nanobody, provide only `--heavy`.

Run `python design_pdb.py --help` or `python design_testset.py --help` for the complete set of arguments.

### Variable-length CDR design

Use `--cdr-lengths` for exact lengths, inclusive ranges, or discrete choices:

```bash
# Four designs; each output independently samples an H3 length from 8 through 16.
./DP.sh \
  --type antibody \
  --region h3 \
  --pdb /path/to/antibody_antigen.pdb \
  --heavy H \
  --light L \
  --num-samples 4 \
  --cdr-lengths 'H3:8-16'

# Multi-CDR design with discrete H3 choices and a fixed L3 length.
./DP.sh \
  --type antibody \
  --region all \
  --pdb /path/to/antibody_antigen.pdb \
  --heavy H \
  --light L \
  --cdr-lengths 'H3:10|12|14,L3:9'
```

The same option is available on `design_pdb.py`. In YAML, use:

```yaml
sampling:
  sample_structure: true
  sample_sequence: true
  cdr_lengths:
    H_CDR3: 8-16
    L_CDR3: 9
  cdr_initial_residues:
    H_CDR3: GLY
    L_CDR3: SER
  cdr_initial_residue_strength: 1.0
```

Supported CDR names are `H1`, `H2`, `H3`, `L1`, `L2`, and `L3`; supported
lengths are 5–30 residues. Range sampling preserves the requested total
`num_samples`: outputs that happen to select the same length combination are
batched together. These YAML values are used directly by `design_pdb.py` and
are preserved when `DP.sh` builds its per-job configuration; a
`--cdr-lengths` argument overrides only `cdr_lengths`.

`cdr_initial_residues` accepts a standard one-letter or three-letter amino-acid
code, such as `G` or `GLY`, and repeats that type across the selected CDR. It
also accepts an exact one-letter sequence when its length matches the target
CDR, for example `H_CDR3: GYSGYSGY`. Exact sequences should normally be paired
with a fixed `cdr_lengths` value rather than a random range.

This is a soft sampling prior, not a hard final-sequence constraint: generated
sequence noise is shifted by
`cdr_initial_residue_strength × one_hot(residue)`. A value of `1.0` gives a
mild bias; larger values give a stronger and more out-of-distribution bias.
Omit the CDR (or leave the mapping empty) for the original unbiased Gaussian
initialization.

The main generated-region amino-acid tensor remains `UNK`, and generated
sequence/structure features remain masked. Added residues use finite N/CA/C
backbone placeholders with valid backbone masks.

Energy guidance is compatible with resized CDRs. The energy preprocessor uses
the selected heavy/light chain IDs and the actual Chothia residue identifiers
written to each intermediate PDB, including insertion codes. Before applying a
gradient, it verifies that the number of selected CDR C-alpha atoms equals the
per-sample `generate_flag` count. The energy backend runs on the same device as
the sampler. Enable it together with variable lengths, for example:

```bash
./DP.sh \
  --type antibody \
  --region h3 \
  --pdb /path/to/antibody_antigen.pdb \
  --heavy H \
  --light L \
  --cdr-lengths 'H3:8-16' \
  --energy true
```

### Classifier-free guidance

ForceFlowAb supports classifier-free guidance (CFG) for both the standard and
sequence-fine-tuned rectified-flow models. During training, a configurable
fraction of examples has its antigen context removed, allowing the same model
to learn conditional and antigen-unconditional predictions. Enable this in the
training configuration:

```yaml
model:
  classifier_free_guidance:
    enabled: true
    condition_dropout_prob: 0.1

dataset:
  train:
    transform:
      - type: mask_multiple_cdrs
      - type: merge_chains
      - type: patch_around_anchor
      - type: random_remove_antigen
```

`random_remove_antigen` must appear after `merge_chains` and
`patch_around_anchor`. When CFG is enabled, `condition_dropout_prob` controls
how often the antigen residues are removed and must be between `0` and `1`.

At inference time, configure the guidance scale under `sampling`:

```yaml
sampling:
  classifier_free_guidance:
    enabled: true
    scale: 1.5
```

The sampler combines the antigen-unconditional and antigen-conditional vector
fields as `(1 - scale) * unconditional + scale * conditional` for amino-acid,
position, and orientation updates. A scale of `1.0` is ordinary conditional
sampling and avoids the second model evaluation; values above `1.0` strengthen
antigen conditioning, while `0.0` uses the antigen-unconditional prediction.
The scale must be finite and non-negative. Use a checkpoint trained with
`model.classifier_free_guidance.enabled: true` whenever the inference scale is
not `1.0`.

CFG and energy guidance are independent and can be enabled together. CFG
changes the learned sampling vector field, whereas energy guidance adds an
external force/torque update during the configured sampling steps. CFG is also
compatible with fixed or variable CDR lengths.

### Docking-guided design

`design_dock.py` can generate antibody-antigen poses with HDOCK and then run the design pipeline. The bundled `bin/hdock` and `bin/createpl` executables belong to the [HDOCK](http://hdock.phys.hust.edu.cn/) docking suite developed by Professor Sheng-You Huang's group at the School of Physics, Huazhong University of Science and Technology. Alternative executable paths can be supplied with `--hdock_bin` and `--createpl_bin`.

```bash
python design_dock.py \
  --antigen /path/to/antigen.pdb \
  --antibody /path/to/antibody.pdb \
  --heavy H \
  --light L \
  --cdrs H1 H2 H3 L1 L2 L3 \
  --epitope_sites A:991 A:992 \
  --config ./configs/test/moe/codesign_multicdrs_0.4.yml \
  --num_docks 10 \
  --out_root ./results
```

Here, `--heavy H` and `--light L` identify the antibody chains. Antigen chains come from the file passed to `--antigen`; chain IDs used in `--epitope_sites` refer to that antigen PDB (for example, residues 991 and 992 on antigen chain `A`). The default antibody chain IDs are `H` and `L`, but they should be changed when the input PDB uses different IDs.

HDOCK is third-party software and is not installed by `env.yaml`. Users should follow the official HDOCK terms and cite the original work:


## Acknowledgements

The repository structure and several core components build on the [DiffAb](https://github.com/luost26/diffab) antibody-design codebase. Its flow-matching approach to antibody CDR sequence-structure co-design also draws on [FlowDesign](https://github.com/nohandsomewujun/FlowDesign).

The docking workflow uses HDOCK from Professor Sheng-You Huang's group at the School of Physics, Huazhong University of Science and Technology. We thank the HDOCK authors for making their docking tools available to the academic community.

> Luo, S. *et al.* Antigen-Specific Antibody Design and Optimization with Diffusion-Based Generative Models for Protein Structures. *Advances in Neural Information Processing Systems* **35** (2022). [NeurIPS paper](https://papers.neurips.cc/paper_files/paper/2022/hash/3fa7d76a0dc1179f1e98d1bc62403756-Abstract-Conference.html)

> Wu, J. *et al.* FlowDesign: Improved design of antibody CDRs through flow matching and better prior distributions. *Cell Systems* **16**, 101270 (2025). [https://doi.org/10.1016/j.cels.2025.101270](https://doi.org/10.1016/j.cels.2025.101270)

> Yan, Y., Zhang, D., Zhou, P., Li, B. & Huang, S.-Y. HDOCK: a web server for protein-protein and protein-DNA/RNA docking based on a hybrid strategy. *Nucleic Acids Research* **45**, W365-W373 (2017). [https://doi.org/10.1093/nar/gkx407](https://doi.org/10.1093/nar/gkx407)

## License

The original ForceFlowAb source code is released under the [MIT License](LICENSE). Third-party code, bundled HDOCK executables, and SAbDab-derived metadata are not relicensed by this repository and remain subject to their respective licenses and terms of use.

## Citation

If you use ForceFlowAb in academic work, please add the project citation here once the corresponding paper or preprint is available.

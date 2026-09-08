# ForceFlowAb

ForceFlowAb is a rectified-flow framework for antibody sequence and structure
design that integrates mixture-of-experts (MoE) modeling, variable-length CDR
generation, classifier-free guidance (CFG), and energy-guided sampling.

## Features

- Joint antibody CDR sequence and backbone structure generation.
- Single-CDR and multi-CDR design.
- Fixed, ranged, or discrete variable-length CDR generation.
- Classifier-free antigen guidance for sequence, position, and orientation.
- Optional MoE routing and energy-guided sampling.
- Design from antibody-antigen complexes or HDOCK-generated poses.

## Installation

The provided environment targets Python 3.8, PyTorch 1.12.1, and CUDA 11.3.
Create it with Conda:

```bash
conda env create -f env.yaml
conda activate ForceFlowAb
```

The repository includes `data/sabdab_summary_all.tsv`, a snapshot of the SAbDab
index. Download the corresponding antibody structure files separately, place
them under `data/`, and update the dataset paths in the selected YAML
configuration.

### Optional: HDOCK

HDOCK is required only for workflows that dock an antibody template to an
antigen. Download `hdock` and `createpl` from the
[official HDOCK site](http://huanglab.phys.hust.edu.cn/software/hdocklite/) and
place them in `bin/`:

```text
bin/
|-- hdock
`-- createpl
```

## Training

Run the two-stage training workflow with:

```bash
ACTIVATE_ENV=0 \
STAGE1_CONFIG=./configs/train/codesign_muti_rectflow_RF.yml \
STAGE2_CONFIG=./configs/train/codesign_muti_rectflow_finetune_RF.yml \
bash ./train_two_stage.sh
```

To run one stage directly:

```bash
python train.py configs/train/codesign_muti_rectflow_RF.yml
```

## Trained Weights

Pretrained checkpoints are hosted at
[SherrySherry123/ForceFlowAb](https://huggingface.co/SherrySherry123/ForceFlowAb).

## Inference

Inference behavior is controlled by files under `configs/test/`. Before
running inference, set `model.checkpoint` in the selected configuration to a
local checkpoint.

### Antibody-antigen complex

```bash
python design_pdb.py /path/to/antibody_antigen.pdb \
  --heavy H \
  --light L \
  --config ./configs/test/H3.yml \
  --out_root ./results
```

`--heavy` and `--light` specify the antibody heavy- and light-chain IDs in the
input PDB. All remaining chains are treated as antigen chains. For a nanobody,
provide only `--heavy`.

### Variable-length CDR design

Use `--cdr-lengths` to specify an exact length, an inclusive range, or a list of
discrete choices:

```bash
# Generate four designs; independently sample the H3 length from 8 through 16.
./DP.sh \
  --type antibody \
  --region h3 \
  --pdb /path/to/antibody_antigen.pdb \
  --heavy H \
  --light L \
  --num-samples 4 \
  --cdr-lengths 'H3:8-16'

# Design multiple CDRs with discrete H3 choices and a fixed L3 length.
./DP.sh \
  --type antibody \
  --region all \
  --pdb /path/to/antibody_antigen.pdb \
  --heavy H \
  --light L \
  --cdr-lengths 'H3:10|12|14,L3:9'
```

The same option is available on `design_pdb.py`. CDR lengths can also be set in
the sampling configuration:

```yaml
sampling:
  cdrs:
    - H_CDR3
    - L_CDR3
  cdr_lengths:
    H_CDR3: 8-16
    L_CDR3: 9
  num_samples: 4
```

Supported CDR names are `H1`, `H2`, `H3`, `L1`, `L2`, and `L3`; accepted
lengths are 5-30 residues. Range and discrete-choice sampling preserve the
requested total number of samples, grouping identical length combinations into
the same model batch. The generated residues retain valid backbone masks and
Chothia residue numbering, including insertion codes when needed.

Variable-length design can be combined with energy guidance:

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

ForceFlowAb supports classifier-free guidance for both the standard and
sequence-fine-tuned rectified-flow models. During training, a configurable
fraction of examples has its antigen context removed so that one model learns
both antigen-conditional and antigen-unconditional predictions.

Enable CFG in the training configuration:

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

`random_remove_antigen` must run after the chains have been merged and the
anchor patch has been selected. `condition_dropout_prob` controls the fraction
of training examples without antigen context and must be between `0` and `1`.

At inference time, set the CFG scale under `sampling`:

```yaml
sampling:
  classifier_free_guidance:
    enabled: true
    scale: 1.5
```

For sequence, position, and orientation updates, the sampler uses
`(1 - scale) * unconditional + scale * conditional`. A scale of `1.0` performs
ordinary conditional sampling without a second model evaluation; values above
`1.0` strengthen antigen conditioning, while `0.0` uses the
antigen-unconditional prediction. The scale must be finite and non-negative.

Use a checkpoint trained with `model.classifier_free_guidance.enabled: true`
whenever the inference scale differs from `1.0`. CFG is compatible with fixed
or variable CDR lengths and can be enabled together with energy guidance.

## Citation

If you use ForceFlowAb in academic work, please cite the archived software
release at [https://doi.org/10.5281/zenodo.21645253](https://doi.org/10.5281/zenodo.21645253).

## Acknowledgements

ForceFlowAb builds on the
[DiffAb](https://github.com/luost26/diffab) antibody-design codebase and draws
on the flow-matching approach introduced by
[FlowDesign](https://github.com/nohandsomewujun/FlowDesign). The optional
docking workflow uses the third-party HDOCK suite.

## License and third-party software

ForceFlowAb-specific contributions are released under the [MIT License](LICENSE).
Third-party code, bundled HDOCK executables, and SAbDab-derived metadata remain
subject to their respective licenses and terms of use.

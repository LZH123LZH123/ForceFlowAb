#!/usr/bin/env python
"""Extract the six Chothia CDR sequences from an antibody PDB file.

Examples:
    python extract_cdr_sequences.py antibody.pdb --heavy H --light L
    python extract_cdr_sequences.py nanobody.pdb --heavy H --light none
    python extract_cdr_sequences.py antibody.pdb --heavy H --light L --format json
"""

import argparse
import json
import os
import sys

from Bio.PDB import PDBParser

from diffab.utils.protein.constants import (
    AA,
    CDR,
    ChothiaCDRRange,
    ressymb_to_resindex,
)


CDR_NAMES = ("H1", "H2", "H3", "L1", "L2", "L3")
CDR_ENUM_TO_NAME = {
    int(CDR.H1): "H1",
    int(CDR.H2): "H2",
    int(CDR.H3): "H3",
    int(CDR.L1): "L1",
    int(CDR.L2): "L2",
    int(CDR.L3): "L3",
}
RESINDEX_TO_SYMBOL = {
    residue_index: symbol
    for symbol, residue_index in ressymb_to_resindex.items()
}


def normalize_chain_id(chain_id):
    if chain_id is None:
        return None
    chain_id = str(chain_id).strip()
    if chain_id.lower() in {"", "none", "null", "-"}:
        return None
    return chain_id


def residue_to_symbol(residue):
    """Return a one-letter residue symbol, or None for a non-protein residue."""
    residue_name = residue.get_resname().strip().upper()
    try:
        residue_type = AA(residue_name)
    except (TypeError, ValueError):
        # Preserve an unrecognized polymer residue as X when it has a CA atom.
        if residue.id[0] == " " and residue.has_id("CA"):
            return "X"
        return None
    return RESINDEX_TO_SYMBOL.get(int(residue_type), "X")


def extract_chain_cdrs(model, chain_id, chain_type):
    names = ("H1", "H2", "H3") if chain_type == "H" else ("L1", "L2", "L3")
    sequences = {name: [] for name in names}
    if chain_id is None:
        return {name: "" for name in names}
    if chain_id not in model:
        available = [chain.id for chain in model]
        raise ValueError(
            f"Chain {chain_id!r} was not found. Available chains: {available}."
        )

    residues = sorted(
        model[chain_id].get_residues(),
        key=lambda residue: (residue.id[1], residue.id[2]),
    )
    for residue in residues:
        cdr_type = ChothiaCDRRange.to_cdr(chain_type, int(residue.id[1]))
        if cdr_type is None:
            continue
        symbol = residue_to_symbol(residue)
        if symbol is None:
            continue
        sequences[CDR_ENUM_TO_NAME[int(cdr_type)]].append(symbol)

    return {name: "".join(symbols) for name, symbols in sequences.items()}


def extract_cdr_sequences(pdb_path, heavy_chain="H", light_chain="L"):
    heavy_chain = normalize_chain_id(heavy_chain)
    light_chain = normalize_chain_id(light_chain)
    if heavy_chain is None and light_chain is None:
        raise ValueError("At least one antibody chain must be specified.")
    if heavy_chain is not None and heavy_chain == light_chain:
        raise ValueError("Heavy and light chain IDs must be different.")
    if not os.path.isfile(pdb_path):
        raise FileNotFoundError(f"PDB file was not found: {pdb_path}")

    model = PDBParser(QUIET=True).get_structure("antibody", pdb_path)[0]
    sequences = {name: "" for name in CDR_NAMES}
    sequences.update(extract_chain_cdrs(model, heavy_chain, "H"))
    sequences.update(extract_chain_cdrs(model, light_chain, "L"))
    return sequences


def format_sequences(sequences, output_format):
    if output_format == "json":
        return json.dumps(
            {
                name: {
                    "sequence": sequences[name],
                    "length": len(sequences[name]),
                }
                for name in CDR_NAMES
            },
            indent=2,
        )
    if output_format == "fasta":
        return "\n".join(
            f">{name} length={len(sequences[name])}\n{sequences[name] or '-'}"
            for name in CDR_NAMES
        )
    return "\n".join(
        f"{name}: {sequences[name] or '-'} (length={len(sequences[name])})"
        for name in CDR_NAMES
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Extract H1/H2/H3/L1/L2/L3 sequences using Chothia residue "
            "numbering. The input PDB must already use Chothia numbering."
        )
    )
    parser.add_argument("pdb_path", help="Input antibody or antibody-antigen PDB.")
    parser.add_argument("--heavy", default="H", help="Heavy-chain ID (default: H).")
    parser.add_argument(
        "--light",
        default="L",
        help="Light-chain ID (default: L); use 'none' for a nanobody.",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json", "fasta"),
        default="text",
        help="Output format (default: text).",
    )
    parser.add_argument("-o", "--output", default=None, help="Optional output file.")
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        sequences = extract_cdr_sequences(
            args.pdb_path,
            heavy_chain=args.heavy,
            light_chain=args.light,
        )
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    expected_names = []
    if normalize_chain_id(args.heavy) is not None:
        expected_names.extend(("H1", "H2", "H3"))
    if normalize_chain_id(args.light) is not None:
        expected_names.extend(("L1", "L2", "L3"))
    empty = [name for name in expected_names if not sequences[name]]
    if empty:
        print(
            "Warning: empty CDR sequence(s): "
            + ", ".join(empty)
            + ". Check chain IDs and Chothia numbering.",
            file=sys.stderr,
        )

    output = format_sequences(sequences, args.format)
    if args.output:
        with open(args.output, "w") as handle:
            handle.write(output + "\n")
    else:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

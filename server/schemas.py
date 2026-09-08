import re
from typing import Literal, Optional

from pydantic import BaseModel, Field, root_validator, validator


class DesignJobRequest(BaseModel):
    type: Literal["nanobody", "antibody"] = "antibody"
    region: Literal["h3", "all"] = "all"
    pdb: str
    heavy: Optional[str] = None
    light: Optional[str] = None
    batch_size: int = Field(8, ge=1)
    num_samples: int = Field(8, ge=1, le=8)
    cdr_lengths: Optional[str] = Field(
        None,
        description="CDR length specification, e.g. H3:8-16 or H3:10|12|14,L3:9",
    )
    energy: bool = False
    energy_start: int = Field(98, ge=0)
    energy_end: int = Field(99, ge=0)
    energy_warmup: int = Field(0, ge=0)
    device: str = "cuda"
    tag: Optional[str] = None
    no_renumber: bool = False

    @validator("cdr_lengths")
    def validate_cdr_lengths(cls, value):
        if value is None or not value.strip():
            return None

        supported_names = {
            "H1", "H2", "H3", "L1", "L2", "L3",
            "H_CDR1", "H_CDR2", "H_CDR3",
            "L_CDR1", "L_CDR2", "L_CDR3",
        }
        for entry in value.split(","):
            if ":" not in entry:
                raise ValueError("use CDR:SPEC syntax, for example H3:8-16")
            raw_name, raw_spec = entry.split(":", 1)
            name = raw_name.strip().upper().replace("-", "_")
            if name not in supported_names:
                raise ValueError(f"unsupported CDR name: {raw_name}")
            spec = raw_spec.strip()
            if not re.fullmatch(r"\d+(?:\s*-\s*\d+)?|\d+(?:\s*[|/]\s*\d+)+", spec):
                raise ValueError(f"invalid CDR length specification: {raw_spec}")
            lengths = [int(number) for number in re.findall(r"\d+", spec)]
            if any(length < 5 or length > 30 for length in lengths):
                raise ValueError("CDR lengths must be between 5 and 30 residues")
            if "-" in spec and lengths[0] > lengths[1]:
                raise ValueError("CDR minimum length must not exceed maximum length")
        return value.strip()

    @root_validator
    def validate_cdr_length_scope(cls, values):
        value = values.get("cdr_lengths")
        if not value:
            return values

        canonical_names = {
            {
                "H_CDR1": "H1",
                "H_CDR2": "H2",
                "H_CDR3": "H3",
                "L_CDR1": "L1",
                "L_CDR2": "L2",
                "L_CDR3": "L3",
            }.get(name, name)
            for name in (
                entry.split(":", 1)[0].strip().upper().replace("-", "_")
                for entry in value.split(",")
            )
        }
        if values.get("region") == "h3" and canonical_names != {"H3"}:
            raise ValueError('region="h3" only accepts an H3 length specification')
        if values.get("type") == "nanobody" and any(
            name.startswith("L") for name in canonical_names
        ):
            raise ValueError("nanobody design does not accept light-chain CDR lengths")
        return values


class DesignJobSubmitResponse(BaseModel):
    job_id: str
    status: str
    job_dir: str


class DesignJobStatus(BaseModel):
    job_id: str
    status: str
    pid: Optional[int] = None
    created_at: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    updated_at: Optional[str] = None
    return_code: Optional[int] = None
    request_json: Optional[str] = None
    config_yml: Optional[str] = None
    run_log: Optional[str] = None
    results_dir: Optional[str] = None

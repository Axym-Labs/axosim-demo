"""Frozen full-connectome AxoSim mushroom-body conditioning experiment."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from axosim.checkpoint import load_checkpoint

from .neural import CHECKPOINT_SHA256, DEFAULT_CHECKPOINT
from .whole_brain import StreamingCrossingDecoder


DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs/mb_conditioning.json"


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class FlyGraph:
    body_id: np.ndarray
    neuron_type: np.ndarray
    neuron_class: np.ndarray
    sign: np.ndarray
    source: np.ndarray
    target: np.ndarray
    contacts: np.ndarray
    sha256: str

    @classmethod
    def load(cls, path, expected_sha256):
        path = Path(path)
        digest = _sha256(path)
        if digest != expected_sha256:
            raise ValueError(f"Connectome SHA256 mismatch: {digest}")
        data = np.load(path, allow_pickle=False)
        required = {"bodyId", "type", "cls", "sign", "pre", "post", "w"}
        if not required.issubset(data.files):
            raise ValueError(f"Connectome lacks {sorted(required - set(data.files))}")
        graph = cls(
            body_id=data["bodyId"],
            neuron_type=data["type"].astype(str),
            neuron_class=data["cls"].astype(str),
            sign=data["sign"].astype(np.int8),
            source=data["pre"].astype(np.int32),
            target=data["post"].astype(np.int32),
            contacts=data["w"].astype(np.int32),
            sha256=digest,
        )
        graph.validate()
        return graph

    def validate(self):
        n = len(self.body_id)
        if len(np.unique(self.body_id)) != n:
            raise ValueError("bodyId must be unique")
        if not all(len(x) == n for x in (self.neuron_type, self.neuron_class, self.sign)):
            raise ValueError("Neuron arrays differ in length")
        if not (len(self.source) == len(self.target) == len(self.contacts)):
            raise ValueError("Edge arrays differ in length")
        if len(self.source) and (
            self.source.min() < 0
            or self.target.min() < 0
            or self.source.max() >= n
            or self.target.max() >= n
        ):
            raise ValueError("Edge index outside neuron table")
        # The mirrored release intentionally retains zero-weight pair rows.
        if not np.all(self.contacts >= 0):
            raise ValueError("Anatomical contact counts must be nonnegative")
        if not np.all(np.isin(self.sign, (-1, 0, 1))):
            raise ValueError("Fast sign must be -1, 0, or 1")

    @property
    def n_neurons(self):
        return len(self.body_id)

    @property
    def n_edges(self):
        return len(self.source)

    @property
    def signed_contacts(self):
        return self.sign[self.source].astype(np.int32) * self.contacts


def _indices_by_type(graph, names):
    names = tuple(names)
    mask = np.isin(graph.neuron_type, names)
    found = set(graph.neuron_type[mask])
    missing = set(names) - found
    if missing:
        raise ValueError(f"Absent neuron types: {sorted(missing)}")
    return np.flatnonzero(mask).astype(np.int64)


def _dan_projection(graph, dan_type, mbon_indices):
    """Continuous direct-contact DAN projection onto every released MBON."""
    dan = graph.neuron_type == dan_type
    mbon_local = np.full(graph.n_neurons, -1, dtype=np.int32)
    mbon_local[mbon_indices] = np.arange(len(mbon_indices), dtype=np.int32)
    edge_mask = dan[graph.source] & (mbon_local[graph.target] >= 0)
    projection = np.bincount(
        mbon_local[graph.target[edge_mask]],
        weights=graph.contacts[edge_mask],
        minlength=len(mbon_indices),
    ).astype(np.float32)
    if projection.max(initial=0) <= 0:
        raise ValueError(f"No direct {dan_type} to MBON contacts")
    return projection / projection.max()


def _bootstrap_mean_interval(values, *, samples, seed):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    choices = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[choices].mean(axis=1)
    return np.quantile(means, (0.025, 0.975)).tolist()


class AxoSimConditioningBrain:
    """One frozen AxoSim state per neuron with plastic KC-to-MBON efficacy."""

    def __init__(self, graph, contract, *, checkpoint=DEFAULT_CHECKPOINT, device="cuda"):
        checkpoint = Path(checkpoint)
        checkpoint_digest = _sha256(checkpoint)
        if checkpoint_digest != CHECKPOINT_SHA256:
            raise ValueError(f"Checkpoint SHA256 mismatch: {checkpoint_digest}")
        model, metadata = load_checkpoint(checkpoint)
        if model.config.patch_size != 4 or model.config.output_dim != 2:
            raise ValueError("Conditioning requires the AxoSim P4 two-output contract")
        self.device = torch.device(device)
        self.graph = graph
        self.contract = contract
        self.model = model.to(self.device).eval()
        self.checkpoint_metadata = metadata
        self.checkpoint_sha256 = checkpoint_digest
        morphology = int(contract["axosim"]["morphology_index"])
        self.morphology_index = morphology
        self.route_feature = self.model.route_features[morphology].mean(0).to(self.device)
        self.adaptation_cache = self.model.compile_adaptation(
            self.model.behavior_adaptation_bank[morphology][None].to(self.device)
        ).expand(graph.n_neurons, -1)

        self.source = torch.as_tensor(graph.source, dtype=torch.long, device=self.device)
        self.target = torch.as_tensor(graph.target, dtype=torch.long, device=self.device)
        self.weight = torch.as_tensor(
            graph.signed_contacts,
            dtype=torch.float32,
            device=self.device,
        ) * float(contract["axosim"]["network_gain"])
        self.efficacy = torch.ones(graph.n_edges, dtype=torch.float32, device=self.device)

        self.kc = np.flatnonzero(graph.neuron_class == "Kenyon_Cell").astype(np.int64)
        self.mbon = np.flatnonzero(graph.neuron_class == "MBON").astype(np.int64)
        if (len(self.kc), len(self.mbon)) != (4064, 97):
            raise ValueError("Released KC/MBON population counts changed")
        kc_local = np.full(graph.n_neurons, -1, dtype=np.int32)
        kc_local[self.kc] = np.arange(len(self.kc), dtype=np.int32)
        mbon_local = np.full(graph.n_neurons, -1, dtype=np.int32)
        mbon_local[self.mbon] = np.arange(len(self.mbon), dtype=np.int32)
        plastic = (kc_local[graph.source] >= 0) & (mbon_local[graph.target] >= 0)
        self.plastic_edge = np.flatnonzero(plastic).astype(np.int64)
        if len(self.plastic_edge) != 33496:
            raise ValueError("Released KC-to-MBON pair-edge count changed")
        self.plastic_edge_t = torch.as_tensor(
            self.plastic_edge, dtype=torch.long, device=self.device
        )
        # AxoSim parameterization: one multiplicative log-efficacy per released
        # KC-to-MBON pair edge. ``efficacy`` is its routing cache so the full
        # 10.86-million-edge graph does not require an exponential every block.
        self.synaptic_log_efficacy = torch.zeros(
            len(self.plastic_edge), dtype=torch.float32, device=self.device
        )
        self.plastic_pre_local = torch.as_tensor(
            kc_local[graph.source[plastic]], dtype=torch.long, device=self.device
        )
        self.plastic_post_local = torch.as_tensor(
            mbon_local[graph.target[plastic]], dtype=torch.long, device=self.device
        )
        self.kc_t = torch.as_tensor(self.kc, dtype=torch.long, device=self.device)
        self.mbon_t = torch.as_tensor(self.mbon, dtype=torch.long, device=self.device)

        aversive = contract["aversive_dan"]
        appetitive = contract["appetitive_dan"]
        self.dan_types = (aversive, appetitive)
        self.dan_indices = {
            name: _indices_by_type(graph, (name,)) for name in self.dan_types
        }
        self.dan_projection = np.stack(
            [_dan_projection(graph, name, self.mbon) for name in self.dan_types]
        )
        self.dan_projection_t = torch.as_tensor(
            self.dan_projection, dtype=torch.float32, device=self.device
        )

        core_fraction = float(contract["plasticity"]["core_readout_fraction"])
        core_a = set(np.flatnonzero(self.dan_projection[0] >= core_fraction).tolist())
        core_p = set(np.flatnonzero(self.dan_projection[1] >= core_fraction).tolist())
        shared = core_a & core_p
        self.core_a_local = np.asarray(sorted(core_a - shared), dtype=np.int64)
        self.core_p_local = np.asarray(sorted(core_p - shared), dtype=np.int64)
        if (len(self.core_a_local), len(self.core_p_local), len(shared)) != (6, 4, 0):
            raise ValueError("Frozen core MBON definition changed")
        self.core_a = self.mbon[self.core_a_local]
        self.core_p = self.mbon[self.core_p_local]
        self.core_a_t = torch.as_tensor(self.core_a, dtype=torch.long, device=self.device)
        self.core_p_t = torch.as_tensor(self.core_p, dtype=torch.long, device=self.device)

        self.odor_indices = {
            name: _indices_by_type(graph, types)
            for name, types in contract["odors"].items()
        }
        self.decoder = StreamingCrossingDecoder(
            graph.n_neurons,
            threshold=float(contract["axosim"]["spike_score_threshold"]),
            device=self.device,
        )
        self.reset_state(reset_plasticity=True)

    def reset_state(self, *, reset_plasticity_traces=True, reset_plasticity=False):
        self.state = self.model.initial_state(
            self.graph.n_neurons, device=self.device, dtype=torch.float32
        )
        self.recurrent_events = torch.zeros(
            self.graph.n_neurons, 4, dtype=torch.bool, device=self.device
        )
        self.pending_events = torch.zeros_like(self.recurrent_events)
        self.decoder.reset()
        if reset_plasticity_traces or not hasattr(self, "kc_trace"):
            self.kc_trace = torch.zeros(
                len(self.kc), dtype=torch.float32, device=self.device
            )
            self.da_trace = torch.zeros(
                len(self.mbon), dtype=torch.float32, device=self.device
            )
            self.da_baseline = torch.zeros_like(self.da_trace)
        if reset_plasticity:
            self.efficacy.fill_(1.0)
            self.synaptic_log_efficacy.zero_()

    @torch.inference_mode()
    def step(self, clamp_indices, clamp_events, *, learn=False):
        clamp_indices = torch.as_tensor(
            clamp_indices, dtype=torch.long, device=self.device
        )
        clamp_events = torch.as_tensor(
            clamp_events, dtype=torch.bool, device=self.device
        )
        if tuple(clamp_events.shape) != (len(clamp_indices), 4):
            raise ValueError("clamp_events must have shape (clamped neurons, 4)")
        events_to_route = self.recurrent_events
        messages = (
            events_to_route.index_select(0, self.source).to(torch.float32)
            * self.weight[:, None]
            * self.efficacy[:, None]
        )
        current = torch.zeros(
            self.graph.n_neurons, 4, dtype=torch.float32, device=self.device
        ).index_add_(0, self.target, messages)
        forecast, self.state = self.model.step_p4(
            current.unsqueeze(-1) * self.route_feature,
            self.state,
            adaptation_cache=self.adaptation_cache,
        )
        if not torch.isfinite(forecast).all():
            raise RuntimeError("Nonfinite AxoSim forecast")
        emitted = self.decoder(forecast[..., 0])
        if len(clamp_indices):
            emitted.index_copy_(0, clamp_indices, clamp_events)
        self.recurrent_events = self.pending_events
        self.pending_events = emitted
        if learn:
            self._learn(emitted)
        return emitted

    @torch.inference_mode()
    def _learn(self, emitted):
        p = self.contract["plasticity"]
        kc_decay = 1.0 - 1.0 / float(p["kc_trace_ms"])
        da_decay = 1.0 - 1.0 / float(p["da_trace_ms"])
        baseline_rate = 1.0 / float(p["da_baseline_ms"])
        rate = float(p["learn_rate_per_ms"])
        floor = float(p["min_efficacy"])
        for millisecond in range(4):
            events = emitted[:, millisecond].to(torch.float32)
            self.kc_trace.mul_(kc_decay).add_(events.index_select(0, self.kc_t) / float(p["kc_trace_ms"]))
            self.da_trace.mul_(da_decay)
            for dan_row, dan_type in enumerate(self.dan_types):
                indices = torch.as_tensor(
                    self.dan_indices[dan_type], dtype=torch.long, device=self.device
                )
                fraction = events.index_select(0, indices).sum() / len(indices)
                self.da_trace.add_(self.dan_projection_t[dan_row] * fraction / float(p["da_trace_ms"]))
            self.da_baseline.add_((self.da_trace - self.da_baseline) * baseline_rate)
            phasic = torch.clamp(self.da_trace - self.da_baseline, min=0.0)
            eligibility = self.kc_trace.index_select(0, self.plastic_pre_local)
            dopamine = phasic.index_select(0, self.plastic_post_local)
            factor = 1.0 - rate * torch.tanh(
                eligibility * float(p["kc_trace_scale"])
                * dopamine * float(p["da_trace_scale"])
            )
            self.synaptic_log_efficacy.add_(torch.log(factor))
            self.synaptic_log_efficacy.clamp_(min=math.log(floor), max=0.0)
            updated = torch.exp(self.synaptic_log_efficacy)
            self.efficacy.index_copy_(
                0, self.plastic_edge_t, updated
            )

    def plastic_efficacy_summary(self):
        values = self.efficacy.index_select(0, self.plastic_edge_t)
        posts = self.plastic_post_local
        mask_a = torch.isin(
            posts,
            torch.as_tensor(self.core_a_local, dtype=torch.long, device=self.device),
        )
        mask_p = torch.isin(
            posts,
            torch.as_tensor(self.core_p_local, dtype=torch.long, device=self.device),
        )
        return {
            "all": float(values.mean().cpu()),
            "PPL105_core": float(values[mask_a].mean().cpu()),
            "PAM08_core": float(values[mask_p].mean().cpu()),
            "min": float(values.min().cpu()),
        }


def _poisson(rng, count, rate_hz):
    probability = -np.expm1(-float(rate_hz) / 1000.0)
    return rng.random((count, 4)) < probability


def run_presentation(
    brain,
    *,
    odor,
    duration_ms,
    rng,
    dan_rng=None,
    dan_type=None,
    learn=False,
    score_start_ms=0,
    record=False,
):
    brain.reset_state(reset_plasticity_traces=True, reset_plasticity=False)
    odor_indices = brain.odor_indices[odor]
    dan_indices = (
        np.empty(0, dtype=np.int64)
        if dan_type is None
        else brain.dan_indices[dan_type]
    )
    clamp_indices = np.concatenate((odor_indices, dan_indices))
    counts_a = 0
    counts_p = 0
    rows = []
    storm_run = 0
    gate = brain.contract["gate"]
    dan_rng = rng if dan_rng is None else dan_rng
    run_ms = int(math.ceil(duration_ms / 4) * 4)
    for block_start in range(0, run_ms, 4):
        odor_events = _poisson(
            rng, len(odor_indices), brain.contract["odor_rate_hz"]
        )
        dan_events = _poisson(
            dan_rng, len(dan_indices), brain.contract["dan_rate_hz"]
        )
        clamp_events = np.concatenate((odor_events, dan_events), axis=0)
        emitted = brain.step(clamp_indices, clamp_events, learn=learn)
        emitted_np = emitted.cpu().numpy()
        for j in range(4):
            t = block_start + j
            if t >= duration_ms:
                continue
            population_fraction = emitted_np[:, j].sum() / brain.graph.n_neurons
            storm_run = storm_run + 1 if population_fraction > gate["storm_fraction"] else 0
            if storm_run >= gate["storm_consecutive_ms"]:
                raise RuntimeError("Population storm guard triggered")
            a = int(emitted_np[brain.core_a, j].sum())
            p = int(emitted_np[brain.core_p, j].sum())
            if t >= score_start_ms:
                counts_a += a
                counts_p += p
            if record:
                rows.append(
                    (
                        t,
                        int(emitted_np[brain.kc, j].sum()),
                        a,
                        p,
                        int(emitted_np[odor_indices, j].sum()),
                        int(emitted_np[dan_indices, j].sum()) if len(dan_indices) else 0,
                    )
                )
    return {
        "A": counts_a,
        "P": counts_p,
        "trace": np.asarray(rows, dtype=np.int32).reshape(-1, 6),
    }


def _disc(counts):
    denominator = counts["CS+"] + counts["CS-"]
    return 0.0 if denominator == 0 else (counts["CS+"] - counts["CS-"]) / denominator


def measure(brain, seed, *, record=False):
    total_ms = brain.contract["settle_ms"] + brain.contract["test_ms"]
    a_counts = {}
    p_counts = {}
    traces = {}
    for offset, odor in enumerate(("CS+", "CS-")):
        result = run_presentation(
            brain,
            odor=odor,
            duration_ms=total_ms,
            rng=np.random.default_rng(seed + 100_000 * offset),
            score_start_ms=brain.contract["settle_ms"],
            record=record,
        )
        a_counts[odor] = result["A"]
        p_counts[odor] = result["P"]
        if record:
            traces[odor] = result["trace"]
    d_a = _disc(a_counts)
    d_p = _disc(p_counts)
    return {
        "D": d_a - d_p,
        "d_PPL105": d_a,
        "d_PAM08": d_p,
        "counts": {"PPL105": a_counts, "PAM08": p_counts},
        "traces": traces,
    }


def _schedule(contract, arm):
    aversive = contract["aversive_dan"]
    appetitive = contract["appetitive_dan"]
    if arm in ("standard", "no_plasticity"):
        return {"CS+": aversive, "CS-": appetitive}
    if arm == "reversed":
        return {"CS+": appetitive, "CS-": aversive}
    if arm == "dopamine_blocked":
        return {"CS+": None, "CS-": None}
    raise ValueError(f"Unknown arm {arm}")


def run_replicate(brain, *, arm, seed, record=False):
    brain.reset_state(reset_plasticity=True)
    pre = measure(brain, seed + 10_000_000, record=record)
    schedule = _schedule(brain.contract, arm)
    plastic = arm != "no_plasticity"
    history = []
    traces = {}
    for trial in range(brain.contract["trials"]):
        for odor_index, odor in enumerate(("CS+", "CS-")):
            # Keep the odor realization identical across causal arms. DAN draws
            # have a separate stream because PPL105 and PAM08 differ in cell count.
            presentation = trial * 2 + odor_index
            odor_rng = np.random.default_rng(seed * 1000 + presentation)
            dan_rng = np.random.default_rng(50_000_000 + seed * 1000 + presentation)
            result = run_presentation(
                brain,
                odor=odor,
                duration_ms=brain.contract["train_ms"],
                rng=odor_rng,
                dan_rng=dan_rng,
                dan_type=schedule[odor],
                learn=plastic,
                record=record and trial in (0, brain.contract["trials"] - 1),
            )
            efficacy = brain.plastic_efficacy_summary()
            history.append(
                {
                    "trial": trial + 1,
                    "odor": odor,
                    "dan_type": schedule[odor],
                    "counts_PPL105": result["A"],
                    "counts_PAM08": result["P"],
                    "efficacy": efficacy,
                }
            )
            if record and len(result["trace"]):
                traces[f"trial{trial + 1}_{odor}"] = result["trace"]
    post = measure(brain, seed + 10_000_000, record=record)
    return {
        "seed": seed,
        "pre": {k: v for k, v in pre.items() if k != "traces"},
        "post": {k: v for k, v in post.items() if k != "traces"},
        "delta_D": post["D"] - pre["D"],
        "history": history,
        "traces": {
            **{f"pre_{k}": v for k, v in pre["traces"].items()},
            **traces,
            **{f"post_{k}": v for k, v in post["traces"].items()},
        },
        "final_efficacy": brain.plastic_efficacy_summary(),
    }


def _analyse(results, contract):
    arms = {}
    for arm, replicates in results.items():
        values = np.asarray([row["delta_D"] for row in replicates])
        arms[arm] = {
            "delta_D": values.tolist(),
            "mean_delta_D": float(values.mean()),
            "bootstrap_95_ci": _bootstrap_mean_interval(
                values,
                samples=contract["gate"]["bootstrap_samples"],
                seed=contract["gate"]["bootstrap_seed"],
            ),
            "mean_final_efficacy": {
                key: float(np.mean([row["final_efficacy"][key] for row in replicates]))
                for key in ("all", "PPL105_core", "PAM08_core", "min")
            },
        }
    pair_signs = int(
        np.sum(
            (np.asarray(arms["standard"]["delta_D"]) < 0)
            & (np.asarray(arms["reversed"]["delta_D"]) > 0)
        )
    )
    trained = (
        arms["standard"]["mean_delta_D"] < 0
        and arms["standard"]["bootstrap_95_ci"][1] < 0
        and arms["reversed"]["mean_delta_D"] > 0
        and arms["reversed"]["bootstrap_95_ci"][0] > 0
        and pair_signs >= contract["gate"]["trained_min_opposite_sign_pairs"]
    )
    control_limit = contract["gate"]["control_max_abs_mean_delta_D"]
    controls = all(
        abs(arms[name]["mean_delta_D"]) <= control_limit
        and arms[name]["bootstrap_95_ci"][0] <= 0 <= arms[name]["bootstrap_95_ci"][1]
        for name in ("no_plasticity", "dopamine_blocked")
    )
    return {
        "arms": arms,
        "trained_opposite_sign_pairs": pair_signs,
        "trained_gate": trained,
        "control_gate": controls,
        "passed": bool(trained and controls),
    }


def run_experiment(*, config=DEFAULT_CONFIG, output, checkpoint=DEFAULT_CHECKPOINT, device="cuda"):
    config = Path(config)
    contract = json.loads(config.read_text())
    graph = FlyGraph.load(
        contract["source"]["path"], contract["source"]["sha256"]
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    brain = AxoSimConditioningBrain(
        graph, contract, checkpoint=checkpoint, device=device
    )
    if brain.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(brain.device)
    started = time.monotonic()
    partial_path = output / "partial-results.json"
    config_sha256 = _sha256(config)
    if partial_path.exists():
        partial = json.loads(partial_path.read_text())
        if partial.get("config_sha256") != config_sha256:
            raise ValueError("Partial conditioning run uses a different config")
        results = partial["results"]
    else:
        results = {}
    arms = ("standard", "reversed", "no_plasticity", "dopamine_blocked")
    for arm in arms:
        results.setdefault(arm, [])
        for i, seed in enumerate(
            contract["replicate_seeds"][len(results[arm]):], start=len(results[arm])
        ):
            print(f"{arm}: replicate {i + 1}/{len(contract['replicate_seeds'])}", flush=True)
            results[arm].append(
                run_replicate(brain, arm=arm, seed=seed, record=i == 0)
            )
            partial_path.write_text(
                json.dumps(
                    {"config_sha256": config_sha256, "results": results},
                    default=lambda value: value.tolist()
                    if isinstance(value, np.ndarray)
                    else float(value),
                )
            )
    analysis = _analyse(results, contract)
    memory = {}
    if brain.device.type == "cuda":
        memory = {
            "max_allocated_bytes": int(torch.cuda.max_memory_allocated(brain.device)),
            "max_reserved_bytes": int(torch.cuda.max_memory_reserved(brain.device)),
        }
    # Time-series arrays remain in the lossless NPZ rather than being duplicated
    # into the human-readable JSON result.
    report_results = {
        arm: [{key: value for key, value in row.items() if key != "traces"} for row in rows]
        for arm, rows in results.items()
    }
    report = {
        "status": "complete",
        "passed": analysis["passed"],
        "scope": "full-connectome AxoSim neural conditioning; no behavior or motor output",
        "config": str(config),
        "contract": contract,
        "checkpoint_sha256": brain.checkpoint_sha256,
        "checkpoint_model_kind": brain.checkpoint_metadata["model_kind"],
        "checkpoint_training_domain": "mammalian neuronal surrogate; fly use is out of domain",
        "graph": {
            "sha256": graph.sha256,
            "neurons": graph.n_neurons,
            "pair_edges": graph.n_edges,
            "nonzero_pair_edges": int(np.sum(graph.contacts > 0)),
            "released_zero_weight_rows": int(np.sum(graph.contacts == 0)),
            "anatomical_contacts": int(graph.contacts.sum(dtype=np.int64)),
            "fast_zero_pair_edges": int(np.sum(graph.sign[graph.source] == 0)),
            "KC": len(brain.kc),
            "MBON": len(brain.mbon),
            "plastic_pair_edges": len(brain.plastic_edge),
            "plastic_contacts": int(graph.contacts[brain.plastic_edge].sum(dtype=np.int64)),
            "pruned": False,
            "coalesced": False,
        },
        "readout": {
            "PPL105_core_body_ids": graph.body_id[brain.core_a].tolist(),
            "PPL105_core_types": graph.neuron_type[brain.core_a].tolist(),
            "PAM08_core_body_ids": graph.body_id[brain.core_p].tolist(),
            "PAM08_core_types": graph.neuron_type[brain.core_p].tolist(),
        },
        "analysis": analysis,
        "results": report_results,
        "runtime": {
            "device": str(brain.device),
            "wall_seconds": time.monotonic() - started,
            **memory,
        },
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2))
    arrays = {}
    for arm, replicates in results.items():
        arrays[f"{arm}_delta_D"] = np.asarray([x["delta_D"] for x in replicates])
        arrays[f"{arm}_history_efficacy_all"] = np.asarray(
            [[p["efficacy"]["all"] for p in x["history"]] for x in replicates]
        )
        arrays[f"{arm}_history_efficacy_A"] = np.asarray(
            [[p["efficacy"]["PPL105_core"] for p in x["history"]] for x in replicates]
        )
        arrays[f"{arm}_history_efficacy_P"] = np.asarray(
            [[p["efficacy"]["PAM08_core"] for p in x["history"]] for x in replicates]
        )
        for name, trace in replicates[0]["traces"].items():
            arrays[f"{arm}_seed0_{name}"] = trace
    np.savez_compressed(output / "raw-results.npz", **arrays)
    partial_path.unlink(missing_ok=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output", default="data/mb_conditioning")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    report = run_experiment(
        config=args.config,
        output=args.output,
        checkpoint=args.checkpoint,
        device=args.device,
    )
    print(json.dumps({"passed": report["passed"], **report["analysis"]}, indent=2))


if __name__ == "__main__":
    main()

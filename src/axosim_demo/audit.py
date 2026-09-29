"""Reproducible graph download and exact connectivity memory/route audit."""
import argparse
import json
from pathlib import Path
import time
from urllib.request import urlopen
import numpy as np
from axosim_demo.connectome import ExactConnectome, sha256

SHIU_COMMIT = '91bdd1e7dcf193f3e7ca5a8933497fcef63b7960'
ASSETS = {
    'neurons_630.csv': ('2023_03_23_completeness_630_final.csv', 'e6b71e17671a9bdb05f55e4bc6774640a1418cb7a05125e0fc994ad40f9bfdfb'),
    'connectivity_630.parquet': ('2023_03_23_connectivity_630_final.parquet', '94db8c650533bc36ffa3223f2e62325d5648b8d6bd31c3a4e1c804628c7557b3'),
}


def fetch_data(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for filename, (upstream, digest) in ASSETS.items():
        target = directory / filename
        if not target.exists():
            url = f'https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/{SHIU_COMMIT}/{upstream}'
            temporary = target.with_suffix(target.suffix + '.part')
            with urlopen(url, timeout=120) as source, temporary.open('wb') as dest:
                for block in iter(lambda: source.read(1024*1024), b''):
                    dest.write(block)
            if sha256(temporary) != digest:
                temporary.unlink()
                raise ValueError(f'Checksum mismatch for {filename}')
            temporary.replace(target)
        if sha256(target) != digest:
            raise ValueError(f'Checksum mismatch for {filename}; refusing changed data')
    return directory


def audit(directory, device='cpu', repeats=20):
    import torch
    directory = Path(directory)
    graph = ExactConnectome.from_shiu(directory/'neurons_630.csv', directory/'connectivity_630.parquet')
    report = graph.memory_report()
    report['upstream_commit'] = SHIU_COMMIT
    if device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats(device)
    router = graph.torch_router(device)
    inputs = np.random.default_rng(42).integers(0,2,graph.n_neurons).astype(np.float32)
    activity = torch.from_numpy(inputs).to(device)
    router(activity)  # warmup
    if device.startswith('cuda'):
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(repeats):
        result = router(activity)
    if device.startswith('cuda'):
        torch.cuda.synchronize(device)
    report['routing_ms_mean'] = (time.perf_counter()-start)*1000/repeats
    report['routing_device'] = device
    report['routing_repeats'] = repeats
    report['max_abs_error_vs_cpu_reference'] = float(np.abs(result.cpu().numpy()-graph.route(inputs)).max())
    report['all_released_edges_retained'] = router.source.numel() == graph.n_edges
    report['scope_measured'] = 'All-edge signed count routing, FP32 unit efficacy; not closed-loop neural simulation or adaptation.'
    if device.startswith('cuda'):
        report['gpu_peak_allocated_bytes'] = torch.cuda.max_memory_allocated(device)
        report['gpu_peak_reserved_bytes'] = torch.cuda.max_memory_reserved(device)
        report['gpu_name'] = torch.cuda.get_device_name(device)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/connectome'))
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.download:
        fetch_data(args.data)
    report = audit(args.data, args.device)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()

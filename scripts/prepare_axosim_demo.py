"""Download checksum-locked Shiu data/reference; optionally stage a Lite checkpoint.

Run with the demo environment from the repository root. No checkpoint is fetched
from an unverified source: --checkpoint-source must identify an existing file.
"""
import argparse
import json
from pathlib import Path
import shutil
from urllib.request import urlopen
from axosim_demo.audit import fetch_data, SHIU_COMMIT
from axosim_demo.connectome import sha256
from axosim_demo.neural import CHECKPOINT_SHA256

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--checkpoint-source', type=Path)
args = parser.parse_args()
fetch_data('data/connectome')
contract = json.loads(Path('configs/shiu_grooming.json').read_text())
for key, name in [('model','model.py'),('notebook','figures.ipynb')]:
    entry = contract['files'][key]
    path = Path(entry['path'])
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        url = f'https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/{SHIU_COMMIT}/{name}'
        temporary = path.with_suffix(path.suffix+'.part')
        with urlopen(url,timeout=60) as source:
            temporary.write_bytes(source.read())
        if sha256(temporary) != entry['sha256']:
            temporary.unlink()
            raise ValueError(f'Checksum mismatch: {name}')
        temporary.replace(path)
    if sha256(path) != entry['sha256']:
        raise ValueError(f'Checksum mismatch: {path}')
if args.checkpoint_source:
    if sha256(args.checkpoint_source) != CHECKPOINT_SHA256:
        raise ValueError('Checkpoint differs from the evaluated AxoSim-Lite initialization')
    target = Path('data/checkpoints/axosim-lite-t24-s8.pt')
    target.parent.mkdir(parents=True,exist_ok=True)
    if args.checkpoint_source.resolve() != target.resolve():
        shutil.copyfile(args.checkpoint_source,target)
print('Verified original Shiu v630 graph and reference source.' + (' Staged verified checkpoint.' if args.checkpoint_source else ''))

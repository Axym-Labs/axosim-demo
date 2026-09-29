"""Guard scientific provenance and original stimulus identities, not a fake score."""
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_notebook_literals_are_extracted_without_executing_code(tmp_path):
    from axosim_demo.shiu_benchmark import notebook_assignments
    notebook = tmp_path / 'figures.ipynb'
    notebook.write_text(json.dumps({'cells': [{'cell_type': 'code', 'source': [
        'neu_JON_CE = [10, 20]\nneu_JON_F = [30]\nraise RuntimeError("never execute notebook")\n'
    ]}]}))
    assert notebook_assignments(notebook, ['neu_JON_CE', 'neu_JON_F']) == {
        'neu_JON_CE': [10, 20], 'neu_JON_F': [30]}


def test_missing_or_nonliteral_population_fails(tmp_path):
    from axosim_demo.shiu_benchmark import notebook_assignments
    notebook = tmp_path / 'figures.ipynb'
    notebook.write_text(json.dumps({'cells': [{'cell_type': 'code', 'source': ['x = arbitrary()']}]}))
    with pytest.raises(ValueError, match='literal'):
        notebook_assignments(notebook, ['x'])


def test_source_integrity_rejects_modified_code(tmp_path):
    from axosim_demo.shiu_benchmark import verify_file
    path = tmp_path / 'model.py'
    path.write_text('changed upstream dynamics')
    with pytest.raises(ValueError, match='SHA256'):
        verify_file(path, '0' * 64)


def test_upstream_stimulation_sets_and_readout_ids_match_pinned_config():
    from axosim_demo.shiu_benchmark import notebook_assignments
    path = ROOT / 'data/reference/shiu_figures.ipynb'
    if not path.exists():
        pytest.skip('Download original reference notebook to run integration identity check')
    contract = json.loads((ROOT / 'configs/shiu_grooming.json').read_text())
    wanted = {c['notebook_variable']: c['root_ids'] for c in contract['conditions'].values() if c['notebook_variable']}
    wanted.update({r['notebook_variable']: r['root_id'] for r in contract['readouts'].values()})
    assert notebook_assignments(path, wanted) == wanted

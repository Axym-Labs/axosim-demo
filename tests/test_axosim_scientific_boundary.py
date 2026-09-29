"""Prevent the retired motor controller from becoming an implicit demo fallback."""
import pytest
from axosim_demo.__main__ import main


def test_unimplemented_scientific_fly_demo_fails_explicitly(monkeypatch,capsys):
    monkeypatch.setattr('sys.argv',['axosim_demo','fly-demo'])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code==2
    message=capsys.readouterr().err
    assert 'not implemented' in message
    assert 'not accepted substitutes' in message


def test_old_recording_flags_do_not_fall_back_to_engineered_control(monkeypatch):
    monkeypatch.setattr('sys.argv',['axosim_demo','--backend','axosim','--output','never-created'])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code==2


def test_command_help_reaches_the_selected_scientific_module(monkeypatch):
    import sys
    from types import SimpleNamespace

    observed=[]
    monkeypatch.setattr('sys.argv',['axosim_demo','neuron-fidelity','--help'])
    monkeypatch.setattr(
        'axosim_demo.__main__.importlib.import_module',
        lambda name: SimpleNamespace(main=lambda: observed.append((name,sys.argv.copy()))),
    )
    main()
    assert observed==[('axosim_demo.neuron_fidelity',['axosim_demo neuron-fidelity','--help'])]

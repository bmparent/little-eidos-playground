import os
import numpy as np
import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import memory
from agents import get


@pytest.fixture
def mem_path(tmp_path, monkeypatch):
    path = tmp_path / '.eidos_memory.json'
    monkeypatch.setattr(memory, 'MEM_PATH', path)
    return path


@pytest.fixture
def github_token(monkeypatch):
    monkeypatch.setenv('GH_BOT_TOKEN', 'test')


def test_entropy_watcher_threshold(tmp_path, mem_path, monkeypatch):
    memory.save_memory({'timestamp': '2024', 'ck': np.array([], dtype=np.complex64),
                        'sigma2': 0.0,
                        'state_amplitudes': np.array([1,0], dtype=np.complex64)})
    monkeypatch.chdir(tmp_path)
    open('generated.eidos', 'w').close()
    watcher = get('entropy_watcher')
    watcher.BURST_GLYPH = 'H'
    watcher.main()
    with open('generated.eidos') as f:
        lines = [l.strip() for l in f if l.strip()]
    assert lines == ['H']


def test_branch_pruner_no_token(github_token, monkeypatch):
    monkeypatch.delenv('GH_BOT_TOKEN')
    pruner = get('branch_pruner')
    with pytest.raises(FileNotFoundError):
        pruner.main(['--repo', 'none/none'])

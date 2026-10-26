import json

import pytest

from paper_tools.reproduce_tools import do_reproduce_metric, make_tools
from paper_tools.spec_validation_tools import do_load_spec
from paper_tools.state import ReproState


@pytest.fixture
def session(tmp_path):
    spec = tmp_path / 'spec.yaml'
    spec.write_text('''name: sample
repo: https://example.com/paper
run:
  command: python evaluate.py
  gpu: false
metrics:
  accuracy:
    pattern: 'accuracy ([0-9.]+)'
    recorded: 88.88
    tolerance: 1.0
  missing:
    pattern: 'nothing ([0-9.]+)'
''')
    return ReproState(str(tmp_path / 'workspace')), str(spec)


def _seed_run(state, output):
    """Record a successful run so do_reproduce_metric reuses its log instead of calling Docker."""
    container = state.spec.image_tag()
    command = state.spec.run_command
    log_path = state.paper_dir('logs') + '/seeded.log'
    with open(log_path, 'w') as f:
        f.write(output)
    state.record_run(container, command, log_path, 0, 1.0)
    return container


def test_reproduce_reads_cached_run_and_persists(session):
    state, path = session
    do_load_spec(state, path)
    _seed_run(state, 'epoch done\naccuracy 89.40\n')
    result = do_reproduce_metric(state, 'accuracy')
    assert result['status'] == 'OK'
    assert result['value'] == 89.40
    assert result['reused_previous_run'] is True
    # The value is persisted where validate_metric looks for it.
    restored = ReproState(state.workspace)
    assert restored.data['reproduced'][state.key('accuracy')]['value'] == 89.40


def test_reproduce_reports_pattern_not_found(session):
    state, path = session
    do_load_spec(state, path)
    _seed_run(state, 'accuracy 89.40\n')
    result = do_reproduce_metric(state, 'missing')
    assert result['status'] == 'NOT_FOUND'
    assert state.key('missing') not in state.data['reproduced']


def test_unknown_metric_is_rejected(session):
    state, path = session
    do_load_spec(state, path)
    result = do_reproduce_metric(state, 'unknown')
    assert result['status'] == 'ERROR'
    assert 'unknown metric' in result['message']


def test_langchain_invocation(session):
    state, path = session
    do_load_spec(state, path)
    _seed_run(state, 'accuracy 89.40\n')
    tools = {tool.name: tool for tool in make_tools(state)}
    assert set(tools) == {'reproduce_metric'}
    reproduced = json.loads(tools['reproduce_metric'].invoke({'metric': 'accuracy'}))
    assert reproduced['status'] == 'OK'
    assert reproduced['value'] == 89.40
